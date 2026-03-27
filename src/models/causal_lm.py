import math
import time
from typing import Any, Dict

import hydra
import torch
from transformers import AutoModelForCausalLM

from src.models.base import LitModuleBase


class LitCausalLM(LitModuleBase):
    """LightningModule for causal language model pre-training.

    Wraps any HuggingFace causal LM. The architecture is fully specified via
    Hydra config: set model_params.config._target_ to any HF config class
    (e.g. transformers.LlamaConfig) and provide its kwargs alongside it.

    To load a pretrained checkpoint instead, set model_params.from_pretrained
    to a HF model ID or local path (model_params.config is then ignored).

    Batch convention: batch['input_ids'] of shape (B, T). HF shifts labels by 1
    internally, so no separate label tensor is needed.
    """

    def _initialize_models(self) -> list:
        p = self.hparams.model_params
        attn = p.get("attn_implementation", "sdpa")
        if p.get("from_pretrained"):
            model = AutoModelForCausalLM.from_pretrained(
                p["from_pretrained"], attn_implementation=attn
            )
        else:
            hf_config = hydra.utils.instantiate(p["config"])
            model = AutoModelForCausalLM.from_config(hf_config, attn_implementation=attn)

        self.model = model
        n_params = sum(param.numel() for param in model.parameters()) / 1e6
        print(f"[LitCausalLM] Parameters: {n_params:.1f}M")
        return [model]

    def setup(self, stage: str) -> None:
        if self.hparams["model_params"].get("compile", False) and stage == "fit":
            self.model = torch.compile(self.model)

    def model_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"]
        # Pass input_ids as labels — HF shifts by 1 internally.
        out = self.model(input_ids=input_ids, labels=input_ids)
        return out.loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch, "learn")
        ppl = math.exp(min(loss.item(), 20))
        self.log("learn/loss", loss, **self.logging_kwargs["learn"])
        self.log("learn/ppl", ppl, **self.logging_kwargs["learn"])
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss = self.model_step(batch, "val")
        ppl = math.exp(min(loss.item(), 20))
        self.log("val/loss", loss, **self.logging_kwargs["val"])
        self.log("val/ppl", ppl, **self.logging_kwargs["val"])

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss = self.model_step(batch, "test")
        ppl = math.exp(min(loss.item(), 20))
        self.log("test/loss", loss, **self.logging_kwargs["test"])
        self.log("test/ppl", ppl, **self.logging_kwargs["test"])

    # --- throughput logging ---

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t0
        tokens = batch["input_ids"].numel()
        self.log(
            "learn/tokens_per_sec",
            tokens / elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,  # per-process metric, not averaged across ranks
        )
