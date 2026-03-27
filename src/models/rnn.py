import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.models.base import LitModuleBase


class RNNLitModel(LitModuleBase):
    """LSTM model for sequence-to-sequence tasks.

    Demonstrated on the cumulative parity task:
      input:  binary sequence  [b_0, b_1, ..., b_{T-1}]
      target: running XOR      [b_0, b_0^b_1, ..., b_0^...^b_{T-1}]

    The model processes one bit per step and predicts the running parity at
    each position, so loss and accuracy are computed over the full sequence.

    Batch keys expected:
      batch['sequence']  — (B, T) int64 tensor of input bits
      batch['cot']       — (B, T-1) int64 tensor of running parities at steps 1..T-1
    """

    def _initialize_models(self):
        p = self.hparams['model_params']
        num_layers = p.get('num_layers', 1)
        self.rnn = nn.LSTM(
            input_size=p['input_size'],
            hidden_size=p['hidden_size'],
            num_layers=num_layers,
            batch_first=True,
            dropout=p.get('dropout', 0.0) if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(p['hidden_size'], 1)
        self.criterion = nn.BCEWithLogitsLoss()
        return [self.rnn, self.head]

    def forward(self, batch, stage='learn') -> torch.Tensor:
        x = batch['sequence'].float().unsqueeze(-1)  # (B, T, 1)
        out, _ = self.rnn(x)                          # (B, T, hidden_size)
        logits = self.head(out).squeeze(-1)            # (B, T)
        return logits

    def model_step(self, batch, stage) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(batch, stage)          # (B, T)

        # Build full target: running parity at every step.
        # cot contains parities for steps 1..T-1; prepend parity at step 0
        # which equals the first input bit itself.
        first   = batch['sequence'][:, 0:1].float()  # (B, 1)
        cot     = batch['cot'].float()               # (B, T-1)
        targets = torch.cat([first, cot], dim=1)     # (B, T)

        loss = self.criterion(logits, targets)
        acc  = ((logits > 0).float() == targets).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, acc = self.model_step(batch, stage='learn')
        self.log('learn/loss', loss, **self.logging_kwargs['learn'])
        self.log('learn/acc',  acc,  **self.logging_kwargs['learn'])
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, acc = self.model_step(batch, stage='val')
        self.log('val/loss', loss, **self.logging_kwargs['val'])
        self.log('val/acc',  acc,  **self.logging_kwargs['val'])

    def test_step(self, batch, batch_idx: int) -> None:
        loss, acc = self.model_step(batch, stage='test')
        self.log('test/loss', loss, **self.logging_kwargs['test'])
        self.log('test/acc',  acc,  **self.logging_kwargs['test'])


class RNNLitLM(LitModuleBase):
    """LSTM language model for next-token prediction on text.

    Uses the same batch interface as LitCausalLM — batch['input_ids'] of shape
    (B, T) — so it is interchangeable with any text datamodule. Unlike
    LitCausalLM the label shift is done explicitly here (HF does it internally).

    Batch keys expected:
      batch['input_ids']  — (B, T) int64 token ids

    Config (model_params):
      vocab_size    — vocabulary size (must match the tokenizer used for data)
      embed_size    — token embedding dimension
      hidden_size   — LSTM hidden dimension
      num_layers    — number of LSTM layers (default 2)
      dropout       — dropout between LSTM layers when num_layers > 1 (default 0)
    """

    def _initialize_models(self):
        p = self.hparams['model_params']
        num_layers = p.get('num_layers', 2)
        self.embedding = nn.Embedding(p['vocab_size'], p['embed_size'])
        self.rnn = nn.LSTM(
            input_size=p['embed_size'],
            hidden_size=p['hidden_size'],
            num_layers=num_layers,
            batch_first=True,
            dropout=p.get('dropout', 0.0) if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(p['hidden_size'], p['vocab_size'])
        self.criterion = nn.CrossEntropyLoss()
        return [self.embedding, self.rnn, self.head]

    def forward(self, batch, stage='learn') -> torch.Tensor:
        x = self.embedding(batch['input_ids'])  # (B, T, embed_size)
        out, _ = self.rnn(x)                     # (B, T, hidden_size)
        return self.head(out)                     # (B, T, vocab_size)

    def model_step(self, batch, stage) -> torch.Tensor:
        input_ids = batch['input_ids']           # (B, T)
        logits = self.forward(batch, stage)      # (B, T, vocab_size)
        # predict token t+1 from context 0..t
        loss = self.criterion(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch, stage='learn')
        self.log('learn/loss', loss, **self.logging_kwargs['learn'])
        self.log('learn/ppl', math.exp(min(loss.item(), 20)), **self.logging_kwargs['learn'])
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss = self.model_step(batch, stage='val')
        self.log('val/loss', loss, **self.logging_kwargs['val'])
        self.log('val/ppl', math.exp(min(loss.item(), 20)), **self.logging_kwargs['val'])

    def test_step(self, batch, batch_idx: int) -> None:
        loss = self.model_step(batch, stage='test')
        self.log('test/loss', loss, **self.logging_kwargs['test'])
        self.log('test/ppl', math.exp(min(loss.item(), 20)), **self.logging_kwargs['test'])
