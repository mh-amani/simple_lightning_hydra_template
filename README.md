# simple-lightning-hydra-template

A PyTorch Lightning + Hydra template for reproducible sequence learning experiments.
All components (model, data, trainer, callbacks, logger) are instantiated from YAML configs via Hydra,
making it easy to swap architectures, datasets, and training setups without touching Python code.

Three working examples are included out of the box:
- **Parity RNN** — LSTM on a cumulative XOR task, runs on CPU in ~30 epochs, no data download needed
- **RNN LM** — LSTM language model on wikitext-2, same data interface as the transformer
- **Causal LM** — Llama-style transformer pre-trained on wikitext-103 (or any HuggingFace dataset), with single-GPU and multi-GPU (FSDP) support


## Quickstart

```bash
# clone project
git clone <repo-url>
cd simple_lightning_hydra_template

# create conda env
conda create -n myenv python=3.11 -y
conda activate myenv

# install dependencies
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your `WANDB_API_KEY`.
Set `entity` and `project` in `configs/train.yaml` to point at your WandB workspace.


## Running experiments

> **zsh note:** always single-quote tag arguments to prevent glob expansion:
> `'tags=["my_tag"]'`

**Parity RNN (CPU, no data download needed):**
```bash
python ./src/train.py experiment=parity_rnn trainer=cpu 'tags=["parity"]'
```

**RNN LM on wikitext-2 (CPU smoke-test):**
```bash
# one-time data download (~5 MB)
python scripts/prepare_data/prepare_text_dataset.py
python ./src/train.py experiment=rnn_lm trainer=cpu 'tags=["rnn_lm"]'
```

**Causal LM — CPU smoke-test:**
```bash
python ./src/train.py experiment=lm_demo trainer=cpu data.batch_size=4 'tags=["lm_demo"]'
```

**Causal LM — single GPU with bf16:**
```bash
python ./src/train.py experiment=lm_demo trainer=gpu_bf16 'tags=["lm_demo"]'
```

**Causal LM — multi-GPU with FSDP:**
```bash
python ./src/train.py experiment=lm_demo trainer=fsdp 'tags=["lm_demo"]'
```

#### Pre-training cost estimates (default ~500M Llama config)

The default [configs/model/causal_lm.yaml](configs/model/causal_lm.yaml) is a ~500M parameter Llama
(24 layers, hidden=1024, intermediate=4096). Measured throughput on A100 is **30–40k tokens/sec per GPU**
with `bf16` and `torch.compile`.

[Chinchilla](https://arxiv.org/abs/2203.15556) scaling laws suggest ~**20 tokens per parameter** for a
compute-optimal run, which gives **~10B tokens** for this model size.

| Setup | Tokens/sec | Steps to 10B tokens | Wall time |
|---|---|---|---|
| 1× A100 (batch=16, seq=1024) | ~35k | ~610k | ~80 h |
| 8× A100 (FSDP, same per-GPU batch) | ~280k | ~76k | ~10 h |

Tokens per step = `batch_size × seq_len` (default 16 × 1024 = 16,384). Scale `trainer.max_steps`
accordingly; the `learn/tokens_per_sec` metric logged during training lets you verify throughput.

**Scale to a larger dataset (e.g. OpenWebText):**
```bash
python ./src/train.py experiment=lm_demo trainer=gpu_bf16 \
  data.dataset_name=Skylion007/openwebtext data.dataset_config_name=null \
  data.val_split=train data.test_split=train \
  trainer.max_steps=100000 'tags=["lm_owt"]'
```

**Inline overrides and sweeps:**
```bash
# override any config value on the command line
python ./src/train.py experiment=parity_rnn trainer=cpu trainer.max_epochs=100 model.optimizer.lr=0.01

# Hydra multirun sweep over learning rates
python ./src/train.py --multirun model.optimizer.lr=0.001,0.0001

# skip the test phase
python ./src/train.py experiment=parity_rnn trainer=cpu test=false
```


## Architecture

```
configs/
├── train.yaml              # root config — composes all groups
├── experiment/             # per-experiment overrides (parity_rnn, lm_demo, …)
├── model/                  # model configs (_target_, model_params, optimizer, scheduler)
├── data/                   # datamodule configs; dataset/ holds sub-group configs
├── trainer/                # default (CPU), gpu, gpu_bf16, fsdp
└── callbacks/              # basic (checkpoint + early stopping + summary), default, none

src/
├── train.py                # entry point — instantiates everything from cfg, runs fit + test
├── models/
│   ├── base.py             # LitModuleBase — base LightningModule
│   ├── rnn.py              # RNNLitModel (parity), RNNLitLM (text LM)
│   └── causal_lm.py        # LitCausalLM — HuggingFace causal LM wrapper
└── data/
    ├── base_datamodule.py          # BaseDatamodule — standard (in-memory) datasets
    ├── streaming_datamodule.py     # StreamingDatamodule — HF streaming for large corpora
    └── dataset/
        ├── parity.py               # ParityDataset — on-the-fly XOR sequences
        ├── text.py                 # TextDataset — map-style, loads from HF disk format
        └── streaming_text.py       # StreamingTextDataset — tokenize + pack on-the-fly
```

### Adding a new experiment

1. Create `configs/model/<name>.yaml` pointing `_target_` at your `LitModuleBase` subclass.
2. Create `configs/data/dataset/<name>.yaml` with `dataset_config._target_` pointing at your dataset class.
3. Create `configs/experiment/<name>.yaml`:
   ```yaml
   # @package _global_
   defaults:
     - override /model: <name>
     - override /data/dataset@data: <name>   # @data package annotation is required
     - override /callbacks: basic
   ```
4. Run with `python ./src/train.py experiment=<name> trainer=cpu 'tags=["<tag>"]'`

### Implementing a model

Subclass `LitModuleBase` ([src/models/base.py](src/models/base.py)) and override:
- `_initialize_models()` — create `nn.Module` objects as `self.<name>`, return them in a list assigned to `self.parametric_models`
- `forward(batch, stage)` and `model_step(batch, stage)` — or override `training_step`/`validation_step`/`test_step` directly if you need to log extra metrics (see [src/models/rnn.py](src/models/rnn.py))

Use `self.logging_kwargs[stage]` for consistent `on_step`/`on_epoch`/`sync_dist` kwargs when calling `self.log(...)`.

### Streaming data

`StreamingDatamodule` + `StreamingTextDataset` handle large corpora that don't fit in memory:
- data is fetched from HuggingFace in streaming mode and tokenized on-the-fly
- tokens are packed into non-overlapping chunks of `seq_len`; each document ends with `eos_token`
- use step-based training (`trainer.max_steps`) and `val_check_interval` instead of epoch-based
- val/test dataloaders always use `num_workers=0` — HF streaming splits often have only one shard and will crash if given multiple workers

### Map-style text data

`TextDataset` is the in-memory alternative for datasets that fit in RAM (e.g. wikitext-2):
- download once with `scripts/prepare_data/prepare_text_dataset.py`, which saves via `datasets.save_to_disk()`
- all tokenization and packing happens at init time; random access and full epoch shuffling work normally
- returns the same `{'input_ids': tensor(seq_len,)}` interface as `StreamingTextDataset`, so models are interchangeable
- `BaseDatamodule` accepts `train_split`/`val_split`/`test_split` params to route the correct HF split to each dataloader

### Trainer presets

| Config | Accelerator | Precision | Strategy |
|---|---|---|---|
| `trainer=cpu` | CPU | fp32 | — |
| `trainer=gpu` | GPU | fp32 | — |
| `trainer=gpu_bf16` | GPU | bf16-mixed | — |
| `trainer=fsdp` | GPU (multi) | bf16-mixed | FSDP |

### Remote debugging (VSCode)

```bash
python ./src/train.py debug=true
# then attach the VSCode debugger on port 5678
# add this piece to .vscode/launch.json:
```
```json
{
  "name": "Remote Debug",
  "type": "python",
  "request": "attach",
  "connect": {
    "host": "localhost",
    "port": 5678
  },
  "pathMappings": [
    {
      "localRoot": "${workspaceFolder}",
      "remoteRoot": "/app/src"
    }
  ]
}
```


## Known issues / placeholders

| Location | Status |
|---|---|
| `configs/model/mlp.yaml` | Empty — MLP config not yet written |
| `configs/model/default.yaml` | Points to a non-existent class; do not use as a base |
| `src/utils/schedulers/` | Module missing — needed by `SchedulerCallback`; use `callbacks: basic` to avoid it |
