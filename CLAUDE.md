# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the parity example (works on CPU, no data download needed):**
```bash
python ./src/train.py experiment=parity_rnn trainer=cpu 'tags=["parity"]'
```

**Run with specific overrides:**
```bash
# Override config values inline
python ./src/train.py experiment=parity_rnn trainer=cpu trainer.max_epochs=100 model.optimizer.lr=0.01

# Hydra multirun sweep
python ./src/train.py --multirun model.optimizer.lr=0.001,0.0001

# Skip testing after training
python ./src/train.py experiment=parity_rnn trainer=cpu test=false
```

**Remote debugging (VSCode):**
```bash
python ./src/train.py debug=true
# Then attach debugger in VSCode on port 5678
```

**Generate data (parity/XOR sequences):**
```bash
python scripts/create_data/cumulative_parity_dataset.py --local_dir data/parity --train_size 1024 --length 256 --bit_width 1 --test_size 512
```

**Submit cluster jobs (Apple Bolt):**
```bash
cd experiments
python bolt/bolt_submit.py 0 1 2             # submit experiments with IDs 0, 1, 2
python bolt/bolt_submit.py all               # submit all
python bolt/bolt_submit.py 0 1 --dry-run     # print without submitting
```

## Architecture

This is a PyTorch Lightning + Hydra template for ML experiments focused on **sequence learning**. All components are instantiated via Hydra from YAML configs.

### Config system (`configs/`)

- `configs/train.yaml` — root config; composes `data`, `model`, `callbacks`, `trainer`, `logger`
- `configs/experiment/` — experiment overrides; use `experiment=<name>` to apply
- `configs/model/` — model configs with `_target_`, `model_params`, `optimizer`, `scheduler`
- `configs/data/` — datamodule config; `dataset` is a required sub-group loaded with `@_here_` package merge (see Hydra notes)
- `configs/trainer/` — `default.yaml` (CPU base), `gpu.yaml` (overrides accelerator to GPU), `cpu.yaml`
- `configs/callbacks/` — individual callback configs; `basic.yaml` composes checkpoint + early stopping + summary; `default.yaml` additionally includes the curriculum learning scheduler

Hydra outputs run artifacts to `logs/<task_name>/runs/<timestamp>/`. WandB is the default logger.

**WandB setup:** Update `entity` and `project` in `configs/train.yaml` (defaults are `epfl-dlab` / `sigmae`). Set `WANDB_API_KEY` in a `.env` file (see `.env.example`). In zsh, always single-quote tag arguments: `'tags=["my_tag"]'`.

### Source code (`src/`)

**Models** — extend `LitModuleBase` ([src/models/base.py](src/models/base.py)):
- Override `_initialize_models()` — create all `nn.Module` objects as `self.<name>`, return them as a list in `self.parametric_models`
- Override `forward(batch, stage)` and `model_step(batch, stage)` for compute logic; or override the individual `training_step` / `validation_step` / `test_step` directly (as `RNNLitModel` does) to log extra metrics
- `configure_optimizers` instantiates `self.hparams.optimizer` and optionally `self.hparams.scheduler` via Hydra; set `scheduler: null` in model config to disable
- Scheduler config supports a `scheduler_config` sub-key for Lightning scheduler dict options (e.g. `monitor`, `interval`)
- Logging stages: `'learn'` (training), `'val'`, `'test'` — use `self.logging_kwargs[stage]` for consistent `on_step`/`on_epoch`/`sync_dist` kwargs
- `on_load_checkpoint` resets optimizer and scheduler states — useful for fine-tuning from a checkpoint without resuming the optimizer
- `setup()` compiles all models in `self.parametric_models` if `model_params.compile=True`
- Batch convention: batch is a dict, e.g. `batch['sequence']`, `batch['cot']`, `batch['x']`, `batch['y']`

**Data** — extend `BaseDatamodule` ([src/data/base_datamodule.py](src/data/base_datamodule.py)):
- Constructor takes `dataset_config`, `batch_size`, `num_workers`
- `setup()` calls `hydra.utils.instantiate(self.dataset_config)` to build train/val/test datasets (all three get independent instances of the same dataset class)
- `persistent_workers` is enabled automatically when `num_workers > 0`

**Datasets** ([src/data/dataset/](src/data/dataset/)):
- `ParityDataset` — generates cumulative XOR/parity sequences on-the-fly (no files needed); returns `{sequence: (T,), cot: (T-1,), answer: scalar}`
- Parquet-based datasets are loaded via `datasets.load_dataset` (see `sphere.yaml` for the config pattern)

**Training entry point** — [src/train.py](src/train.py):
- Instantiates datamodule, model, callbacks, logger, trainer from `cfg`
- Sets `SafeCheckpointIO` on the trainer strategy for PyTorch 2.6+ compatibility (see note below)
- Runs `trainer.fit()` if `cfg.train=True`, then `trainer.test()` from best checkpoint if `cfg.test=True`
- Returns metric dict for Hydra Optuna sweeper compatibility

**Reference experiment** — [src/models/rnn.py](src/models/rnn.py) + [configs/experiment/parity_rnn.yaml](configs/experiment/parity_rnn.yaml):
- LSTM that reads a binary sequence one bit at a time and predicts the running XOR at each step
- Logs `loss` and `acc` per stage; reaches ~99% accuracy in ~30 epochs on CPU
- Shows how to override `training_step`/`validation_step`/`test_step` when `model_step` returns more than just loss

### Callbacks ([src/callbacks/](src/callbacks/))

- `SchedulerCallback` — anneals an arbitrary model attribute (by dotted path) each epoch using a scheduler from `src/utils/schedulers/`
- `SupervisionProbabilitySchedulerCallback` — anneals a dict of batch-type probabilities stored in `pl_module.batch_probabilities`; useful for curriculum learning

Both require `src/utils/schedulers/` (not yet implemented — see Known Issues). Use `callbacks: basic` in experiments to avoid this.

### Custom OmegaConf Resolvers ([src/utils/hydra_custom_resolvers.py](src/utils/hydra_custom_resolvers.py))

Registered resolvers usable in YAML configs:
- `${add:a,b}`, `${add_int:a,b}` — addition (float or int)
- `${mult:a,b}`, `${mult_int:a,b}` — multiplication (float or int)
- `${floor_div:a,b}`, `${float_div:a,b}` — division
- `${num_files:path}` — count files in a directory (0 if not exists)
- `${as_tuple:a,b,...}` — Python tuple
- `${get_class_from_name:dotted.path}` — import and return a class
- `${get_dict_except:cfg,key1,key2}` — config copy without specified keys
- `${get_method:dotted.path}` — get a callable by dotted path
- `${get_obj_attr:cfg,attr1,attr2}` — instantiate from cfg, return first non-null attribute
- `${get_module_attr:module.attr}` — import a module attribute at runtime

### Experiment tracking

- `experiments/exps.jsonl` — flat registry; format `id: python src/train.py ...`, lines starting with `//` are comments
- `experiments/bolt/` — Apple Bolt cluster scripts; `bolt_submit.py` reads `exps.jsonl`, fills a YAML template, submits via `bolt task submit`
- `experiments/runai/` — RunAI cluster alternative

### Adding a new experiment

1. Create `configs/model/<name>.yaml` with `_target_` pointing to your `LitModuleBase` subclass
2. Create `configs/data/dataset/<name>.yaml` with `dataset_config._target_` pointing to your dataset class
3. Create `configs/experiment/<name>.yaml`:
   ```yaml
   # @package _global_
   defaults:
     - override /model: <name>
     - override /data/dataset@data: <name>   # note the @data package annotation
     - override /callbacks: basic
   ```
4. Run with `python ./src/train.py experiment=<name> trainer=cpu 'tags=["<tag>"]'`

### Key notes

- `rootutils` sets `PROJECT_ROOT` env var from the `.project-root` marker file; all config paths resolve relative to it
- `extras.enforce_tags=True` in `configs/train.yaml` prompts for tags if none provided — always pass `'tags=["<tag>"]'` on the CLI (single-quoted in zsh to prevent glob expansion)
- Checkpoints saved to `${paths.output_dir}/checkpoints/`, monitoring `${model.model_params.monitor}` (set in the model config)
- LR is logged at every validation epoch end as `lr`
- Training logs go to `logs/<task_name>/runs/<timestamp>/`; WandB also logs everything

### Hydra dataset package annotation

`configs/data/default.yaml` uses `dataset@_here_: ???` in its defaults list. The `@_here_` directive tells Hydra to merge the dataset sub-config's keys directly into the `data` package (rather than nesting them under `data.dataset`). This is what makes `cfg.data.dataset_config` work correctly.

Consequence: when overriding the dataset in an experiment, you must include the package annotation in the override key:
```yaml
- override /data/dataset@data: parity   # correct
- override /data/dataset: parity         # wrong — Hydra won't find the group
```

### PyTorch 2.6+ checkpoint loading

PyTorch 2.6 changed `torch.load` to default `weights_only=True`, which fails for checkpoints that contain `OmegaConf` objects (saved hparams). `train.py` installs a `SafeCheckpointIO` on the trainer strategy after instantiation to restore `weights_only=False` for our own checkpoints. This is transparent and requires no config changes.

## Known Issues / Placeholders

| Location | Issue |
|---|---|
| `configs/model/mlp.yaml` | Empty file — MLP config not yet written |
| `configs/model/default.yaml` | Points to `src.model.base_lightning_module.BaseLightningModule` which does not exist; do not use as a base |
| `src/utils/schedulers/` | Entire module missing — needed by `SchedulerCallback` and `SupervisionProbabilitySchedulerCallback`; needs `AbstractScheduler` and `DictScheduler` |
