from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from omegaconf import DictConfig


# PyTorch 2.6 changed the default of torch.load to weights_only=True, which breaks
# checkpoints that contain OmegaConf objects (e.g. saved hparams). This subclass
# restores the old behaviour for checkpoints we produce ourselves.
class SafeCheckpointIO(TorchCheckpointIO):
    def load_checkpoint(self, path, map_location=None, **kwargs):
        return torch.load(path, map_location=map_location, weights_only=False)


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks, # instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils import hydra_custom_resolvers

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # log.info("Instantiating loggers...")
    # logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    log.info(f"Instantiating logger <{cfg.get('logger')._target_}>")
    logger = hydra.utils.instantiate(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.strategy.checkpoint_io = SafeCheckpointIO()

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    
    if cfg.get("debug", False):
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))  # Or another port
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()


# to enable remote debugging, e.g., from VSCode you need to install `debugpy` package
# and set a breakpoint in your code, then run the script and attach the debugger; here's a config for the debugger:
# {
#             "name": "Python Debugger: Remote Attach",
#             "type": "debugpy",
#             "request": "attach",
#             "connect": {
#                 "host": "0.0.0.0",
#                 "port": 5678
#             },
#             // "pathMappings":[{"localRoot":"${workspaceFolder}","remoteRoot":"."}]
#             "pathMappings": [
#                 {
#                     "localRoot": "/dlabscratch1/amani/simple_hydra_template",
#                     "remoteRoot": "."
#                 }
#             ]
#         },