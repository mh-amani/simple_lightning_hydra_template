from typing import Type
import functools
from pytorch_lightning import Callback, LightningModule, Trainer
from src.utils.schedulers.abstract_scheduler import AbstractScheduler


class SchedulerCallback(Callback):
    """
    PL Callback to anneal a hyperparameter during the training of the model (e.g. temperature of the gumbel softmax)
    """

    def __init__(self, hyperparameter_location: str, hyperparameter_location_pl: str, scheduler: Type[AbstractScheduler]):
        """
        Args:
            hyperparameter_location: path to our hyperparameter in our model
            scheduler: type of scheduler we want to use
        """
        super().__init__()
        hyperaparam_path = hyperparameter_location.split('.')
        self.hyperparameter_path = hyperaparam_path[:-1] if len(hyperaparam_path) > 1 else []
        self.hyperparameter_name =  hyperaparam_path[-1]
        self.hyperparameter_layer = None
        
        # why do we need this?
        hyperparameter_pl_path = hyperparameter_location_pl.split('.')
        
        self.hyperparameter_pl_path = hyperparameter_pl_path[:-1] if len(hyperaparam_path) > 1 else []
        self.hyperparameter_pl_name = hyperparameter_pl_path[-1]
        self.hyperparameter_layer_pl = None
      
        self.scheduler = scheduler
        
        
    def hook_to_hyperparameter(self,pl_module: LightningModule) -> None:
        
        if len(self.hyperparameter_path) == 0:
            self.hyperparameter_layer = pl_module
        else:
            self.hyperparameter_layer = functools.reduce(getattr, self.hyperparameter_path, pl_module)
        
        if len(self.hyperparameter_pl_path) == 0:
            self.hyperparameter_layer_pl = pl_module.hparams
        else:
            self.hyperparameter_layer_pl = functools.reduce(getattr, self.hyperparameter_pl_path, pl_module.hparams)

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
        ) -> None:
        
        
        if self.hyperparameter_layer is None:
            self.hook_to_hyperparameter(pl_module)
        
        current_hyperparameter_val = getattr(self.hyperparameter_layer, self.hyperparameter_name)
        new_hyperparameter_value = self.scheduler.update(current_hyperparameter_val, pl_module.current_epoch)
        setattr(self.hyperparameter_layer, self.hyperparameter_name, new_hyperparameter_value)
        
        setattr(self.hyperparameter_layer_pl ,self.hyperparameter_pl_name ,current_hyperparameter_val)
        pl_module.save_hyperparameters()

        pl_module.log_dict({self.hyperparameter_name: new_hyperparameter_value, 'global_step': float(pl_module.global_step)})
        # wandb.log({self.hyperparameter_name: new_hyperparameter_value, 'global_step': pl_module.global_step})
        
