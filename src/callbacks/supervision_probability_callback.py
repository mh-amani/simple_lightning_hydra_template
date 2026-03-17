from typing import Type
from pytorch_lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from src.utils.schedulers.abstract_scheduler import AbstractScheduler

class SupervisionProbabilitySchedulerCallback(Callback):
    """
    PL Callback to anneal supervision probabilities for different dataset types during training.
    Uses a dictionary-based scheduler to handle multiple batch types.
    """

    def __init__(self, scheduler: Type[AbstractScheduler]):
        """
        Args:
            scheduler: Scheduler instance that handles dictionary of probabilities
        """
        super().__init__()
        self.scheduler = scheduler

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        """
        Update batch probabilities at the start of each training epoch.
        """
        # Get current probabilities from the model
        current_probs = pl_module.batch_probabilities
        
        # Update all probabilities using the scheduler
        updated_probs = self.scheduler.update(current_probs, pl_module.current_epoch)

        # make sure probabilities sum to 1
        total_prob = sum(updated_probs.values())
        if total_prob > 0:
            updated_probs = {key: value / total_prob for key, value in updated_probs.items()}
        
        # Set the updated probabilities back to the model
        pl_module.batch_probabilities = updated_probs
        
        # Log all probabilities
        log_dict = {f'prob_{key}': value for key, value in updated_probs.items()}
        log_dict['global_step'] = float(pl_module.global_step)
        pl_module.log_dict(log_dict)