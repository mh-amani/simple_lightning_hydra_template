from src.models.base import LitModuleBase
import torch
from typing import Any, Dict, Tuple


class LinearLitModel(LitModuleBase):

    def __init__(
        self,
        model_params: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """Initialize a `LinearModel`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param collator: The collator to use for training.
        :param tokenizer: The tokenizer to use for training.
        """
        super().__init__(model_params, optimizer, scheduler)

            
    def _initialize_models(self,) -> None:
        self.model = torch.nn.Linear(
            in_features=self.hparams['model_params']['in_features'],
            out_features=self.hparams['model_params']['out_features'],
            bias=self.hparams['model_params'].get('bias', False)
        )
        return [self.model]


    def forward(self, batch, stage='learn') -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = batch['x']
        logits = self.model(x)
        # write the forward pass logic here if needed!
        loss = self.criterion(logits, batch['y'])
        return loss


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        self.forward(batch, stage)