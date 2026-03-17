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
