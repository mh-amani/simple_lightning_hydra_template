from lightning import LightningDataModule
import hydra
from torch.utils.data import DataLoader
from typing import Optional


class BaseDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset_config,
        batch_size: int,
        num_workers: int,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
        test_split: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def _instantiate(self, split: Optional[str]):
        if split is not None:
            return hydra.utils.instantiate(self.dataset_config, split=split)
        return hydra.utils.instantiate(self.dataset_config)

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self._instantiate(self.train_split)
        self.val_dataset = self._instantiate(self.val_split)
        self.test_dataset = self._instantiate(self.test_split)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                            persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)