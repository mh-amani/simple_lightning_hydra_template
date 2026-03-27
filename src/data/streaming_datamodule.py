import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.dataset.streaming_text import StreamingTextDataset


class StreamingDatamodule(LightningDataModule):
    """Datamodule for streaming text datasets.

    Replaces BaseDatamodule for large-scale LM pre-training where the dataset
    does not fit in memory. Uses HuggingFace streaming datasets with on-the-fly
    tokenization and packing.
    """

    def __init__(
        self,
        dataset_name: str,
        seq_len: int,
        tokenizer_name: str,
        batch_size: int = 8,
        num_workers: int = 4,
        dataset_config_name: str = None,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "test",
        text_column: str = "text",
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self, split: str, shuffle: bool, rank: int = 0, world_size: int = 1) -> StreamingTextDataset:
        dataset = StreamingTextDataset(
            dataset_name=self.hparams.dataset_name,
            dataset_config_name=self.hparams.dataset_config_name,
            split=split,
            seq_len=self.hparams.seq_len,
            tokenizer_name=self.hparams.tokenizer_name,
            text_column=self.hparams.text_column,
            buffer_size=self.hparams.buffer_size if shuffle else 0,
            rank=rank,
            world_size=world_size,
        )
        return dataset

    def setup(self, stage=None):
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1

        self.train_dataset = self._make_dataset(self.hparams.train_split, shuffle=True, rank=rank, world_size=world_size)
        self.val_dataset = self._make_dataset(self.hparams.val_split, shuffle=False, rank=rank, world_size=world_size)
        self.test_dataset = self._make_dataset(self.hparams.test_split, shuffle=False, rank=rank, world_size=world_size)

    def _make_loader(self, dataset: StreamingTextDataset, num_workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        # Val/test splits often have only 1 shard. Using >1 worker causes HF to
        # SIGKILL the excess workers, which PyTorch surfaces as a segfault crash.
        return self._make_loader(self.val_dataset, num_workers=0)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, num_workers=0)
