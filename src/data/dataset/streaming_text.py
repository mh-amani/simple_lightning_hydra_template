import torch
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer


class StreamingTextDataset(IterableDataset):
    """Streams text from a HuggingFace dataset, tokenizes on-the-fly, and packs
    tokens into fixed-length chunks for next-token prediction.

    Returns batches of {'input_ids': (seq_len,)} where the model is expected to
    predict each token from the previous ones (labels = input_ids, shifted internally
    by the HuggingFace causal LM loss).

    When rank/world_size are provided, the underlying HF dataset is sharded across
    ranks so each GPU sees a unique subset of the data.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        seq_len: int,
        tokenizer_name: str,
        dataset_config_name: str = None,
        text_column: str = "text",
        buffer_size: int = 1000,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.seq_len = seq_len
        self.text_column = text_column
        self.buffer_size = buffer_size
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # We do our own chunking, so disable the tokenizer's length warning.
        self.tokenizer.model_max_length = int(1e9)

    def __iter__(self):
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        if self.world_size > 1:
            ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)
        if self.buffer_size > 0:
            ds = ds.shuffle(buffer_size=self.buffer_size, seed=42)

        buffer = []
        for example in ds:
            text = example[self.text_column]
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.tokenizer.eos_token_id)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}
