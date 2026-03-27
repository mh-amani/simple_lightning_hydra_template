import datasets
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Map-style dataset that loads a pre-saved HF dataset from disk, tokenizes
    all text at init time, and packs tokens into fixed-length chunks.

    Returns {'input_ids': tensor(seq_len,)}, the same interface as
    StreamingTextDataset, so it works with any model that expects input_ids
    (e.g. LitCausalLM, RNNLitLM).

    Usage:
        # Save a dataset to disk first (see scripts/prepare_data/prepare_text_dataset.py)
        # Then point this class at the saved directory and the desired split.
    """

    def __init__(
        self,
        path: str,
        split: str,
        seq_len: int,
        tokenizer_name: str,
        text_column: str = "text",
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = int(1e9)  # suppress truncation warnings

        ds = datasets.load_from_disk(path)[split]

        # Tokenize all documents and concatenate with eos separators.
        all_ids: list[int] = []
        for example in ds:
            text = example[text_column]
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            all_ids.extend(ids)

        # Discard the trailing remainder that doesn't fill a full chunk.
        n_chunks = len(all_ids) // seq_len
        tokens = torch.tensor(all_ids[: n_chunks * seq_len], dtype=torch.long)
        self.chunks = tokens.view(n_chunks, seq_len)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.chunks[idx]}
