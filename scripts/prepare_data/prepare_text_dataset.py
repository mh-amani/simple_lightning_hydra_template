#!/usr/bin/env python3
"""Download a text dataset from HuggingFace and save it to disk in HF format.

The saved dataset can then be loaded by TextDataset (src/data/dataset/text.py)
via datasets.load_from_disk(), which is faster and more reliable than streaming
for datasets that fit in memory.

Examples:
    # wikitext-2 (small, ~2MB, good for quick experiments)
    python scripts/prepare_data/prepare_text_dataset.py

    # wikitext-103 (medium, ~500MB)
    python scripts/prepare_data/prepare_text_dataset.py \
        --dataset Salesforce/wikitext \
        --config wikitext-103-raw-v1 \
        --output_dir ./data/wikitext103
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download and save an HF text dataset to disk.")
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="HuggingFace dataset name")
    parser.add_argument("--config", default="wikitext-2-raw-v1", help="Dataset config/subset name")
    parser.add_argument("--output_dir", default="./data/wikitext2", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        print(f"Output directory already exists: {out}. Delete it to re-download.")
        return

    print(f"Downloading {args.dataset} ({args.config})...")
    ds = load_dataset(args.dataset, args.config)

    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"Saved to {out}/")
    for split, d in ds.items():
        print(f"  {split}: {len(d):,} examples")


if __name__ == "__main__":
    main()
