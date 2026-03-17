from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerBase
from typing import Dict, Tuple

import os
import argparse
from datasets import load_dataset
import datasets
import json

import code


def batch_iterator(dataset, key, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]

def create_tokenizer(dataset, tokenizer_type="word_level", **kwargs):
    key = kwargs['key']
    
    if tokenizer_type == "word_level":
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        trainer = trainers.WordLevelTrainer(special_tokens=list(kwargs['special_tokens']), vocab_size=kwargs['max_vocab_size'])
    elif tokenizer_type == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(special_tokens=list(kwargs['special_tokens']), vocab_size=kwargs['max_vocab_size'])
    else:
        raise ValueError("Invalid tokenizer_type. Choose 'word_level' or 'unigram'.")

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    iterator = batch_iterator(dataset, key=key, batch_size=kwargs['batch_size'])
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(dataset))
    
    # Add special tokens
    tokenizer.add_special_tokens(kwargs['special_tokens'])
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    from tokenizers import processors
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        pair=f"[BOS]:0 $A:0 [EOS]:0 $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_token_id), ("[EOS]", eos_token_id)],
    )
    
    # Create a CustomPreTrainedTokenizer
    from transformers import PreTrainedTokenizerFast
    custom_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]"
    )


    return custom_tokenizer

# Usage:
# tokenizer, special_tokens = create_tokenizer(dataset, tokenizer_type="word_level", key='text', batch_size=1000, special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], max_vocab_size=30000)
# tokenizer.save_pretrained("./tokenizer")

# Later, to load:
# loaded_tokenizer = CustomPreTrainedTokenizer.from_pretrained("./tokenizer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer on a dataset")
    parser.add_argument("--tokenizer_type", choices=["word", "unigram"], required=True, help="Type of tokenizer to train")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset or HuggingFace dataset name")
    parser.add_argument("--save_path", required=True, help="Path to save the trained tokenizer")
    parser.add_argument("--key", default="input", help="Key for the text column in the dataset")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training")
    parser.add_argument("--max_vocab_size", type=int, default=30000, help="Maximum vocabulary size")
    parser.add_argument("--special_tokens", nargs="+", default=["[PAD]", "[UNK]", "[BOS]", "[EOS]"], help="Special tokens")
    parser.add_argument("--dataset_config", type=str, default="{}", help="JSON string of dataset configuration")
    
    args = parser.parse_args()

    # Parse dataset config
    dataset_config = json.loads(args.dataset_config)

    # Load dataset
    if os.path.exists(args.dataset_path):
        try: 
            dataset = datasets.load_from_disk(args.dataset_path)["train"]
        except:
            dataset = load_dataset('csv', data_files=args.dataset_path, **dataset_config)["train"]
    else:
        dataset = load_dataset(args.dataset_path, **dataset_config)["train"]

    # dataset_name_or_path = args.dataset_path
    # if isinstance(dataset_name_or_path, str) or (os.path.isdir(dataset_name_or_path) and os.path.exists(os.path.join(dataset_name_or_path, 'dataset_dict.json'))): 
    #     try:
    #         dataset = datasets.load_from_disk(dataset_name_or_path)
    #     except:
    #         dataset = load_dataset(dataset_name_or_path, trust_remote_code=True)
    # else:
    #     # Handle file path loading
    #     if os.path.isfile(dataset_name_or_path):
    #         # Single file case
    #         file_extension = os.path.splitext(dataset_name_or_path)[1].lower()
    #         if file_extension in ['.csv', '.tsv']:
    #             dataset = load_dataset('csv', data_files=dataset_name_or_path)
    #         else:
    #             raise ValueError(f"Unsupported file type: {file_extension}")
    #     elif 'data_files' in kwargs:
    #         # Multiple files case
    #         for split, file_name in kwargs['data_files'].items():
    #             kwargs['data_files'][split] = os.path.join(dataset_name_or_path, file_name)
    #         kwargs['data_files'] = dict(kwargs['data_files']) # resolve the paths. does nothing but without it load_dataset fails
    #         dataset = load_dataset('csv', **kwargs)
    #     else:
    #         raise ValueError("Invalid dataset_name_or_path or missing data_files in kwargs")

    # code.interact(local=locals())
    # Train tokenizer
    if args.tokenizer_type == "word":
        tokenizer = create_tokenizer(dataset, tokenizer_type="word_level", key=args.key, batch_size=args.batch_size, 
                                             max_vocab_size=args.max_vocab_size, special_tokens=args.special_tokens)
    else:
        tokenizer = create_tokenizer(dataset, tokenizer_type="unigram", key=args.key, batch_size=args.batch_size, 
                                           max_vocab_size=args.max_vocab_size, special_tokens=args.special_tokens)

    # Save tokenizer
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    tokenizer.save_pretrained(args.save_path)
    print(f"Tokenizer saved to {args.save_path}")

