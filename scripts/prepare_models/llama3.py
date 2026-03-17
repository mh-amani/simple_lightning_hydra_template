

import argparse
import os
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast   
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--local_dir", default="./models/llama/")

    args = parser.parse_args()
    local_dir = args.local_dir

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # last_layer_weights = model.lm_head.weight
    # embedding_weights = model.model.embed_tokens.weight

    # 128000: AddedToken("<|begin_of_text|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	# 128001: AddedToken("<|end_of_text|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	# 128002: AddedToken("<|reserved_special_token_0|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    # ....
    # Note that after 120001, all the embeddings are unused and they are the same!
    # we choose 128002 to be the pad token

    PAD = "<|reserved_special_token_0|>"
    tokenizer.pad_token = PAD
    tokenizer.add_special_tokens({"pad_token": PAD})
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    model.config.pad_token_id = pad_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = pad_id

    # sample some text to verify the tokenizer and model work well
    sample_text_1 = "Hello, how are you?"
    sample_text_2 = "The quick brown fox jumps over the lazy dog."
    # batch encoding
    inputs = tokenizer([sample_text_1, sample_text_2], padding=True, return_tensors="pt")
    outputs = model(**inputs)
    print("Tokenized input IDs:", inputs['input_ids'], inputs['attention_mask'])
    print("Model outputs (logits) shape:", outputs.logits.shape)

    # save the model and tokenizer
    dir_to_save = local_dir + args.model_path.split("/")[-1]
    model.save_pretrained(dir_to_save)
    tokenizer.save_pretrained(dir_to_save)

    print(f"Model saved to {dir_to_save}")
    print(f"New tokenizer saved to {dir_to_save}")