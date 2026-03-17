

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
    parser.add_argument("--model_path", default="masani/SFT_cumulative_parity_length_16_bitwidth_1_1024_512_Llama-3.2-1B_epoch_3_global_step_12")
    parser.add_argument("--local_dir", default="./data/models/binary_SFT_cumulative_parity_length_16_bitwidth_1_1024_512_Llama-3.2-1B_epoch_3_global_step_12")
    parser.add_argument("--token_ids_to_keep", default=[15, 16, 220])
    parser.add_argument("--repo_id", default=None)

    args = parser.parse_args()

    model_path = args.model_path
    local_dir = args.local_dir
    token_ids_to_keep = args.token_ids_to_keep
    repo_id = args.repo_id

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    last_layer_weights = model.lm_head.weight
    embedding_weights = model.model.embed_tokens.weight

    additional_token_ids = [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    token_ids_to_keep = token_ids_to_keep + additional_token_ids

    new_embedding_weights = embedding_weights[token_ids_to_keep]
    new_last_layer_weights = last_layer_weights[token_ids_to_keep]

    # remove the unused tokens
    model.resize_token_embeddings(len(token_ids_to_keep))

    # update the weights
    model.model.embed_tokens.weight = torch.nn.Parameter(new_embedding_weights)
    model.lm_head.weight = torch.nn.Parameter(new_last_layer_weights)

    # Create vocabulary mapping
    vocab = {}
    for new_id, old_id in enumerate(token_ids_to_keep):
        token = tokenizer._convert_id_to_token(old_id)
        vocab[token] = new_id
    
    # Create new tokenizer from vocabulary
    vocab_keep_items = len(token_ids_to_keep)

    assert tokenizer.is_fast, "This only works for fast tokenizers."
    # directly changing the json file of the tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    # vocab = tokenizer_json["model"]["vocab"]
    if tokenizer_json["model"]["type"] == "BPE":
        new_vocab = vocab
        merges = tokenizer_json["model"]["merges"]
        new_merges = []
        for i in range(len(merges)):
            a, b = merges[i]
            new_token = "".join((a, b))
            if a in new_vocab and b in new_vocab and new_token in new_vocab:
                new_merges.append(merges[i])
        tokenizer_json["model"]["merges"] = new_merges
    elif tokenizer_json["model"]["type"] == "Unigram":
        new_vocab = vocab[:vocab_keep_items]
    elif tokenizer_json["model"]["type"] == "WordPiece" or tokenizer_json["model"]["type"] == "WordLevel":
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
    else:
        raise ValueError(f"don't know how to handle {tokenizer_json['model']['type']}")
    tokenizer_json["model"]["vocab"] = new_vocab
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    from tokenizers.processors import TemplateProcessing

    # Create a new TemplateProcessing object with correct special token IDs
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"<|begin_of_text|> $A",
        pair=f"<|begin_of_text|> $A <|begin_of_text|> $B",
        special_tokens=[
            ("<|begin_of_text|>", tokenizer.bos_token_id),  # <- Your corrected BOS token ID
        ]
    )
    data = '1 0 1 0'
    print(tokenizer.encode(data, add_special_tokens=False))
    print(tokenizer.encode(data, add_special_tokens=True))
    tokenizer.save_pretrained(local_dir)
    

    # save the model
    model.save_pretrained(local_dir)


    # push to hub
    if repo_id is not None:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    print(f"Model saved to {local_dir}")
    print(f"New tokenizer saved to {local_dir}")
    print(f"Model pushed to {repo_id}")
    print(f"New tokenizer pushed to {repo_id}")
