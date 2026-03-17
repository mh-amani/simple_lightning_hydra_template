# scan:
# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type word \
#     --dataset_path "scan" \
#     --save_path "data/tokenizers/scan/actions" \
#     --key "actions" \
#     --batch_size 1000 \
#     --max_vocab_size 1000 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
#     --dataset_config '{"name": "simple"}'

# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type word \
#     --dataset_path "scan" \
#     --save_path "data/tokenizers/scan/commands" \
#     --key "commands" \
#     --batch_size 1000 \
#     --max_vocab_size 1000 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
#     --dataset_config '{"name": "simple"}'


#  pcfg:
# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type word \
#     --dataset_path "/dlabscratch1/amani/sigmae/data/pcfgset/pcfgset_train_test" \
#     --save_path "data/tokenizers/pcfg/inputs" \
#     --key "input" \
#     --batch_size 1000 \
#     --max_vocab_size 1000 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]"

# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type word \
#     --dataset_path "/dlabscratch1/amani/sigmae/data/pcfgset/pcfgset_train_test" \
#     --save_path "data/tokenizers/pcfg/outputs" \
#     --key "output" \
#     --batch_size 1000 \
#     --max_vocab_size 1000 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]"



# # cogs:
# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type unigram \
#     --dataset_path "/dlabscratch1/amani/sigmae/data/cogs_train_test" \
#     --save_path "data/tokenizers/cogs/cogs_unigram_tokenizer_input_500" \
#     --key "input" \
#     --batch_size 1000 \
#     --max_vocab_size 500 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \

# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type unigram \
#     --dataset_path "/dlabscratch1/amani/sigmae/data/cogs_train_test" \
#     --save_path "data/tokenizers/cogs/cogs_unigram_tokenizer_output_500" \
#     --key "output" \
#     --batch_size 1000 \
#     --max_vocab_size 500 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \



# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type unigram \
#     --dataset_path "data/cogs/train.tsv" \
#     --save_path "data/tokenizers/cogs/cogs_unigram_tokenizer_input_500" \
#     --key "input" \
#     --batch_size 1000 \
#     --max_vocab_size 500 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
#     # --dataset_config '{"delimiter": "\t", "column_names": ["input", "output", "in_dist_or_out"]}'

# python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
#     --tokenizer_type unigram \
#     --dataset_path "data/cogs/train.tsv" \
#     --save_path "data/tokenizers/cogs/cogs_unigram_tokenizer_output_500" \
#     --key "output" \
#     --batch_size 1000 \
#     --max_vocab_size 500 \
#     --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
#     # --dataset_config '{"delimiter": "\t", "column_names": ["input", "output", "in_dist_or_out"]}'



# cfq:
python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
    --tokenizer_type word \
    --dataset_path "cfq" \
    --save_path "data/tokenizers/cfq/question" \
    --key "question" \
    --batch_size 1000 \
    --max_vocab_size 500 \
    --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
    --dataset_config '{"name": "random_split"}'

python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
    --tokenizer_type word \
    --dataset_path "cfq" \
    --save_path "data/tokenizers/cfq/query" \
    --key "query" \
    --batch_size 1000 \
    --max_vocab_size 500 \
    --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
    --dataset_config '{"name": "random_split"}'