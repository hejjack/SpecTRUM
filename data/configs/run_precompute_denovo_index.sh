python ../precompute_denovo_index.py \
           --reference ../datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../datasets/NIST/NIST_split_filip/valid.jsonl \
           --outfile ../datasets/NIST/NIST_split_filip/denovo_data/test_with_denovo_info.jsonl \
           --num_processes 64 \
