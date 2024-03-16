SPLIT_NAME=test
python ../data/precompute_denovo_index.py \
           --reference ../data/datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../data/datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../data/datasets/NIST/NIST_split_filip/denovo_data/morgan_cosine/${SPLIT_NAME}_with_denovo_info.jsonl \
           --num_processes 32
