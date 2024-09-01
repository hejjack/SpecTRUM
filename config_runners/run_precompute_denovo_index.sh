SPLIT_NAME=valid
FP_TYPE=morgan
SIMIL_FUN=tanimoto

python ../precompute_db_index.py \
           --reference ../data/datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../data/datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../data/datasets/NIST/NIST_split_filip/db_index/${SPLIT_NAME}_with_db_index.jsonl \
           --num_processes 16 \
           --fingerprint_type ${FP_TYPE} \
           --simil_function ${SIMIL_FUN}