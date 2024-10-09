SPLIT_NAME=valid
FP_TYPE=morgan
SIMIL_FUN=tanimoto

python precompute_db_index.py \
           --reference data/nist/train.jsonl \
           --query data/nist/${SPLIT_NAME}.jsonl \
           --outfile data/nist/${SPLIT_NAME}_with_db_index.jsonl \
           --num_processes 64 \
           --fingerprint_type ${FP_TYPE} \
           --fp_simil_function ${SIMIL_FUN}