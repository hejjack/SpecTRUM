SPLIT_NAME=debug/valid_debug
FP_TYPE=morgan
SIMIL_FUN=tanimoto

python precompute_db_index.py \
           --reference data/nist/valid.jsonl \
           --query data/nist/${SPLIT_NAME}.jsonl \
           --outfile data/nist/${SPLIT_NAME}_with_db_index2.jsonl \
           --num_processes 10 \
           --fingerprint_type ${FP_TYPE} \
           --fp_simil_function ${SIMIL_FUN}