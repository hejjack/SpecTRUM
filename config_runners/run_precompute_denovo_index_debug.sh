SPLIT_NAME=valid
FP_TYPE=morgan                  # morgan, daylight
FP_SIMIL_FUN=tanimoto           # tanimoto, cosine
SPEC_SIMIL_FUN=modified_cosine  # cosine, modified_cosine, cosine&modified_cosine

python ../precompute_db_index.py \
           --reference ../data/datasets/DEBUG/DEBUG_reference.jsonl \
           --query ../data/datasets/DEBUG/DEBUG_valid.jsonl \
           --outfile ../data/datasets/DEBUG/DEBUG_valid_with_db_index.jsonl \
           --num_processes 16 \
           --fingerprint_type ${FP_TYPE} \
           --fp_simil_function ${FP_SIMIL_FUN} \
           --spectra_simil_functions ${SPEC_SIMIL_FUN} \
           --num_db_candidates 10 \