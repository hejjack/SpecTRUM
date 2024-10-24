
# ./predictions/db_search_morgan_tanimoto/NIST/1729606578_valid_full_50cand/predictions.jsonl
# ./predictions/db_search_morgan_tanimoto/NIST/1729630722_valid_full_10cand/predictions.jsonl
# ./predictions/db_search_morgan_tanimoto/NIST/1729630748_valid_full_1cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1729629585_valid_full_50cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1729676364_valid_full_1cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1729676352_valid_full_10cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1729676322_valid_full_1cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1729676337_valid_full_10cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1729605198_valid_full_50cand/predictions.jsonl

PREDICTIONS=(./predictions/db_search_morgan_tanimoto/NIST/1729606578_valid_full_50cand/predictions.jsonl
./predictions/db_search_morgan_tanimoto/NIST/1729630722_valid_full_10cand/predictions.jsonl
./predictions/db_search_morgan_tanimoto/NIST/1729630748_valid_full_1cand/predictions.jsonl
./predictions/db_search_sss/NIST/1729629585_valid_full_50cand/predictions.jsonl
./predictions/db_search_sss/NIST/1729676364_valid_full_1cand/predictions.jsonl
./predictions/db_search_sss/NIST/1729676352_valid_full_10cand/predictions.jsonl
./predictions/db_search_hss/NIST/1729676322_valid_full_1cand/predictions.jsonl
./predictions/db_search_hss/NIST/1729676337_valid_full_10cand/predictions.jsonl
./predictions/db_search_hss/NIST/1729605198_valid_full_50cand/predictions.jsonl)


# Correct for loop with proper array reference
for prediction in "${PREDICTIONS[@]}"; do
    echo "Processing prediction: $prediction `wc -l $prediction`"
    python ../evaluate_predictions.py --predictions-path $prediction \
                                    --labels-path data/nist/valid_with_db_index.jsonl \
                                    --config-file configs/evaluate_nist.yaml &
done

# Optional: Wait for all background processes to finish
wait