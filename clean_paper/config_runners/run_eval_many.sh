
# ./predictions/absurd-wildflower-536_exp2_lin_1000/NIST/1728026100_valid_full_greedy/predictions.jsonl
# ./predictions/apricot-frost-534_exp2_log_39_1.2/NIST/1728026101_valid_full_greedy/predictions.jsonl
# ./predictions/apricot-frost-534_exp2_log_39_1.2/NIST/1728026101_valid_full_greedy/predictions.jsonl
# ./predictions/crisp-meadow-535_exp2_lin_100/NIST/1728026099_valid_full_greedy/predictions.jsonl
# ./predictions/devout-disco-532_exp2_log_20_1.43/NIST/1728026100_valid_full_greedy/predictions.jsonl
# ./predictions/absurd-wildflower-536_exp2_lin_1000/NIST/1728026100_valid_full_greedy/predictions.jsonl
# ./predictions/fresh-haze-530_exp1_int_emb_exp2_log_9_2.2/NIST/1728026100_valid_full_greedy/predictions.jsonl
# ./predictions/gallant-lion-533_exp2_log_29_1.28_exp3_mf10M/NIST/1727961993_valid_full_greedy/predictions.jsonl
# ./predictions/jolly-lion-562_exp6_45perc_frozen/NIST/1728025474_valid_full_greedy/predictions.jsonl
# ./predictions/smooth-totem-563_exp6_72perc_frozen/NIST/1728025473_valid_full_greedy/predictions.jsonl

PREDICTIONS=(./predictions/absurd-wildflower-536_exp2_lin_1000/NIST/1728026100_valid_full_greedy/predictions.jsonl
./predictions/apricot-frost-534_exp2_log_39_1.2/NIST/1728026101_valid_full_greedy/predictions.jsonl
./predictions/apricot-frost-534_exp2_log_39_1.2/NIST/1728026101_valid_full_greedy/predictions.jsonl
./predictions/crisp-meadow-535_exp2_lin_100/NIST/1728026099_valid_full_greedy/predictions.jsonl
./predictions/devout-disco-532_exp2_log_20_1.43/NIST/1728026100_valid_full_greedy/predictions.jsonl
./predictions/absurd-wildflower-536_exp2_lin_1000/NIST/1728026100_valid_full_greedy/predictions.jsonl
./predictions/fresh-haze-530_exp1_int_emb_exp2_log_9_2.2/NIST/1728026100_valid_full_greedy/predictions.jsonl
./predictions/gallant-lion-533_exp2_log_29_1.28_exp3_mf10M/NIST/1727961993_valid_full_greedy/predictions.jsonl
./predictions/jolly-lion-562_exp6_45perc_frozen/NIST/1728025474_valid_full_greedy/predictions.jsonl
./predictions/smooth-totem-563_exp6_72perc_frozen/NIST/1728025473_valid_full_greedy/predictions.jsonl)


# Correct for loop with proper array reference
for prediction in "${PREDICTIONS[@]}"; do
    echo "Processing prediction: $prediction `wc -l $prediction`"
    python ../evaluate_predictions.py --predictions-path $prediction \
                                    --labels-path data/nist/valid_with_db_index.jsonl \
                                    --config-file configs/evaluate_nist.yaml &
done

# Optional: Wait for all background processes to finish
wait