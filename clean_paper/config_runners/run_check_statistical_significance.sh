
# ./predictions/major-capybara-531_exp1_pos_emb/NIST/1727968160_valid_full_greedy/predictions.jsonl
# ./predictions/fresh-haze-530_exp1_int_emb_exp2_log_9_2.2/NIST/1728026100_valid_full_greedy/predictions.jsonl
# ./predictions/devout-disco-532_exp2_log_20_1.43/NIST/1728026100_valid_full_greedy/predictions.jsonl
# ./predictions/gallant-lion-533_exp2_log_29_1.28_exp3_mf10M/NIST/1727961993_valid_full_greedy/predictions.jsonl
# ./predictions/apricot-frost-534_exp2_log_39_1.2/NIST/1728026101_valid_full_greedy/predictions.jsonl
# ./predictions/crisp-meadow-535_exp2_lin_100/NIST/1728026099_valid_full_greedy/predictions.jsonl
# ./predictions/absurd-wildflower-536_exp2_lin_1000/NIST/1728026100_valid_full_greedy/predictions.jsonl
# ./predictions/exalted-elevator-545_exp3_mf10/NIST/1727961994_valid_full_greedy/predictions.jsonl
# ./predictions/dashing-grass-547_exp3_mf100/NIST/1727961993_valid_full_greedy/predictions.jsonl
# ./predictions/noble-glitter-546_exp3_mf10K/NIST/1727961994_valid_full_greedy/predictions.jsonl
# ./predictions/devoted-feather-548_exp3_selfies/NIST/1727966508_valid_full_greedy/predictions.jsonl
# ./predictions/valiant-totem-557_exp4_neims/NIST/1727959179_valid_full_greedy/predictions.jsonl
# ./predictions/avid-rain-560_exp4_rassp/NIST/1727959179_valid_full_greedy/predictions.jsonl
# ./predictions/effortless-river-558_exp4_rassp_neims/NIST/1727959179_valid_full_greedy/predictions.jsonl
# ./predictions/hopeful-rain-557_exp4_rassp_neims_nist/NIST/1727883450_valid_full_greedy/predictions.jsonl
# ./predictions/autumn-dawn-564_exp5_one_src_token/NIST/1727883404_valid_full_greedy/predictions.jsonl
# ./predictions/jolly-lion-562_exp6_45perc_frozen/NIST/1728025474_valid_full_greedy/predictions.jsonl
# ./predictions/smooth-totem-563_exp6_72perc_frozen/NIST/1728025473_valid_full_greedy/predictions.jsonl

python ../check_statistical_significance.py \
                --additional-info almost_all \
                ./predictions/major-capybara-531_exp1_pos_emb/NIST/1727968160_valid_full_greedy \
                ./predictions/fresh-haze-530_exp1_int_emb_exp2_log_9_2.2/NIST/1728026100_valid_full_greedy \
                ./predictions/devout-disco-532_exp2_log_20_1.43/NIST/1728026100_valid_full_greedy \
                ./predictions/gallant-lion-533_exp2_log_29_1.28_exp3_mf10M/NIST/1727961993_valid_full_greedy \
                ./predictions/apricot-frost-534_exp2_log_39_1.2/NIST/1728026101_valid_full_greedy \
                ./predictions/crisp-meadow-535_exp2_lin_100/NIST/1728026099_valid_full_greedy \
                ./predictions/absurd-wildflower-536_exp2_lin_1000/NIST/1728026100_valid_full_greedy \
                ./predictions/exalted-elevator-545_exp3_mf10/NIST/1727961994_valid_full_greedy \
                ./predictions/dashing-grass-547_exp3_mf100/NIST/1727961993_valid_full_greedy \
                ./predictions/noble-glitter-546_exp3_mf10K/NIST/1727961994_valid_full_greedy \
                ./predictions/devoted-feather-548_exp3_selfies/NIST/1727966508_valid_full_greedy \
                ./predictions/valiant-totem-557_exp4_neims/NIST/1727959179_valid_full_greedy \
                ./predictions/avid-rain-560_exp4_rassp/NIST/1727959179_valid_full_greedy \
                ./predictions/effortless-river-558_exp4_rassp_neims/NIST/1727959179_valid_full_greedy \
                ./predictions/hopeful-rain-557_exp4_rassp_neims_nist/NIST/1727883450_valid_full_greedy \
                ./predictions/autumn-dawn-564_exp5_one_src_token/NIST/1727883404_valid_full_greedy \
                ./predictions/jolly-lion-562_exp6_45perc_frozen/NIST/1728025474_valid_full_greedy \
                ./predictions/smooth-totem-563_exp6_72perc_frozen/NIST/1728025473_valid_full_greedy