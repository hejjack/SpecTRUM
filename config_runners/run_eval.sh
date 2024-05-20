# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST_denovo/1697808199_valid_0:9248_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --do-denovo

# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST_denovo/1697801918_valid_0:50_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --do-denovo

# python ../evaluate_predictions.py --predictions-path ../predictions/pretty-brook-385_rassp1_neims1_112kPretrain_mf10M/NIST_denovo/1710934120_valid_full_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --config-file ../configs/eval_config.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/legendary-disco-449_rassp1_neims1_nist01_224k/NIST_denovo/1713340943_valid_full_beam10_peaks300/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --config-file ../configs/eval_config_daylight.yaml \


# debug
# python ../evaluate_predictions.py --predictions-path ../predictions/bart_2023-04-07-18_27_23_30Mneims/DEBUG/1692277311_valid_20:50_ahoj/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \


# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/NIST_denovo/1715767948_test_full_beam50/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/test_with_denovo_info.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \


python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/NIST_denovo/1715767947_test_full_beam10/predictions.jsonl \
                                  --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/test_with_denovo_info.jsonl \
                                  --config-file ../configs/eval_config_morgan.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/NIST_denovo/1715857966_test_full_greedy/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/test_with_denovo_info.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \



                                  