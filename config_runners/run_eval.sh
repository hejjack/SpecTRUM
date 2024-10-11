# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST/1697808199_valid_0:9248_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/valid_with_db_index.jsonl \
#                                   --do-db_search

# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST/1697801918_valid_0:50_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/valid_with_db_index.jsonl \
#                                   --do-db_search

# python ../evaluate_predictions.py --predictions-path ../predictions/pretty-brook-385_rassp1_neims1_112kPretrain_mf10M/NIST/1710934120_valid_full_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/valid_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/legendary-disco-449_rassp1_neims1_nist01_224k/NIST/1713340943_valid_full_beam10_peaks300/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/valid_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_daylight.yaml \


# debug
# python ../evaluate_predictions.py --predictions-path ../predictions/bart_2023-04-07-18_27_23_30Mneims/DEBUG/1692277311_valid_20:50_ahoj/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/valid_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \


# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/NIST/1715767948_test_full_beam50/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/test_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \


# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/NIST/1715767947_test_full_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/test_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/NIST/1715857966_test_full_greedy/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/db_index/test_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

# swgdrug
# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/SWGDRUG/1726432120_test_full_greedy/predictions.jsonl \
#                                   --labels-path ../data/datasets/extra_libraries/SWGDRUG_3/SWGDRUG_3_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/SWGDRUG/1726433172_test_full_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/extra_libraries/SWGDRUG_3/SWGDRUG_3_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/SWGDRUG/1726435306_test_full_beam50/predictions.jsonl \
#                                   --labels-path ../data/datasets/extra_libraries/SWGDRUG_3/SWGDRUG_3_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

#noD
# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/SWGDRUG/1726499003_noD_full_greedy/predictions.jsonl \
#                                   --labels-path ../data/datasets/extra_libraries/SWGDRUG_3/SWGDRUG_3_noD_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/Cayman/1726519724_test_full_greedy/predictions.jsonl \
#                                   --labels-path ../data/datasets/extra_libraries/Cayman_library/Cayman_library_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/Cayman/1726521813_test_full_beam50/predictions.jsonl \
#                                   --labels-path ../data/datasets/extra_libraries/Cayman_library/Cayman_library_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml \

# DEBUG
# python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/SWGDRUG/1726433172_test_full_beam10_DEBUG/predictions.jsonl \
#                                   --labels-path ../data/datasets/extra_libraries/SWGDRUG_3/SWGDRUG_3_with_db_index.jsonl \
#                                   --config-file ../configs/eval_config_morgan.yaml

# DEBUG 2
python ../evaluate_predictions.py --predictions-path ../predictions/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/SWGDRUG/1726432121_test_full_greedy_DEBUG/predictions.jsonl \
                                  --labels-path ../data/datasets/extra_libraries/SWGDRUG_3/SWGDRUG_3_with_db_index.jsonl \
                                  --config-file ../configs/eval_config_morgan.yaml