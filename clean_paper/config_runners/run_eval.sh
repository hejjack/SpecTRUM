# MACE
python ../evaluate_predictions.py --predictions-path predictions/hopeful-rain-557_exp4_rassp_neims_nist/MACE/1727882289_all_full_beam10/predictions.jsonl \
                                  --labels-path data/mace/MACE_r05_with_db_index.jsonl \
                                  --config-file configs/evaluate_mace.yaml

# NIST
# python ../evaluate_predictions.py --predictions-path predictions/autumn-dawn-564_exp5_one_src_token/MACE/1727882452_all_full_beam10/predictions.jsonl \
#                                   --labels-path data/nist/test_with_db_index.jsonl \
#                                   --config-file configs/evaluate_mace.yaml
