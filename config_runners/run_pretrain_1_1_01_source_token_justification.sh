# pretrain RASSP_1 NEIMS_1 NIST_0.1
python ../train_bart.py --config-file ../configs/train_config_pretrain_1_1_01_source_token_justification.yaml \
                                               --additional-info "source_token_justification" \
                                               --additional-tags "scratch:H100:meta" \
                                               --wandb-group pretrain \
