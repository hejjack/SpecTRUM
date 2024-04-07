# pretrain RASSP_1 NEIMS_1 NIST_0.1
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_nist01.yaml \
                                               --additional-info "_rassp1_neims1_neims01" \
                                               --additional-tags "scratch:H100:meta" \
                                               --wandb-group pretrain \
