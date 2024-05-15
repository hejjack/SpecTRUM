
python ../train_bart.py --config-file ../configs/train_config_pretrain_neims_40bin.yaml \
                                               --additional-info "_neims_scratch_40bin" \
                                               --additional-tags "H100:NEIMS:scratch:meta" \
                                               --wandb-group pretrain \

