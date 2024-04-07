
python ../train_bart.py --config-file ../configs/train_config_pretrain_neims.yaml \
                                               --additional-info "_neims_scratch" \
                                               --additional-tags "H100:NEIMS:scratch:meta" \
                                               --wandb-group pretrain \

