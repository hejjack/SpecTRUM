python ../train_bart.py --config-file ../configs/train_config_finetune_21bin.yaml \
                                               --additional-info _from_scratch_21bin \
                                               --wandb-group finetune \
                                               --additional-tags "H100:meta:mf10M"