python ../train_bart.py --config-file ../configs/train_config_finetune_30bin.yaml \
                                               --additional-info _from_scratch_30bin \
                                               --wandb-group finetune \
                                               --additional-tags "H100:meta:mf10M"