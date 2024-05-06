python ../train_bart.py --config-file ../configs/train_config_finetune_40bin.yaml \
                                               --additional-info _from_scratch_40bin \
                                               --wandb-group finetune \
                                               --additional-tags "H100:meta:mf10M"