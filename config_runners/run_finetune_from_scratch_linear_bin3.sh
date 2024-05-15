python ../train_bart.py --config-file ../configs/train_config_finetune_linear_bin3.yaml \
                                               --additional-info _from_scratch_linear_bin3 \
                                               --wandb-group finetune \
                                               --additional-tags "H100:meta:mf10M"