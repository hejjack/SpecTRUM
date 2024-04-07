python ../train_bart.py --config-file ../configs/train_config_finetune_selfies.yaml \
                                               --additional-info _from_scratch_selfies \
                                               --wandb-group finetune \
                                               --additional-tags "H100:meta:selfies" \