python ../train_bart.py --config-file ../configs/train_config_finetune_15bin.yaml \
                                               --additional-info _from_scratch_lb1_6_15shift \
                                               --wandb-group finetune \
                                               --additional-tags "H100:meta:mf10M:lb1_6"