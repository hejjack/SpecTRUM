python ../train_bart.py --config-file ../configs/train_config_finetune_mf10.yaml \
                        --additional-info _from_scratch_mf10 \
                        --wandb-group finetune \
                        --additional-tags "H100:meta:mf10"