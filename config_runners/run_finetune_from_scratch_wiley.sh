python ../train_bart.py --config-file ../configs/train_config_finetune_wiley.yaml \
                        --additional-info _from_scratch_nist_wiley \
                        --wandb-group finetune \
                        --additional-tags "H100:capy:wiley:nist:from_scratch"