CUDA_VISIBLE_DEVICES=2 python ../train_bart.py --config-file ../configs/train_config_finetune_shuffle.yaml \
                                               --additional-info _from_scratch_shuffle \
                                               --wandb-group finetune \
                                               --additional-tags "A100:alfa:shuffle"