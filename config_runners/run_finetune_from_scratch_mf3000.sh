CUDA_VISIBLE_DEVICES=3 python ../train_bart.py --config-file ../configs/train_config_finetune_mf3000.yaml \
                                               --additional-info _from_scratch_mf3000 \
                                               --wandb-group finetune \
                                               --additional-tags "A100:alfa:mf3000"