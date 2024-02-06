CUDA_VISIBLE_DEVICES=0,3 python ../train_bart.py --config-file ../_gpu_test/train_config_debug.yaml \
                                               --additional-info _from_scratch \
                                               --wandb-group GPU_test \
                                               --additional-tags "A100:alfa"