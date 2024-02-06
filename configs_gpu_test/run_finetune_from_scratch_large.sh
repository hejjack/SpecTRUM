CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs_gpu_test/train_config_finetune_large.yaml \
                                               --additional-info _1xA100_80GB_from_scratch_large \
                                               --wandb-group GPU_test \
                                               --additional-tags "large:A100:alfa"