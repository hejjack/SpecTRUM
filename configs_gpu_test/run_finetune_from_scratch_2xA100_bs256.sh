CUDA_VISIBLE_DEVICES=0,1 python ../train_bart.py --config-file ../configs_gpu_test/train_config_finetune_2xA100_bs265.yaml \
                                               --additional-info _2xA100_80GB_from_scratch_BS256 \
                                               --wandb-group GPU_test \
                                               --additional-tags "A100:alfa:gpu_test"
