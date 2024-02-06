# CUDA_VISIBLE_DEVICES=2 
python ../train_bart.py --config-file train_config_finetune_GPU_test_large.yaml \
                                               --additional-info _1xH100_from_scratch_large \
                                               --wandb-group GPU_test \
                                               --additional-tags "gpu_test:large"