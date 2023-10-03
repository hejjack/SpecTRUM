# NIEMS30m
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_finetune.yaml \
                                               --checkpoint ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000 \
                                               --additional-info "_neims30M" \
                                               --wandb-group finetune  \
                                               --additional-tags "A100:alfa" \