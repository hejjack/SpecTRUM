# finetune model ran on 4.8M rassp split generated by NEIMS dataset
CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file ../configs/train_config_finetune_clever_freeze.yaml \
                                               --checkpoint ../checkpoints/pretrain/bright-dumpling-376_rassp1_neims1/checkpoint-224000 \
                                               --additional-info "_clever_freeze" \
                                               --wandb-group finetune  \
                                               --additional-tags "clever_freeze" \
