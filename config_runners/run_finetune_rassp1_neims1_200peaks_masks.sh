CUDA_VISIBLE_DEVICES=2 python ../train_bart.py --config-file ../configs/train_config_finetune_mf10000000_200peaks.yaml \
                        --checkpoint ../checkpoints/pretrain/eager-sunset-413_rassp1_neims1_200peaks_masks/checkpoint-224000 \
                        --additional-info "_rassp1_neims1_200peaks_masks" \
                        --wandb-group finetune  \
                        --additional-tags "alfa" \