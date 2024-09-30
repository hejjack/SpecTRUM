python train_bart.py --config-file configs/finetune_exp3_mf10M.yaml \
                        --checkpoint ../checkpoints/pretrain_clean/vital-plant-549_exp4_neims/checkpoint-112000 \
                        --additional-info "_exp4_neims" \
                        --wandb-group finetune_clean  \
                        --additional-tags "exp4:neims:from_pretrained" \

