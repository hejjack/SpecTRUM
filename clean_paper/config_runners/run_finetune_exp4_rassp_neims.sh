python train_bart.py --config-file configs/finetune_exp3_mf10M.yaml \
                        --checkpoint ../checkpoints/pretrain_clean/peachy-rain-551_exp4_rassp_neims/checkpoint-112000 \
                        --additional-info "_exp4_rassp_neims" \
                        --wandb-group finetune_clean  \
                        --additional-tags "exp4:rassp:neims:from_pretrained" \

