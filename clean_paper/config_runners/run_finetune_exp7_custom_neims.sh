python train_bart.py --config-file configs/finetune_exp3_mf10M.yaml \
                     --checkpoint ../checkpoints/pretrain_clean/grateful-field-566_exp7_custom_neims/checkpoint-112000 \
                     --additional-info "_exp7_custom_neims" \
                     --wandb-group finetune_clean  \
                     --additional-tags "exp7:neims_custom:from_pretrained" \
