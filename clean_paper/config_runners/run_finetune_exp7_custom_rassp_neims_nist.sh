python train_bart.py --config-file configs/finetune_exp3_mf10M.yaml \
                     --checkpoint ../checkpoints/pretrain_clean/drawn-fire-571_exp7_custom_rassp_neims_nist/checkpoint-112000 \
                     --additional-info "_exp7_custom_rassp_neims_nist" \
                     --wandb-group finetune_clean  \
                     --additional-tags "exp7:custom_rassp:custom_neims:from_pretrained" \
