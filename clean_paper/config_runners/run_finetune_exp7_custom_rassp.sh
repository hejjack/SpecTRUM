python train_bart.py --config-file configs/finetune_exp3_mf10M.yaml \
                     --checkpoint ../checkpoints/pretrain_clean/golden-morning-569_exp7_custom_rassp/checkpoint-112000 \
                     --additional-info "_exp7_custom_rassp" \
                     --wandb-group finetune_clean  \
                     --additional-tags "exp7:custom_rassp:from_pretrained" \

