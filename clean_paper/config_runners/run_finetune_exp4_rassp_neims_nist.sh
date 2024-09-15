python ../train_bart.py --config-file configs/finetune_exp3_mf10M.yaml \
                        --checkpoint ../checkpoints/pretrain_clean/???/checkpoint-112000 \
                        --additional-info "_exp4_rassp_neims_nist" \
                        --wandb-group finetune_clean  \
                        --additional-tags "exp4:rassp:neims:nist:from_pretrained" \

