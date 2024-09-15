python ../train_bart.py --config-file configs/finetune_exp3_mf10M.yaml \
                        --checkpoint ../checkpoints/pretrain_clean/feasible-breeze-527_exp4_rassp/checkpoint-112000 \
                        --additional-info "_exp4_rassp" \
                        --wandb-group finetune_clean  \
                        --additional-tags "exp4:rassp:from_pretrained" \

