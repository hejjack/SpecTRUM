python train_bart.py --config-file configs/finetune_final.yaml \
                     --checkpoint ../checkpoints/pretrain_clean/pleasant-resonance-556_exp4_rassp_neims_nist/checkpoint-224000 \
                     --additional-info _final \
                     --additional-tags log_29_1.28:mf10M:nist:from_pretrained:final \
                     --wandb-group finetune_clean

