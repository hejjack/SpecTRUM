python train_bart.py --config-file configs/finetune_exp6_72perc_frozen.yaml \
                     --checkpoint ../checkpoints/pretrain_clean/pleasant-resonance-556_exp4_rassp_neims_nist/checkpoint-112000 \
                     --additional-info _exp6_72perc_frozen \
                     --wandb-group finetune_clean \
                     --additional-tags exp6:frozen:72perc:from_pretrained 