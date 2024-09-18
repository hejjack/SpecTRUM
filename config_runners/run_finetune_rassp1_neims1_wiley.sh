python ../train_bart.py --config-file ../configs/train_config_finetune_wiley.yaml \
                        --checkpoint ../checkpoints/pretrain_clean/confused-flower-523_exp4_rassp_neims/checkpoint-112000 \
                        --additional-info "rassp1_neims1_nist_wiley" \
                        --wandb-group finetune  \
                        --additional-tags "H100:capy:wiley:nist:from_pretrained" \

