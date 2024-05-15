python ../train_bart.py --config-file ../configs/train_config_finetune_40bin.yaml \
                        --checkpoint ../checkpoints/pretrain/devoted-dew-481_rassp1_neims1_40bin/checkpoint-112000 \
                        --additional-info "rassp1_neims1_112k_40bin" \
                        --wandb-group finetune  \
                        --additional-tags "H100:meta:rassp:neims" \
