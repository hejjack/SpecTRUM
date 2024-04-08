python ../train_bart.py --config-file ../configs/train_config_finetune_mf10000000.yaml \
                        --checkpoint ../checkpoints/pretrain/blooming-thunder-439_rassp1_neims1/checkpoint-112000 \
                        --additional-info "rassp1_neims1_112kPretrain" \
                        --wandb-group finetune  \
                        --additional-tags "H100:meta:rassp:neims" \
