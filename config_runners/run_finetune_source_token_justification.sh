python ../train_bart.py --config-file ../configs/train_config_finetune_source_token_justification.yaml \
                        --checkpoint ../checkpoints/pretrain/azure-pine-491source_token_justification/checkpoint-112000 \
                        --additional-info "source_token_justification" \
                        --wandb-group finetune  \
                        --additional-tags "H100:meta:rassp:neims" \

