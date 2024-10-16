# finetune model ran on 4.8M rassp split generated by NEIMS dataset
python ../train_bart.py --config-file ../configs/train_config_finetune_mf10000000.yaml \
                        --checkpoint ../checkpoints/pretrain/stilted-pine-480_rassp_scratch_40bin/checkpoint-112000 \
                        --additional-info "_rassp_40bin" \
                        --wandb-group finetune  \
                        --additional-tags "H100:meta:rassp1" \