# finetune model ran on 1-1-0.1 dataset (NEIMS1, RASSP1, NIST0.1)
python ../train_bart.py --config-file ../configs/train_config_finetune_linear_bin3.yaml \
                        --checkpoint ../checkpoints/pretrain/pious-universe-467_rassp1_neims1_neims01_linear_bin3/checkpoint-112000 \
                        --additional-info "_rassp1_neims1_nist01_linear_bin3" \
                        --wandb-group finetune  \
                        --additional-tags "meta:H100:rassp1:neims1:nist01" \