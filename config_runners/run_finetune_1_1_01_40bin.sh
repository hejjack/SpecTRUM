# finetune model ran on 1-1-0.1 dataset (NEIMS1, RASSP1, NIST0.1)
python ../train_bart.py --config-file ../configs/train_config_finetune_40bin.yaml \
                        --checkpoint ../checkpoints/pretrain/major-dragon-468_rassp1_neims1_neims01_40bin/checkpoint-112000 \
                        --additional-info "_rassp1_neims1_nist01_40bin" \
                        --wandb-group finetune  \
                        --additional-tags "meta:H100:rassp1:neims1:nist01" \