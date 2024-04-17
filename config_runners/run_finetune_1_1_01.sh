# finetune model ran on 1-1-0.1 dataset (NEIMS1, RASSP1, NIST0.1)
python ../train_bart.py --config-file ../configs/train_config_finetune_mf10000000.yaml \
                        --checkpoint ../checkpoints/pretrain/rural-dawn-440_rassp1_neims1_neims01/checkpoint-224000 \
                        --additional-info "_rassp1_neims1_nist01_224k" \
                        --wandb-group finetune  \
                        --additional-tags "meta:H100:rassp1:neims1:nist01" \