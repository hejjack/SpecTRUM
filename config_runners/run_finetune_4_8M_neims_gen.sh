# finetune model ran on 4.8M rassp split generated by NEIMS dataset
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_finetune.yaml \
                        --checkpoint ../checkpoints/finetune/blooming-feather-255_4_8M_niems_gen/checkpoint-9792 \
                        --additional-info "_4_8M_niems_gen" \
                        --wandb-group finetune  \
                        --resume-id ibcdhnwa \
                        --additional-tags "" \