# 30mPT resume
CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file train_config_finetune.yaml \
                                               --checkpoint ../checkpoints/finetune/solar-sponge-191/checkpoint-34272 \
                                               --additional-info "" \
                                               --wandb-group finetune  \
                                               --resume-id xtu5llkf
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \

# run on A40
CUDA_VISIBLE_DEVICES=2 python ../train_bart.py --config-file train_config_finetune.yaml \
                                               --checkpoint ../checkpoints/finetune/icy-vortex-203_from_scratch/checkpoint-3264 \
                                               --additional-info "_from_scratch" \
                                               --wandb-group finetune \
                                               --resume-id gcnzcu74
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \

# 30mPT resume
CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file train_config_finetune.yaml \
                                               --checkpoint ../checkpoints/finetune/solar-sponge-191/checkpoint-34272 \
                                               --additional-info "" \
                                               --wandb-group finetune  \
                                               --resume-id xtu5llkf
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \

# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best