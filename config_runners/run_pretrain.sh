# NEIMS pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_neims.yaml \
                                               --additional-info "_neims_scratch" \
                                               --wandb-group pretrain
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \
                                             #   --checkpoint ../checkpoints/finetune/solar-sponge-191/checkpoint-34272 \


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best