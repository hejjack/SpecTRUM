# RASSP pretrain from scratch
CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp.yaml \
                                               --additional-info "_rassp_scratch" \
                                               --wandb-group pretrain \
                                               --additional-tags "A100:scratch"
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \
                                             #   --checkpoint ../checkpoints/finetune/solar-sponge-191/checkpoint-34272 \


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best