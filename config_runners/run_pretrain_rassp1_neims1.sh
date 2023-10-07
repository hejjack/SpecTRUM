# RASSP pretrain from scratch
CUDA_VISIBLE_DEVICES=2,3 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1.yaml \
                                               --additional-info "_rassp1_neims1" \
                                               --additional-tags "A100:scratch" \
                                               --resume-id pwrbt7zt \
                                               --wandb-group pretrain \
                                               --checkpoint ../checkpoints/pretrain/northern-cloud-230_rassp1_neims1/checkpoint-112000 \

# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best
