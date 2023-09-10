# RASSP pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp.yaml \
                                               --additional-info "_rassp_scratch" \
                                               --additional-tags "A100:scratch" \
                                               --wandb-group pretrain \
                                            #    --device cpu \
                                             #   --resume-id qa0wkyd9
                                             #   --checkpoint ../checkpoints/pretrain/glamorous-dragon-222_rassp_scratch/checkpoint-4000 \
                                               # --checkpoints_dir ../checkpoints \


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best
