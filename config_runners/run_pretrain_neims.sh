# NEIMS pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_neims.yaml \
                                               --checkpoint ../checkpoints/pretrain/misunderstood-voice-218_neims_scratch/checkpoint-54000 \
                                               --additional-info "_neims_scratch" \
                                               --wandb-group pretrain \
                                               --resume-id lw7cyi1d \
                                               --additional-tags ["A100"] \
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best

