
# NEIMS pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_neims.yaml \
                                               --additional-info "_neims_scratch2_validation_w_neims_token" \
                                               --wandb-group pretrain \
                                               --additional-tags "A100:scratch:apollo:neims_token_valid" \
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \


                                             #   --resume-id lw7cyi1d \
                                             #   --checkpoint ../checkpoints/pretrain/misunderstood-voice-218_neims_scratch/checkpoint-74000 \
# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best

