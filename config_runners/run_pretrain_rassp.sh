<<<<<<<< HEAD:config_runners/run_pretrain_neims.sh
# NEIMS pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_neims.yaml \
                                               --checkpoint ../checkpoints/pretrain/misunderstood-voice-218_neims_scratch/checkpoint-54000 \
                                               --additional-info "_neims_scratch" \
                                               --wandb-group pretrain \
                                               --resume-id lw7cyi1d \
                                               --additional-tags ["A100"] \
========
# RASSP pretrain from scratch
CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp.yaml \
                                               --additional-info "_rassp_scratch" \
                                               --wandb-group pretrain \
                                               --additional-tags "A100:scratch"
>>>>>>>> 8296a92fbc4868d28224e47f767fb26437e2c494:config_runners/run_pretrain_rassp.sh
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best

