# pretrain RASSP_1 NEIMS_1 NIST_0.1 on 2 A40 on meta
# CUDA_VISIBLE_DEVICES=1 # META
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_nist01.yaml \
                                               --checkpoint ../checkpoints/pretrain/fallen-star-250_rassp1_neims1_neims01/checkpoint-104000 \
                                               --additional-info "_rassp1_neims1_neims01" \
                                               --additional-tags "scratch" \
                                               --wandb-group pretrain \
                                               --resume-id 0is7ec25


                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best
