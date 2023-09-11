# pretrain RASSP_1 NEIMS_1 NIST_0.1 on 2 A40 on meta
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_nist01.yaml \
                        --additional-info "_rassp1_neims1_neims01" \
                        --additional-tags "meta:scratch:2GPU" \
                        --wandb-group pretrain \
                                            #    --device cpu \
                                             #   --resume-id qa0wkyd9
                                             #   --checkpoint ../checkpoints/pretrain/glamorous-dragon-222_rassp_scratch/checkpoint-4000 \
                                               # --checkpoints_dir ../checkpoints \


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best
