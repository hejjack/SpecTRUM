# pretrain RASSP_1 NEIMS_1 NIST_0.1
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_nist01_224k.yaml \
                        --checkpoint ../checkpoints/pretrain/rural-dawn-440_rassp1_neims1_neims01/checkpoint-112000 \
                        --additional-info "_rassp1_neims1_neims01_224k" \
                        --additional-tags "scratch:H100:meta" \
                        --wandb-group pretrain \
                        --resume-id z0gv7gte


