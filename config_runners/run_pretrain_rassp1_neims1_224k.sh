# RASSP pretrain from scratch
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_224k_40bin.yaml \
                        --checkpoint ../checkpoints/pretrain/devoted-dew-481_rassp1_neims1_40bin/checkpoint-112000 \
                        --resume-id hfd44n0p \
                        --additional-info "_rassp1_neims1_224k_40bin" \
                        --additional-tags "H100:scratch:rassp1:neims1:capy" \
                        --wandb-group pretrain \