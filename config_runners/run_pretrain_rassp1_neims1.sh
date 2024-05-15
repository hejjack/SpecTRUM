# RASSP pretrain from scratch
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_40bin.yaml \
                        --additional-info "_rassp1_neims1_40bin" \
                        --additional-tags "H100:scratch:rassp1:neims1:capy" \
                        --wandb-group pretrain \