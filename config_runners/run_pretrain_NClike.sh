# RASSP pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_NClike.yaml \
                        --additional-info "_NClike" \
                        --additional-tags "A100:scratch:RASSP1NEIMS1:meta" \
                        --wandb-group pretrain \
                        --resume-id 7j1lujju \
                        --checkpoint ../checkpoints/pretrain/avid-aardvark-415_NClike/checkpoint-112000