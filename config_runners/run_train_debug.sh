CUDA_VISIBLE_DEVICES=3 python ../train_bart.py --config-file ../configs/train_config_debug.yaml \
                                               --checkpoint ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1670000 \
                                               --additional-info _test_apollo \
                                               --device cuda \
                                               --wandb-group debug     # finetune
                                               # --checkpoints_dir ../checkpoints \
                                               # --resume-id 2yqhwcas