CUDA_VISIBLE_DEVICES=0
python ../train_bart.py --config-file train_config.yaml \
                        --checkpoint ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1670000 \
                        --additional-info _debug \
                        --device cpu \
                        --wandb-group pretrain     # finetune
                        # --checkpoints_dir ../checkpoints \
                        # --resume-id 2yqhwcas \
