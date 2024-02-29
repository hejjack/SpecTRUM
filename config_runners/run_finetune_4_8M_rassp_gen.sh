# finetune model ran on 4.8M rassp split generated by NEIMS dataset
CUDA_VISIBLE_DEVICES=2 python ../train_bart.py --config-file ../configs/train_config_finetune_mf10000000.yaml \
                                               --checkpoint ../checkpoints/pretrain/chromatic-dragon-375_rassp_scratch/checkpoint-112000 \
                                               --additional-info "_rassp" \
                                               --wandb-group finetune  \
                                               --additional-tags "A100:alfa:rassp1" \
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \