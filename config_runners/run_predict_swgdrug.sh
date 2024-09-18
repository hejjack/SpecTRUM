CUDA_VISIBLE_DEVICES=0 python ../predict.py --checkpoint ../checkpoints/finetune/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/checkpoint-147476 \
                                            --output-folder ../predictions \
                                            --config-file ../configs/predict_config_swgdrug_greedy.yaml \
