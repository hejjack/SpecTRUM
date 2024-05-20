CUDA_VISIBLE_DEVICES=3 python ../predict.py --checkpoint ../checkpoints/finetune/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/checkpoint-147476 \
                     --output-folder ../predictions \
                     --config-file ../configs/predict_config_nist_test_beam50.yaml \
