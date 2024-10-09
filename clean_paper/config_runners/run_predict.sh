CUDA_VISIBLE_DEVICES=0 python ../predict.py --checkpoint ../checkpoints/finetune_clean/hopeful-rain-557_exp4_rassp_neims_nist/checkpoint-73738 \
                                          --output-folder predictions \
                                          --config-file configs/predict_nist.yaml \
