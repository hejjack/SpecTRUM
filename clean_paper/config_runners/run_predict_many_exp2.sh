# ../checkpoints/finetune_clean/absurd-wildflower-536_exp2_lin_1000
# ../checkpoints/finetune_clean/crisp-meadow-535_exp2_lin_100
# ../checkpoints/finetune_clean/fresh-haze-530_exp1_int_emb_exp2_log_9_2.2
# ../checkpoints/finetune_clean/devout-disco-532_exp2_log_20_1.43
# ../checkpoints/finetune_clean/apricot-frost-534_exp2_log_39_1.2


CVD=0

model="../checkpoints/finetune_clean/absurd-wildflower-536_exp2_lin_1000"
config="configs/predict_nist_valid_beam10_exp2_lin_1000.yaml"
echo "Processing model #1: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/crisp-meadow-535_exp2_lin_100"
config="configs/predict_nist_valid_beam10_exp2_lin_100.yaml"
echo "Processing model #2: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/fresh-haze-530_exp1_int_emb_exp2_log_9_2.2"
config="configs/predict_nist_valid_beam10_exp2_log_9_2.2.yaml"
echo "Processing model #3: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/devout-disco-532_exp2_log_20_1.43"
config="configs/predict_nist_valid_beam10_exp2_log_20_1.43.yaml"
echo "Processing model #4: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/apricot-frost-534_exp2_log_39_1.2"
config="configs/predict_nist_valid_beam10_exp2_log_39_1.2.yaml"
echo "Processing model #5: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &