# checkpoints/finetune_clean/exalted-elevator-545_exp3_mf10
# checkpoints/finetune_clean/dashing-grass-547_exp3_mf100
# checkpoints/finetune_clean/noble-glitter-546_exp3_mf10K
# checkpoints/finetune_clean/gallant-lion-533_exp2_log_29_1.28_exp3_mf10M
# checkpoints/finetune_clean/devoted-feather-548_exp3_selfies

CVD=1

model="../checkpoints/finetune_clean/exalted-elevator-545_exp3_mf10"
config="configs/predict_nist_valid_beam10_exp3_mf10.yaml"
echo "Processing model #1: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/dashing-grass-547_exp3_mf100"
config="configs/predict_nist_valid_beam10_exp3_mf100.yaml"
echo "Processing model #2: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/noble-glitter-546_exp3_mf10K"
config="configs/predict_nist_valid_beam10_exp3_mf10K.yaml"
echo "Processing model #3: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/gallant-lion-533_exp2_log_29_1.28_exp3_mf10M"
config="configs/predict_nist_valid_beam10_exp3_mf10M.yaml"
echo "Processing model #4: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

model="../checkpoints/finetune_clean/devoted-feather-548_exp3_selfies"
config="configs/predict_nist_valid_beam10_exp3_selfies.yaml"
echo "Processing model #5: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config &

# Optional: Wait for all background processes to finish
wait
