# ../checkpoints/finetune/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k/checkpoint-147476
CVD=0

model="../checkpoints/finetune/fearless-wildflower-490_rassp1_neims1_224kPretrain_148k"
config="configs/predict_nist_valid_beam10.yaml"
echo "Processing model #1: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-147476" \
                                            --output-folder predictions \
                                            --config-file $config &

config="configs/predict_nist_valid_greedy.yaml"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-147476" \
                                            --output-folder predictions \
                                            --config-file $config &