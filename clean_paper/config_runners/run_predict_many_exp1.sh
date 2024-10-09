# ../checkpoints/finetune_clean/major-capybara-531_exp1_pos_emb


CVD=0

model="../checkpoints/finetune_clean/major-capybara-531_exp1_pos_emb"
config="configs/predict_nist_valid_beam10_exp1_pos_emb.yaml"
echo "Processing model #1: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint "$model/checkpoint-73738" \
                                            --output-folder predictions \
                                            --config-file $config