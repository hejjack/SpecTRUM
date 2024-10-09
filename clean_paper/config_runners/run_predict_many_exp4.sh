MODELS=("../checkpoints/finetune_clean/avid-rain-560_exp4_rassp" \
        "../checkpoints/finetune_clean/effortless-river-558_exp4_rassp_neims" \
        "../checkpoints/finetune_clean/valiant-totem-557_exp4_neims")

# Correct for loop with proper array reference
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"
    CUDA_VISIBLE_DEVICES=1 python predict.py --checkpoint "$model/checkpoint-73738" \
                                             --output-folder predictions \
                                             --config-file configs/predict_nist_valid_beam10.yaml &
done

# Optional: Wait for all background processes to finish
wait