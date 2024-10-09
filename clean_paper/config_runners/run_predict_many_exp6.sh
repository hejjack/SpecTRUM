MODELS=(../checkpoints/finetune_clean/jolly-lion-562_exp6_45perc_frozen \
        ../checkpoints/finetune_clean/smooth-totem-563_exp6_72perc_frozen)

# Correct for loop with proper array reference
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"
    CUDA_VISIBLE_DEVICES=0 python predict.py --checkpoint "$model/checkpoint-73738" \
                                             --output-folder predictions \
                                             --config-file configs/predict_nist_valid_beam10.yaml &
done

# Optional: Wait for all background processes to finish
wait
