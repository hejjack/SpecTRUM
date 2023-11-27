CUDA_VISIBLE_DEVICES=0 python ../predict.py --checkpoint ../checkpoints/finetune/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/checkpoint-73440 \
                                            --output-folder ../predictions \
                                            --config-file ../configs/predict_config_nist_valid.yaml \
                                            # --data-range "0:40"