CUDA_VISIBLE_DEVICES=0 python ../predict.py --checkpoint ../checkpoints/finetune/different-mountain-417_NClike/checkpoint-73440 \
                                            --output-folder ../predictions \
                                            --config-file ../configs/predict_config_nist_valid_old_gen.yaml \
