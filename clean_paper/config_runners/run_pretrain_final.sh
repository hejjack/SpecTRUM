# pretrain RASSP_1 NEIMS_1 NIST_0.1
python train_bart.py --config-file configs/pretrain_final.yaml \
                     --resume-id k3vjifm8 \
                     --checkpoint ../checkpoints/pretrain_clean/pleasant-resonance-556_exp4_rassp_neims_nist/checkpoint-112000 \
                     --additional-info "_final_224k" \
                     --additional-tags "final:rassp:neims:nist:from_pretrain" \
                     --wandb-group pretrain_clean \