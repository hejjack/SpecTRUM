# pretrain RASSP_1 NEIMS_1 NIST_0.1
python train_bart.py --config-file configs/pretrain_exp4_rassp_neims_nist.yaml \
                        --additional-info "_exp4_rassp_neims_nist" \
                        --additional-tags "exp4:rassp:neims:nist:from_scratch" \
                        --wandb-group pretrain_clean \