echo $CONDA_PREFIX should be /storage/brno2/home/ahajek/miniconda3/envs/NEIMSpy3
OMP_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0  # sth for OMP to not throw INFOs
ID=30M_rassp
python ../smi_preprocess_neims.py \
                        --smiles-path ../datasets/${ID}/${ID}.smi \
                        --dataset-id $ID \
                        --num-workers 32 \
                        --config-file ./config_preprocess_NEIMS.yaml \
                        --auto-remove \
