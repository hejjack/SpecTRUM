echo $CONDA_PREFIX should be /storage/brno2/home/ahajek/miniconda3/envs/NEIMSpy3
OMP_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0  # sth for OMP to not throw INFOs
ID=30M_rassp
python ../data_preprocess_all.py \
                        --smiles-path ../datasets/${ID}/${ID}.smi \
                        --dataset-id $ID \
                        --num-workers 32 \
                        --config-file ./config_preprocess_NEIMS.yaml \
                        --auto-remove \

# df = PandasTools.LoadSDF("../datasets/tmp/1K_chunk0_after_phase4_neims.sdf", idName="id", molColName='Molecule')
#/storage/projects/msml/mg_neims_branch/MassGenie/data/ZINC15/30M_slice/30M.smi \
                        # --auto-remove \

                        # --smiles-path ../trial_set/${ID}.smi \
