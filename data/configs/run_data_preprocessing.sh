echo $CONDA_PREFIX should be /storage/brno12-cerit/home/ahajek/.conda/envs/NEIMSpy3
OMP_NUM_THREADS=1
ID=30M
python ../data_preprocess_all.py \
                        --smiles-path ../ZINC15/${ID}_slice/${ID}.smi \
                        --dataset-id $ID \
                        --num-workers 80 \
                        --config-file ./config_preprocess_NEIMS.yaml \
                        --auto-remove \

# df = PandasTools.LoadSDF("../datasets/tmp/1K_chunk0_after_phase4_neims.sdf", idName="id", molColName='Molecule')
#/storage/projects/msml/mg_neims_branch/MassGenie/data/ZINC15/30M_slice/30M.smi \
                        # --auto-remove \

                        # --smiles-path ../trial_set/${ID}.smi \
