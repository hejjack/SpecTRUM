echo $CONDA_PREFIX should be /storage/brno12-cerit/home/ahajek/.conda/envs/NEIMSpy3
ID=30M
python ../data_preprocess_all.py \
                        --dataset-id $ID \
                        --smiles-path ../datasets/30M/${ID}.smi \
                        --num-workers 90 \
                        --config-file ./config_preprocess_NEIMS.yaml \
                        --auto-remove \

# df = PandasTools.LoadSDF("../datasets/tmp/1K_chunk0_after_phase4_neims.sdf", idName="id", molColName='Molecule')
#/storage/projects/msml/mg_neims_branch/MassGenie/data/ZINC15/30M_slice/30M.smi \
                        # --auto-remove \
