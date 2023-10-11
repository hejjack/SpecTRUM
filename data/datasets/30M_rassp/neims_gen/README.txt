This dataset was created by filtering 30M ZINC slice dataset. The filtering process 
suits the RASSP model trained by Ales Krenek (following the official RASSP model repo)

This dataset is created by the following steps:
1. Get the 30M_rassp.smi (A.Krenek's filtering method used on a 30M ZINC slice dataset)
2. feed this to run_neims_preprocess.sh

```bash
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
```

3. Set a config to config_preprocess_NEIMS.yaml:
```yaml
tmp_dir: "../datasets/tmp"
log_dir: "../datasets/log"
output_dir: "../datasets"
phases_to_perform: [1,2,3,4,5,6]   # 1_6, order is not important
lost_chunks: []
data_has_header: False     # whether the input has clean smiles file or csv (with <smiles zinc_id> structure)
max_smiles_len: 100
max_mz: 500
max_peaks: 200
seq_len: 200
log_base: 1.7
spectra_generator: "neims"
source_token: "<neims>"
train_split_ratio: 0.9
valid_split_ratio: 0.05
test_split_ratio: 0.05
tokenizer_path: "../../tokenizer/bbpe_tokenizer/bart_bbpe_1M_tokenizer.model"
neims_dir: "../../../NEIMS"
seed: 42
```



In the end, there is 4.8M SMILES in the dataset. The preprocessing further cut the number of molecules,
finally the lengths of splits are:
    - train 4283848
    - valid 237990
    - test  237990