# NIST split created by FILIP JOZEFOV
- NIST20 without unidentifiable spectra (~60k)
- jsonl files are created by the function msp_file_to_jsonl from spectra_process_utils.py

```
dataset_path = Path("Spektro/MassGenie/data/datasets/NIST/NIST_split_filip")
dataset_type = "train"
source_token = "<nist>"
msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                tokenizer_path,
                source_token,
                path_jsonl=dataset_path / f"{dataset_type}_{source_token}.jsonl"
                )
```

# PREPROCESSING STATS
 - test_<nist>.jsonl
    0 no smiles
    36 smiles too long
    3 spectra corrupted
    762 spectra w/ too high mz
    2489 spectra w/ too many peaks
    totally 3290 issues
    discarded 3000/29040 spectra 
    LENGTH: 26040
 - train
    0 no smiles
    456 smiles too long
    13 spectra corrupted
    6601 spectra w/ too high mz
    22209 spectra w/ too many peaks
    totally 29279 issues
    discarded 26364/261272 spectra
    LENGTH: 234908

# train-valid split
 - the train split was further split into train and valid with random state 42
 - train_<nist>.jsonl : 208908
 - valid_<nist>.jsonl : 26000

 ```
 from sklearn.model_selection import train_test_split

train, valid = train_test_split(df, test_size=26000, random_state=42)

train.to_pickle(dataset_path / f"train_<nist>.jsonl")
valid.to_pickle(dataset_path / f"valid_<nist>.jsonl")
 ```