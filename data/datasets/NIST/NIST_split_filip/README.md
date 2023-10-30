# NIST split created by FILIP JOZEFOV's notebook Noteboks/data_preprocessing.ipynb
- NIST20 without unidentifiable spectra (~60k)
- train:valid:test split -> 0.8:0.1:0.1

## This dataset is created by the following steps:
1. Create jsonl files by the function msp_file_to_jsonl from spectra_process_utils.py (for 'train' and then 'test' and 'valid'):

```python
from spectra_process_utils import msp_file_to_jsonl
from pathlib import Path

dataset_path = Path("data/datasets/NIST/NIST_split_filip")
dataset_type = "train"
source_token = "<nist>"
msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                tokenizer_path,
                source_token,
                path_jsonl=dataset_path / f"{dataset_type}_{source_token}.jsonl",
                keep_spectra=True
                )
```

# PREPROCESSING STATS
 - test.jsonl
    0 no smiles
    48 smiles too long
    1 spectra corrupted
    697 spectra w/ too high mz
    2267 spectra w/ too many peaks
    totally 3013 issues
    discarded 2693/26365 spectra
    LENGTH: 23870
 - valid.jsonl
    0 no smiles
    39 smiles too long
    1 spectra corrupted
    663 spectra w/ too high mz
    2205 spectra w/ too many peaks
    totally 2908 issues
    discarded 2623/26493 spectra
    LENGTH: 23870
 - train.jsonl
    0 no smiles
    406 smiles too long
    14 spectra corrupted
    6004 spectra w/ too high mz
    20226 spectra w/ too many peaks
    totally 26650 issues
    discarded 24049/237455 spectra
    LENGTH: 213406