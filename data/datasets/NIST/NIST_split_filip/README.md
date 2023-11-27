## This dataset is created by the following steps:
1. Create msp splits using my updated version of FILIP JOZEFOV's notebook data/nist_cleaning_splitting.ipynb
   - it dorps ~60k spectra that don't have some form of proper identifier (smiles, inchikey)
   - split 0.8:0.1:0.1
2. Create jsonl files by the function msp_file_to_jsonl from spectra_process_utils.py (for 'train' and then 'test' and 'valid'):

```python
from spectra_process_utils import msp_file_to_jsonl
from pathlib import Path

dataset_path = Path("data/datasets/NIST/NIST_split_filip")
dataset_type = "train"
source_token = "<nist>"
msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                tokenizer_path,
                source_token,
                path_jsonl=dataset_path / f"{dataset_type}.jsonl",
                keep_spectra=True
                )
```

# PREPROCESSING STATS
 - test.jsonl
   0 no smiles
   52 smiles too long
   0 spectra corrupted
   780 spectra w/ too high mz
   2463 spectra w/ too many peaks
   totally 3295 issues
   discarded 2957/29218 spectra 
   LENGTH: 26261
 - valid.jsonl
   0 no smiles
   55 smiles too long
   0 spectra corrupted
   709 spectra w/ too high mz
   2339 spectra w/ too many peaks
   totally 3103 issues
   discarded 2809/29053 spectra 
   LENGTH: 26244
 - train.jsonl
    0 no smiles
    406 smiles too long              ??? deprecated
    14 spectra corrupted             ??? deprecated
    6004 spectra w/ too high mz      ??? deprecated
    20226 spectra w/ too many peaks  ??? deprecated
    totally 26650 issues             ??? deprecated
    discarded 24049/237455 spectra   ??? deprecated
    LENGTH: 208448                   CORRECT       