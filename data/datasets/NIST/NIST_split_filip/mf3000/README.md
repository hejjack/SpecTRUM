## This dataset is created by the following steps:
1. Create msp splits using my updated version of FILIP JOZEFOV's notebook data/nist_cleaning_splitting.ipynb
   - it dorps ~60k spectra that don't have some form of proper identifier (smiles, inchikey)
   - split 0.8:0.1:0.1
2. Create jsonl files by the function msp_file_to_jsonl from spectra_process_utils.py (for 'train' and then 'test' and 'valid'):

```python
from spectra_process_utils import msp_file_to_jsonl
from pathlib import Path
from train_bart import build_tokenizer

tokenizer_path = "tokenizer/bbpe_tokenizer/bart_bbpe_tokenizer_1M_mf3000.model"
tokenizer = build_tokenizer(tokenizer_path)


for dataset_type in ["train", "valid", "test"]:
    dataset_path = Path("data/datasets/NIST/NIST_split_filip")
    source_token = "<nist>"
    msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                    tokenizer,
                    source_token,
                    path_jsonl=dataset_path / "mf3000" / f"{dataset_type}.jsonl",
                    keep_spectra=True
                    )
```

###################### TOTO JE DIVNY #############################
# PREPROCESSING STATS
 - test.jsonl
   0 no mol_repr
   104 mol_reprs too long
   0 spectra corrupted
   1560 spectra w/ too high mz
   4926 spectra w/ too many peaks
   totally 6590 issues
   discarded 5914/58436 spectra
   LENGTH: 52522                    
 - valid.jsonl
   0 no mol_repr
   110 mol_reprs too long
   0 spectra corrupted
   1418 spectra w/ too high mz
   4678 spectra w/ too many peaks
   totally 6206 issues
   discarded 5618/58106 spectra 
   LENGTH: 52488                   
 - train.jsonl
   0 no mol_repr
   726 mol_reprs too long
   0 spectra corrupted
   11746 spectra w/ too high mz
   39788 spectra w/ too many peaks
   totally 52260 issues
   discarded 47154/464050 spectra
    LENGTH: 416896                          
