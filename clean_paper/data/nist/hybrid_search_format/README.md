These files were created with the save_as_msp() function from matchms.exporting. The only difference is that the style is set to "nist".
Python calls:

```python
from matchms.importing import load_from_msp
from matchms.exporting import save_as_msp

# Save as NIST style
nist_test_path = "../clean_paper/data/nist/test.msp"
nist_train_path = "../clean_paper/data/nist/train.msp"

output_dir = "../clean_paper/data/nist/hybrid_search_format/"

nist_test = list(load_from_msp(nist_test_path, metadata_harmonization=False))
nist_train = list(load_from_msp(nist_train_path, metadata_harmonization=False))

save_as_msp(nist_test, output_dir + "test_hybrid.msp", style="nist")
save_as_msp(nist_train, output_dir + "train_hybrid.msp", style="nist")
```