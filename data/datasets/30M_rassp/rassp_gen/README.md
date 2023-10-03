This dataset was created by filtering 30M ZINC slice dataset. The filtering process 
suits the RASSP model trained by Ales Krenek (following the official RASSP model repo)

WARNING! : this dataset was used for the initial set of experiments before discovering a dataleak => moved to DEPRECATED

This dataset is created by the following steps:
1. Get the 30M_rassp.smi (A.Krenek's filtering method used on a 30M ZINC slice dataset)
2. feed this to run_neims_preprocess.sh in a following way:

```bash
python msp_preprocess_rassp.py --input-dir datasets/30M_rassp/rassp_gen/msps \
                                     --output-dir datasets/30M_rassp/rassp_gen/jsonls \
                                     --num-processes 64 \
                                     --concat \
```

3. split the data to train, test, valid:

```python
def data_split(df, train_test_valid_ratio: list):
    """split the df into train, test and valid sets"""
    if sum(train_test_valid_ratio) != 1:
        print("train_test_valid_ratio does not sum to 1")
        return None, None, None
    train_set = df.sample(
        frac=train_test_valid_ratio[0], random_state=42)
    rest = df.drop(train_set.index)

    test_set = rest.sample(frac=train_test_valid_ratio[1]/(train_test_valid_ratio[1]+train_test_valid_ratio[2]),
                           random_state=42)
    valid_set = rest.drop(test_set.index)
    print(f"train len: {len(train_set)}, test len: {len(test_set)}, valid len: {len(valid_set)}")
    return train_set, test_set, valid_set

df = pd.read_json("data/datasets/30M_rassp/rassp_gen/jsonls/all.jsonl", lines=True)

train, test, valid = data_split(df, [0.9, 0.05, 0.05])

train.to_json("data/datasets/30M_rassp/rassp_gen/train.jsonl", orient="records", lines=True)
test.to_json("data/datasets/30M_rassp/rassp_gen/test.jsonl", orient="records", lines=True)
valid.to_json("data/datasets/30M_rassp/rassp_gen/valid.jsonl", orient="records", lines=True)
```


There is 4.8M SMILES in the dataset. The preprocessing further cut the number of molecules,
finally the lengths of splits are:
    - train 4322581
    - valid 240143
    - test 240144 