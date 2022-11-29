import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import re
from collections import defaultdict
from typing import Union, Dict, List
# from rdkit import Chem
# from rdkit.Chem import PandasTools
# from rdkit import Chem
# from rdkit.Chem.inchi import MolToInchiKey
import argparse
from tqdm import tqdm


import sys
sys.path.append("../bart_spektro")
from bart_spektro_tokenizer import BartSpektroTokenizer
from tokenizers import Tokenizer
from deepchem.feat.smiles_tokenizer import SmilesTokenizer


def print_args(args):
    """Print arguments.
    (borrowed from Megatron code https://github.com/NVIDIA/Megatron-LM)"""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)

        
def data_split(df, args):
    if args.train_split_ratio + args.test_split_ratio + args.valid_split_ratio != 1:
        print("!!!!!!! split ratios don't sum to 1 !!!!!!!")
        print("----> TRAIN SET will be extracted according to paramter, TEST and VALID wil be split proportionally")
    
    train_set = df.sample(frac=args.train_split_ratio, random_state=42)
    rest = df.drop(train_set.index)

    test_set = rest.sample(frac=args.test_split_ratio/(args.test_split_ratio+args.valid_split_ratio), random_state=42)
    valid_set = rest.drop(test_set.index)

    print("##### SPLITTING STATS #####")
    print("###########################")
    print(f"train len: {len(train_set)}\ntest len: {len(test_set)}\nvalid len: {len(valid_set)}\n")
    print(len(train_set) + len(test_set) + len(valid_set) == len(df), ": the sum matches len of the df")
    print("###########################")
    
    return train_set, test_set, valid_set
    

def my_position_ids_creator(intensities, log_base, log_shift, seq_len):
    x = (np.log(intensities)/log_base).astype(int) + log_shift
    x = x * (x > 0)
    return np.concatenate((x, [-1]*(seq_len-len(intensities)))).astype("int32")
    
def main():
    parser = argparse.ArgumentParser(description="Parse data preparation args")

    parser.add_argument("--df-path", type=str, required=True,
                        help="absolute path for loading a DataFrame file with generated in phase3")
    parser.add_argument("--save-pickle-path", type=str, required=True,
                        help="absolute path for saving a final file with filtered and enriched data")
    parser.add_argument("--seq-len", type=int, required=False, default=200,
                        help="sequence length of the encoder and decoder models")
    parser.add_argument("--log-base", type=float, required=False, default=1.7,
                        help="base of the logarithm for intensity binning")
    parser.add_argument("--train-split-ratio", type=float, required=False, default=0.8,
                        help="ratio of the train set in the final data split")
    parser.add_argument("--test-split-ratio", type=float, required=False, default=0.1,
                        help="ratio of the test set in the final data split")
    parser.add_argument("--valid-split-ratio", type=float, required=False, default=0.1,
                        help="ratio of the valid set in the final data split")
    parser.add_argument("--tokenizer", type=str, required=True, default="spektro", choices=['spektro', 'bbpe', 'wordpiece'],
                        help="type of SMILES tokenizer to use for preprocessing")
    
    
    args = parser.parse_args()

    print_args(args)
    
    # load the pickle from the first phase
    print("\n\n##### PHASE 4: LOADING DF #####")
    df = pd.read_pickle(args.df_path)
    
    if args.tokenizer == "spektro":
        tokenizer = BartSpektroTokenizer().init_tokenizer()
        print("##### TOKENIZATION AND PADDING #####")
        # pad the tokenized smiles and mzs
        pt = tokenizer.smiles_to_ids(tokenizer.pad_tok)[0]
        bt = tokenizer.smiles_to_ids(tokenizer.bos_tok)[0]
        et = tokenizer.smiles_to_ids(tokenizer.eos_tok)[0]

        df["tok_smiles"] = [tokenizer.smiles_to_ids(s) for s in tqdm(df["destereo_smiles"])]

        
    elif args.tokenizer == "wordpiece":
        raise Exception("WordPiece not implemented yet!")
        wp_path = "../tokenizer/wp_tokenizer/vocab.txt"
        tokenizer = SmilesTokenizer(vocab_path)
    
    elif args.tokenizer == "bbpe":
        bbpe_path = "../tokenizer/bbpe_tokenizer/bart_bbpe_1M_tokenizer.model"
        tokenizer = Tokenizer.from_file(bbpe_path)
        
        print("##### TOKENIZATION AND PADDING #####")
        # pad the tokenized smiles and mzs
        pt = tokenizer.token_to_id("<pad>")
        bt = tokenizer.token_to_id("<bos>")
        et = tokenizer.token_to_id("<eos>")

        df["tok_smiles"] = [tokenizer.encode(s).ids for s in tqdm(df["destereo_smiles"])]
                
    df["tok_smiles"] = [[bt] + ts + [et] + (args.seq_len-2-len(ts)) * [pt] for ts in tqdm(df.tok_smiles)]
    df["mz"] = [ts + (args.seq_len-len(ts)) * [pt] for ts in df.mz]
     
    print("##### CREATING MASKS #####")
    df["encoder_attention_mask"] = [[int(id_ != pt) for id_ in arr] for arr in tqdm(df.mz)]
    print("DONE eam")
    df["decoder_attention_mask"] = [[int(id_ != pt) for id_ in arr] for arr in tqdm(df.tok_smiles)]
    print("DONE dam")
    df["labels"] = [[id_ if int(id_ != pt) else -100 for id_ in arr] for arr in tqdm(df.tok_smiles)]
    print("DONE ll")    
        
    print("##### INTENSITY LOG BINNING #####")
    log_base = np.log(args.log_base)

    # add integral bins according to bounds (position_ids parameterzz)
    log_shift = 9 # =9 for 1.7 -> shift after logarithmization!!!!!!!!!!BACHA!!!!!!!!!!!!!!!!!!
    
    # empty positions (padding) get -1 (useless, BART doesn't have positions ids -> we create them)
    df["position_ids"] = [my_position_ids_creator(arr, log_base, log_shift, args.seq_len) for arr in df.intensity] 
    
    
    print("##### RENAMING, DROPPING USELESS COLUMNS #####")
    df.rename(columns={"mz": "input_ids", "tok_smiles": "decoder_input_ids"}, inplace=True)
    df.drop(columns=["intensity"], inplace=True)

    print(f"##### len after PHASE4: {len(df)}")
    
    df_train, df_test, df_valid = data_split(df, args)
    
    print("##### PICKELING PREPARED DATA #####")
    path_splt = args.save_pickle_path.split(".")
    path_template = "".join(path_splt[:-1])
    path_ext = path_splt[-1]
    
    parent_dir = "/".join(args.save_pickle_path.split("/")[:-1])  ######## FUNGUJE????
    os.makedirs(parent_dir, exist_ok=True)                        ######## FUNGUJE????
    df_train.to_pickle(path_template + "_train." + path_ext)
    df_test.to_pickle(path_template + "_test." + path_ext)
    df_valid.to_pickle(path_template + "_valid." + path_ext)

if __name__ == "__main__":
    main()    

   
