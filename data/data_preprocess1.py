import pandas as pd
# import numpy as np
from pandas import DataFrame
import os
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
import argparse
from tqdm import tqdm

def print_args(args):
    """Print arguments.
    (borrowed from Megatron code https://github.com/NVIDIA/Megatron-LM)"""
    all_args = "arguments:\n"
    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        line = '  {} {} {}'.format(arg, dots, getattr(args, arg))
        print(line, flush=True)
        all_args = all_args + line + "\n"
    return all_args
        
def clean_smiles(data_path):
    data_path_clean = data_path + ".clean"
    with open(data_path, "r") as dirty, open(data_path_clean, "w") as clean:
        clean.write("smiles zinc_id\n")
        for line in dirty:
            if line[0] == "s" or line[0] == "<":
                continue
            else:
                clean.write(line)
    os.remove(data_path)
    os.rename(data_path_clean, data_path)
    

def main():
    parser = argparse.ArgumentParser(description="Parse data preparation args")
    #####################################
    parser.add_argument("--data-path", type=str, required=True,
                        help="absolute path to a list of zinc_ids and smiles")
    parser.add_argument("--save-pickle-path", type=str, required=True,
                        help="absolute path for saving a dataframe before destereochemicalization")
    parser.add_argument("--save-canon-path", type=str, required=True,
                        help="absolute path for saving canon smiles prepared for obabel destereochemicalization")
   

    args = parser.parse_args()

    print_args(args)
    
    print("\n\n##### PHASE 1 #####")        
    # clean_smiles(args.data_path) # change or don't use, depending on the data file

    print("##### READING CSV #####")  
    df = pd.read_csv(args.data_path, sep=" ")
    print("##### READING DONE #####")  
    
    print("##### CANONICALIZATION #####")  
    cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in tqdm(df["smiles"])]
    df["canon_smiles"] = cans
    df = df[["zinc_id", "smiles", "canon_smiles"]]
    print("##### CANONICALIZATION DONE #####")  

    print("##### SAVING CANON SMILES ALONE #####")  
    sms = df["canon_smiles"].tolist()
    with open(args.save_canon_path, "w+") as out:
        for s in tqdm(sms):
            out.write(s + "\n")
    
    # save to pickle after first phase
    print("##### PICKELING DATA AFTER INIT PHASE #####")        
    print(f"##### len after PHASE1: {len(df)}")
    df.to_pickle(args.save_pickle_path) 

if __name__ == "__main__":
    main()
        
    
    
    