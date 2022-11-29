import pandas as pd
# import numpy as np
from pandas import DataFrame
import os
from rdkit import Chem
from rdkit.Chem import PandasTools
# from rdkit.Chem.inchi import MolToInchiKey
import argparse
from tqdm import tqdm

def print_args(args):
    """Print arguments.
    (borrowed from Megatron code https://github.com/NVIDIA/Megatron-LM)"""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)
        

def filter_corrupted_smiles(df):
    for index, row in tqdm(df.iterrows()):
#         try:
#             mol = Chem.MolFromSmiles(row["destereo_smiles"])
#         except:
#             df.drop(index, inplace=True)
        if not Chem.MolFromSmiles(row["destereo_smiles"]):
            df.drop(index, inplace=True)    
    print("Length after filtering out corrupted molecules:", len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Parse data preparation args")
    
    parser.add_argument("--df-path", type=str, required=True,
                        help="absolute path to a dataframe form phase1")
    parser.add_argument("--destereo-path", type=str, required=True,
                        help="absolute path to a list of destereochemicalized smiles")
    parser.add_argument("--save-pickle-path", type=str, required=False,
                        help="absolute path for saving a dataframe before spectra generation")
    parser.add_argument("--load-pickle-path", type=str, required=False,
                        help="absolute path for loading a dataframe (with destereo) before sdf generation")
    parser.add_argument("--save-plain-sdf-path", type=str, required=True,
                        help="absolute path for saving a sdf for the next (spectra generation) step")
    parser.add_argument("--max-smiles-len", type=int, required=False, default=100,
                        help="max len of SMILES strings")
    
    args = parser.parse_args()
    print_args(args)
    
    if not args.load_pickle_path:
        print("\n\n##### PHASE 2: LOADING DATAFRAME #####")  
        with open(args.df_path, "rb") as dff:
            df = pd.read_pickle(dff)

        print("##### LOADING DESTEREO SMILES #####")
        with open(args.destereo_path, "r") as destereo_f:
            sms = destereo_f.readlines()
            sms = ["".join(s.split()) for s in tqdm(sms)] # strip whitespaces
        df["destereo_smiles"] = sms
        df.drop(columns=["canon_smiles"], inplace=True)

        # deduplication
        df.drop_duplicates(subset=["destereo_smiles"], inplace=True)
        print("##### DEDUPLICATION DONE #####")  
        print(f"data len: {len(df)}")

        # long SMILES filtering
        print("##### FILTERING LONG SMILES #####")
#         print(df.columns)
        df = df.loc[df['destereo_smiles'].str.len() < args.max_smiles_len]
        print(f"data len: {len(df)}")
        
        print("##### INIT PHASE DONE #####")  
        print("dataset is now canonicalized, deduplicated, and long SMILES (over 99) are filtered out.")

        # save to pickle after second phase
        if args.save_pickle_path:
            print("##### PICKELING DATA AFTER PHASE2 #####")        
            df.to_pickle(args.save_pickle_path)
        else:
            print("##### NO SAVE PICKLE PATH => NO PICKELING #####")
    else:
        print("##### LOADING PICKLED DF WITH DESTEREO #####")
        with open(args.load_pickle_path, "rb") as dff:
            df = pd.read_pickle(dff)
            
    # filter corrupted SMILES
    print("##### FILTERING CORRUPTED SMILES #####")  
    df = filter_corrupted_smiles(df)
    print(f"data len: {len(df)}")
    
    # add molecule structure to DF
    print("##### ADDING MOLECULE STRUCTURE #####")  
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='destereo_smiles', molCol='ROMol')

    # exporting to SDF
    print(f"##### EXPORTING TO SDF #####")
    PandasTools.WriteSDF(df, args.save_plain_sdf_path, idName="zinc_id", properties=list(df.columns))
    
    print(f"##### len after PHASE2: {len(df)}")

if __name__ == "__main__":
    main()

    

    
