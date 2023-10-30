# Precompute index of similarities with reference library for denovo evaluation
# For every sample in dataset find the most similar molecule in the reference library,
# and save its index, spectral imilarity and SMILES similarity.
# It takes a lot of time => we utilize parallelization. 


import argparse
import json
import pandas as pd
import numpy as np
from matchms.similarity import CosineGreedy
from matchms import Spectrum
from tqdm import tqdm
from rdkit import Chem, DataStructs
import multiprocessing
from pathlib import Path

def find_best_indexes_and_similarities(df_query, ref_spectra, ref_fps, outfile_path, process_id=None):
    """find the best match (according to spectra similarity) for every sample in df_query
      and add its index, spectral similarity and SMILES similarity to the row. 
      Write the row immediately to a new file.
      
      Parameters
      ----------
      df_query : pd.DataFrame
          dataframe with query samples
      ref_spectra : list[Spectrum]
          list of reference spectra
      ref_fps : list[Chem.RDKFingerprint]
          list of reference fingerprints
      outfile_path : str
          path to the output file
      process_id : int, optional
          id of the process to print when multiprocessing, by default None"""
    
    if process_id is not None:
        print(f"process {process_id} started")

    outfile = open(outfile_path, "w+")

    cosine_greedy = CosineGreedy()
    best_spec_simils = []
    best_smiles_simils = []
    best_indexes = []
    for _, query_row in tqdm(df_query.iterrows(), desc="outer loop"):
        query_spec = Spectrum(mz=np.array(query_row.mz),
                        intensities=np.array(query_row.intensity),
                        metadata_harmonization=False)
        query_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(query_row.smiles))
        best_spec_simil = 0
        best_index = 0
        for index, ref_spec in enumerate(ref_spectra):
            spec_score = float(cosine_greedy.pair(query_spec, ref_spec)["score"])
            if spec_score > best_spec_simil:
                best_spec_simil = spec_score
                best_index = index
        smiles_score = DataStructs.FingerprintSimilarity(query_fp, ref_fps[best_index])
        
        best_spec_simils.append(best_spec_simil)
        best_indexes.append(best_index)  
        best_smiles_simils.append(smiles_score)
        
        # update row and write it to file
        query_row["index_of_closest"] = best_index
        query_row["spectra_sim_of_closest"] = best_spec_simil
        query_row["smiles_sim_of_closest"] = smiles_score
        query_row = dict(query_row)
        outfile.write(json.dumps(query_row))
        outfile.write("\n")

    assert len(best_spec_simils) == len(df_query), "best_simils and df have different lengths"
    assert len(best_indexes) == len(df_query), "best_indexes and df have different lengths"
    assert len(best_smiles_simils) == len(df_query), "best_smiles_simils and df have different lengths"

    outfile.close()

    if process_id is not None:
        print(f"process {process_id} finished")

    return pd.Series(best_indexes), pd.Series(best_spec_simils, dtype=np.float64), pd.Series(best_smiles_simils, dtype=np.float64)


def denovo_preprocess_mp(df_reference, df_query, outfile_path, num_processes=1):
    # create fingerprints and spectra fo reference dataset
    ref_spectra = [Spectrum(mz=np.array(ref_row.mz),
                            intensities=np.array(ref_row.intensity),
                            metadata_harmonization=False) 
                            for _, ref_row in tqdm(df_reference.iterrows(), desc="precomputing ref_spectra")]
    ref_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(ref_row.smiles)) 
               for _, ref_row in tqdm(df_reference.iterrows(), desc="precomputing ref_fps")]
    assert len(ref_spectra) == len(df_reference), "ref_spectra and df have different lengths"
    assert len(ref_fps) == len(df_reference), "ref_fps and df have different lengths"

    # split data 
    idxs = np.array_split(np.arange(len(df_query)), num_processes)
    
    # create file names
    outfile_path = Path(outfile_path)
    outfile_paths = [outfile_path.parent / f"{outfile_path.stem}_{i}{outfile_path.suffix}" for i in range(num_processes)]

    # run multiprocess
    print("STARTING MULTIPROCESSING")
    processes = {}
    for i in range(num_processes):
        processes[f"process{i}"] = multiprocessing.Process(target=find_best_indexes_and_similarities, 
                                                           args=(df_query.iloc[idxs[i]], 
                                                                 ref_spectra,
                                                                 ref_fps,
                                                                 outfile_paths[i]),
                                                           kwargs=dict(process_id=i))
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    # concat files
    print("CONCATENATING FILES")
    with open(outfile_path, "w+") as outfile:
        for i in range(num_processes):
            with open(outfile_paths[i], "r") as f:
                    for line in f:
                        outfile.write(line)
    
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, required=True, help="path to the reference dataset")
    parser.add_argument("--query", type=str, required=True, help="path to the query dataset")
    parser.add_argument("--outfile", type=str, required=True, help="path to the output file")
    parser.add_argument("--num_processes", type=int, default=1, help="number of processes to use")
    args = parser.parse_args()

    # load data
    print("LOADING DATA")
    df_reference = pd.read_json(args.reference, lines=True, orient="records")
    df_query = pd.read_json(args.query, lines=True, orient="records")

    # run precompute
    denovo_preprocess_mp(df_reference, df_query, args.outfile, args.num_processes)