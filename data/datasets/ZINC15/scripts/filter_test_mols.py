# this script takes a big file with SMILES (input) and another smaller file with SMILES in .msp file (forbidden)
# that we dont want to have in the input file (basically we filter test set of downstream evaluation
# out of training data)

from matchms.importing import load_from_msp
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import multiprocessing

inputFile = "../43M_slice/all_clean_43M.smi"
outputFile = "../43M_slice/all_clean_notest_43M.smi"
forbidden_msp_file = "../../../../../NIST_split/test.msp"

n_processes = 120

# test
#inputFile = "./input.smi"
#outputFile = "./output.smi"
#forbidden_msp_file = "../../../../../NIST_split/test.msp"

def filter_forbidden(input_array, forbidden_array, chunk_id, return_dict):
    print(f"## FILTERING CHUNK num {chunk_id}")
    output_smiles = input_array[~np.isin(input_array, forbidden_array)].tolist()
    print(f"## CHUNK num {chunk_id} DONE (length: {len(output_smiles)})")
    return_dict[chunk_id] = "\n".join(output_smiles)+"\n"

if __name__ == "__main__":
    forbidden_gen = load_from_msp(forbidden_msp_file, metadata_harmonization=False)
    forbidden_smiles = [s.metadata["smiles"] for s in forbidden_gen]

    # canonization of forbidden smiles
    forbidden_cans = []
    for smi in tqdm(forbidden_smiles):
        try: 
            forbidden_cans.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi),True))
        except:
            forbidden_cans.append(smi)
    print(forbidden_cans[:5])
    forbidden_cans = np.array(forbidden_cans)
    with open(inputFile, "r") as inputf:
        print("## READING input FILE")
        input_smiles = np.array(inputf.read().splitlines())

        print(f"## SPLITTING INPUT ARRAY TO {n_processes} subarrays")
        input_chunks = np.array_split(input_smiles, n_processes)

    print("## INITIALIZING multiprocessing")
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = {}
    for i in range(1, n_processes+1):
        processes[f"process{i}"] = multiprocessing.Process(target=filter_forbidden, args=(input_chunks[i-1],
                                                                                          forbidden_cans,
                                                                                          i,
                                                                                          return_dict))
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    print(f"WRITING ALL TO FILE: {outputFile}")
    with open(outputFile, "w") as outputf:
        for val in return_dict.values():
            outputf.write(val)
