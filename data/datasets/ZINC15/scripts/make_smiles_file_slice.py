# this script is used to take N random lines from a file and save it to a new file. 

import numpy as np
import os

seed = 42
N = 30000000

inputFile = "../43M_slice/all_clean_notest_43M.smi"
outputDir = "../30M_slice"
outputFileName = "all_clean_30M.smi"

if __name__ == "__main__":
    # init the output dir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    np.random.seed(seed)
    with open(inputFile, "r") as inputf, open(f"{outputDir}/{outputFileName}", "w") as outputf:
        print("## READING FILE")
        smiles = np.array(inputf.readlines())
        print("## SLICING")
        N_smiles = np.random.choice(smiles, N, replace=False)
        print(f"## WRITING (length: {len(N_smiles)})")
        outputf.writelines(N_smiles.tolist())

