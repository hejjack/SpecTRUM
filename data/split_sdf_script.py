import math
import os
from pathlib import Path
from tqdm import tqdm

data_type = "12M_derivatized"
sdf_path = f"trial_set/tmp/{data_type}_plain.sdf"
num_of_chunks = 15 # 8M .. 10, 2M .. 5
num_of_separators = 10035233 # count with grep in terminal or sth (8M .. 8049006, 2M .. 1893040, 12M .. 10035233)
mols_in_one_chunk = math.ceil((num_of_separators+1)/num_of_chunks)
chunks_dir = f"trial_set/tmp/{data_type}_sdf_chunks"
Path(chunks_dir).mkdir(parents=True, exist_ok=True)


print(f">>> splitting into {num_of_chunks} chunks")
print(f">>> each chunk has at most {mols_in_one_chunk} mols")
with open(sdf_path, "r") as original:
    # iterate through chunks
    for i in tqdm(range(num_of_chunks)):
#         print(f"chunk {i} is in process")
        with open(f"{chunks_dir}/{data_type}_chunk{i}", "w+") as chunk_file:
            # copy lines until the number of $$$$ hits the mols_in_one_chunk (or END)
            dollar_count = 0
            while dollar_count < mols_in_one_chunk:
                # Get next line from file
                line = original.readline()
                if not line:
                    break

                chunk_file.write(line)
                if line[0]=="$" and line[:4] == "$$$$":
                    dollar_count += 1
print("all done")
