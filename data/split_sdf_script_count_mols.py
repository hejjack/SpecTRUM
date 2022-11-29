import math
import os

data_type = "12M_derivatized"
sdf_path = f"trial_set/tmp/{data_type}_plain.sdf"

dollar_count = 0 
with open(sdf_path, "r") as original:
    while True:
        line = original.readline()
        if not line:
            break
        if line[0]=="$" and line[:4] == "$$$$":        
            dollar_count += 1
print("number of dollars: ", dollar_count)

