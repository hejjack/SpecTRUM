
# this script is used to create a 10% slice of a 1.8B ZINC15 database (2D-clean-annotated-druglike)
# it slices 
# creates a random sample from each file acording to sampleRatio. 
# (... the next step is to add the firstline and concat the files) 

import random
import os
import glob
import multiprocessing

seed = 42
sampleRatio = 4/10

inputDir = "../100M_slice"
outputDir = "../40M_slice"

def sampleData(subdir):
    print("subdir {} RUNNING".format(subdir))
    subdir_name = subdir.split("/")[-1]
    outputFile = outputDir + f"/{subdir_name}.smi"

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    files = sorted(glob.glob(subdir + "/*"))
    with open(outputFile, "w+", encoding="utf-8") as newfile:
        for inputFile in files:
            with open(inputFile, "r+", encoding="utf-8") as oldfile:
                data = oldfile.readlines()
                try:
                    del data[0] # first line is csv legend, we don't want to mix that in the slice
                except Exception as e: # work on python 3.x
                    print(str(e) + f"\n#### Error happened on file {inputFile}")
                    continue
                # del data[-1] # last line is a newline, we don't wanna sample that
                length = int(len(data) * sampleRatio)
                random.seed(seed)
                used = random.sample(data, length)
                newfile.writelines("%s" % line for line in used)
    print(f"subdir {subdir} DONE")

if __name__ == "__main__":
    subdirs = sorted(glob.glob(inputDir + "/*"))
    print(f"subdirs: {subdirs}")
    # init the output dir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    processes = {}
    for i in range(1, len(subdirs)+1):
        processes[f"process{i}"] = multiprocessing.Process(target=sampleData, args=(subdirs[i-1],))
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    print("all done")
