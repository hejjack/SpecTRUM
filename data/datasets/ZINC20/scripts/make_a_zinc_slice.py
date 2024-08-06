
# this script is used to create a x% slice of a 1.8B ZINC15 database (2D-clean-annotated-druglike)
# it slices 
# creates a random sample from each file acording to sample_ratio. 
# (... the next step is to add the firstline and concat the files) 

import random
import os
import glob
from multiprocessing import Pool
import typer
import pathlib

app = typer.Typer(pretty_exceptions_enable=False)

def sampleData(subdir, output_dir, sample_ratio, seed):
    print("subdir {} RUNNING".format(subdir))
    subdir_name = subdir.split("/")[-1]
    outputFile = output_dir + f"/{subdir_name}.smi"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
                length = int(len(data) * sample_ratio)
                random.seed(seed)
                used = random.sample(data, length)
                newfile.writelines("%s" % line for line in used)
    print(f"subdir {subdir} DONE")
    return subdir

@app.command()
def main(
    input_dir: str = typer.Option(..., help="Path to the directory with tranches to sample from"),
    output_dir: str = typer.Option(..., help="Path to the directory where to save the sampled tranches"),
    sample_ratio: float = typer.Option(..., help="Ratio of the data to sample"),
    num_workers: int = typer.Option(8, help="Number of workers to use"),
    seed: int = typer.Option(42, help="Seed for the random number generator"),
) -> None:
    
    subdirs = sorted(glob.glob(input_dir + "/*"))
    print(f"subdirs: {subdirs}")
    # init the output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with Pool(processes=num_workers) as p:
        p.map(sampleData, [(subdir, output_dir, sample_ratio, seed) for subdir in subdirs])
    processes = {}
    for i in range(1, len(subdirs)+1):
        processes[f"process{i}"] = multiprocessing.Process(target=sampleData, 
                                                           args=(subdirs[i-1], output_dir, sample_ratio, seed))
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    print("all done")


if __name__ == "__main__":
    app()

    
