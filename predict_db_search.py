import sys
import io
import os
import shutil
import time
from pathlib import Path
import typer
import yaml
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
from copy import deepcopy
import multiprocessing
import heapq

from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from matchms.similarity import ModifiedCosine, CosineGreedy
from matchms import Spectrum

from utils.spectra_process_utils import get_fp_generator, get_fp_simil_function
from utils.data_utils import SpectroDataCollator, filter_datapoints
from bart_spektro.modeling_bart_spektro import BartSpektroForConditionalGeneration
from utils.general_utils import build_tokenizer, get_sequence_probs, timestamp_to_readable, hours_minutes_seconds
from utils.general_utils import move_file_pointer, line_count, dummy_generator, timestamp_to_readable, hours_minutes_seconds
from predict import open_files, get_unique_predictions, get_canon_predictions

RDLogger.DisableLog('rdApp.*')

app = typer.Typer(pretty_exceptions_enable=False)


# def get_unified_similarity_measure(similarity_type, simil_function):
#     """Get the similarity measure based on the ranking function.
#     Currently supports modified cosine (hss), cosine greedy (sss) and morgan tanimoto (morgan_tanimoto).
#     The outputed function takes two dicts (df rows) with corresponding precomputed fields and returns the similarity score.

#     The ranking functions need:
#     - hss: spectrum field in both dicts with matchms.Specrta that include mz, intensities and metadata with precursor_mz
#     - sss: spectrum field in both dicts with matchms.Specrta that include mz and intensities
#     - morgan_tanimoto: fp field in both dicts with the precomputed fingerprints
#     Parameters
#     ----------
#     similarity_type : str
#         The type of similarity function to use
#     """
#     if similarity_type == "modified_cosine":
#         def similarity_measure(ref, query):
#             return .(ref["spectrum"], query["spectrum"])
#         similarity_measure = ModifiedCosine(tolerance=0.005)
#     elif similarity_type == "cosine_greedy":
#         similarity_measure = CosineGreedy(tolerance=0.005)
#     else:
#         raise ValueError(f"Unknown similarity type: {similarity_type}")
#     return similarity_measure


def find_candidates_in_database_for_one_query(query_row, df_references, ranking_function, similarity_measure, num_candidates):

    if ranking_function in ["hss", "sss"]:
        query = Spectrum(mz=np.array(query_row.mz),
                        intensities=np.array(query_row.intensity),
                        metadata={"precursor_mz": round(ExactMolWt(Chem.MolFromSmiles(query_row.smiles)))},
                        metadata_harmonization=False)
    else:
        query = query_row["fp"]
    candidates = []

    for i, ref_row in df_references.iterrows():
        if ranking_function in ["hss", "sss"]:
            similarity_score = similarity_measure(query, ref_row["spectrum"])
        else:
            similarity_score = similarity_measure(query, ref_row["fp"])

        if len(candidates) < num_candidates:
            heapq.heappush(candidates, (similarity_score, ref_row["smiles"]))
        else:
            if similarity_score > candidates[0][0]:
                heapq.heappushpop(candidates, (similarity_score, ref_row["smiles"]))

    return candidates


def find_candidates_in_database_for_all_queries(df_queries,
                                                df_references,
                                                ranking_function,
                                                num_candidates,
                                                outfile_path,
                                                process_id=0):
    """Find the top num_candidates most similar spectra in the database to each query spectrum"""

    if ranking_function == "hss":
        similarity_measure = lambda x,y: ModifiedCosine().pair(x, y)["score"].round(5).item()
        ref_spectra = [Spectrum(mz=np.array(ref_row.mz),
                        intensities=np.array(ref_row.intensity),
                        metadata={"precursor_mz": round(ExactMolWt(Chem.MolFromSmiles(ref_row.smiles)))},    # rounded exact mass came out as the most reliable
                        metadata_harmonization=False)
                        for _, ref_row in tqdm(df_references.iterrows(), desc="precomputing ref_spectra")]
        df_references["spectrum"] = ref_spectra
    elif ranking_function == "sss":
        similarity_measure = lambda x,y: CosineGreedy().pair(x, y)["score"].round(5).item()
        ref_spectra = [Spectrum(mz=np.array(ref_row.mz),
                        intensities=np.array(ref_row.intensity),
                        metadata_harmonization=False)
                        for _, ref_row in tqdm(df_references.iterrows(), desc="precomputing ref_spectra")]

        df_references["spectrum"] = ref_spectra
    elif ranking_function == "morgan_tanimoto":
        similarity_measure = lambda x, y: round(get_fp_simil_function("tanimoto")(x, y), 5)
        fpgen = get_fp_generator("morgan")

        ref_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm(df_references["smiles"], desc="Reference data fps precomputing")]
        query_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm(df_queries["smiles"], desc="Query data fps precomputing")]

        df_references["fp"], df_queries["fp"] = ref_fps, query_fps
    else:
        raise ValueError(f"Unknown similarity type: {ranking_function}")

    # go through all queries
    outfile = open(outfile_path, "w+")
    for i, query_row in tqdm(df_queries.iterrows(), desc=f"Process {process_id}"):
        candidates = find_candidates_in_database_for_one_query(query_row, df_references, ranking_function, similarity_measure, num_candidates)
        candidates_dict = {smiles: score for score, smiles in candidates}
        outfile.write(json.dumps(candidates_dict) + "\n")

    outfile.close()


def parallel_db_search(df_references,
                       df_queries,
                       config,
                       outfile_path,
                       num_workers):
    """Extract queries and references based on the ranking function and run the database search in parallel"""

    # split data
    idxs = np.array_split(np.arange(len(df_queries)), num_workers)

    # create file names
    tmp_folder_path = outfile_path.parent / "tmp"
    tmp_folder_path.mkdir(parents=True, exist_ok=True)
    tmp_paths = [tmp_folder_path / f"{outfile_path.stem}_{i}{outfile_path.suffix}" for i in range(num_workers)]

    # run multiprocess
    print("STARTING MULTIPROCESSING")
    processes = {}
    for i in range(num_workers):
        processes[f"process{i}"] = multiprocessing.Process(target=find_candidates_in_database_for_all_queries,
                                                           args=(df_queries.iloc[idxs[i]],
                                                                 df_references,
                                                                 config["general"]["ranking_function"],
                                                                 config["general"]["num_candidates"],
                                                                 tmp_paths[i]),
                                                           kwargs=dict(process_id=i))
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    # concat files
    print("CONCATENATING FILES")
    with open(outfile_path, "w+") as outfile:
        for i in range(num_workers):
            with open(tmp_paths[i], "r") as f:
                    for line in f:
                        outfile.write(line)

    # force remove tmp folder
    print("REMOVING TMP FILES")
    shutil.rmtree(str(tmp_folder_path))


@app.command()
def main(
    output_folder: Path = typer.Option(..., dir_okay=True, file_okay=True, exists=False, writable=True, help="Path to the folder where the predictions will be saved"),
    config_file: Path = typer.Option(..., help="Path to the config file"),
    data_range: str = typer.Option("", help="Range of data to generate predictions for. Format: <start>:<end>"),
    num_workers: int = typer.Option(1, help="Number of processes to use for the database search"),
) -> None:

    # load config
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    general_config = config["general"]
    dataset_config = config["dataset"]
    filtering_args = deepcopy(config["filtering_args"]) # deepcopy so tokenizer is not saved to logs

    config["command"] = " ".join(sys.argv)
    start_time = time.time()
    config["start_loading_time"] = timestamp_to_readable(start_time)
    additional_info = general_config["additional_naming_info"]

    # load data
    print("LOADING DATA")
    df_reference = pd.read_json(dataset_config["reference_data"], lines=True, orient="records")
    df_all_queries = pd.read_json(dataset_config["reference_data"], lines=True, orient="records")
    tqdm.pandas(desc="Filtering queries")
    df_queries = df_all_queries[df_all_queries.progress_apply(lambda row: filter_datapoints(row, filtering_args), axis=1)]

    # load range
    if data_range:
        data_range_min, data_range_max = list(map(int, data_range.split(":")))
        df_queries = df_queries.iloc[data_range_min:data_range_max]
    else:
        data_range_min, data_range_max = None, None

    # set output files
    db_search_name = f"db_search_{general_config['ranking_function']}"
    placeholder_path = Path(db_search_name) / "placeholder"  # in order to comply with the open files function
    additional_info = f"{general_config['num_candidates']}cand" + additional_info
    log_file, outfile = open_files(output_folder, placeholder_path, dataset_config, data_range, additional_info)
    outfile_path = Path(outfile.name)
    outfile.close() # we dont need it yet

    yaml.dump(config, log_file)

    # Start generating
    start_generation_time = time.time()
    config["start_generation_time"] = timestamp_to_readable(start_generation_time)

    parallel_db_search(df_reference, df_queries, config, outfile_path, num_workers)

    finished_time = time.time()

    log_config = {
        "finished_time_utc": timestamp_to_readable(finished_time),
        "generation_time": f"{hours_minutes_seconds(finished_time - start_generation_time)}",
        "wall_time_utc": f"{hours_minutes_seconds(finished_time - start_time)}"}
    yaml.dump(log_config, log_file)
    log_file.close()


if __name__ == "__main__":
    app()
