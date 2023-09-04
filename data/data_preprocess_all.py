import os
import re
import sys
import glob
import pathlib
import multiprocessing
import logging
import time, datetime
import subprocess as subp
from typing import List
from multiprocessing import Process, Lock

import typer
import yaml
from tqdm import tqdm

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem import PandasTools

from tokenizers import Tokenizer

tqdm.pandas()
app = typer.Typer()


def df_to_n_chunks(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    chunk_size = len(df) // n + 1
    dfs = []
    for i in range(n):
        new_df = df.iloc[i*chunk_size:(i+1)*chunk_size]
        dfs.append(new_df)
    return dfs


def create_logging(log_dir: pathlib.Path, dataset_id: str):
    """create logging file and logger"""

    log_file_id = 0
    while os.path.isfile(os.path.join(log_dir, f'{log_file_id :04d}.log')):
        log_file_id += 1

    # Create config
    log_path = os.path.join(log_dir, f'{log_file_id :04d}.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def check_phase(file_path: pathlib.Path, phase_n: int, process_id: int, logging, lock):
    """check if phase n is already performed"""
    if not file_path.is_file():
        log_safely(lock, logging.error,
                   f"PHASE {phase_n} NOT YET PERFORMED process:{process_id}")
        raise ValueError(
            f"PHASE {phase_n} NOT YET PERFORMED process:{process_id}")


def process_spec(df):
    """process spectra generated by NEIMS and divide it to mz and normalized intensities"""
    df2 = df.copy()
    new_df = pd.DataFrame()
    all_i = []
    all_mz = []
    for row in tqdm(range(len(df2))):
        spec = df2["PREDICTED SPECTRUM"][row].split("\n")
        mz = []
        i = []
        spec_max = 0
        for t in spec:
            j, d = t.split()
            j, d = int(j), float(d)
            if spec_max < d:
                spec_max = d
            mz.append(j)
            i.append(d)
        all_mz.append(mz)
        all_i.append(np.around(np.array(i)/spec_max, 2))
    new_df = pd.DataFrame.from_dict({"mz": all_mz, "intensity": all_i})
    df2 = pd.concat([df2, new_df], axis=1)
    df2 = df2.drop(["PREDICTED SPECTRUM"], axis=1)
    return df2


def my_position_ids_creator(intensities, log_base, log_shift, seq_len):
    """create position ids for our Transformer model"""
    x = (np.log(intensities)/log_base).astype(int) + log_shift
    x = x * (x > 0)
    return np.concatenate((x, [-1]*(seq_len-len(intensities)))).astype("int32")


def data_split(df, config, logging, process_id, lock):
    """split the df into train, test and valid sets"""
    if config["train_split_ratio"] + config["test_split_ratio"] + config["valid_split_ratio"] != 1:
        log_safely(lock, logging.warning,
                   f"!!!!!!! split ratios don't sum to 1 process:{process_id} !!!!!!!")
        log_safely(lock, logging.warning,
                   f"----> TRAIN SET will be extracted according to paramter, TEST and VALID wil be split proportionally; process:{process_id}")

    train_set = df.sample(
        frac=config["train_split_ratio"], random_state=config["seed"])
    rest = df.drop(train_set.index)

    test_set = rest.sample(frac=config["test_split_ratio"]/(
        config["test_split_ratio"]+config["valid_split_ratio"]), random_state=config["seed"])
    valid_set = rest.drop(test_set.index)

    log_safely(lock, logging.info, f"SPLITTING STATS process:{process_id}\n" +
               "################################################\n" +
               f"train len: {len(train_set)}\ntest len: {len(test_set)}\nvalid len: {len(valid_set)}\n" +
               f"{len(train_set)} + {len(test_set)} + {len(valid_set)} == {len(df)} : the sum matches len of the df\n" +
               "################################################")

    return train_set, test_set, valid_set


def log_safely(lock: Lock, func, text: str):
    """log safely in terms of multiprocessing"""
    with lock:
        return func(f"##### {text} #####")


def autoremove_file(file_path: pathlib.Path, logging, lock: Lock):
    """delete file safely and log the action"""
    with lock:
        if os.path.isfile(file_path):
            os.remove(str(file_path))
        else:
            log_safely(lock, logging.error, f"Error while removing: {file_path} : file does not exist")
    log_safely(lock, logging.info, f"removed succesfully: {file_path}")


def smiles_to_labels(smiles, tokenizer, source_id, seq_len):
    """Converts smiles to labels for the model"""
    eos_token = tokenizer.token_to_id("<eos>")
    encoded_smiles = tokenizer.encode(smiles).ids
    padding_len = (seq_len-2-len(encoded_smiles))
    labels = [source_id] + encoded_smiles + [eos_token] + padding_len * [-100]
    mask = [1] * (seq_len - padding_len) + [0] * padding_len
    assert len(labels) == seq_len, "Labeles have wrong length"
    assert len(mask) == seq_len, "Mask has wrong length"
    return labels, mask


@app.command()
def main(dataset_id: str = typer.Option(..., help="Name of the dataset to identify it and its tmp files (e.g. 8M)"),
         smiles_path: pathlib.Path = typer.Option(..., dir_okay=False, help="Path to the smiles file (e.g. to ./8M.smi) \
         containing smiles (and optionally zinc_ids)"),
         num_workers: int = typer.Option(
             1, "-w", "--num-workers", help="Number of workers to use for multiprocessing"),
         auto_remove: bool = typer.Option(
                False, "-r", "--auto-remove", help="Remove tmp files when we don't need them anymore"),
         config_file: pathlib.Path = typer.Option(
             ..., dir_okay=False, help="Path to the config file")
         ):

    """Main function to run the pipeline"""

    # unpack config file
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError("Error in configuration file:", exc) from exc

    # set logging
    log_dir = pathlib.Path(config["log_dir"]) / dataset_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logging = create_logging(log_dir, dataset_id)

    # set tmp directory
    tmp_dir = pathlib.Path(config["tmp_dir"]) / dataset_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # log config and parameters
    config_str = ""
    for key, value in config.items():
        config_str += f"{key}: {value}\n"
    logging.info(f"CONFIG:\n{config_str}")
    logging.info(f"PARAMETERS:\ndataset_id = {dataset_id}\n" +
                 f"smiles_path = {smiles_path}\n" +
                 f"num_workers = {num_workers}\n" +
                 f"config_file = {config_file}")

    # open smiles file (ONLY SMILES COLUMN)
    logging.info("READING CSV")
    if config["data_has_header"] is True:
        df = pd.read_csv(smiles_path, header=0, sep=" ")[["smiles"]]
    if config["data_has_header"] is False:
        df = pd.read_csv(smiles_path, header=None, names=["smiles"])
    logging.info("READING DONE")

    # set lost chunks - if some of the chunks don't get properly through (process fails)
    try:
        if config["lost_chunks"] != []:
            process_ids = config["lost_chunks"]
            logging.info(f"SETTING PROCESES IDS USING <LOST_CHUNKS> TO: {process_ids}")
        else:
            process_ids = list(range(num_workers))
            logging.info(f"SETTING PROCESES IDS BY DEFAULT TO: {process_ids}")
    except KeyError:
        process_ids = list(range(num_workers))
        logging.info(f"SETTING PROCESES IDS BY DEFAULT TO: {process_ids}")


    # set multiprocessing
    logging.info(f"SETTING MULTIPROCESSING with {num_workers} workers")
    df_chunks = df_to_n_chunks(df, num_workers)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    lock = Lock()
    processes = [Process(target=process, args=(df_chunks[i],
                                               dataset_id,
                                               config["phases_to_perform"],  # might cause trouble (type)
                                               config,
                                               tmp_dir,
                                               logging,
                                               return_dict,
                                               lock,
                                               auto_remove,
                                               i)) for i in process_ids]
    start_time = time.time()

    logging.info("RUNNING")
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # concat results
    done_split_files = glob.glob(str(tmp_dir / f"{dataset_id}_chunk*_after_phase6_train.pkl")) # done train/test/valid chunks so far
    done_split_ids = set([int(re.findall(r"chunk(\d+)", file)[0]) for file in done_split_files])
    df_splits = {"df_train": None, "df_test": None, "df_valid": None}
    output_dir_with_id = pathlib.Path(config["output_dir"]) / dataset_id
    os.makedirs(output_dir_with_id, exist_ok=True)
    # if all chunks got through 6th phase, concat it from return_dict
    if set(process_ids) == set(range(num_workers)) and 6 in config["phases_to_perform"]:
        try:
            for data_type in tqdm(["train", "valid", "test"]):
                df_split = pd.concat([return_dict[i][data_type] for i in tqdm(range(num_workers))])
                df_split.to_pickle(output_dir_with_id / f"{dataset_id}_{data_type}.pkl")
        except ValueError:
            logging.info("SOME CHUNKS DIDN'T GET THROUGH 6TH PHASE, TRYING TO LOAD THEM")
    # else if all the split files are saved, prepared for concat, load them and concat them
    elif done_split_ids == set(range(num_workers)):
        del return_dict
        logging.info("LOADING CHUNKS FROM FILES TO CONCAT IT")
        for data_type in ["valid", "test", "train"]:
            logging.info(f"{data_type.upper()} ")
            file_path = tmp_dir / f"{dataset_id}_chunk*_after_phase6_{data_type}.pkl"   # pickled df
            files = glob.glob(str(file_path))
            for f in files:
                df_split = pd.concat([pd.read_pickle(f) for f in tqdm(files)])
                df_split.to_pickle(output_dir_with_id / f"{dataset_id}_{data_type}.pkl")

    # some chunks didn't get through 6th phase, try to run them again
    else:
        raise Exception(f"Not all chunks got through preprocessing, try to run these" +
                        f" specific ones via <lost_chunks> config option.\n" + 
                        f"Those lost are: {set(range(num_workers)).difference(done_split_ids)}")

    # removing after 6phase files
    if auto_remove:
        logging.info("REMOVING ALL AFTER 6PHASE SPLITS")
        for i in range(num_workers):
            file_path = tmp_dir / f"{dataset_id}_chunk{i}_after_phase6_*.pkl"   # pickled df
            files = glob.glob(str(file_path))
            for f in files:
                os.remove(f)
                logging.info(f"Deleted: {f}")

    # get runtime
    runtime = time.time() - start_time
    logging.info("ALL DONE, it all took: " + f"{datetime.timedelta(seconds=runtime)}")



def process(df: pd.DataFrame,
            dataset_id: str,
            phases_to_perform: List[int],
            config: dict,
            tmp_dir: pathlib.Path,
            logging,  # module 
            return_dict: dict,
            lock: Lock,
            auto_remove: bool,
            process_id: int = 0):
    """
    This script performs the data preprocessing for the MassGenie project. 
    It performs the following steps: 
        - Phase 1: CANONICALIZATION
        - Phase 2: DESTEREOCHEMICALIZATION
        - Phase 3: DEDUPLICATION, LONG SMILES FILTERING, SDF/PLAIN SMILES PREPARATION
        - Phase 4: SPECTRA GENERATION       
        - Phase 5: FILTERING
        - Phase 6: DATA CREATION
    """

    phase3_extension = "sdf" if config["spectra_generator"] == "neims" else "smi"

    # paths
    after_phase1_smiles = tmp_dir / f"{dataset_id}_chunk{process_id}_after_phase1.smi"  # canonicalized smiles (plain)
    after_phase2_smiles = tmp_dir / f"{dataset_id}_chunk{process_id}_after_phase2.smi"  # destereo smiles (plain)
    after_phase3_file = tmp_dir / f"{dataset_id}_chunk{process_id}_after_phase3_{config['spectra_generator']}.{phase3_extension}"  # file prepared for generation
    after_phase4_sdf = tmp_dir / f"{dataset_id}_chunk{process_id}_after_phase4_neims.sdf" # enriched sdf w/ generated spectra
    after_phase5_pickle = tmp_dir / f"{dataset_id}_chunk{process_id}_after_phase5.pkl"  # pickled df
    after_phase6_pickle = tmp_dir / f"{dataset_id}_chunk{process_id}_after_phase6.pkl"  # pickled df

    df_after_phase5 = None  # in case phase 5 is not performed
    if 1 in phases_to_perform:
        phase1(df, after_phase1_smiles, logging, lock, process_id)
    else:
        logging.info("Skipping phase 1")
    if 2 in phases_to_perform:
        phase2(after_phase1_smiles, after_phase2_smiles,
               logging, lock, process_id)
        if auto_remove:
            autoremove_file(after_phase1_smiles, logging, lock)
    else:
        logging.info("Skipping phase 2")
    if 3 in phases_to_perform:
        phase3(after_phase2_smiles, after_phase3_file,
               logging, config, lock, process_id)
        if auto_remove:
            autoremove_file(after_phase2_smiles, logging, lock)
        if config["spectra_generator"] == "RASSP":
            return            # TODO: implement RASSP generation
    else:
        logging.info("Skipping phase 3")

    if 4 in phases_to_perform:
        phase4(after_phase3_file, after_phase4_sdf,
               logging, config, lock, process_id)
        if auto_remove:
            autoremove_file(after_phase3_file, logging, lock)
        
    else: 
        logging.info("Skipping phase 4")

    if 5 in phases_to_perform:
        df_after_phase5 = phase5(
            after_phase4_sdf, after_phase5_pickle, logging, config, lock, process_id)
        if auto_remove:
            autoremove_file(after_phase4_sdf, logging, lock)
    else:
        logging.info("Skipping phase 5")

    if 6 in phases_to_perform:
        df_train, df_test, df_valid = phase6(df_after_phase5,
                                             after_phase5_pickle,
                                             after_phase6_pickle,
                                             logging,
                                             config,
                                             lock,
                                             process_id)
        if auto_remove:
            autoremove_file(after_phase5_pickle, logging, lock)
    else:
        logging.info("Skipping phase 6")
        return_dict[process_id] = {"train": None,
                            "test": None, 
                            "valid": None}
        return

    return_dict[process_id] = {"train": df_train,
                               "test": df_test, 
                               "valid": df_valid}


def phase1(df: pd.DataFrame,
           after_phase1_smiles: pathlib.Path,
           logging,  # module 
           lock: Lock,
           process_id: int = 0):
    """ Phase 1: CANONICALIZATION """

    log_safely(lock, logging.info, f"PHASE 1 process:{process_id}")

    # canonicalization
    log_safely(lock, logging.debug,
               f"CANONICALIZATION process:{process_id}")
    cans = []
    for smi in tqdm(df["smiles"]):
        try:
            can = Chem.MolToSmiles(Chem.MolFromSmiles(smi), True)
            cans.append(can)
        except Exception as e:
            logging.warning(f"SMILES {smi} couldn't be canonicalized due to: {e}")
    log_safely(lock, logging.debug,
               f"CANONICALIZATION DONE process:{process_id}")

    # save plain smiles
    log_safely(lock, logging.debug,
               f"SAVING CANON SMILES process:{process_id}")
    with open(after_phase1_smiles, "w+") as out:
        for s in tqdm(cans):
            out.write(s + "\n")


def phase2(after_phase1_smiles: pathlib.Path,
           after_phase2_smiles: pathlib.Path,
           logging,  # module 
           lock: Lock,
           process_id: int = 0):
    """ Phase 2: DESTEREOCHEMICALIZATION """

    log_safely(lock, logging.info, f"PHASE 2 process:{process_id}")
    check_phase(after_phase1_smiles, 1, logging, process_id,
                lock)  # check whether phase 1 was performed

    # destereochemicalize
    log_safely(lock, logging.debug,
               f"DESTEREOCHEMICALIZATION process:{process_id}")
    subp.check_call(
        f"obabel -i smi {after_phase1_smiles} -o can -xi > {after_phase2_smiles}", shell=True)
    log_safely(lock, logging.debug,
               f"DESTEREOCHEMICALIZATION DONE process:{process_id}")


def phase3(after_phase2_smiles: pathlib.Path,
           after_phase3_file: pathlib.Path,
           logging,  # module 
           config: dict,
           lock: Lock,
           process_id: int = 0):
    """ Phase 3: DEDUPLICATION, LONG SMILES FILTERING, SDF/PLAIN SMILES PREPARATION """

    log_safely(lock, logging.info, f"PHASE 3 process:{process_id}")
    check_phase(after_phase2_smiles, 2, logging, process_id,
                lock)  # check whether phase 2 was performed

    # load destereo smiles
    log_safely(lock, logging.debug,
               f"LOADING DESTEREO SMILES process:{process_id}")
    df = pd.read_csv(after_phase2_smiles, header=0, names=["smiles"])

    # deduplicate
    log_safely(lock, logging.debug,
               f"DEDUPLICATION process:{process_id}")
   
    df.drop_duplicates(subset=["smiles"], inplace=True)
    log_safely(lock, logging.debug,
               f"DEDUPLICATION DONE process:{process_id}")

    # long smiles filtering
    log_safely(lock, logging.debug,
               f"LONG SMILES FILTERING process:{process_id}")
    df = df[df["smiles"].progress_apply(lambda x: len(x) <= config['max_smiles_len'])]
    
    log_safely(lock, logging.debug,
               f"LONG SMILES FILTERING DONE process:{process_id}")

    # filter corrupted smiles
    log_safely(lock, logging.debug,
               f"CORRUPTED SMILES FILTERING process:{process_id}")
    df = df[df["smiles"].progress_apply(
        lambda x: Chem.MolFromSmiles(x) is not None)]

    if config["spectra_generator"] == "rassp":
        # save plain smiles
        log_safely(lock, logging.debug,
                   f"SAVING PLAIN SMILES process:{process_id}")
        with open(after_phase3_file, "w") as f:
            f.write('\n'.join(df.smiles.tolist()))
        log_safely(lock, logging.info,
                   f"len after PHASE3 process {process_id}: {len(df)}")
    elif config["spectra_generator"] == "neims":
        # save sdf
        log_safely(lock, logging.debug,
                   f"SAVING SDF process:{process_id}")
        log_safely(lock, logging.debug,
                   f"ADDING MOLECULE STRUCTURE process:{process_id}")
        PandasTools.AddMoleculeColumnToFrame(
            df, smilesCol='smiles', molCol='ROMol')

        # exporting to SDF
        log_safely(lock, logging.debug,
                   f"EXPORTING TO SDF process:{process_id}")
        df["id"] = df.index
        PandasTools.WriteSDF(df, str(after_phase3_file), idName="id", properties=list(
            df.columns))  # might be trouble with index
        log_safely(lock, logging.debug,
                   f"SAVING SDF DONE process:{process_id}")
        log_safely(lock, logging.info,
                   f"len after PHASE3 process {process_id}: {len(df)}")
    else:
        raise ValueError(
            f"SPECTRA GENERATOR {config['spectra_generator']} NOT RECOGNIZED")


def phase4(after_phase3_file: pathlib.Path,
           after_phase4_sdf: pathlib.Path,
           logging,  # module 
           config: dict,
           lock: Lock,
           process_id: int = 0):
    """ Phase 4: SPECTRA GENERATION """
    log_safely(lock, logging.info, f"PHASE 4 process:{process_id}")

    # neims generation
    if config["spectra_generator"] == "neims":
        # check whether phase 3 was performed
        check_phase(after_phase3_file, 3, logging, process_id, lock)

        # generate spectra
        log_safely(lock, logging.debug,
                   f"SPECTRA GENERATION process:{process_id}")   
        # TODO: oddelat hardcode a nejak presit tu condu
        subp.check_call(f"python {config['neims_dir']}/make_spectra_prediction.py \
                        --input_file={after_phase3_file} \
                        --output_file={after_phase4_sdf} \
                        --weights_dir={config['neims_dir']}/NEIMS_weights/massspec_weights", shell=True) \
        # conda deactivate")
        log_safely(lock, logging.debug,
                   f"SPECTRA GENERATION DONE process:{process_id}")
    elif config["spectra_generator"] == "rassp":
        log_safely(lock, logging.info,
                   f"A.K HAS TO DO THE RASSP SPECTRA GENERATION process:{process_id}")
        return
    else:
        raise ValueError(
            f"SPECTRA GENERATOR {config['spectra_generator']} NOT RECOGNIZED")


def phase5(after_phase4_sdf: pathlib.Path,
           after_phase5_pickle: pathlib.Path,
           logging,  # module 
           config: dict,
           lock: Lock,
           process_id: int = 0):
    """ Phase 5: SPECTRA FILTERING """
    log_safely(lock, logging.info, f"PHASE 5 process:{process_id}")
    # check whether phase 4 was performed
    check_phase(after_phase4_sdf, 4, process_id, logging, lock)

    # load the pickle from the first phase
    log_safely(lock, logging.debug,
               f"\n\nLOADING GENERATED SPECTRA process:{process_id}")

    df = PandasTools.LoadSDF(str(after_phase4_sdf), idName="id", molColName='Molecule')
    
    # processing spectra
    log_safely(lock, logging.debug,
               f"PROCESSING SPECTRA process:{process_id}")
    df = process_spec(df)

    # filtering high MZs
    log_safely(lock, logging.debug,
               f"FILTERING HIGH MZs process:{process_id}")
    df = df.loc[[x[-1] <= config["max_mz"] for x in tqdm(df["mz"])]]
    log_safely(lock, logging.info,
               f"len after MZ filtering; process {process_id}: {len(df)}")

    # filtering long spectra
    log_safely(lock, logging.debug,
               f"FILTERING LONG SPECTRA process:{process_id}")
    df = df.loc[[len(x) <= config["max_peaks"] for x in tqdm(df["mz"])]]
    log_safely(lock, logging.info,
               f"len after long spectra filtering; process {process_id}: {len(df)}")

    # drop unnecessary columns
    df = df[["smiles", "mz", "intensity"]]

    # strip potential \t from smiles
    df["smiles"] = df["smiles"].progress_apply(lambda x: x.strip()) 

    # save the df
    log_safely(lock, logging.info,
               f"len after PHASE5 process {process_id}: {len(df)}")
    log_safely(lock, logging.debug,
               f"SAVING destereo_smiles mz intensity; process:{process_id}")
    df.to_pickle(after_phase5_pickle)

    return df


def phase6(df_after_phase5: pd.DataFrame,
           after_phase5_pickle: pathlib.Path,
           after_phase6_pickle: pathlib.Path,
           logging,  # module 
           config: dict,
           lock: Lock,
           process_id: int = 0):
    """ Phase 6: DATASET_CREATION """
    log_safely(lock, logging.info, f"PHASE 6 process:{process_id}")
    check_phase(after_phase5_pickle, 5, process_id, logging,
                lock)  # check whether phase 5 was performed

    # load df from the phase 5 (check if we got the df, first - faster)
    if df_after_phase5 is None:
        log_safely(lock, logging.debug,
                   f"\n\nLOADING GENERATED SPECTRA process:{process_id}")
        df = pd.read_pickle(after_phase5_pickle)
    else:
        df = df_after_phase5

    log_safely(lock, logging.debug,
               f"TOKENIZATION AND PADDING process:{process_id}")
    # load tokenizer
    tokenizer = Tokenizer.from_file(config["tokenizer_path"])

    # set special tokens
    pt = tokenizer.token_to_id("<pad>")

    source_token = config["source_token"]
    source_id = tokenizer.token_to_id(source_token)
    seq_len = config["seq_len"]

    # create labels and decoder masks
    labels, dec_masks = [], []
    for i, smiles in enumerate(tqdm(df["smiles"])):
        l, m = smiles_to_labels(smiles, tokenizer, source_id, seq_len)
        labels.append(l)
        dec_masks.append(m)
    df["labels"], df["decoder_attention_mask"] = labels, dec_masks

    # create mz and encoder masks
    df["mz"] = [ts + (seq_len-len(ts)) * [pt] for ts in df.mz]
    df["attention_mask"] = [
        [int(id_ != pt) for id_ in arr] for arr in tqdm(df.mz)]
    
    log_safely(lock, logging.debug,
               f"LOG BINNING INTENSITIES process:{process_id}")
    log_base = np.log(config["log_base"])

    # add integral bins according to bounds (position_ids parameter)
    log_shift = 9  # = 9 for 1.7 -> shift after logarithmization

    # empty positions (padding) get -1 (=> no information there. BART doesn't have position ids -> we create them)
    df["position_ids"] = [my_position_ids_creator(
        arr, log_base, log_shift, seq_len) for arr in df.intensity]

    log_safely(lock, logging.debug,
               f"DROPPING USELESS COLUMNS process:{process_id}")
    df.rename(columns={"mz": "input_ids"}, inplace=True)
    df.drop(columns=["intensity"], inplace=True)

    log_safely(lock, logging.info, f"len after PHASE6 {process_id}: {len(df)}")

    df_train, df_test, df_valid = data_split(
        df, config, logging, process_id, lock)

    log_safely(lock, logging.debug,
               f"PICKELING PREPARED DATA process:{process_id}")
    path_wo_ext = after_phase6_pickle.with_suffix('')
    path_ext = after_phase6_pickle.suffix
    parent_dir = after_phase6_pickle.parents[0]
    os.makedirs(parent_dir, exist_ok=True)

    df_train.to_pickle(str(path_wo_ext) + "_train" + path_ext)
    df_test.to_pickle(str(path_wo_ext) + "_test" + path_ext)
    df_valid.to_pickle(str(path_wo_ext) + "_valid" + path_ext)

    log_safely(lock, logging.debug,
               f"PHASE6 DONE process:{process_id}")

    return df_train, df_test, df_valid


if __name__ == "__main__":
    app()
