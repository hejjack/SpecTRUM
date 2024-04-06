# imports
import pandas as pd
import pathlib
import typer
import json
from statistics import mean
from tqdm import tqdm
from rdkit import Chem, RDLogger
import numpy as np
from collections import defaultdict
from icecream import ic
import plotly.express as px
import yaml
import time
from datetime import datetime
from collections.abc import Iterator

from spectra_process_utils import get_fp_generator, get_simil_function
from general_utils import move_file_pointer, line_count, dummy_generator
from eval_utils import load_labels_from_dataset, load_labels_to_datapipe

RDLogger.DisableLog('rdApp.*')


app = typer.Typer(pretty_exceptions_enable=False)

def parse_predictions_path(predictions_path: pathlib.Path) -> tuple:
    """Parse the predictions path to get the dataset name, split and range
    for 'full' range, returns range(0)"""
    try:
        data_name = predictions_path.parent.parent.stem
        data_split, data_range_str = predictions_path.parent.stem.split("_")[1:3]
        if data_range_str == "full":
            data_range = range(0)
        else:
            data_range = range(*map(int, data_range_str.split(":")))

    except Exception as exc:
        raise ValueError("Couldn't deduce labels info from predictions_path, " +
                         "please hold the file naming convention " +
                         "<dataset_name>/<id>_<data_split>_<data_range>_*/predictions.jsonl") from exc
    return data_name, data_split, data_range


def update_counter(sorted_simil: np.ndarray, all_simils: dict) -> None:
    """Add simil values to lists with the same index as their ranking"""
    for i, simil in enumerate(sorted_simil):
        all_simils[i].append(simil)


def diagram_from_dict(d: dict, title: str): # TODO: rename
    """Create a plotly figure from a cumulatively stored simils dict"""
    simils = []
    ks = []
    for k, s in d.items():
        simils += s
        ks += [k] * len(s)
    df = pd.DataFrame({"simil": simils, "k": ks})
    fig = px.box(df, x="k", y="simil", points="all")
    return fig


def get_eval_tag(old_logs: dict) -> str:
    """Get the evaluation tag for the eval new logs"""
    if not old_logs.get("evaluation_0", False):
        return "evaluation_0"
    else:
        return f"evaluation_{max([int(key.split('evaluation_')[1]) for key in old_logs.keys() if 'evaluation' in key]) + 1}"


@app.command()
def main(
    predictions_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="Path to the jsonl file with caption predictions"),
    labels_path: pathlib.Path = typer.Option(None, dir_okay=False, file_okay=True, readable=True, help="either .smi file or .jsonl (DataFrame with 'smiles' column)"),
    datasets_folder: pathlib.Path = typer.Option("data/datasets", dir_okay=True, file_okay=False, readable=True, help="Path to the folder containing the datasets (used for automatic labels deduction)"),
    config_file: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="Path to the config file"),
) -> None:

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    do_denovo = config["do_denovo"]
    on_the_fly = config["on_the_fly"]


    data_name, data_split, data_range = parse_predictions_path(predictions_path)
    num_lines_predictions = line_count(predictions_path)
    if data_range:
        assert num_lines_predictions == len(data_range), "Number of predictions does not match the data range"
    else:
        if on_the_fly: # filtering will reduce the num of labels. Does not have to have the same length, but RISKY
            print("WARNING: on-the-fly preprocessing is enabled, data_range is not specified, be sure to have exactly the same preprocessing setup as for generation, otherwise will return incorrrect results")
        else: # expecting full range, already preprocessed - has to have the same number of lines
            assert num_lines_predictions == line_count(labels_path), "No data_range specified, num of predictions does not match num of labels "
    
    if labels_path.suffix == ".jsonl":
        if on_the_fly:
            labels_iterator, smiles_sim_of_closest = load_labels_to_datapipe(labels_path, 
                                                                             data_range, 
                                                                             do_denovo=do_denovo, 
                                                                             fp_type=config["fingerprint_type"], 
                                                                             simil_func=config["simil_function"],
                                                                             filtering_args=config["filtering_args"]
                                                                             )
        else:
            labels_iterator, smiles_sim_of_closest = load_labels_from_dataset(labels_path, 
                                                                              data_range, 
                                                                              do_denovo=do_denovo, 
                                                                              fp_type=config["fingerprint_type"], 
                                                                              simil_func=config["simil_function"])
    elif labels_path.suffix == ".smi":
        labels_iterator = labels_path.open("r")
        move_file_pointer(data_range.start, labels_iterator)
    else: 
        raise ValueError("Labels have to be either .jsonl or .smi file")
    
    # set up fingerprint generator and similarity function
    fp_simil_args_info = f"{config['fingerprint_type']}_{config['simil_function']}"
    fpgen = get_fp_generator(config["fingerprint_type"])
    simil_function = get_simil_function(config["simil_function"])
    print(f">> Setting up   {config['fingerprint_type']}   fingerprint generator. Do your data CORRESPOND?")
    print(f">> Setting up   {config['simil_function']}   similarity function. Do your data CORRESPOND?")

    
    parent_dir = predictions_path.parent
    log_file_path = parent_dir / "log_file.yaml"
    pred_f = predictions_path.open("r")
    log_file = (log_file_path).open("a+")
    fp_simil_fails_f = (parent_dir / f"fp_simil_fails_{fp_simil_args_info}.csv").open("w+")
    fp_simil_fails_f.write("pred,label\n")
    pred_jsonl = {}
    counter_empty_preds = 0
    start_time = time.time()

    # set empty lists for best predictions dataframe
    if config["save_best_predictions"]:
        save_best_to_new_df = not (parent_dir / "df_best_predictions.jsonl").exists()
        if save_best_to_new_df:
            all_gt_smiless = []
        simil_best_smiless = []
        prob_best_smiless = []

    simil_all_simils = defaultdict(list) # all simils sorted by similarity with gt (at each ranking)
    prob_all_simils = defaultdict(list)   # all simils sorted by probability (at each ranking)
    counter_fp_simil_fails_preds = 0 # number of situations when fingerprint similarity is 1 for different molecules


    for i in tqdm(dummy_generator()):  # basically a while True
        pred_jsonl = pred_f.readline()
        if not pred_jsonl:
            break
        preds = json.loads(pred_jsonl)
        gt_smiles = next(labels_iterator)
        pred_smiless = list(preds.keys())
        pred_mols = [Chem.MolFromSmiles(smiles) for smiles in pred_smiless]
        pred_fps = [fpgen.GetFingerprint(mol) for mol in pred_mols if mol is not None]

        if not preds or not pred_fps:
            if preds and not pred_fps:
                print(f"WARNING: No valid prediction out of {len(pred_mols)} for {gt_smiles}, line {i}")
            counter_empty_preds += 1
            simil_all_simils[0].append(0)
            prob_all_simils[0].append(0)
            if config["save_best_predictions"]:
                if save_best_to_new_df:
                    all_gt_smiless.append(gt_smiles)
                simil_best_smiless.append(None)
                prob_best_smiless.append(None)
            continue

        gt_mol = Chem.MolFromSmiles(gt_smiles)
        gt_fp = fpgen.GetFingerprint(gt_mol)
        smiles_simils = [simil_function(fp, gt_fp) for fp in pred_fps]


        prob_simil = np.stack(list(zip(preds.values(), smiles_simils)))

        simil_decreasing_index = np.argsort(-prob_simil[:, 1])
        prob_decreasing_index = np.argsort(-prob_simil[:, 0])

        # when simil is 1, check if canon smiles are the same 
        if prob_simil[simil_decreasing_index[0]][1] == 1:
            best_pred_smiles = Chem.MolToSmiles(pred_mols[simil_decreasing_index[0]])
            gt_smiles_canon = Chem.MolToSmiles(gt_mol)
            if best_pred_smiles != gt_smiles_canon:
                counter_fp_simil_fails_preds += 1
                fp_simil_fails_f.write(f"{best_pred_smiles},{gt_smiles_canon}\n")

        update_counter(prob_simil[simil_decreasing_index][:, 1], simil_all_simils)
        update_counter(prob_simil[prob_decreasing_index][:, 1], prob_all_simils)

        if config["save_best_predictions"]:
            if save_best_to_new_df:
                all_gt_smiless.append(gt_smiles)
            simil_best_smiless.append(pred_smiless[simil_decreasing_index[0]])
            prob_best_smiless.append(pred_smiless[prob_decreasing_index[0]])
    
    simil_average_simil_kth = [mean(simil_all_simils[k]) for k in sorted(simil_all_simils.keys())]
    prob_average_simil_kth = [mean(prob_all_simils[k]) for k in sorted(prob_all_simils.keys())]
    num_predictions_at_k_counter = [len(l[1]) for l in sorted(list(simil_all_simils.items()), key=lambda x: x[0])]

    # create plots
    fig_similsort = diagram_from_dict(simil_all_simils, title="Similarity on the k-th position (sorted by ground truth similarity)")
    fig_probsort = diagram_from_dict(prob_all_simils, title="Similarity on the k-th position (sorted by generation probability)")
    df_top1 = pd.DataFrame({"simil": simil_all_simils[0], "prob": prob_all_simils[0]})
    fig_top1_simil_simils = px.histogram(df_top1, x="simil", nbins=100, labels={'x':'similarity', 'y':'count'})
    fig_top1_prob_simils = px.histogram(df_top1, x="prob", nbins=100, labels={'x':'similarity', 'y':'count'})

    # save plots
    fig_similsort.write_image(str(parent_dir / f"topk_similsort_{fp_simil_args_info}.png"))
    fig_probsort.write_image(str(parent_dir / f"topk_probsort_{fp_simil_args_info}.png"))
    fig_top1_simil_simils.write_image(str(parent_dir / f"top1_simil_simils_{fp_simil_args_info}.png"))
    fig_top1_prob_simils.write_image(str(parent_dir / f"top1_prob_simils_{fp_simil_args_info}.png"))

    finish_time = time.time()
    num_precise_preds_similsort = sum(np.array(simil_all_simils[0]) == 1) 
    num_precise_preds_probsort = sum(np.array(prob_all_simils[0]) == 1)
    num_better_than_threshold_similsort = sum(np.array(simil_all_simils[0]) > config["threshold"])
    num_better_than_threshold_probsort = sum(np.array(prob_all_simils[0]) > config["threshold"])

    
    if config["save_best_predictions"]:
        print("INFO: Saving best predictions")
        if save_best_to_new_df:
            # create a dataframe 
            df_best_predictions = pd.DataFrame({"gt_smiles": all_gt_smiless})
        else:
            # open the dataframe and add columns to it
            df_best_predictions = pd.read_json(parent_dir / "df_best_predictions.jsonl", lines=True, orient="records")
        df_best_predictions[f"simil_best_smiless_{fp_simil_args_info}"] = simil_best_smiless 
        df_best_predictions[f"prob_best_smiless_{fp_simil_args_info}"] = prob_best_smiless
        df_best_predictions[f"simil_best_simil_{fp_simil_args_info}"] = simil_all_simils[0]
        df_best_predictions[f"prob_best_simil_{fp_simil_args_info}"] = prob_all_simils[0]
        df_best_predictions.to_json(parent_dir / "df_best_predictions.jsonl", lines=True, orient="records")

    with open(log_file_path, "r", encoding="utf-8") as f:
        old_logs = yaml.safe_load(f)
    eval_tag = get_eval_tag(old_logs)
    logs = {eval_tag:
            {"eval_config": config,
             "topk_similsort": str(simil_average_simil_kth),
             "topk_probsort": str(prob_average_simil_kth),
             "num_predictions_at_k_counter": str(num_predictions_at_k_counter),
             "counter_empty_preds": str(counter_empty_preds),
             "counter_datapoints_tested": str(num_lines_predictions),
             "counter_fp_simil_fails_preds": str(counter_fp_simil_fails_preds),
             "labels_path": str(labels_path),
             "num_similsort_precise_preds": str(num_precise_preds_similsort),
             "num_probsort_precise_preds": str(num_precise_preds_probsort),
             "num_better_than_threshold_similsort": str(num_better_than_threshold_similsort),
             "num_better_than_threshold_probsort": str(num_better_than_threshold_probsort),
             "percentage_of_precise_preds_similsort": str(num_precise_preds_similsort / len(simil_all_simils[0])),
             "percentage_of_precise_preds_probsort": str(num_precise_preds_probsort / len(prob_all_simils[0])),
             "percentage_of_better_than_threshold_similsort": str(num_better_than_threshold_similsort / len(simil_all_simils[0])),
             "percentage_of_better_than_threshold_probsort": str(num_better_than_threshold_probsort / len(prob_all_simils[0])),
             "start_time_utc": datetime.utcfromtimestamp(start_time).strftime("%d/%m/%Y %H:%M:%S"),
             "eval_time": f"{time.strftime('%H:%M:%S', time.gmtime(finish_time - start_time))}"
            }}
    
    if do_denovo:
        smiles_sim_of_closest = np.array(list(smiles_sim_of_closest))
        simil_preds_minus_closest = np.array(simil_all_simils[0]) - smiles_sim_of_closest
        prob_preds_minus_closest = np.array(prob_all_simils[0]) - smiles_sim_of_closest
        fig_simil_preds_minus_closest = px.histogram(x=simil_preds_minus_closest, nbins=100, labels={'x':'FPSD score (FingerPrintSimilDiff)', 'y':'count'})
        fig_prob_preds_minus_closest = px.histogram(x=prob_preds_minus_closest, nbins=100, labels={'x':'FPSD score (FingerPrintSimilDiff)', 'y':'count'})
        fig_simil_preds_minus_closest.write_image(str(parent_dir / f"fpsd_score_similsort_{fp_simil_args_info}.png"))
        fig_prob_preds_minus_closest.write_image(str(parent_dir / f"fpsd_score_probsort_{fp_simil_args_info}.png"))
        logs[eval_tag]["denovo"] = {"mean_fpsd_score_similsort": str(simil_preds_minus_closest.mean()),
                                        "mean_fpsd_score_probsort": str(prob_preds_minus_closest.mean()),
                                        "mean_db_score": str(smiles_sim_of_closest.mean()),
                                        "percentage_of_BART_wins_similsort": str(sum(simil_preds_minus_closest > 0) / len(simil_preds_minus_closest)),
                                        "percentage_of_BART_wins_probsort": str(sum(prob_preds_minus_closest > 0) / len(prob_preds_minus_closest)),
                                        }
        
    yaml.dump(logs, log_file)
    print(logs)
    log_file.close()
    pred_f.close()
    

if __name__ == "__main__":
    app()
