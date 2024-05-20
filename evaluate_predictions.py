# imports
import pandas as pd
import pathlib
import typer
import json
from statistics import mean
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import numpy as np
from collections import defaultdict
from icecream import ic
import plotly.express as px
import plotly.io as pio
import yaml
import time
import pytz
from datetime import datetime
from collections.abc import Iterator

from spectra_process_utils import get_fp_generator, get_simil_function
from general_utils import move_file_pointer, line_count, dummy_generator, timestamp_to_readable, hours_minutes_seconds 
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


def create_topk_diagram_from_dict(d: dict, title: str): # TODO: rename
    """Create a plotly figure from a cumulatively stored simils dict"""
    simils = []
    ks = []
    for k, s in d.items():
        simils += s
        ks += [k+1] * len(s)
    fig = px.box(x=ks, y=simils, labels={"x": "k", "y": "similarity"}, points="outliers", title=title)
    return fig


def get_eval_tag(old_logs: dict) -> str:
    """Get the evaluation tag for the eval new logs"""
    if not old_logs.get("evaluation_0", False):
        return "evaluation_0"
    else:
        return f"evaluation_{max([int(key.split('evaluation_')[1]) for key in old_logs.keys() if 'evaluation' in key]) + 1}"

def compute_hit_at_k(councounter_first_hit_index_probsort: dict, num_datapoints: int) -> dict:
    """Compute hit at k for all k"""
    hit_at_k = {}
    cumulative_hit = 0
    for k in sorted(councounter_first_hit_index_probsort.keys()):
        cumulative_hit += councounter_first_hit_index_probsort[k]
        hit_at_k[k+1] = cumulative_hit / num_datapoints
    return hit_at_k

def did_we_hit_corrrect(gt, preds, simil_decreasing_index, prob_simil):
    """
    Sometimes there is more than one prediction with similarity 1.0,
    the simil sort is too weak to differentiate between them, so we need
    to check, whether some of the predictions with label 1.0 are the same
    as the ground truth. If so, we hit the best prediction as first.
    This function is usable for SMILES, formulas and other non-unique labels.
    """
    for i in range(len(simil_decreasing_index)):
        if prob_simil[simil_decreasing_index[i]][1] != 1:
            break
        if preds[simil_decreasing_index[i]] == gt:
            return True
    return False


def get_descriptors(smiless, fpgen):
    """Get mols, fps and formulas for predictions"""
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiless]
    fps = [fpgen.GetFingerprint(mol) for mol in mols if mol is not None]
    formulas = [rdMolDescriptors.CalcMolFormula(mol) for mol in mols if mol is not None]
    return mols, fps, formulas


@app.command()
def main(
    predictions_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="Path to the jsonl file with caption predictions"),
    labels_path: pathlib.Path = typer.Option(None, dir_okay=False, file_okay=True, readable=True, help="either .smi file or .jsonl (DataFrame with 'smiles' column)"),
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
    fp_simil_fails_simil_f = (parent_dir / f"fp_simil_fails_simil_{fp_simil_args_info}.csv").open("w+")
    fp_simil_fails_prob_f = (parent_dir / f"fp_simil_fails_prob_{fp_simil_args_info}.csv").open("w+")
    fp_simil_fails_simil_f.write("pred,label\n")
    fp_simil_fails_prob_f.write("pred,label\n")
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
    counter_fp_simil_fail_simil = 0 # number of situations when fingerprint similarity is 1 for different molecules (similsort)
    counter_fp_simil_fail_prob = 0 # number of situations when fingerprint similarity is 1 for different molecules (probsort)
    counter_multiple_hits = defaultdict(lambda: 0) # number of situations when there are multiple hits with similarity 1
    counter_first_hit_index_probsort = defaultdict(lambda: 0) # number of situations when the first hit has similarity 1
    counter_all_predictions = 0
    counter_all_correct_formulas = 0
    counter_correct_formulas_best_simil = 0
    counter_correct_formulas_best_prob = 0
    counter_at_least_one_correct_formula = 0

    for line in tqdm(dummy_generator()):  # basically a while True      
        pred_jsonl = pred_f.readline()
        if not pred_jsonl:
            break
        preds = json.loads(pred_jsonl)

        pred_smiless = list(preds.keys())
        gt_smiles = next(labels_iterator)
        
        pred_mols, pred_fps, pred_formulas = get_descriptors(pred_smiless, fpgen)      
        [gt_mol], [gt_fp], [gt_formula] = get_descriptors([gt_smiles], fpgen)

        counter_all_predictions += len(pred_mols)

        # process empty predictions
        if not preds or not pred_fps:
            if preds and not pred_fps:
                print(f"WARNING: No valid prediction out of {len(pred_mols)} for {gt_smiles}, line {line}")
            counter_empty_preds += 1
            simil_all_simils[0].append(0)
            prob_all_simils[0].append(0)
            if config["save_best_predictions"]:
                if save_best_to_new_df:
                    all_gt_smiless.append(gt_smiles)
                simil_best_smiless.append(None)
                prob_best_smiless.append(None)
            continue
        
        # SMILES simils
        smiles_simils = [simil_function(fp, gt_fp) for fp in pred_fps]
        prob_simil = np.stack(list(zip(preds.values(), smiles_simils)))

        simil_decreasing_index = np.argsort(-prob_simil[:, 1])
        prob_decreasing_index = np.argsort(-prob_simil[:, 0])


        # when simil is 1, check if there's a true match
        if prob_simil[simil_decreasing_index[0]][1] == 1:
            ### test
            # if len(simil_decreasing_index) > 1 and prob_simil[simil_decreasing_index[1]][1] == 1:
            #     all_hits = {"gt": gt_smiles, "hits": [pred_smiless[i] for i in simil_decreasing_index if prob_simil[i][1] == 1]}
            #     print(f"{line}: {all_hits}")
            ### 

            # update hit@k counter
            first_hit_index = np.where(prob_simil[:, 1][prob_decreasing_index] == 1)[0][0]
            counter_first_hit_index_probsort[first_hit_index] += 1 

            num_of_hits = sum(prob_simil[:, 1] == 1)
            counter_multiple_hits[num_of_hits] += 1

            gt_smiles_canon = Chem.MolToSmiles(gt_mol)
            we_hit_correct_smiles = did_we_hit_corrrect(gt_smiles_canon, pred_smiless, simil_decreasing_index, prob_simil)
            
            if not we_hit_correct_smiles:
                counter_fp_simil_fail_simil += 1
        else:
            we_hit_correct_smiles = False

        # check for precise match if highest prob similarity is 1. Further match checking doesn't make sense here 
        if prob_simil[prob_decreasing_index[0]][1] == 1:
            best_pred_prob_smiles = Chem.MolToSmiles(pred_mols[prob_decreasing_index[0]])
            gt_smiles_canon = Chem.MolToSmiles(gt_mol)
            if best_pred_prob_smiles != gt_smiles_canon:
                counter_fp_simil_fail_prob += 1
                fp_simil_fails_prob_f.write(f"{best_pred_prob_smiles},{gt_smiles_canon}\n")

        # update counters
        update_counter(prob_simil[simil_decreasing_index][:, 1], simil_all_simils)
        update_counter(prob_simil[prob_decreasing_index][:, 1], prob_all_simils)

        if config["save_best_predictions"]:
            if save_best_to_new_df:
                all_gt_smiless.append(gt_smiles)
            simil_best_smiless.append(gt_smiles if we_hit_correct_smiles else pred_smiless[simil_decreasing_index[0]])
            prob_best_smiless.append(pred_smiless[prob_decreasing_index[0]])
    
        # FORMULA stats
        is_correct_formula = [gt_formula == formula for formula in pred_formulas]
        counter_all_correct_formulas += sum(is_correct_formula)
        counter_at_least_one_correct_formula += any(is_correct_formula)
        we_hit_correct_formula = did_we_hit_corrrect(gt_formula, pred_formulas, simil_decreasing_index, prob_simil)
        counter_correct_formulas_best_simil += 1 if we_hit_correct_formula else gt_formula == pred_formulas[prob_decreasing_index[0]]
        counter_correct_formulas_best_prob += 1 if gt_formula == pred_formulas[prob_decreasing_index[0]] else 0




    simil_average_simil_kth = [mean(simil_all_simils[k]) for k in sorted(simil_all_simils.keys())]
    prob_average_simil_kth = [mean(prob_all_simils[k]) for k in sorted(prob_all_simils.keys())]
    num_predictions_at_k_counter = [len(l[1]) for l in sorted(list(simil_all_simils.items()), key=lambda x: x[0])]

    hit_at_k_prob = compute_hit_at_k(counter_first_hit_index_probsort, num_lines_predictions) 

    # create plots
    print("INFO: Creating plots...")
    fig_similsort = create_topk_diagram_from_dict(simil_all_simils, title="Similarity of the k-th best prediction (similsort)")
    fig_probsort = create_topk_diagram_from_dict(prob_all_simils, title="Similarity of the k-th best prediction (probsort)")
    fig_top1_simil_simils = px.histogram(x=simil_all_simils[0], nbins=100, labels={'x':'similarity', 'y':'count'}, title="Similarity of the best prediction (similsort)")
    fig_top1_prob_simils = px.histogram(x=prob_all_simils[0], nbins=100, labels={'x':'similarity', 'y':'count'}, title="Similarity of the best prediction (probsort)")
    fig_hit_at_k = px.line(x=list(hit_at_k_prob.keys()), y=list(hit_at_k_prob.values()), labels={'x':'k', 'y':'hit@k'}, title="Hit at k (probsort)")
    fig_num_simil_hits = px.bar(x=list(counter_multiple_hits.keys()), y=list(counter_multiple_hits.values()), text=list(counter_multiple_hits.values()),  labels={'x':'number of hits with similarity 1', 'y':'count'}, title="Number of hits with similarity 1")

    # increase plot sizes
    all_figs = [fig_similsort, fig_probsort, fig_top1_simil_simils, fig_top1_prob_simils, fig_hit_at_k, fig_num_simil_hits]
    general_font_size, title_font_size, tickfont_size, axis_title_font_size = 24, 29, 24, 26
    for fig in all_figs:
        fig.update_layout(title_font_size=title_font_size, title_x=0.5, title_y=0.9, title_xanchor='center', title_yanchor='top', font=dict(size=general_font_size))
        fig.update_xaxes(title_font_size=axis_title_font_size, tickfont_size=tickfont_size)
        fig.update_yaxes(title_font_size=axis_title_font_size, tickfont_size=tickfont_size)


    # fig.update_layout(barmode='stack',
    #                   title=dict(text="Maximum m/z filter", font=dict(size=27)),
    #                   xaxis_title=dict(text="max m/z", font=dict(size=22)),
    #                   yaxis_title=dict(text="count", font=dict(size=22)),
    #                       legend=dict(
    #                         x=0.7,
    #                         y=0.9,
    #                         traceorder='normal',
    #                         font=dict(size=18)),
    #                   )

    # save plots
    print("INFO: Saving plots...")
    scale = 6
    fig_similsort.write_image(str(parent_dir / f"topk_similsort_{fp_simil_args_info}.png"), scale=scale)
    fig_probsort.write_image(str(parent_dir / f"topk_probsort_{fp_simil_args_info}.png"), scale=scale)
    fig_top1_simil_simils.write_image(str(parent_dir / f"top1_simil_simils_{fp_simil_args_info}.png"), scale=scale)
    fig_top1_prob_simils.write_image(str(parent_dir / f"top1_prob_simils_{fp_simil_args_info}.png"), scale=scale)
    fig_hit_at_k.write_image(str(parent_dir / f"hit_at_k_{fp_simil_args_info}.png"), scale=scale)
    fig_num_simil_hits.write_image(str(parent_dir / f"num_simil_hits_{fp_simil_args_info}.png"), scale=scale)

    finish_time = time.time()
    num_1_hits_as_first_similsort = sum(np.array(simil_all_simils[0]) == 1)
    num_1_hits_as_first_probsort = sum(np.array(prob_all_simils[0]) == 1)
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
                "num_empty_preds": str(counter_empty_preds),
                "num_datapoints_tested": str(num_lines_predictions),
                "average_num_of_predictions": str(counter_all_predictions / num_lines_predictions),
                "labels_path": str(labels_path),
                "threshold_stats": {"threshold": str(config["threshold"]),
                                    "num_better_than_threshold_similsort": str(num_better_than_threshold_similsort),
                                    "num_better_than_threshold_probsort": str(num_better_than_threshold_probsort),
                                    "percentage_of_better_than_threshold_similsort": str(num_better_than_threshold_similsort / len(simil_all_simils[0])),
                                    "percentage_of_better_than_threshold_probsort": str(num_better_than_threshold_probsort / len(prob_all_simils[0]))
                                    },
                "simil_1_hits": {"num_1_hits_as_first_similsort": str(num_1_hits_as_first_similsort),
                                 "num_1_hits_as_first_probsort": str(num_1_hits_as_first_probsort),
                                 "percentage_of_1_hits_as_first_similsort": str(num_1_hits_as_first_similsort / num_lines_predictions),
                                 "percentage_of_1_hits_as_first_probsort": str(num_1_hits_as_first_probsort / num_lines_predictions),
                                 "counter_multiple_hits": str(counter_multiple_hits.items()),
                                 "num_fp_simil_fail_simil": str(counter_fp_simil_fail_simil),
                                 "num_fp_simil_fail_prob": str(counter_fp_simil_fail_prob),
                                },
                "precise_preds_stats": {"num_precise_preds_similsort": str(num_1_hits_as_first_similsort - counter_fp_simil_fail_simil),
                                        "num_precise_preds_probsort": str(num_1_hits_as_first_probsort - counter_fp_simil_fail_prob),
                                        "percentage_of_precise_preds_similsort": str((num_1_hits_as_first_similsort - counter_fp_simil_fail_simil) / num_lines_predictions),
                                        "percentage_of_precise_preds_probsort": str((num_1_hits_as_first_probsort - counter_fp_simil_fail_prob) / num_lines_predictions),
                                        },
                "start_time_utc": timestamp_to_readable(start_time),
                "eval_time": f"{hours_minutes_seconds(finish_time - start_time)}",
                "hit_at_k_prob": str(sorted(zip(hit_at_k_prob.keys(), hit_at_k_prob.values()), key=lambda x: x[0])),
                "formula_stats": {"num_all_correct_formulas": f"{counter_all_correct_formulas} / {counter_all_predictions}",
                                  "num_at_least_one_correct_formula": str(counter_at_least_one_correct_formula),
                                  "num_correct_formulas_at_best_simil": str(counter_correct_formulas_best_simil),
                                  "num_correct_formulas_at_best_prob": str(counter_correct_formulas_best_prob),
                                  "percentage_of_all_correct_formulas": str(counter_all_correct_formulas / counter_all_predictions),
                                  "percentage_of_correct_formulas_at_best_simil": str(counter_correct_formulas_best_simil / num_lines_predictions),
                                  "percentage_of_correct_formulas_at_best_prob": str(counter_correct_formulas_best_prob / num_lines_predictions),
                                  "percentage_of_at_least_one_correct_formula": str(counter_at_least_one_correct_formula / num_lines_predictions),
                                 }
                }
            }
            
    if do_denovo:
        smiles_sim_of_closest = np.array(list(smiles_sim_of_closest))
        simil_preds_minus_closest = np.array(simil_all_simils[0]) - smiles_sim_of_closest
        prob_preds_minus_closest = np.array(prob_all_simils[0]) - smiles_sim_of_closest
        simil_fpsd_tie_simils = np.array(simil_all_simils[0])[simil_preds_minus_closest == 0]
        prob_fpsd_tie_simils = np.array(prob_all_simils[0])[(prob_preds_minus_closest >= -0.005) & (prob_preds_minus_closest <= 0.005)]


        fig_simil_preds_minus_closest = px.histogram(x=simil_preds_minus_closest, 
                                                     nbins=100, 
                                                     labels={'x':'FPSD score', 'y':'count'}, 
                                                     title="FPSD: SpecTRUM vs. database search (similsort)")
        fig_prob_preds_minus_closest = px.histogram(x=prob_preds_minus_closest, 
                                                    nbins=100, 
                                                    labels={'x':'FPSD score', 'y':'count'}, 
                                                    title="FPSD: SpecTRUM vs. database search (probsort)")
        fig_simil_fpsd_ties = px.histogram(x=simil_fpsd_tie_simils, 
                                                     nbins=100, 
                                                     labels={'x':'similarity', 'y':'count'}, 
                                                     title="FPSD ties: SpecTRUM vs. database search (similsort)")
        fig_prob_fpsd_ties = px.histogram(x=prob_fpsd_tie_simils,
                                                    nbins=100, 
                                                    labels={'x':'similarity', 'y':'count'}, 
                                                    title="FPSD ties: SpecTRUM vs. database search (probsort)")

        for fig in [fig_simil_preds_minus_closest, fig_prob_preds_minus_closest, fig_simil_fpsd_ties, fig_prob_fpsd_ties]:
            fig.update_layout(title_font_size=title_font_size, title_x=0.5, title_y=0.9, title_xanchor='center', title_yanchor='top', font=dict(size=general_font_size))
            fig.update_xaxes(title_font_size=axis_title_font_size, tickfont_size=tickfont_size)
            fig.update_yaxes(title_font_size=axis_title_font_size, tickfont_size=tickfont_size)

        fig_simil_preds_minus_closest.write_image(str(parent_dir / f"fpsd_score_similsort_{fp_simil_args_info}.png"), scale=scale)
        fig_prob_preds_minus_closest.write_image(str(parent_dir / f"fpsd_score_probsort_{fp_simil_args_info}.png"), scale=scale)
        fig_simil_fpsd_ties.write_image(str(parent_dir / f"fpsd_ties_similsort_{fp_simil_args_info}.png"), scale=scale)
        fig_prob_fpsd_ties.write_image(str(parent_dir / f"fpsd_ties_probsort_{fp_simil_args_info}.png"), scale=scale)

        logs[eval_tag]["denovo"] = {"mean_fpsd_score_similsort": str(simil_preds_minus_closest.mean()),
                                        "mean_fpsd_score_probsort": str(prob_preds_minus_closest.mean()),
                                        "mean_db_score": str(smiles_sim_of_closest.mean()),
                                        "percentage_of_BART_wins_similsort": str(sum(simil_preds_minus_closest > 0) / len(simil_preds_minus_closest)),
                                        "percentage_of_BART_wins_probsort": str(sum(prob_preds_minus_closest > 0) / len(prob_preds_minus_closest)),
                                        "percentage_of_ties_similsort": str(len(simil_fpsd_tie_simils) / len(simil_preds_minus_closest)),
                                        "percentage_of_ties_probsort": str(len(prob_fpsd_tie_simils) / len(prob_preds_minus_closest)),
                                        "ties": {"num_of_ties_similsort": str(len(simil_fpsd_tie_simils)),
                                                  "num_of_ties_probsort": str(len(prob_fpsd_tie_simils)),
                                                  "num_of_ties_simils_equal_to_1_similsort": str(sum(simil_fpsd_tie_simils == 1)),
                                                  "num_of_ties_simils_equal_to_1_probsort": str(sum(prob_fpsd_tie_simils == 1)),
                                                  "percentage_of_ties_simils_equal_to_1_similsort": str(sum(simil_fpsd_tie_simils == 1) / len(simil_fpsd_tie_simils)),
                                                  "percentage_of_ties_simils_equal_to_1_probsort": str(sum(prob_fpsd_tie_simils == 1) / len(prob_fpsd_tie_simils)),
                                                  "mean_tie_simils_similsort": str(simil_fpsd_tie_simils.mean()),
                                                  "mean_tie_simils_probsort": str(prob_fpsd_tie_simils.mean()),
                                                  }
                                    }
            
    yaml.dump(logs, log_file, indent=4)
    print(logs)
    log_file.close()
    pred_f.close()
    

if __name__ == "__main__":
    app()
