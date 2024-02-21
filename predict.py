import sys
import io
import os
import time 
from datetime import datetime
from pathlib import Path
import typer
import yaml
import torch
import numpy as np
import pandas as pd
import json
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers.utils import ModelOutput
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
from icecream import ic
from rdkit import Chem, RDLogger
from data_utils import SpectroDataset, SpectroDataCollator, build_single_datapipe
# from bart_spektro import BartSpektroForConditionalGeneration
from bart_spektro.modeling_bart_spektro import BartSpektroForConditionalGeneration
from train_bart import build_tokenizer

RDLogger.DisableLog('rdApp.*')

app = typer.Typer(pretty_exceptions_enable=False)


def open_files(output_folder: Path, 
               checkpoint: Path, 
               dataset_config: Dict[str, Any],
               additional_info: str = "") -> Tuple[io.TextIOWrapper, io.TextIOWrapper]:
    """Opens log and predictions files and returns their handles"""
    timestamp = time.time()
    model_name = checkpoint.parent.name
    run_name = str(round(timestamp)) + \
        "_" + dataset_config["data_split"]
    run_name += (f"_{data_range}") if data_range else "_full"   
    run_name += ("_" + additional_info) if additional_info else ""
    
    run_folder = output_folder / model_name / dataset_config["dataset_name"] / run_name
    run_folder.mkdir(parents=True, exist_ok=True)
    log_file = (run_folder / "log_file.yaml").open("w")
    predictions_file = (run_folder / "predictions.jsonl").open("w")
    return log_file, predictions_file


def get_unique_predictions(preds: List[List]):
    """takes a list of SMILES predictions for each input in batch.
    Returns a list of unique predictions and their indices """
    unique = [np.unique(np.array(p), axis=0, return_index=True) for p in preds]
    unique_preds, unique_idxs = zip(*unique)
    return unique_preds, unique_idxs


def get_canon_predictions(preds: List[List], idxs: List[List]):
    """
    Takes a list of SMILES predictions for each input in batch and
    their indexes within the batch.
    Filters non-valid SMILESes and returns a list of canonicalized
    predictions with respectively filtered indexes.
    """
    idxs = [list(a) for a in idxs]
    canon_preds = [[] for _ in range(len(preds))]
    canon_idxs = [[] for _ in range(len(preds))]
    for i, group in enumerate(preds):
        for j, smi in enumerate(group):
            try:
                canon_preds[i].append(Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False))
                canon_idxs[i].append(idxs[i][j])
            except Exception:
                pass
    return canon_preds, canon_idxs


def get_sequence_probs(model,
                       generated_outputs: ModelOutput,
                       batch_size: int,
                       is_beam_search: bool):
    """ collect the generation probability of all generated sequences """
    transition_scores = model.compute_transition_scores(
                                generated_outputs.sequences,
                                generated_outputs.scores,
                                beam_indices=generated_outputs.beam_indices if is_beam_search else None,
                                normalize_logits=True)  # type: ignore

    transition_scores = transition_scores.reshape(batch_size, -1, transition_scores.shape[-1])
    transition_scores[torch.isinf(transition_scores)] = 0
    if is_beam_search:
        output_length = (transition_scores < 0).sum(dim=-1)
        length_penalty = model.generation_config.length_penalty
        all_probs = transition_scores.sum(dim=2).exp() / output_length**length_penalty
    else:
        all_probs = transition_scores.sum(dim=2).exp()
    return all_probs


def prepare_decoder_input(decoder_input_token: str, tokenizer: PreTrainedTokenizerFast, batch_size: int):
    """ prepare forced prefix input for the decoder"""
    if decoder_input_token:  # prepare dataset-specific prefixes for decoding
        decoder_input_ids_single = tokenizer.encode(decoder_input_token) if decoder_input_token else None
        decoder_input_ids_batch = torch.tensor([decoder_input_ids_single] * batch_size)
    else:
        decoder_input_ids_batch = None
    return decoder_input_ids_batch


def timestamp_to_readable(timestamp: float) -> str:    
    return datetime.utcfromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S")


def hours_minutes_seconds(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


@app.command()
def main(
    checkpoint: Path = typer.Option(..., dir_okay=True, file_okay=True, readable=True, help="Path to the checkpoint file"),
    output_folder: Path = typer.Option(..., dir_okay=True, file_okay=True, exists=False, writable=True, help="Path to the folder where the predictions will be saved"),
    config_file: Path = typer.Option(..., dir_okay=False, file_okay=True, exists=True, readable=True, help="File with all the needed configurations"),
    data_range: str = typer.Option("", help="Range of data to generate predictions for. Format: <start>:<end>"),
) -> None:

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generation_config = config["generation_args"]
    general_config = config["general"]
    dataloader_config = config["dataloader"]
    dataset_config = config["dataset"]

    config["command"] = " ".join(sys.argv)
    config["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    start_time = time.time()
    config["start_loading_time"] = timestamp_to_readable(start_time)
    
    batch_size = dataloader_config["batch_size"]
    if batch_size != 1:
        raise ValueError("For different batch sizes the prediction gives wrong results. Please set batch_size=1")
    device = general_config["device"]
    additional_info = general_config["additional_naming_info"]
    
    datapipe = build_single_datapipe(dataset_config["data_path"],
                                     shuffle=False,
                                     )
    if data_range:
        data_range_min, data_range_max = list(map(int, data_range.split(":")))
    else:
        data_range_min, data_range_max = None, None

    # set output files
    log_file, predictions_file = open_files(output_folder, checkpoint, dataset_config, data_range, additional_info)

    model = BartSpektroForConditionalGeneration.from_pretrained(checkpoint)
    model.generation_config.length_penalty = generation_config["length_penalty"]
    tokenizer = build_tokenizer(config["tokenizer_path"])
    loader = torch.utils.data.DataLoader(datapipe, **dataloader_config, collate_fn=SpectroDataCollator(inference_mode=True), drop_last=False, shuffle=False) # type: ignore

    decoder_input_token = generation_config.pop("decoder_input_token", "")
    decoder_input_ids = prepare_decoder_input(decoder_input_token, tokenizer, batch_size)

    yaml.dump(config, log_file)

    # Start generating
    start_generation_time = time.time()
    config["start_generation_time"] = timestamp_to_readable(start_generation_time)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            # take care of data range since datapipe has no len()
            if data_range_min is not None and i < data_range_min:
                continue
            if data_range_max is not None and i >= data_range_max:
                break
            
            # proceed with generation
            model_input = {key: value.to(device) for key, value in batch.items()} # move tensors from batch to device
            generated_outputs = model.generate( # type: ignore
                decoder_input_ids=decoder_input_ids.to(device),
                **model_input,
                **generation_config,
                output_scores=True,
                return_dict_in_generate=True,
            )

            preds = tokenizer.decode_batch(generated_outputs.sequences.tolist(), skip_special_tokens=True) # type: ignore
            preds = np.array(preds).reshape(batch_size, generation_config["num_return_sequences"])
            
            unique_preds, unique_idxs = get_unique_predictions(preds)
            canon_preds, canon_idxs = get_canon_predictions(unique_preds, unique_idxs) # filter invalid and canonicalize
            all_probs = get_sequence_probs(model, generated_outputs, batch_size, generation_config["num_beams"] > 1)

            result_jsonl = ""
            for i, group in enumerate(canon_preds):
                result_jsonl += json.dumps({
                    group[j]: all_probs[i, canon_idx].item()
                    for j, canon_idx in enumerate(canon_idxs[i])}) + "\n"
            
            predictions_file.write(result_jsonl)

    finished_time = time.time()
    
    predictions_file.close()
    log_config = {
        "finished_time_utc": timestamp_to_readable(finished_time),
        "generation_time": f"{hours_minutes_seconds(finished_time - start_generation_time)}",
        "wall_time_utc": f"{hours_minutes_seconds(finished_time - start_time)}"}
    yaml.dump(log_config, log_file)
    log_file.close()


if __name__ == "__main__":
    app()
