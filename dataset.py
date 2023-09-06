# Class representing custom dataset inheriting from torch Dataset
# crucial part: __getitem__ changes data to tensor (could not be
# stored as jsonl)

import json
import torch
from torch.utils.data import Dataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, SampleMultiplexer, Concater
from typing import Dict, Union, Any, Optional, List, Tuple, Iterable
import pandas as pd
import numpy as np
from pathlib import Path 


# from dataset file

class SpectroDataset(Dataset):
    def __init__(self, df_or_pth_jsonl, eval_mode=False):
        """
        Parameters
        ----------
        df_or_pth_jsonl: pd.DataFrame 
            dataframe with prepared data or a path to DFs stored as jsonl (prepared with run_prepare_data.sh)
        eval_mode: bool
            evaluation mode where we work only with input_ids and position_ids
        """
        if isinstance(df_or_pth_jsonl, pd.DataFrame):
            self.data = df_or_pth_jsonl
        else:
            self.data = pd.read_json(df_or_pth_jsonl, orient="records", lines=True)
        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        out = {"input_ids": row["input_ids"],
                "position_ids": row["position_ids"],
                "attention_mask": row["attention_mask"],
                }
        if not self.eval_mode:
            out["decoder_attention_mask"] = row["decoder_attention_mask"],
            out["labels"] = row["labels"]
        return out


class SpectroDataCollator:
    """
    A class that from a list of dataset elements forms a batch
    """

    def __init__(self, eval_mode=False):
        self.eval_mode = eval_mode

    def __call__(
        self, batch: List[Dict[str, list]]
    ) -> Dict[str, torch.Tensor]:
        return self._collate_batch(batch)
                   
    def _collate_batch(self, batch: List[Dict[str, list]],
                eval_mode: bool = False ) -> Dict[str, torch.Tensor]:
        """Collate `examples` into a batch"""
        inputs = []
        position_ids = []
        attention_masks = []
        if eval_mode:
            for e in batch:
                inputs.append(e["input_ids"])
                position_ids.append(e["position_ids"])
                attention_masks.append(e["attention_mask"])
            return {"input_ids": torch.tensor(inputs),
                    "position_ids": torch.tensor(position_ids),
                    "attention_mask": torch.tensor(attention_masks)}
        else:
            dec_att = []
            labels = []
            for e in batch:
                inputs.append(e["input_ids"])
                position_ids.append(e["position_ids"])
                attention_masks.append(e["attention_mask"])
                dec_att.append(e["decoder_attention_mask"])
                labels.append(e["labels"])
            return {"input_ids": torch.tensor(inputs),
                    "position_ids": torch.tensor(position_ids),
                    "attention_mask": torch.tensor(attention_masks),
                    "decoder_attention_mask": torch.tensor(dec_att),
                    "labels": torch.tensor(labels)}


def json_loader(row):
    return json.loads(row[1])


def build_single_datapipe(json_file: str, buffer_size=2):
    datapipe = IterableWrapper([json_file])
    datapipe = datapipe.open_files(mode='rt')
    datapipe = datapipe.readlines()
    if buffer_size:
        datapipe = datapipe.shuffle(buffer_size=2)
    datapipe = datapipe.sharding_filter()  # after shuffle, before expensive operations
    datapipe = datapipe.map(json_loader)
    return datapipe


def build_datapipe_mixture(datapipes: List[IterDataPipe], 
                           weights: Union[Tuple[float], None], 
                           concat: bool = False):
    """
    Build a datapipe that samples from a mixture of datapipes.
    Parameters
    ----------
    datapipes: List[IterDataPipe]
        List of datapipes to sample from.
    weights: List[float]
        List of weights for each datapipe.
    concat: bool
        Whether to concatenate the datapipes (if False, the pipes will be interleaved).
    """
    if concat:
        return Concater(*datapipes)
    else:
        assert weights is not None, "weights must be provided if concat=False"
        assert len(datapipes) == len(weights)
        
        return SampleMultiplexer({
            pipe: weight for pipe, weight in zip(datapipes, weights)
        })
    

def load_all_datapipes(data_args: Dict[str, Any]) -> Dict[str, IterDataPipe]:
    """
    Load all datapipes from a dict of json files.
    Parameters
    ----------
    data_args: Dict[str,Dict]
        Dictionary containing datasets' paths, weights and other config info.
    buffer_size: int
        Buffer size for shuffling.
    
    Returns
    -------
    Dict[str, IterDataPipe]
        Dictionary with train and valid datapipes.
        {"train": train_datapipe,
         "valid":
             {"dataset1": valid_datapipe1,
             "dataset2": valid_datapipe2}
        }
    """
    buffer_size = data_args["buffer_size"]
    datapipes = {}

    info = [(dataset["train_path"], 
             dataset["valid_path"],
             dataset["weight"],
             dataset_name)
           for dataset_name, dataset in data_args["datasets"].items()]
    train_paths, valid_paths, weights, dataset_names = list(zip(*info))
    
    train_pipes = [build_single_datapipe(path, buffer_size=buffer_size) for path in train_paths]
    valid_pipes = {name: build_single_datapipe(path, buffer_size=buffer_size) 
                   for name, path in zip(dataset_names, valid_paths)}

    datapipes["train"] = build_datapipe_mixture(train_pipes, weights, concat=False)
    datapipes["valid"] = valid_pipes
    return datapipes