# Class representing custom dataset inheriting from torch Dataset
# crucial part: __getitem__ changes data to tensor (could not be
# stored as jsonl)

import json
import torch
from torch.utils.data import Dataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, SampleMultiplexer, Concater, Header
from typing import Dict, Sized, Union, Any, Optional, List, Tuple, Iterator, Hashable, TypeVar
import warnings
import pandas as pd
import numpy as np
from pathlib import Path 

T_co = TypeVar("T_co", covariant=True)

 
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


def build_single_datapipe(json_file: str, 
                          shuffle: bool,
                          buffer_size: Optional[int] = None,
                          limit: Optional[int] = None,
                        ):
    datapipe = IterableWrapper([json_file])
    datapipe = datapipe.open_files(mode='rt')
    datapipe = datapipe.readlines()
    if buffer_size and shuffle:
        print(f"shuffling {json_file} with buffer_size={buffer_size}")
        datapipe = datapipe.shuffle(buffer_size=buffer_size)
    elif buffer_size or shuffle:
        warnings.warn("SHUFFLE and its buffer_size are IGNORED if either not specified")
    datapipe = datapipe.sharding_filter()  # after shuffle, before expensive operations
    if limit is not None:
        datapipe = datapipe.header(limit)
    datapipe = datapipe.map(json_loader)
    return datapipe


def build_datapipe_mixture(datapipes: List[IterDataPipe], 
                           weights: Union[Tuple[float], None], 
                           concat: bool = False,
                           seed: Optional[int] = 42):
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
        return Concater(*datapipes) # type: ignore
    else:
        assert weights is not None, "weights must be provided if concat=False"
        assert len(datapipes) == len(weights)
        
        # filter zero weight datasets
        nonzero_datapipes = [(pipe, weight) for pipe, weight in zip(datapipes, weights) if weight]
        if len(nonzero_datapipes)==0:
            raise ValueError("All train weights are zero! Please provide at least one non-zero weight.")
        print("Number of non-zero weight datasets: ", len(nonzero_datapipes))

        return SampleMultiplexer({
            pipe.cycle(): weight for pipe, weight in nonzero_datapipes
        }, seed=seed)


def load_all_datapipes(data_args: Dict[str, Any]) -> Dict[str, IterDataPipe]:
    """
    Load all datapipes from a dict of json files.
    Parameters
    ----------
    data_args: Dict[str,Dict]
        Dictionary containing datasets' paths, weights and other config info (like shuffle and buffer_size       Buffer size for shuffling.
    
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
    seed = data_args["data_seed"] if data_args.get("data_seed") else 42
    buffer_size = data_args["buffer_size"]
    shuffle_train = data_args["shuffle_train"]
    datapipes = {}

    info = [(dataset_name,
             dataset["train_path"], 
             dataset["valid_path"],
             dataset["weight"],
             dataset["limit_train_split"],
             dataset["limit_val_split"],
             dataset["limit_example_split"]
             )
           for dataset_name, dataset in data_args["datasets"].items()]
    dataset_names, train_paths, valid_paths, weights, limit_train_splits, limit_val_splits, limit_example_splits = list(zip(*info))
    
    train_pipes = [build_single_datapipe(path, 
                                         shuffle=shuffle_train,
                                         buffer_size=buffer_size,
                                         limit=limit,
                                         ) 
                   for path, limit in zip(train_paths, limit_train_splits)]
    valid_pipes = {name: build_single_datapipe(path, 
                                               shuffle=False,
                                               buffer_size=buffer_size,
                                               limit=limit)
                   for name, path, limit in zip(dataset_names, valid_paths, limit_val_splits)}
    example_pipes = {name: build_single_datapipe(path, 
                                                 shuffle=False,
                                                 buffer_size=buffer_size,
                                                 limit=limit)
                     for name, path, limit in zip(dataset_names, valid_paths, limit_example_splits)}

    datapipes["train"] = build_datapipe_mixture(train_pipes, weights, concat=False, seed=seed)
    datapipes["valid"] = valid_pipes
    datapipes["example"] = example_pipes
    return datapipes