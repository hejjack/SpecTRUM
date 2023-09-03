# Class representing custom dataset inheriting from torch Dataset
# crucial part: __getitem__ changes data to tensor (could not be
# stored as pickle)

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, Union, Any, Optional, List, Tuple
import pandas as pd
import numpy as np


# from dataset file

class SpectroDataset(Dataset):
    def __init__(self, df_or_pth_pkl, eval_mode=False):
        """
        Parameters
        ----------
        df_or_pth_pkl: pd.DataFrame 
            dataframe with prepared data or a path to DFs stored as pkl (prepared with run_prepare_data.sh)
        eval_mode: bool
            evaluation mode where we work only with input_ids and position_ids
        """
        if isinstance(df_or_pth_pkl, pd.DataFrame):
            self.data = df_or_pth_pkl
        else:
            self.data = pd.read_pickle(df_or_pth_pkl)
        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.eval_mode:
            out = {"input_ids": torch.tensor(row["input_ids"]),
                   "position_ids": torch.tensor(row["position_ids"]),
                   "attention_mask": torch.tensor(row["attention_mask"]),
                   }
        else:
            out = {"input_ids": torch.tensor(row["input_ids"].tolist()),
                   "attention_mask": torch.tensor(row["attention_mask"].tolist()),
                   "decoder_attention_mask": torch.tensor(row["decoder_attention_mask"].tolist()),
                   "position_ids": torch.tensor(row["position_ids"].tolist()),
                   "labels": torch.tensor(row["labels"].tolist())}
        
        return out


class SpectroDataCollator:
    """
    A class that from a list of dataset elements forms a batch
    """

    def __init__(self, eval_mode=False):
        self.eval_mode = eval_mode

    def __call__(
        self, examples: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self._collate_batch(examples)
        return batch
                   
    def _collate_batch(self, examples):
        """Collate `examples` into a batch"""
        inputs = []
        position_ids = []
        attention_masks = []
        if self.eval_mode:
            for e in examples:
                inputs.append(e["input_ids"])
                position_ids.append(e["position_ids"])
                attention_masks.append(e["attention_mask"])
            return {"input_ids": torch.stack(inputs, dim=0),
                    "position_ids": torch.stack(position_ids, dim=0),
                    "attention_mask": torch.stack(attention_masks, dim=0)}
        else:
            dec_att = []
            labels = []
            for e in examples:
                inputs.append(e["input_ids"])
                attention_masks.append(e["attention_mask"])
                dec_att.append(e["decoder_attention_mask"])
                labels.append(e["labels"])
            return {"input_ids": torch.stack(inputs, dim=0),
                    "position_ids": torch.stack(position_ids, dim=0),
                    "attention_mask": torch.stack(attention_masks, dim=0),
                    "decoder_attention_mask": torch.stack(dec_att, dim=0),
                    "labels": torch.stack(labels, dim=0)}
