# Class representing custom dataset inheriting from torch Dataset
# crucial part: __getitem__ changes data to tensor (could not be
# stored as pickle)

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, Union, Any, Optional, List, Tuple
import pandas as pd


# from dataset file

class SpectroDataset(Dataset):
    def __init__(self, df_or_pth_pkl, eval_mode=False, original=False):
        """ 
        Parameters
        ----------
        df_or_pth_pkl: pd.DataFrame 
            dataframe with prepared data or a path to DFs stored as pkl (prepared with run_prepare_data.sh)
        eval_mode: bool
            evaluation mode where we work only with input_ids and position_ids
        original: bool
            in case we don't want to use position_ids
        """
        if isinstance(df_or_pth_pkl, pd.DataFrame):
            self.data = df_or_pth_pkl
        else:
            self.data = pd.read_pickle(df_or_pth_pkl)
        self.len = len(self.data)
        self.eval_mode = eval_mode
        self.original = original

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        if self.eval_mode:
            out = {"input_ids": torch.tensor(row["input_ids"]),
                  "position_ids": torch.tensor(row["position_ids"]),
                  "attention_mask": torch.tensor(row["attention_mask"]),
                  }
        elif self.original:
            out = {"input_ids": torch.tensor(row["input_ids"]),
                   "attention_mask": torch.tensor(row["encoder_attention_mask"]),
                   "decoder_attention_mask": torch.tensor(row["decoder_attention_mask"]),
                   "labels": torch.tensor(row["labels"])}
        else:
            out = {"input_ids": torch.tensor(row["input_ids"]),
                   "attention_mask": torch.tensor(row["encoder_attention_mask"]),
#                    "decoder_input_ids": torch.tensor(row["decoder_input_ids"]),
                   "decoder_attention_mask": torch.tensor(row["decoder_attention_mask"]),
                   "position_ids": torch.tensor(row["position_ids"]),
                   "labels": torch.tensor(row["labels"])}
        
        return out


class SpectroDataCollator:
    """
    A class that from a list of dataset elements forms a batch
    """

    def __init__(self, eval_mode=False, original=False):
        self.eval_mode = eval_mode
        self.original = original

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
                    "attention_mask": torch.stack(attention_mask, dim=0)}
        else:
            dec_inputs = []
            enc_att = []
            dec_att = []
            labels = []
            for e in examples:
                inputs.append(e["input_ids"])
#                 dec_inputs.append(e["decoder_input_ids"])
                enc_att.append(e["attention_mask"])
                dec_att.append(e["decoder_attention_mask"])
                if not self.original:
                    position_ids.append(e["position_ids"])
                labels.append(e["labels"])
            if self.original:
                return {"input_ids": torch.stack(inputs, dim=0),
#                     "decoder_input_ids": torch.stack(dec_inputs, dim=0),
                    "attention_mask": torch.stack(enc_att, dim=0),
                    "decoder_attention_mask": torch.stack(dec_att, dim=0),
                    "labels": torch.stack(labels, dim=0)}
            
            return {"input_ids": torch.stack(inputs, dim=0),
#                     "decoder_input_ids": torch.stack(dec_inputs, dim=0),
                    "attention_mask": torch.stack(enc_att, dim=0),
                    "decoder_attention_mask": torch.stack(dec_att, dim=0),
                    "position_ids": torch.stack(position_ids, dim=0),
                    "labels": torch.stack(labels, dim=0)}