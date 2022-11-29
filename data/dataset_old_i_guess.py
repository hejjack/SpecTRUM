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
    """ params
        df_or_pth_pkl: DataFrame with prepared data or a path to DFs stored as pkl (prepared with run_prepare_data.sh)
        eval_mode: evaluation mode where we work only with input_ids and position_ids
        sep_id: id of separator token in tokenizer's vocab
    """
    def __init__(self, df_or_pth_pkl, eval_mode=False, sep_id=50257):
        if isinstance(df_or_pth_pkl, pd.DataFrame):
            self.data = df_or_pth_pkl
        else:
            self.data = pd.read_pickle(df_or_pth_pkl)
        self.len = len(self.data)
        self.eval_mode = eval_mode
        
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        if self.eval_mode:
            out = {"input_ids": torch.tensor(row["input_ids"]),
                   "position_ids": torch.tensor(row["position_ids"])}
            
        else:
            out = {"input_ids": torch.tensor(row["input_ids"]),
                   "attention_mask": torch.tensor(row["encoder_attention_mask"]),
                   "decoder_input_ids": torch.tensor(row["decoder_input_ids"]),
                   "decoder_attention_mask": torch.tensor(row["decoder_attention_mask"]),
                   "position_ids": torch.tensor(row["position_ids"]),
                   "lm_labels": torch.tensor(row["lm_labels"])}
        
        return out


class SpectroDataCollator:
    """
    a class that from a list of dataset elements forms a batch
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
        
        if self.eval_mode:
            for e in examples:
                inputs.append(e["input_ids"])
                position_ids.append(e["position_ids"])
            return {"input_ids": torch.stack(inputs, dim=0),
                    "position_ids": torch.stack(position_ids, dim=0)} 
        else:
            dec_inputs = []
            enc_att = []
            dec_att = []
            position_ids = []
            labels = []
            for e in examples:
                inputs.append(e["input_ids"])
                dec_inputs.append(e["decoder_input_ids"])
                enc_att.append(e["attention_mask"])
                dec_att.append(e["decoder_attention_mask"])
                position_ids.append(e["position_ids"])
                labels.append(e["lm_labels"])
            return {"input_ids": torch.stack(inputs, dim=0),
                    "decoder_input_ids": torch.stack(dec_inputs, dim=0),
                    "attention_mask": torch.stack(enc_att, dim=0),
                    "decoder_attention_mask": torch.stack(dec_att, dim=0),
                    "position_ids": torch.stack(position_ids, dim=0),
                    "lm_labels": torch.stack(labels, dim=0)}