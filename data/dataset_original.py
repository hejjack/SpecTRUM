# Class representing custom dataset inheriting from torch Dataset
# crucial part: __getitem__ changes data to tensor (could not be
# stored as pickle)

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, Union, Any, Optional, List, Tuple
import pandas as pd


# from dataset file

class OriginalDataset(Dataset):
    """ params
        data_path: path to the data stored in jsons prepared with prepare_data.py
        gen_mode: creates daaset with direct information about the separation token for summary generation
        sep_id: id of separator token in tokenizer's vocab
    """
    def __init__(self, data_path, gen_mode=False, sep_id=50257):
        self.data = self._load_data(data_path)
        self.len = len(self.data)
#         self.gen_mode = gen_mode

    def _load_data(self, data_path):
        df = pd.read_pickle(data_path)
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        if self.gen_mode:
            out = {"input_ids": torch.tensor(row["input_ids"])}
            
        else:
            out = {"input_ids": torch.tensor(row["input_ids"]),
                   "attention_mask": torch.tensor(row["encoder_attention_mask"]),
                   "decoder_input_ids": torch.tensor(row["decoder_input_ids"]),
                   "decoder_attention_mask": torch.tensor(row["decoder_attention_mask"]),
#                    "position_ids": torch.tensor(row["position_ids"]),
                   "lm_labels": torch.tensor(row["lm_labels"])}
        
        return out


class OriginalDataCollator:
    """
    a class that from a list of dataset elements forms a batch
    """

    def __init__(self, gen_mode=False):
        self.gen_mode = gen_mode

    def __call__(
        self, examples: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        batch = self._collate_batch(examples)
        return batch
                   
    def _collate_batch(self, examples):
        """Collate `examples` into a batch"""
        if self.gen_mode:
            inputs = []
            for e in examples:
                inputs.append(e["input_ids"])
            return {"input_ids": torch.stack(inputs, dim=0)}
        
        else:
            inputs = []
            dec_inputs = []
            enc_att = []
            dec_att = []
#             position_ids = []
            labels = []
            for e in examples:
                inputs.append(e["input_ids"])
                dec_inputs.append(e["decoder_input_ids"])
                enc_att.append(e["attention_mask"])
                dec_att.append(e["decoder_attention_mask"])
#                 position_ids.append(e["position_ids"])
                labels.append(e["lm_labels"])
            return {"input_ids": torch.stack(inputs, dim=0),
                    "decoder_input_ids": torch.stack(dec_inputs, dim=0),
                    "attention_mask": torch.stack(enc_att, dim=0),
                    "decoder_attention_mask": torch.stack(dec_att, dim=0),
#                     "position_ids": torch.stack(position_ids, dim=0),
                    "lm_labels": torch.stack(labels, dim=0)}
