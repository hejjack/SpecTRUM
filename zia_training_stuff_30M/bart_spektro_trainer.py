from typing import Optional, List, Dict
import math
import time 
import copy

import torch
from transformers.trainer_utils import speed_metrics
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
import wandb
import pandas as pd
from rdkit import Chem

from data_utils import SpectroDataset


class BartSpectroTrainer(Trainer):
    def __init__(self,
                 *args,
                 eval_subset_size: int = None,
                 generate_kwargs: Dict = {},
                 eval_log_predictions_size: int = None,
                 eval_tokenizer = None,
                 w_run = None, 
                 **kwargs):
        """
        Customized Trainer class for BartSpectro model.

        Kwargs
        ------
        eval_subset_size: int
            number of samples to use for evaluation. If None, use all samples.
        generate_kwargs: Dict
            kwargs to pass to model.generate() method in evaluate() method when logging predictions to wandb.
        eval_log_predictions_size: int
            number of predictions to log to wandb. If None, don't log predicitons.
        eval_tokenizer: tokenizers.Tokenizer
            tokenizer to use for decoding predictions. If None, use self.tokenizer.
            we add it, because when the normal trainer.tokenizer is present the behavior changes in 
            many different ways, not compatible with tokenizers.Tokenizer
        w_run: wandb.run instance for loogging
        """

# TAKE THE NEWEST VERSION OF THIS SCRIPT FROM THE REPO