import os
import sys
import inspect

import argparse
from datetime import datetime
import json
import os
import pickle
import random
import time
import wandb
from pathlib import Path
import gc
import glob
from tqdm import tqdm
import torch
import numpy as np
from transformers import TrainingArguments, Trainer, BartConfig, BartForConditionalGeneration
import typer
import yaml
import torchdata.datapipes as dp
from pathlib import Path


# custom code
from dataset import SpectroDataset, SpectroDataCollator, load_all_datapipes
from bart_spektro.modeling_bart_spektro import BartSpektoForConditionalGeneration
from bart_spektro.configuration_bart_spektro import BartSpektroConfig
from bart_spektro.bart_spektro_tokenizer import BartSpektroTokenizer
from bart_spektro.bart_spektro_trainer import BartSpectroTrainer
from tokenizers import Tokenizer

app = typer.Typer()


def get_nice_time():
    now = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    now = now.replace(":","_").replace(" ", "-")
    return now

def get_spectro_config(model_args, tokenizer):
    BartSpektroConfig(vocab_size = len(tokenizer.get_vocab()),
                      max_position_embeddings = model_args["seq_len"],
                      max_length = model_args["seq_len"],
                      min_len = 0,
                      encoder_layers = model_args["encoder_layers"],
                      encoder_ffn_dim = model_args["encoder_ffn_dim"],
                      encoder_attention_heads = model_args["encoder_attention_heads"],
                      decoder_layers = model_args["decoder_layers"],
                      decoder_ffn_dim = model_args["decoder_ffn_dim"],
                      decoder_attention_heads = model_args["decoder_attention_heads"],
                      encoder_layerdrop = 0.0,
                      decoder_layerdrop = 0.0,
                      activation_function = 'gelu',
                      d_model = 1024,
                      dropout = 0.2,
                      attention_dropout = 0.0,
                      activation_dropout = 0.0,
                      init_std = 0.02,
                      classifier_dropout = 0.0,
                      scale_embedding = False,
                      use_cache = True,
                      pad_token_id = 2,
                      bos_token_id = 3,
                      eos_token_id = 0,
                      is_encoder_decoder = True,
                      decoder_start_token_id = 3,
                      forced_eos_token_id = 0,
                      max_log_id=9)


@app.command()
def main(config_file: Path = typer.Option(..., dir_okay=False, help="Path to the config file"),
         checkpoint: Path = typer.Option(None, help="Path to the checkpoint directory"),
         resume_id: str = typer.Option(None, help="Wandb id of the run to resume, if not None, resume will be attempted"),
         checkpoints_dir: Path = typer.Option("../checkpoints", help="Path to the checkpoints directory"),
         save_name: str = typer.Option(None, help="Path to the save directory")
         ):

    # load config
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError("Error in configuration file:", exc) from exc

    hw_args = config["hw_args"]
    training_args = config["training_args"]
    dataset_args = config["data_args"]
    tokenizer_path = config["tokenizer_path"]
    model_args = config["model_args"]
    
    # load tokenizer, data, model
    tokenizer = Tokenizer.from_file(tokenizer_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # surpressing a warning
    datapipes = load_all_datapipes(dataset_args)
    bart_spectro_config = get_spectro_config(model_args, tokenizer)
    model = BartSpektoForConditionalGeneration(config)
    model.to(hw_args["device"])



    if resume_id:
        if not checkpoint:
            raise ValueError("Checkpoint must be provided when resuming training")
            
        save_path = checkpoint.parent
        save_name = checkpoint.parent
    else:
        if save_name:
            save_path = checkpoints_dir / save_name
        else:
            save_path = checkpoints_dir / f"bart_{get_nice_time()}"
    print(f"save path: {save_path}")
    # Resume training

    # Init wandb
    wandb.login()
    run = wandb.init(id=resume_id, resume="allow", entity="hajekad", project="BART_for_gcms")
    wandb.run.name = save_name



    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")


if __name__ == "__main__":
    app()