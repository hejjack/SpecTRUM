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
from typing import Dict, List, Optional, Tuple, Union
from callbacks import PredictionLogger
from metrics import SpectroMetrics


# custom code
from dataset import SpectroDataset, SpectroDataCollator, load_all_datapipes
from bart_spektro.modeling_bart_spektro import BartSpektoForConditionalGeneration
from bart_spektro.configuration_bart_spektro import BartSpektroConfig
from bart_spektro.bart_spektro_tokenizer import BartSpektroTokenizer
from tokenizers import Tokenizer

app = typer.Typer()


def get_nice_time():
    now = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    now = now.replace(":","_").replace(" ", "-")
    return now

def get_spectro_config(model_args: Dict, tokenizer: Tokenizer) -> BartSpektroConfig:
    return BartSpektroConfig(vocab_size = len(tokenizer.get_vocab()),
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


def load_prediction_loggers(datapipes: Dict[str, dp.iter.IterableWrapper],
                            hf_training_args: Dict,
                            dataset_args: Dict,
                            example_gen_args: Dict,
                            tokenizer: Tokenizer,
                            show_raw_preds: bool = False,
                            log_to_wandb: bool = True) -> List[PredictionLogger]:
    """Load prediction loggers for all datasets in datapipes"""
    callbacks = []
    for name, pipe in datapipes["example"].items():
        example_gen_args["forced_decoder_ids"] = [[1, tokenizer.token_to_id(dataset_args["datasets"][name]["prefix_token"])]]
        callbacks.append(PredictionLogger(
            log_prefix=name,
            dataset=pipe,
            collator=SpectroDataCollator(),
            log_every_n_steps=hf_training_args["eval_steps"],
            show_raw_preds=show_raw_preds,
            log_to_wandb=log_to_wandb,
            generate_kwargs=example_gen_args,
        ))
    return callbacks


@app.command()
def main(config_file: Path = typer.Option(..., dir_okay=False, help="Path to the config file"),
         checkpoint: Path = typer.Option(None, help="Path to the checkpoint directory"),
         resume_id: str = typer.Option(None, help="Wandb id of the run to resume, if not None, resume will be attempted"),
         checkpoints_dir: Path = typer.Option("../checkpoints", help="Path to the checkpoints directory"),
         additional_info: str = typer.Option(None, help="use format '_info'; additional info to add to run_name"),
         device: str = typer.Option("cuda", help="Device to use for training"),
         wandb_group: str = typer.Option(..., help="Wandb group to use for logging")
         ):

    # load config
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError("Error in configuration file:", exc) from exc

    hf_training_args = config["hf_training_args"]
    dataset_args = config["data_args"]
    model_args = config["model_args"]
    example_gen_args = config["example_gen_args"]
    trainer_args = config["trainer_args"]
    tokenizer_path = model_args["tokenizer_path"]

    # load tokenizer, data
    tokenizer = Tokenizer.from_file(tokenizer_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # surpressing a warning
    datapipes = load_all_datapipes(dataset_args)
    bart_spectro_config = get_spectro_config(model_args, tokenizer)


    print("Loading model...")
    model = BartSpektoForConditionalGeneration(bart_spectro_config)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    run_name = f"{get_nice_time()}{additional_info}"
    # Resume training
    if resume_id:
        if not checkpoint:
            raise ValueError("Checkpoint must be provided when resuming training")
        save_path = checkpoint.parent
    else:
        save_path = checkpoints_dir / run_name
    print(f"save path: {save_path}")

    # Init wandb
    log_tags = [d for d in dataset_args["datasets"].keys()]
    log_tags.append(wandb_group)
    log_tags.append(f"params={num_params}")
    log_tags.append(f"lr={hf_training_args['learning_rate']}")
    log_tags.append(f"pd_bs={hf_training_args['per_device_train_batch_size']}")

    wandb.login()
    run = wandb.init(
            id=resume_id, 
            resume="must" if resume_id else "never",
            entity="hajekad", 
            project="BART_for_gcms",
            tags=log_tags,
            save_code=True,
            config=config,
            group=wandb_group,
            name = run_name
        )
    
    # set callbacks
    example_gen_args["forced_decoder_ids"] = [[1, ]]
    callbacks = load_prediction_loggers(datapipes, 
                                        hf_training_args,
                                        dataset_args, 
                                        example_gen_args, 
                                        tokenizer,
                                        show_raw_preds=True) ##!!DEBUG!!##

    compute_metrics = SpectroMetrics(tokenizer)
    training_args = TrainingArguments(**hf_training_args,
                                      output_dir=str(save_path),
                                      run_name=run_name
                                    )

    trainer = Trainer(
                model=model,                   
                args=trainer_args,                
                train_dataset=datapipes["train"],
                eval_dataset=datapipes["valid"], 
                callbacks=callbacks,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                data_collator = SpectroDataCollator(),
            )
    
    if checkpoint:
        trainer.train(str(checkpoint))
    else:
        trainer.train()


if __name__ == "__main__":
    app()