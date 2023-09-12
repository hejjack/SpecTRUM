import os

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
import transformers
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizerFast, Trainer
import typer
import yaml
import torchdata.datapipes as dp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from callbacks import PredictionLogger
from metrics import SpectroMetrics
from icecream import ic


# custom code
from dataset import SpectroDataset, SpectroDataCollator, load_all_datapipes
from bart_spektro.modeling_bart_spektro import BartSpektroForConditionalGeneration
from bart_spektro.configuration_bart_spektro import BartSpektroConfig
from bart_spektro.bart_spektro_tokenizer import BartSpektroTokenizer
from tokenizers import Tokenizer


app = typer.Typer()


def get_nice_time():
    now = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    now = now.replace(":", "_").replace(" ", "-")
    return now


def enrich_best_metric_name(metric_name: str, dataset_name: str) -> str:
    subnames = metric_name.split("_")
    subnames = subnames[:1] + [dataset_name] + subnames[1:]
    metric_name = "_".join(subnames)
    return metric_name


def build_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    bpe_tokenizer = Tokenizer.from_file(tokenizer_path)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe_tokenizer,
                                        bos_token="<bos>",
                                        eos_token="<eos>",
                                        unk_token="<ukn>",
                                        pad_token="<pad>",
                                        is_split_into_words=True)
    return tokenizer


def get_spectro_config(model_args: Dict, tokenizer: PreTrainedTokenizerFast) -> BartSpektroConfig:
    return BartSpektroConfig(vocab_size=len(tokenizer.get_vocab()),
                             max_position_embeddings=model_args["seq_len"],
                             max_length=model_args["seq_len"],
                             min_len=0,
                             encoder_layers=model_args["encoder_layers"],
                             encoder_ffn_dim=model_args["encoder_ffn_dim"],
                             encoder_attention_heads=model_args["encoder_attention_heads"],
                             decoder_layers=model_args["decoder_layers"],
                             decoder_ffn_dim=model_args["decoder_ffn_dim"],
                             decoder_attention_heads=model_args["decoder_attention_heads"],
                             encoder_layerdrop=0.0,
                             decoder_layerdrop=0.0,
                             activation_function='gelu',
                             d_model=1024,
                             dropout=0.2,
                             attention_dropout=0.0,
                             activation_dropout=0.0,
                             init_std=0.02,
                             classifier_dropout=0.0,
                             scale_embedding=False,
                             use_cache=True,
                             pad_token_id=2,
                             bos_token_id=3,
                             eos_token_id=0,
                             is_encoder_decoder=True,
                             decoder_start_token_id=3,
                             forced_eos_token_id=0,
                             max_log_id=9)
 

@app.command()
def main(config_file: Path = typer.Option(..., dir_okay=False, help="Path to the config file"),
         checkpoint: Path = typer.Option(None, help="Path to the checkpoint directory"),
         resume_id: str = typer.Option(None, help="Wandb id of the run to resume, if not None, resume will be attempted"),
         checkpoints_dir: Path = typer.Option("../checkpoints", help="Path to the checkpoints directory"),
         additional_info: str = typer.Option(None, help="use format '_info'; additional info to add to run_name"),
         additional_tags: str = typer.Option(None, help="Tags to add to the wandb run, one string, delimited by ':'"),
         device: str = typer.Option("cuda", help="Device to use for training"),
         wandb_group: str = typer.Option(..., help="Wandb group to use for logging"),
         ):
    
if additional_tags:
        add_tags = additional_tags.split(":")
    else:
        add_tags = []

    cvd = os.environ['CUDA_VISIBLE_DEVICES']
    print(f"CUDA_VISIBLE_DEVICES set to: {cvd}")
    if len(cvd) < 60:
        add_tags.append("CVD=" + cvd)
    else:
        add_tags.append("CVD=weird_meta_id")

    for i in range(torch.cuda.device_count()):
        print(f"device: {device }")
        print(torch.cuda.get_device_properties(i))


    # load config
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError("Error in configuration file:", exc) from exc

    hf_training_args = config["hf_training_args"]
    dataset_args = config["data_args"]
    model_args = config["model_args"]
    example_gen_args = config["example_generation_args"]
    tokenizer_path = model_args["tokenizer_path"]
    use_wandb = hf_training_args["report_to"] == "wandb"
    
    # GPU specific batch size
    gpu_ram = torch.cuda.get_device_properties(0).total_memory
    if gpu_ram > 70*1e9:
        print("WARNING!!!: Using automatically specific batch size")
        hf_training_args["per_device_train_batch_size"] = 128
        hf_training_args["per_device_eval_batch_size"] = 64
        hf_training_args["gradient_accumulation_steps"] = 1
        hf_training_args["eval_accumulation_steps"] = 1
        print(f"train batch size: {hf_training_args['per_device_train_batch_size']}")
        print(f"eval batch size: {hf_training_args['per_device_eval_batch_size']}")

    else:
        print("WARNING!!!: Using automatically specific batch size")
        hf_training_args["per_device_train_batch_size"] = 64
        hf_training_args["per_device_eval_batch_size"] = 64
        hf_training_args["gradient_accumulation_steps"] = 2
        hf_training_args["eval_accumulation_steps"] = 1
        print(f"train batch size: {hf_training_args['per_device_train_batch_size']}")
        print(f"eval batch size: {hf_training_args['per_device_eval_batch_size']}")
    
    # set the name for metric choosing the best model (add chosen dataset name)
    if dataset_args.get("dataset_for_choosing_best_model", None):
        hf_training_args["metric_for_best_model"] = enrich_best_metric_name(hf_training_args["metric_for_best_model"], 
                                                                            dataset_args["dataset_for_choosing_best_model"])
        print(f"Metric for choosing best model: {hf_training_args['metric_for_best_model']}")
    else:
        raise ValueError("dataset_for_choosing_best_model must be provided in data_args.")

    # load tokenizer, data
    tokenizer = build_tokenizer(tokenizer_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # surpressing a warning
    datapipes = load_all_datapipes(dataset_args)
    bart_spectro_config = get_spectro_config(model_args, tokenizer)

    print("Loading model...")
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        model = BartSpektroForConditionalGeneration.from_pretrained(checkpoint)
    else:
        model = BartSpektroForConditionalGeneration(bart_spectro_config)
    model.to(device)
    ######
    # ic("model embedding shape", model.model.encoder.embed_tokens.weight.shape) ####
    # ic("tokenizer vocab size: ", len(tokenizer.get_vocab()))
    ######

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")


    # Init wandb
    if use_wandb:
        log_tags = [d for d in dataset_args["datasets"].keys()]
        log_tags.extend(add_tags)
        log_tags.append(wandb_group)
        log_tags.append(f"params={num_params}")
        log_tags.append(f"lr={hf_training_args['learning_rate']}")
        log_tags.append(f"pd_bs={hf_training_args['per_device_train_batch_size']}")
        if additional_info:
            log_tags.append(additional_info)

        wandb.login()
        run = wandb.init(
                id=resume_id, 
                resume="must" if resume_id else "never",
                entity="hajekad", 
                project="BART_for_gcms",
                tags=log_tags,
                save_code=True,
                dir=checkpoints_dir.parent,
                config=config,
                group=wandb_group,
            )
        
        # to not add additional info to the run name if it is already there
        if run.name.endswith(additional_info):
            run_name = run.name
        else:
            run_name = run.name + additional_info
        run.name = run_name
        run.tags += (f"run_id={run.id}",)
    else:
        run_name = get_nice_time() + additional_info
    print(f"Run name: {run_name}")
        
    # Resume training
    if resume_id:
        if not checkpoint:
            raise ValueError("Checkpoint must be provided when resuming training")
        save_path = checkpoint.parent
    else:
        save_path = checkpoints_dir / wandb_group / run_name
    print(f"save path: {save_path}")
    
    # set callbacks
    sorted_dataset_names = sorted([name for name in datapipes["example"].keys()])
    prediction_callback = PredictionLogger(datasets=[datapipes["example"][name] for name in sorted_dataset_names],
                                           prefix_tokens=[dataset_args["datasets"][name]["prefix_token"] for name in sorted_dataset_names],
                                           log_prefixes=sorted_dataset_names, # type: ignore
                                           collator=SpectroDataCollator(),
                                           log_every_n_steps=hf_training_args["eval_steps"],
                                           show_raw_preds=dataset_args["show_raw_preds"],
                                           log_to_wandb=use_wandb,
                                           generate_kwargs=example_gen_args,
                                        )

    compute_metrics = SpectroMetrics(tokenizer)
    seq2seq_training_args = Seq2SeqTrainingArguments(**hf_training_args,
                                                     output_dir=str(save_path),
                                                     run_name=run_name,
                                                     data_seed=dataset_args["data_seed"]
                                                    )
    

    trainer = Seq2SeqTrainer(
                model=model,                   
                args=seq2seq_training_args,                
                train_dataset=datapipes["train"],
                eval_dataset=datapipes["valid"], 
                callbacks=[prediction_callback],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                data_collator = SpectroDataCollator(),
            )
    
    
    if checkpoint and resume_id:
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        trainer.train()


if __name__ == "__main__":
    app()