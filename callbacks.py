from __future__ import annotations

import math
import pathlib
import shutil
from typing import Callable

from tqdm.auto import tqdm
import pandas as pd
import transformers
import tokenizers
import wandb
import torch
import torch.utils.data
import warnings
from rdkit import Chem, DataStructs

from bart_spektro.modeling_bart_spektro import BartSpektoForConditionalGeneration


class PredictionLogger(transformers.TrainerCallback):
    def __init__(
        self,
        log_prefix: str,
        log_every_n_steps: int,
        show_raw_preds: bool,
        dataset: torch.utils.data.Dataset | torch.utils.data.IterableDataset,
        collator: Callable,
        generate_kwargs: dict | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.log_prefix = log_prefix
        self.logging_steps = log_every_n_steps
        self.show_raw_preds = show_raw_preds
        self.dataset = dataset
        self.collator = collator

        if generate_kwargs is None:
            generate_kwargs = {}
        self.generate_kwargs = generate_kwargs

        if "max_length" not in generate_kwargs:
            warnings.warn(
                "you might have forgot to set `max_length` in `generate_kwargs` "
                f"inside {self.__class__.__name__}"
            )

        self.num_examples = sum(1 for _ in dataset)

    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ) -> None:
        if state.global_step % self.logging_steps != 0:
            return
        
        model: BartSpektoForConditionalGeneration = kwargs["model"]
        tokenizer: tokenizers.Tokenizer = kwargs["tokenizer"] # if missing add to the class args
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.collator,
        )

        model.eval()
        
        num_batches = math.ceil(self.num_examples / args.per_device_eval_batch_size)
        progress = tqdm(dataloader, total=num_batches, desc="Generating preds for logging", leave=False)
        
        tokenizer = kwargs["eval_tokenizer"]
        all_raw_preds = []
        all_preds = []
        all_decoded_labels = []
        all_pred_molecules = []
        all_gt_molecules = []
        all_simils = []

        with torch.no_grad():
            for batch in progress:     
                kwargs = self.generate_kwargs.copy()
                preds = model.generate(input_ids=batch["input_ids"].to(args.device),
                                            position_ids=batch["position_ids"].to(args.device),
                                            attention_mask=batch["attention_mask"].to(args.device),
                                            max_length=model.config.max_length,
                                            **kwargs)

                raw_preds_str = tokenizer.decode_batch(preds.tolist(), skip_special_tokens=False)
                preds_str = tokenizer.decode_batch(preds.tolist(), skip_special_tokens=True)
                gts_str = [tokenizer.decode((label*mask).tolist()) for label, mask in zip(batch["labels"], batch["decoder_attention_mask"])]
                all_raw_preds.extend(raw_preds_str)
                all_preds.extend(preds_str)
                all_decoded_labels.extend(gts_str)

                 # compute SMILES simil
                gen_mols = [Chem.MolFromSmiles(preds_str) for smiles in preds_str]
                gt_mols = [Chem.MolFromSmiles(gt_smiles) for gt_smiles in gts_str]
                gen_fps = [Chem.RDKFingerprint(x) for x in gen_mols if x]
                gt_fps = [Chem.RDKFingerprint(x) for x in gt_mols if x]
                smiles_simils = [DataStructs.FingerprintSimilarity(gen, gt) for gen, gt in zip(gen_fps, gt_fps)]            
                all_simils.extend(smiles_simils)
                # create a mol for prediction if it's valid, otherwise None
                for mol in gen_mols:
                    try:
                        img = wandb.Image(Chem.Draw.MolToImage(mol))
                    except ValueError:
                        img = None
                    all_pred_molecules.append(img)
                    
                # create a mol for labels if it's valid, otherwise None (it should be, but things can happen)
                for mol in gt_mols:
                    try:
                        img = wandb.Image(Chem.Draw.MolToImage(mol))
                    except ValueError:
                        img = None
                    all_gt_molecules.append(img)

        df_log = pd.DataFrame({"gt_smiles": all_decoded_labels,
                               "predicted_smiles": all_preds,
                               "gt_molecule": all_gt_molecules,
                               "predicted_molecule": all_pred_molecules,
                               "cos_simil": all_simils
                               })
        if self.show_raw_preds:
            df_log["raw_predicted_smiles"] = all_raw_preds
        table = wandb.Table(dataframe=df_log)

        wandb.log({f"eval_tables/{self.log_prefix}/global_step_{state.global_step}": table})
        wandb.log({f"eval/{self.log_prefix}/example_similarity: {sum(all_simils)/len(all_simils)}"})
