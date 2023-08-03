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

from dataset import SpectroDataset


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

        if eval_subset_size is not None and \
           eval_log_predictions_size is not None and \
           eval_subset_size < eval_log_predictions_size:
            raise ValueError("eval_subset_size must be greater than or equal to eval_log_predictions_size")
        
        super().__init__(*args, **kwargs)
        self.generate_kwargs = generate_kwargs
        self.eval_subset_size = eval_subset_size
        self.eval_log_predictions_size = eval_log_predictions_size
        self.eval_tokenizer = eval_tokenizer
        self.w_run = w_run

    @torch.no_grad()
    def eval_log_predictions(self, eval_dataloader: DataLoader):
        """
        Replacement for a callback (callback wouldn't let us take the logs from the sampled eval set)
        Logs predicted SMILES and visualized molecules to wandb.Table.
        """

        # model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        self.model.eval()
        tokenizer = self.eval_tokenizer
        all_preds = []
        all_decoded_labels = []
        all_pred_molecules = []
        all_gt_molecules = []
        for batch in eval_dataloader:
            kwargs = self.generate_kwargs.copy()
            preds = self.model.generate(input_ids=batch["input_ids"].to(self.args.device),
                                        position_ids=batch["position_ids"].to(self.args.device),
                                        attention_mask=batch["attention_mask"].to(self.args.device),
                                        max_length=self.model.config.max_length,
                                        **kwargs)
            preds_str = tokenizer.decode_batch(preds.tolist(), skip_special_tokens=True)
            gts_str = [tokenizer.decode((label*mask).tolist()) for label, mask in zip(batch["labels"], batch["decoder_attention_mask"])]
            all_preds.extend(preds_str)
            all_decoded_labels.extend(gts_str)
            
            # create a mol for prediction if it's valid, otherwise None
            for smiles in preds_str:
                try:
                    mol = wandb.Image(Chem.Draw.MolToImage(Chem.MolFromSmiles(smiles)))
                except ValueError:
                    mol = None
                all_pred_molecules.append(mol)
                
            # create a mol for labels if it's valid, otherwise None (it should be, but things can happen)
            for smiles in gts_str:
                try:
                    mol = wandb.Image(Chem.Draw.MolToImage(Chem.MolFromSmiles(smiles)))
                except ValueError:
                    mol = None
                all_gt_molecules.append(mol)

        df_log = pd.DataFrame({"gt_smiles": all_decoded_labels,
                               "predicted_smiles": all_preds,
                               "gt_molecule": all_gt_molecules,
                               "predicted_molecule": all_pred_molecules
                               })
        table = wandb.Table(dataframe=df_log)

        # self.w_run.log({"eval_predictions": table}, step=self.state.global_step)
        self.log({f"eval_tables/predictions_global_step_{self.state.global_step}": table})

        # remove the last log, which is the one with the table 
        # (dirty hack, table prevents trainer from saving)
        self.state.log_history.pop(-1) 

    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval",
                 ) -> Dict[str, float]:

        """
        Customization of TRAINER EVALUATE method for BartSpectro model.

        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # make a random slice of the eval dataset - ADAM
        if self.eval_dataset is not None and self.eval_subset_size is not None:
            seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            eval_pd_slice = self.eval_dataset.data.sample(self.eval_subset_size, random_state=seed, axis=0)
            eval_dataset = SpectroDataset(eval_pd_slice)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)

        # log predictions to wandb - ADAM
        if self.eval_log_predictions_size is not None and eval_dataset is not None:
            if  len(eval_dataset) < self.eval_log_predictions_size:
                print(f"WARNING: eval_log_predictions_size ({self.eval_log_predictions_size}) is greater \
                       than eval_dataset size ({len(eval_dataset)}). Not logging predictions.")

            log_loader = self.get_eval_dataloader(SpectroDataset(eval_dataset.data.iloc[:self.eval_log_predictions_size]))
            self.eval_log_predictions(log_loader)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
