from __future__ import annotations

import transformers
from tokenizers import Tokenizer
from rdkit import Chem, DataStructs
import numpy as np

class SpectroMetrics:

    def __init__(
        self,
        tokenizer: Tokenizer,
    ) -> None:

        self.tokenizer = tokenizer
    
    def compute_cos_simil(self, preds: list[str], trues: list[str]) -> float:
        """Compute the average cosine similarity between the predicted and true SMILES strings"""
        assert len(preds) == len(trues)
        simils = []
        for pred, true in zip(preds, trues):
            pred_mol = Chem.MolFromSmiles(pred)
            true_mol = Chem.MolFromSmiles(true)
            if pred_mol is None or true_mol is None:
                simils.append(0.0)
                continue
            pred_fp = Chem.RDKFingerprint(pred_mol)
            true_fp = Chem.RDKFingerprint(true_mol)
            simils.append(DataStructs.FingerprintSimilarity(pred_fp, true_fp))
        return np.mean(simils)

    def __call__(self, eval_preds: transformers.EvalPrediction) -> dict[str, float]:
        preds_all = eval_preds.predictions
        trues_all = eval_preds.label_ids

        if isinstance(preds_all, tuple):
            preds_all = preds_all[0]

        assert self.tokenizer.pad_token_id is not None
        pad_token_id = self.tokenizer.token_to_id("<pad>")
        preds_all = np.where(preds_all != -100, preds_all, pad_token_id)
        trues_all = np.where(trues_all != -100, trues_all, pad_token_id)

        preds_str_all = self.tokenizer.batch_decode(preds_all, skip_special_tokens=True)
        trues_str_all = self.tokenizer.batch_decode(trues_all, skip_special_tokens=True)

        metrics = {}
        metrics["cos_simil"] = compute_cos_simil(preds_str_all, trues_str_all)

        return metrics
