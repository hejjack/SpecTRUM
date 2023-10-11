from __future__ import annotations

import transformers
from rdkit import Chem, DataStructs
import numpy as np
from icecream import ic
import selfies as sf
from bart_spektro.selfies_tokenizer import SelfiesTokenizer

def compute_cos_simils(preds: list[str], 
                       trues: list[str], 
                       return_mols=False) -> (tuple[list[float], 
                                                    list[Chem.rdchem.Mol], 
                                                    list[Chem.rdchem.Mol]] | 
                                              list[float]):
    """
    Compute the average cosine similarity between the predicted and true SMILES strings
    
    Parameters
    ----------
    preds : list[str]
        List of predicted SMILES strings
    trues : list[str]
        List of ground truth SMILES strings
    return_mols : bool
        Whether to return the predicted and true molecules as well
    """

    assert len(preds) == len(trues)
    simils = []
    pred_mols = []
    true_mols = []
    for pred, true in zip(preds, trues):
        pred_mol = Chem.MolFromSmiles(pred)
        true_mol = Chem.MolFromSmiles(true)
        if pred_mol is None or true_mol is None:
            simils.append(0.0)
            pred_mols.append(None)
            true_mols.append(None)
            continue
        pred_fp = Chem.RDKFingerprint(pred_mol)
        true_fp = Chem.RDKFingerprint(true_mol)
        simil = DataStructs.FingerprintSimilarity(pred_fp, true_fp)
        simils.append(simil)
        pred_mols.append(pred_mol)
        true_mols.append(true_mol)
        # print("pred: " + pred, "true: " + true, "simil: " + str(simil), sep="\n")    
    
    if return_mols:
        return simils, pred_mols, true_mols
    else:
        return simils


class SpectroMetrics:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerFast,
    ) -> None:

        self.tokenizer = tokenizer

        if isinstance(tokenizer, SelfiesTokenizer):
            self.mol_repr = "selfies"
            assert sf.get_semantic_constraints()["I"] == 5, "Selfies tokenizer constraints are not set properly!"
        else: 
            self.mol_repr = "smiles"


    def __call__(self, eval_preds: transformers.EvalPrediction) -> dict[str, float]:
        preds_all = eval_preds.predictions
        trues_all = eval_preds.label_ids

        if isinstance(preds_all, tuple):
            preds_all = preds_all[0]

        pad_token_id = self.tokenizer.pad_token_id
        preds_all = np.where(preds_all != -100, preds_all, pad_token_id)
        trues_all = np.where(trues_all != -100, trues_all, pad_token_id)

        preds_str_all = self.tokenizer.batch_decode(preds_all, skip_special_tokens=True)
        trues_str_all = self.tokenizer.batch_decode(trues_all, skip_special_tokens=True)

        if self.mol_repr == "selfies":
            preds_str_all = [sf.decoder(pred) for pred in preds_str_all]
            trues_str_all = [sf.decoder(true) for true in trues_str_all]

        metrics = {}
        metrics["cos_simil"] = np.mean(compute_cos_simils(preds_str_all, trues_str_all))

        return metrics
