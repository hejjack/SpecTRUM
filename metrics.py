from __future__ import annotations

import transformers
from rdkit import Chem, DataStructs
import numpy as np
from icecream import ic
import selfies as sf
from bart_spektro.selfies_tokenizer import SelfiesTokenizer
from spectra_process_utils import get_fp_generator, get_simil_function

def compute_fp_simils(preds: list[str], 
                      trues: list[str],
                      fp_type: str = "daylight",
                      simil_function_type: str = "tanimoto",
                      fp_kwargs: dict = {},
                      input_mols: bool = False, 
                      return_mols: bool = False):
    """
    Compute the average cosine similarity between the predicted and true SMILES strings (or molecules)
    
    Parameters
    ----------
    preds : list[str | Chem.rdchem.Mol]
        List of predicted SMILES strings or RDKit molecules (if input mols flag is set)
    trues : list[str | Chem.rdchem.Mol]
        List of ground truth SMILES strings RDKit molecules (if input mols flag is set)
    fp_type : str
        The type of fingerprint to use
    simil_function_type : str
        The type of similarity function to use
    input_mols : bool
        Whether the inputs are lists of RDKit molecules or not
    return_mols : bool
        Whether to return the predicted and true molecules as well
    """

    assert len(preds) == len(trues)
    fpgen = get_fp_generator(fp_type, fp_kwargs)
    simil_function = get_simil_function(simil_function_type)
    simils = []
    pred_mols = []
    true_mols = []
    for pred, true in zip(preds, trues):
        if not input_mols:
            pred_mol = Chem.MolFromSmiles(pred)
            true_mol = Chem.MolFromSmiles(true)
        else:   
            pred_mol = pred
            true_mol = true
        if pred_mol is None or true_mol is None:
            simils.append(0.0)
            pred_mols.append(None)
            true_mols.append(None)
            continue
        pred_fp = fpgen.GetFingerprint(pred_mol)
        true_fp = fpgen.GetFingerprint(true_mol)
        simil = simil_function(pred_fp, true_fp)
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
        tokenizer: transformers.PreTrainedTokenizerFast | SelfiesTokenizer,
    ) -> None:
        self.tokenizer = tokenizer

    def __call__(self, eval_preds: transformers.EvalPrediction) -> dict[str, float]:
        preds_all = eval_preds.predictions
        trues_all = eval_preds.label_ids

        if isinstance(preds_all, tuple):
            preds_all = preds_all[0]


        pad_token_id = self.tokenizer.pad_token_id
        assert pad_token_id is not None
        preds_all = np.where(preds_all != -100, preds_all, pad_token_id)
        trues_all = np.where(trues_all != -100, trues_all, pad_token_id)

        preds_str_all = self.tokenizer.batch_decode(preds_all, skip_special_tokens=True)
        trues_str_all = self.tokenizer.batch_decode(trues_all, skip_special_tokens=True)

        if isinstance(self.tokenizer, SelfiesTokenizer):
            preds_str_all = [sf.decoder(x) for x in preds_str_all]
            trues_str_all = [sf.decoder(x) for x in trues_str_all]

        daylight_tanimoto_simils, pred_mols, gt_mols = compute_fp_simils(preds_str_all, trues_str_all, return_mols=True)
        morgan_tanimoto_simils = compute_fp_simils(pred_mols, gt_mols, fp_type="morgan", fp_kwargs={"radius": 2, "fpSize": 2048}, input_mols=True, return_mols=False)
        precise_morgan_tanimoto_hits = np.sum(np.array(morgan_tanimoto_simils) == 1) / len (morgan_tanimoto_simils)
        precise_daylight_tanimoto_hits = np.sum(np.array(daylight_tanimoto_simils) == 1) / len (daylight_tanimoto_simils)

        metrics = {}
        metrics["daylight_tanimoto_simil"] = np.mean(daylight_tanimoto_simils)
        metrics["morgan_tanimoto_simil"] = np.mean(morgan_tanimoto_simils)
        metrics["precise_morgan_tanimoto_hits"] = precise_morgan_tanimoto_hits
        metrics["precise_daylight_tanimoto_hits"] = precise_daylight_tanimoto_hits        

        return metrics