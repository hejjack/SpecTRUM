# imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from numbers import Number
from rdkit import Chem, DataStructs
from tokenizers import Tokenizer
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_range_opt2_prob(model, model_name, tokenizer, data, data_type, data_range, additional_info="", figure_save_dir="figures",  printing=True, gt_list=[], num_generated=100, top_k=None, top_p=0.8, do_sample=True, num_beams=1, temperature=None, device="cuda"):
    """
    Evaluates model on a given range of given dataset. For every datapoint it sorts 
    generated SMILES according to the probability of each SMILES to be generated. 
    The quality of the sorting is expressed by RMSE with the ground truth sorting.
    
    The outputed graph (saved at particular location) contains several statistics.
    
    Parameters
    ----------
    model : BartSpektoForConditionalGeneration
        model to be evaluated
    model_name : str
        name used as an identifier of given model in titles and figure file name
    tokeznizer : tokenizers.Tokenizer
        tokenizer to use for decoding the ground truth smiles
    data : SpectroDataset
        dataset from which we take a slice according to data_range
    data_type : str
        name used as an identifier of given dataset in titles and figure file name
    data_range : range
    additional_info : stringgure (gets to file name and figure title)
        additional info to the fi
    figure_save_dir : str
        directory to use for saving stat figure
    printing : bool
        whether or not print the final statistics 
    gt_list : List[str]
        a parameter to help eval_mode - a list of ground truth SMILES for a reference
    num_generated : int
        number of generated SMILES for each datapoint
    top_k : int
        parametr of Hugging face generate function
    top_p : float
        parametr of Hugging face generate function
    do_sample : bool
        parametr of Hugging face generate function
    num_beams : int
        parametr of Hugging face generate function
    temperature : float
        parametr of Hugging face generate function
    device : str
        cuda or cpu
    
    Returns
    -------
    histplot : matplotlib.figure.Figure
        statistics about the evaluation, the histogram shows counts of best generated spectra with particular similarities
    """
    all_datapoint_ids = []
    all_smiles = []
    all_sequence_probs = []
    min_range = min(data_range)
    gt_smiless = []  
    
    
    for id_ in tqdm(data_range):
        inputs = data[id_]
        if gt_list:
            gt_smiles = gt_list[id_ - min_range]
        else:
            gt_smiles = tokenizer.decode(list(np.array(inputs["labels"].tolist())*np.array(inputs["decoder_attention_mask"].tolist()))[1:-1])
        
        gt_smiless.append(gt_smiles)
        generated_smiless = []

        # generate
        input_ids = inputs["input_ids"].unsqueeze(0).to(device=device)
        generated_outputs = model.generate(
                               input_ids=input_ids,
                               num_return_sequences = num_generated,
                               position_ids=inputs["position_ids"].unsqueeze(0).to(device=device),
                               attention_mask=inputs["attention_mask"].unsqueeze(0).to(device=device),
                               top_p=top_p,
                               top_k=top_k,
            #                    min_length=20,
            #                    max_length=200,
                               do_sample=do_sample,
                               num_beams=num_beams,
                               temperature=temperature,
                               return_dict_in_generate=True,
                               output_scores=True)

        # decode the generated SMILESs
        generated_smiless = [tokenizer.decode(generated[1:-1]) for generated in generated_outputs.sequences.tolist()]
        generated_smiless_enum = list(enumerate(generated_smiless)) # enum for analizing which smiles drops after validation
        
        # filter invalid
        valid_smiless = [s for s in generated_smiless_enum if Chem.MolFromSmiles(s[1])]
        valid_idxs = [s[0] for s in valid_smiless]
        
        # canonize
        canon_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi[1]),True) for smi in valid_smiless]
    
        ### PROBS computig
        # let's stack the logits generated at each step to a tensor and transform logits to probs
        raw_probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]

        # collect the probability of the generated token
        token_probs = torch.gather(raw_probs, 2, generated_outputs.sequences[:, 1:, None]).squeeze(-1)
        
        # filter out invalid smiles
        valid_token_probs = torch.index_select(token_probs, 0, torch.tensor(valid_idxs, dtype=torch.int64).to(device))

        # normalize the probs ?or not?
#         valid_token_probs = valid_token_probs / valid_token_probs.sum(0)
#         assert valid_token_probs[:, 0].sum() == 1.0, "probs should be normalized"
        
        # multiply all the probs in each sequence except for the zero values
        valid_token_probs[valid_token_probs==0] = 1 # replace zeros for neutral element -> 1

        sequence_probs = valid_token_probs.prod(dim=1).tolist()
        
        all_smiles += canon_smiles
        all_datapoint_ids += [id_]*len(canon_smiles)
        all_sequence_probs += sequence_probs
    
    df_all_data = pd.DataFrame({"datapoint_ID":all_datapoint_ids, 
                                "smiles": all_smiles,
                                "smiles_prob": all_sequence_probs})   
    
    rmses = []
    topNSmi = []
    top1sumSmi = 0
    top3sumSmi = 0
    all_top1Smi = []
    
    diff_unique_vals = 0 # check how many pairs of simil arrays have different num of unique vals
    success1 = 0
    success3 = 0
    unique_vals = 0
    unable_to_generate_anything = []
    
    
    for id_ in data_range:
        inputs = data[id_]
        gt_smiles = gt_smiless[id_ - min_range]
        df_id_subset = df_all_data[df_all_data['datapoint_ID'] == id_].copy()

        
        # compute SMILES simil
        ms = [Chem.MolFromSmiles(smiles) for smiles in df_id_subset["smiles"]]
        gt_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(gt_smiles)) 
        fps = [Chem.RDKFingerprint(x) for x in ms if x]
        smiles_simils = [DataStructs.FingerprintSimilarity(fp, gt_fp) for fp in fps]            
        
        # get computed probs from df
        smiles_probs = df_id_subset["smiles_prob"]

        # compare the two simil lists (RMSE)
        rmse, (idxs_prob, idxs_GT) = rmse_simils_unique(smiles_probs, smiles_simils)
        rmses.append(rmse)
        
        if idxs_prob.size > 0:
            ### STATS        
            if 0 == idxs_prob[0]:
                success1 += 1 
            if 0 in idxs_prob[:3]:
                success3 += 1
            unique_vals_curr = max(idxs_prob) + 1
            unique_vals += unique_vals_curr

            ###### STATS 2
            smiles_simils.sort(reverse=True)
            topNSmi = np.concatenate((topNSmi, smiles_simils))
            top1sumSmi += smiles_simils[0]
            top3sumSmi += sum(smiles_simils[:3])
            all_top1Smi.append(smiles_simils[0])
        else: 
            unable_to_generate_anything.append(gt_smiles)
        
    ############## OUTPUT ################
    topNaverageSmi = sum(topNSmi)/len(topNSmi)
    top1averageSmi = top1sumSmi/len(data_range)
    top3averageSmi = top3sumSmi/(len(data_range)*3)
    
    num_of_datapoints = max(data_range)-min(data_range)+1
    output_text = \
      f"model: {model_name}\n" + \
      f"additional info: {additional_info}\n" + \
      f"data range: ({min(data_range)}, {max(data_range)})\n"+ \
      f"generated for each example: {num_generated} samples\n"+\
      f"data type: {data_type}\n" +\
      f"mean RMSE of prob_simil ranking compared to GT: {sum(rmses)/len(rmses)}\n" +\
      f"recall@1: {success1/num_of_datapoints}\n"+\
      f"recall@3: {success3/num_of_datapoints}\n"+\
      f"average smiles simil from {'all unique'} samples: {topNaverageSmi}\n"+\
      f"average simles simil from the 3 best samples: {top3averageSmi}\n"+\
      f"average smiles simil from the 1 best sample: {top1averageSmi}\n"+\
      f"mean num of unique vals: {unique_vals/num_of_datapoints}\n"
    
    if printing:
        print(f"###### RESULTS ######\n" + output_text)
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    histplot = sns.histplot(all_top1Smi, bins=100).set_title(output_text)
    fig = histplot.get_figure()
    plt.xlabel("Best predictions (smiles similarity)")
    fig.savefig(f"{figure_save_dir}/option2_prob{additional_info}_{model_name}_{data_type}_generated{num_generated}_({min(data_range)}, {max(data_range)}).png", bbox_inches='tight') 
    return histplot


def oneD_spectra_to_mz_int(df : pd.DataFrame) -> pd.DataFrame:
    """
    Function that takes a DF and splits the one-array-representation of spectra into mz and intensity parts
    
    Parameters
    ----------
    df : pd.DataFrame
         dataframe containing 'PREDICTED SPECTRUM' column with sdf spectra representation
         -> is used after loading enriched sdf file with PandasTools.LoadSDF
    
    Returns
    -------
    df2 : pd.DataFrame
          dataframe containing columns 'mz' and 'intensity' that contain decomposed spectra representation, two arrays of the same length
    """
    df2 = df.copy()
    all_i = []
    all_mz = []
    for row in range(len(df2)):
        spec = df2["PREDICTED SPECTRUM"][row].split("\n")
        mz = []
        i = []
        spec_max = 0
        for t in spec:
            j,d = t.split()
            j,d = int(j), float(d)
            if spec_max < d:
                spec_max  = d
            mz.append(j)
            i.append(d)
        all_mz.append(mz)
        all_i.append(np.around(np.array(i)/spec_max, 2))
    new_df = pd.DataFrame.from_dict({"mz": all_mz, "intensity": all_i})
    df2 = pd.concat([df2, new_df], axis=1)
    df2 = df2.drop(["PREDICTED SPECTRUM"], axis=1)
    return df2


def unique_nonsorted(array : np.ndarray) -> np.ndarray:
    """
    Version of np.unique that doesn't change the order of the items
    """
    indexes = np.unique(array, return_index=True)[1]
    return np.array([array[index] for index in sorted(indexes)])


def rmse_simils(inspected_simils : List[Number], gt_simils : List[Number]):
    """
    (In the end this is not used, ather the unique version)
    Computes RMSE for indexes of two lists of values. It is used to compare how well are two sorted lists (e.g. of SMILES) aligned 
    -> we try to measure how well the experimental sorting fits the ground truth sorting    
    Outputs rmse and index lists sorted by gt_simils

    Parameters
    ----------
    inspected_simils : List[Number]
           list of sorted similarities (inspected similarity measure)
           has to be the same length as the gt_sorted_by_gt_indexes
    gt_simils : List[Number]
           list of sorted similarities (ground truth similarity measure)
           has to be the same length as the gt_sorted_by_gt_indexes
    Returns
    -------
    rmse : float
    is_sorted_by_gt_indexes : List[int]
        indexes of inspected similarities array sorted by the ground truth

    gt_sorted_by_gt_indexes : List[int]
        indexes of sorted ground truth values
    """    
    zipped_simils = sorted(list(zip(gt_simils, inspected_simils)), key=lambda x: x[0], reverse=True)
    is_sorted_by_gt = [x[1] for x in zipped_simils]
    is_sorted_by_is = sorted(inspected_simils, reverse=True)
    gt_sorted_by_gt = [x[0] for x in zipped_simils]
        
    # create indexes (each element is replaced by its order in unique list of elements)
    unique_is_sorted_is = list(np.unique(is_sorted_by_is))
    unique_is_sorted_is.reverse()
    is_sorted_by_gt_indexes = np.array([unique_is_sorted_is.index(x) for x in np.array(is_sorted_by_gt)])
    
    # the same for gt
    unique_gt_sorted_gt = list(np.unique(gt_sorted_by_gt))
    unique_gt_sorted_gt.reverse()
    gt_sorted_by_gt_indexes = np.array([unique_gt_sorted_gt.index(x) for x in np.array(gt_sorted_by_gt)]) 
    
    rmse = np.sqrt(np.sum((is_sorted_by_gt_indexes - gt_sorted_by_gt_indexes)**2)/len(is_sorted_by_gt_indexes))
#     print(is_sorted_by_gt_indexes, gt_sorted_by_gt_indexes)
        
    return rmse, (is_sorted_by_gt_indexes, gt_sorted_by_gt_indexes)


"""
Computes RMSE for indexes of two lists of values (makes them unique too).
"""
def rmse_simils_unique(inspected_simils, gt_simils):
    """
    Computes RMSE for indexes of two lists of values and makes it UNOQUE too.
    It is used to compare how well are two sorted lists (e.g. of SMILES) aligned 
    -> we try to measure how well the experimental sorting fits the ground truth sorting    
    Outputs rmse and index lists sorted by gt_simils

    Parameters
    ----------
    inspected_simils : List[Number]
           list of sorted similarities (inspected similarity measure)
           has to be the same length as the gt_sorted_by_gt_indexes
    gt_simils : List[Number]
           list of sorted similarities (ground truth similarity measure)
           has to be the same length as the gt_sorted_by_gt_indexes
    Returns
    -------
    rmse : float
    is_sorted_by_gt_indexes : List[int]
        indexes of inspected similarities array sorted by the ground truth

    gt_sorted_by_gt_indexes : List[int]
        indexes of sorted ground truth values
    """    
    zipped_simils = sorted(list(zip(gt_simils, inspected_simils)), key=lambda x: x[0], reverse=True)
    is_sorted_by_gt = [x[1] for x in zipped_simils]
    is_sorted_by_is = sorted(inspected_simils, reverse=True)
    gt_sorted_by_gt = [x[0] for x in zipped_simils]
            
    # create indexes (each element is replaced by its order in unique list of elements)
    unique_is_sorted_by_gt = unique_nonsorted(is_sorted_by_gt)
    unique_is_sorted_is = list(np.unique(is_sorted_by_is))
    unique_is_sorted_is.reverse()
    is_sorted_by_gt_indexes = np.array([unique_is_sorted_is.index(x) for x in unique_is_sorted_by_gt])
        
    # the same for gt
    num_of_gt_unique = len(np.unique(gt_sorted_by_gt))
    gt_sorted_by_gt_indexes = np.array(np.arange(num_of_gt_unique))
    
    # the addiitonal unique step (it could be WAY simpler if coded with this intention from the beginning)
    if len(gt_sorted_by_gt_indexes) < len(is_sorted_by_gt_indexes):
        is_sorted_by_gt_indexes = is_sorted_by_gt_indexes[:len(gt_sorted_by_gt_indexes)]
    if len(gt_sorted_by_gt_indexes) > len(is_sorted_by_gt_indexes):
        gt_sorted_by_gt_indexes = gt_sorted_by_gt_indexes[:len(is_sorted_by_gt_indexes)]
    
    rmse = np.sqrt(np.sum((is_sorted_by_gt_indexes - gt_sorted_by_gt_indexes)**2)/len(is_sorted_by_gt_indexes))
#     print(is_sorted_by_gt_indexes, gt_sorted_by_gt_indexes)
        
    return rmse, (is_sorted_by_gt_indexes, gt_sorted_by_gt_indexes)


def my_position_ids_restorer(log_intensities, log_base=np.log(1.7), log_shift=9):
    """
    Takes logged intensities and restores them to centers of the bins
    -> reverse process to the one done during preprocessing
    """
    intensities = np.around(np.exp((log_intensities - log_shift) * log_base), decimals=4)
    return intensities
