from typing import List
from matchms import Spectrum
from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm

# make a fction that preprocess one SPECTRUM
def preprocess_spectrum(s : Spectrum, tokenizer, seq_len=200, max_smiles_len=100, max_mz=500):
    """
    Preprocess one matchms.Spectrum according to BART_spektro preprocessing pipeline
    
    Parameters
    ----------
    
    s : matchms.Spectrum
        spectrum that has a corresponding SMILES saved in its metadata
    tokenizer : tokenizers.Tokenizer
        tokenizer used for tokenizing the mz values to fit input_ids
    seq_len : int
        maximal seq_len (num of peaks) - specified by the BART model architecture
    max_smiles_len : int
        max len of SMILES string representation
    max_mz : the highest allowed value of mz (input element) - specified by the tokenizer's vocabulary
    
    Returns
    -------
    mz : List[int]
        tokenized input to a BART spektro model
    intensities : List[int]
        logged and binned intensities, prepared as position_id for BART spektro model
    attention_mask : List[int]
        list of ones and zeros of len seq_len. Zeros mask pad tokens for the future model so it doesn't 
        incorporate it into the computations
    canon_smiles : str
        canonical SMILES representation of the spectra
    error_dict : Dict[str : bool]
        dict of flags detecting if spectrum has any of the five specified errors
    """
    # canonicalization
    canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(s.metadata["smiles"]),True)
    
    goes_out = 0 # flag for discarding the datapoint
    error_dict = {"long_smiles": 0,
                  "corrupted": 0,
                  "high_mz": 0,
                  "too_many_peaks": 0,
                  "no_smiles": 0}
    
    # no simles filtering        
    if len(canon_smiles) == 0:
        goes_out = 1
        error_dict["no_smiles"] = True
    # long simles filtering        
    if len(canon_smiles) > max_smiles_len:
        goes_out = 1
        error_dict["long_smiles"] = True
    # filter corrupted
    if not Chem.MolFromSmiles(canon_smiles):
        goes_out = 1
        error_dict["corrupted"] = True

        #     # destereothng ???
    
    # filter high MZ
    if s.peaks.mz[-1] > max_mz:
        goes_out = 1
        error_dict["high_mz"] = True
    # filter long spectra
    if len(s.peaks.mz) > seq_len:
        goes_out = 1
        error_dict["too_many_peaks"] = True

    if goes_out:
        return ([],[],[],[],error_dict)
    
    # creating attention mask
    pad_len = seq_len-len(s.peaks.mz)
    attention_mask = len(s.peaks.mz)*[1] + pad_len*[0]
    
    # creating MZs inputs
    mz = [round(x) for x in s.peaks.mz]
    pt = tokenizer.token_to_id("<pad>")
    mz += pad_len * [pt]
    
    # scaling intensities
    intensities = s.peaks.intensities/max(s.peaks.intensities)    
    
    # log bining the intensities
    log_base = np.log(1.7)
    log_shift = 9
    x = (np.log(intensities)/log_base).astype(int) + log_shift
    x = x * (x > 0)
    intensities = np.concatenate((x, [-1]*pad_len)).astype("int32")
    return (mz, intensities, attention_mask, canon_smiles, error_dict)

def preprocess_spectra(spectra: List[Spectrum], tokenizer, spectra_name="spectra"):
    """
    Preprocess a list of matchms.Spectrum according to BART_spektro preprocessing pipeline
    Catch errors, sort them into 5 categories and print a report
    
    Parameters
    ----------
    spectra : List[matchms.Spectrum] or Generator[matchms.Spectrum]
        list of spectra to be preprocessed
    tokenizer : tokenizers.Tokenizer
        tokenizer used for tokenizing the mz values in preprocess_spectrum fction
    spectra__name : str
        a name that is used in the resulting dataframe for the column with spectra
    
    Returns
    -------
    df_out : pd.DataFrame
        a dataframe we are able to feed into SpectroDataset and then to SpectroBart
    """
    
    mzs = []
    ints = []
    atts = []
    smiless = []
    clean_spectra = []
    
    # STATS
    no_smiles = 0
    long_smiles = 0
    corrupted = 0
    high_mz = 0
    too_many_peaks = 0
    
    num_spectra=0
    for d in tqdm(spectra): 
        (mz, i, a, cs, ed) = preprocess_spectrum(d, tokenizer)
        if not mz:
            long_smiles += ed["long_smiles"]
            corrupted += ed["corrupted"]
            high_mz += ed["high_mz"]
            too_many_peaks += ed["too_many_peaks"]
            no_smiles += ed["no_smiles"]
        else:
            mzs.append(mz)
            ints.append(i)
            atts.append(a)
            smiless.append(cs)
            clean_spectra.append(d) #########
        num_spectra += 1 # just a counter
           
            
    df_out = pd.DataFrame({"input_ids": mzs, 
                           "position_ids": ints,
                           "attention_mask": atts,
                           "smiles": smiless,
                           f"{spectra_name}": clean_spectra})
    
    # print STATS
    print(f"{no_smiles} no smiles")
    print(f"{long_smiles} smiles too long")
    print(f"{corrupted} spectra corrupted")
    print(f"{high_mz} spectra w/ too high mz")
    print(f"{too_many_peaks} spectra w/ too many peaks")
    print(f"totally {long_smiles + corrupted + high_mz + too_many_peaks} issues")
    print(f"discarded {num_spectra - len(df_out)}/{num_spectra} spectra ")
    return df_out


def process_neims_spec(spec, metadata):
    """
    take a sdf spectrum as it comes out of NEIMS and create a matchms.Spectrum
    
    Parameters
    ----------
    spec : str
        string representation of spectra as it comes out of the NEIMS model
    metadata : dict
        a dict that will be attached as metadata to the spectrum
    
    Return
    ------
    matchms.Spectrum
        matchms representation of the input NEIMS spectra that contains the metadata 
        specified in the corresponding parameter
    """
    spec = spec.split("\n")
    i = []
    mz = []
    for t in spec:
        j,d = t.split()
        mz.append(j)
        i.append(d)
    return Spectrum(mz=np.array(mz).astype(np.float),
                    intensities=np.array(i).astype(np.float),
                    metadata=metadata,
                    metadata_harmonization=False)