from matchms.importing import load_from_msp
from matchms.exporting import save_as_msp
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from tqdm import tqdm
import cirpy
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Function to resolve compound name to SMILES
def name_to_smiles(name):
    try:
        return cirpy.resolve(name, 'smiles')
    except:
        return None

# Function to convert SMILES to molecular formula
def smiles_to_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        return rdMolDescriptors.CalcMolFormula(mol)

# Function to process a single spectrum
def process_spectrum(s):
    s_new = s.clone()

    smiles = name_to_smiles(s.metadata['compound_name'])
    if smiles is None:
        return None

    new_formula = smiles_to_formula(smiles)
    old_formula = s.metadata.get("formula", None)
    if old_formula is None or new_formula is None or old_formula != new_formula:
        return None

    s_new.metadata['smiles'] = smiles
    return s_new

# Function to run the multiprocessing with timeout using ProcessPoolExecutor
def parallel_process_spectrums(spectrums, num_workers=4, timeout=10):
    cleaned_spectrums = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_spectrum, s): s for s in spectrums}

        for future in tqdm(as_completed(futures)):
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    cleaned_spectrums.append(result)
            except TimeoutError:
                continue  # Skip if the task exceeded the timeout
            except Exception as e:
                continue  # Handle other exceptions, e.g., log them

    return cleaned_spectrums

if __name__ == '__main__':
    original_path = "datasets/WILEY/WILEY12_1.msp"
    original_spectrums = load_from_msp(original_path, metadata_harmonization=False)
    cleaned_spectrums = parallel_process_spectrums(original_spectrums, num_workers=32)

    save_as_msp(cleaned_spectrums, "datasets/WILEY/WILEY12_1_cleaned_adam2.msp")

