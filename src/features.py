import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np

def calculate_lipinski_descriptors(smiles):
    """
    Calculate Lipinski descriptors for a given SMILES string.
    Returns a dictionary of descriptors or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    
    return {
        'MW': mw,
        'logP': logp,
        'HBD': hbd,
        'HBA': hba,
        'NumRotatableBonds': rb
    }

def calculate_ecfp4(smiles, radius=2, nBits=1024):
    """
    Calculate ECFP4 fingerprint for a given SMILES string.
    Returns a numpy array of length nBits or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def calculate_maccs(smiles):
    """
    Calculate MACCS keys for a given SMILES string.
    Returns a numpy array of length 167 or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def lipinski_pass(descriptors):
    """
    Check if a molecule passes Lipinski's Rule of Five.
    descriptors: dict with keys 'MW', 'logP', 'HBD', 'HBA', 'NumRotatableBonds'
    Returns True if passes, False otherwise.
    """
    if descriptors is None:
        return False
    return (descriptors['MW'] <= 500 and
            descriptors['logP'] <= 5 and
            descriptors['HBD'] <= 5 and
            descriptors['HBA'] <= 10 and
            descriptors['NumRotatableBonds'] <= 10)

def featurize_dataframe(df, smiles_col='smiles'):
    """
    Add Lipinski descriptors, ECFP4, MACCS, and Lipinski pass flag to a DataFrame.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate Lipinski descriptors
    lipinski_list = []
    ecfp_list = []
    maccs_list = []
    pass_list = []
    
    for smiles in df[smiles_col]:
        desc = calculate_lipinski_descriptors(smiles)
        lipinski_list.append(desc)
        
        ecfp = calculate_ecfp4(smiles)
        ecfp_list.append(ecfp)
        
        maccs = calculate_maccs(smiles)
        maccs_list.append(maccs)
        
        pass_list.append(lipinski_pass(desc))
    
    # Convert lists to DataFrame columns
    lipinski_df = pd.DataFrame(lipinski_list, index=df.index)
    # Prefix the columns to avoid name clashes
    lipinski_df = lipinski_df.add_prefix('lipinski_')
    
    # For ECFP and MACCS, we'll create separate columns for each bit
    # But note: having 1024 + 167 columns might be too wide for some purposes.
    # Alternatively, we could store them as arrays in a column, but for ML we want separate features.
    # We'll create columns for each bit.
    ecfp_df = pd.DataFrame(ecfp_list, index=df.index)
    ecfp_df = ecfp_df.add_prefix('ecfp4_')
    
    maccs_df = pd.DataFrame(maccs_list, index=df.index)
    maccs_df = maccs_df.add_prefix('maccs_')
    
    # Lipinski pass flag
    pass_df = pd.DataFrame(pass_list, index=df.index, columns=['lipinski_pass'])
    
    # Concatenate all features
    features_df = pd.concat([df, lipinski_df, ecfp_df, maccs_df, pass_df], axis=1)
    
    return features_df

if __name__ == "__main__":
    # Load the preprocessed data
    input_file = "data/processed/bace_processed.csv"
    df = pd.read_csv(input_file)
    print(f"Loaded data shape: {df.shape}")
    
    # Featurize
    features_df = featurize_dataframe(df, smiles_col='smiles')
    print(f"Features data shape: {features_df.shape}")
    
    # Save the features
    output_file = "data/processed/bace_features.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Saved features to {output_file}")
    
    # Print some statistics
    print(f"Number of compounds passing Lipinski: {features_df['lipinski_pass'].sum()}")
    print(f"Number of compounds failing Lipinski: {len(features_df) - features_df['lipinski_pass'].sum()}")