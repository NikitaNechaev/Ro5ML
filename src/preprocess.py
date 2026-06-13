import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import numpy as np

def load_and_preprocess(filepath):
    """
    Load the BACE dataset, clean SMILES, remove duplicates, and filter invalid molecules.
    """
    # Load the data
    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")
    
    # Standardize SMILES: convert to RDKit molecule and back to SMILES to get canonical form
    def standardize_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    
    df['standardized_smiles'] = df['mol'].apply(standardize_smiles)
    
    # Drop rows where standardization failed (invalid SMILES)
    initial_count = len(df)
    df = df.dropna(subset=['standardized_smiles'])
    print(f"Dropped {initial_count - len(df)} invalid SMILES.")
    
    # Remove duplicates based on standardized SMILES
    initial_count = len(df)
    df = df.drop_duplicates(subset=['standardized_smiles'])
    print(f"Dropped {initial_count - len(df)} duplicate SMILES.")
    
    # Optionally, we can also filter by Lipinski rules later, but for now we keep all for feature calculation.
    # We'll keep the original columns and add the standardized SMILES.
    # For modeling, we might want to use the standardized SMILES as the input.
    # Let's rename the column for clarity and keep the original 'mol' as well? We'll use standardized for features.
    # We'll create a new DataFrame with the standardized SMILES and the activity class.
    # The BACE dataset has a 'Class' column (1 for active, 0 for inactive) and also pIC50.
    # We'll use the binary class for classification.
    
    # Select relevant columns: standardized_smiles and Class (and maybe pIC50 for regression, but we'll do classification)
    processed_df = df[['standardized_smiles', 'Class']].copy()
    processed_df.rename(columns={'standardized_smiles': 'smiles'}, inplace=True)
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Active compounds: {processed_df['Class'].sum()}")
    print(f"Inactive compounds: {len(processed_df) - processed_df['Class'].sum()}")
    
    return processed_df

if __name__ == "__main__":
    input_file = "data/raw/bace.csv"
    output_file = "data/processed/bace_processed.csv"
    
    processed_df = load_and_preprocess(input_file)
    
    # Save the processed data
    processed_df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")