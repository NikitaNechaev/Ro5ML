#!/usr/bin/env python3
"""
Command-line interface for predicting drug-likeness and biological activity of molecules.
Usage:
    python run.py --smiles "CCO"
    python run.py --smiles "CCO" --model random_forest
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from features import (
    calculate_lipinski_descriptors,
    calculate_ecfp4,
    calculate_maccs,
    lipinski_pass,
    featurize_dataframe
)

def load_model(model_name='random_forest'):
    """Load the trained model."""
    model_path = os.path.join('models', f'{model_name}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_feature_names():
    """Load the feature names used during training."""
    feature_names_path = os.path.join('data', 'processed', 'feature_names.npy')
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    return np.load(feature_names_path, allow_pickle=True)

def predict_activity(smiles, model_name='random_forest'):
    """
    Predict if a molecule is a potential drug candidate.
    Returns a dictionary with Lipinski pass, ML probability, and final decision.
    """
    # Step 1: Lipinski filter (primary)
    lipinski_desc = calculate_lipinski_descriptors(smiles)
    if lipinski_desc is None:
        return {
            'smiles': smiles,
            'lipinski_pass': False,
            'lipinski_descriptors': None,
            'ml_probability': None,
            'is_candidate': False,
            'reason': 'Invalid SMILES'
        }
    
    passes_lipinski = lipinski_pass(lipinski_desc)
    
    # If fails Lipinski, we can stop early (primary filter)
    if not passes_lipinski:
        return {
            'smiles': smiles,
            'lipinski_pass': False,
            'lipinski_descriptors': lipinski_desc,
            'ml_probability': None,
            'is_candidate': False,
            'reason': 'Failed Lipinski Rule of Five'
        }
    
    # Step 2: Calculate additional features (ECFP4, MACCS)
    ecfp = calculate_ecfp4(smiles)
    maccs = calculate_maccs(smiles)
    
    if ecfp is None or maccs is None:
        return {
            'smiles': smiles,
            'lipinski_pass': True,
            'lipinski_descriptors': lipinski_desc,
            'ml_probability': None,
            'is_candidate': False,
            'reason': 'Failed to compute molecular features'
        }
    
    # Step 3: Create feature vector in the same order as training
    # We'll create a DataFrame with one row and then use the featurize function
    # to ensure consistency with training feature engineering.
    df = pd.DataFrame({
        'smiles': [smiles],
        'Class': [0]  # Dummy value, will be ignored
    })
    
    # Featurize (this will add all the Lipinski, ECFP4, MACCS columns)
    features_df = featurize_dataframe(df, smiles_col='smiles')
    
    # Load feature names to ensure we have the same columns in the same order
    feature_names = load_feature_names()
    
    # Select only the features that were used in training, in the correct order
    # We expect features_df to have all the required columns plus possibly extra ones
    # We'll take the intersection and order by feature_names
    available_features = [f for f in feature_names if f in features_df.columns]
    if len(available_features) != len(feature_names):
        missing = set(feature_names) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    
    X = features_df[available_features].values
    
    # Step 4: Load model and predict
    model = load_model(model_name)
    # Get probability of positive class (active)
    try:
        proba = model.predict_proba(X)[:, 1]
    except AttributeError:
        # If model doesn't have predict_proba, use decision function or predict
        # For simplicity, we'll use predict and convert to probability-like score
        pred = model.predict(X)
        proba = pred.astype(float)  # This is not ideal but works for binary classification
    
    ml_probability = float(proba[0])
    
    # Step 5: Final decision (combine Lipinski and ML)
    # We consider it a candidate if it passes Lipinski AND ML probability > threshold
    threshold = 0.5
    is_candidate = passes_lipinski and (ml_probability >= threshold)
    
    return {
        'smiles': smiles,
        'lipinski_pass': passes_lipinski,
        'lipinski_descriptors': lipinski_desc,
        'ml_probability': ml_probability,
        'is_candidate': is_candidate,
        'reason': 'Passed Lipinski and ML prediction active' if is_candidate else 
                  ('Failed Lipinski' if not passes_lipinski else 'ML prediction inactive')
    }

def main():
    parser = argparse.ArgumentParser(description='Predict drug candidate potential from SMILES')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string of the molecule')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'neural_network'],
                        help='Model to use for prediction (default: random_forest)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for positive class (default: 0.5)')
    
    args = parser.parse_args()
    
    result = predict_activity(args.smiles, args.model)
    
    # Print results in a user-friendly format
    print("\n=== Drug Candidate Prediction ===")
    print(f"SMILES: {result['smiles']}")
    print(f"Lipinski Pass: {result['lipinski_pass']}")
    if result['lipinski_descriptors']:
        print("Lipinski Descriptors:")
        for key, value in result['lipinski_descriptors'].items():
            print(f"  {key}: {value:.2f}")
    print(f"ML Probability (Active): {result['ml_probability']:.3f}" if result['ml_probability'] is not None else "ML Probability: N/A")
    print(f"Is Drug Candidate: {result['is_candidate']}")
    print(f"Reason: {result['reason']}")
    print("=" * 40)

if __name__ == '__main__':
    main()