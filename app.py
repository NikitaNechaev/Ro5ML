#!/usr/bin/env python3
"""
Flask web interface for predicting drug-likeness and biological activity of molecules.
"""

import sys
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from features import (
    calculate_lipinski_descriptors,
    calculate_ecfp4,
    calculate_maccs,
    lipinski_pass,
    featurize_dataframe
)
import joblib

app = Flask(__name__)

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
    df = pd.DataFrame({
        'smiles': [smiles],
        'Class': [0]  # Dummy value, will be ignored
    })
    
    # Featurize (this will add all the Lipinski, ECFP4, MACCS columns)
    features_df = featurize_dataframe(df, smiles_col='smiles')
    
    # Load feature names to ensure we have the same columns in the same order
    feature_names = load_feature_names()
    
    # Select only the features that were used in training, in the correct order
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
        pred = model.predict(X)
        proba = pred.astype(float)
    
    ml_probability = float(proba[0])
    
    # Step 5: Final decision (combine Lipinski and ML)
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    smiles = data.get('smiles', '')
    model_name = data.get('model', 'random_forest')
    
    if not smiles:
        return jsonify({'error': 'No SMILES provided'}), 400
    
    try:
        result = predict_activity(smiles, model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)