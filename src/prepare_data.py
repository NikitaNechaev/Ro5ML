import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def prepare_data(features_file, test_size=0.2, random_state=42, scale_features=False):
    """
    Load the features, separate features and target, and split into train and test sets.
    Optionally scale the features (using StandardScaler fit on training data).
    """
    # Load the data
    df = pd.read_csv(features_file)
    print(f"Loaded data shape: {df.shape}")
    
    # The target is the 'Class' column
    y = df['Class'].values
    
    # Features: we want to exclude the SMILES column and the target column.
    # Also exclude the Lipinski pass flag if we are going to use it as a feature? 
    # We'll include all Lipinski descriptors, ECFP4, MACCS, and the Lipinski pass flag as features.
    # But note: the Lipinski pass flag is derived from the descriptors, so there might be multicollinearity.
    # We'll include it anyway and let the model decide.
    
    # Drop the SMILES column and the Class column
    X = df.drop(columns=['smiles', 'Class'])
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Features have been standardized.")
        # We'll return the scaler as well for later use
        return X_train, X_test, y_train, y_test, scaler
    else:
        return X_train, X_test, y_train, y_test, None

if __name__ == "__main__":
    features_file = "data/processed/bace_features.csv"
    # We'll not scale for now because we are going to use tree-based models and neural networks might need scaling.
    # We'll handle scaling in the model scripts if needed.
    X_train, X_test, y_train, y_test, scaler = prepare_data(features_file, scale_features=False)
    
    # Save the split data
    processed_dir = "data/processed"
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    # Also save the feature names for later use
    feature_names = pd.read_csv(features_file).drop(columns=['smiles', 'Class']).columns
    np.save(os.path.join(processed_dir, "feature_names.npy"), feature_names)
    
    print("Saved split data to data/processed/")