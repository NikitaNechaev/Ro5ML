import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

def load_split_data(processed_dir="data/processed"):
    """
    Load the split data and feature names.
    """
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"), allow_pickle=True)
    feature_names = np.load(os.path.join(processed_dir, "feature_names.npy"), allow_pickle=True)
    return X_train, X_test, y_train, y_test, feature_names

def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest classifier.
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        **kwargs
    )
    rf.fit(X_train, y_train)
    return rf

def train_gradient_boosting(X_train, y_train, **kwargs):
    """
    Train a Gradient Boosting classifier.
    """
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        **kwargs
    )
    gb.fit(X_train, y_train)
    return gb

def train_neural_network(X_train, y_train, **kwargs):
    """
    Train a simple Neural Network (MLP) classifier.
    Note: For better performance, we might want to scale the data, but we'll leave that to the user.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=42,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        **kwargs
    )
    mlp.fit(X_train, y_train)
    return mlp

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the model and print metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print()
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_split_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))
    print("Saved Random Forest model.\n")
    
    # Train Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    joblib.dump(gb_model, os.path.join(models_dir, "gradient_boosting.pkl"))
    print("Saved Gradient Boosting model.\n")
    
    # Train Neural Network
    print("Training Neural Network (MLP)...")
    mlp_model = train_neural_network(X_train, y_train)
    mlp_metrics = evaluate_model(mlp_model, X_test, y_test, "Neural Network")
    joblib.dump(mlp_model, os.path.join(models_dir, "neural_network.pkl"))
    print("Saved Neural Network model.\n")
    
    # Optionally, we can save the metrics to a file
    import json
    metrics = {
        'random_forest': rf_metrics,
        'gradient_boosting': gb_metrics,
        'neural_network': mlp_metrics
    }
    with open(os.path.join(models_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to models/metrics.json")

if __name__ == "__main__":
    main()