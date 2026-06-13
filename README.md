# Ro5ML - Drug Candidate Predictor

A machine learning-based system for predicting drug-likeness and biological activity of molecules from SMILES strings. The system applies Lipinski's Rule of Five as a primary filter and uses machine learning models (Random Forest, Gradient Boosting, Neural Network) to predict biological activity.

## Project Structure

```
Ro5ML/
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed data and features
├── models/                   # Trained models and metrics
├── notebooks/                # Jupyter notebooks for EDA and training
├── results/                  # Results and reports
├── src/                      # Source code
│   ├── features.py           # Feature calculation (Lipinski, ECFP4, MACCS)
│   ├── preprocess.py         # Data loading and preprocessing
│   ├── prepare_data.py       # Data splitting and feature preparation
│   └── train_models.py       # Model training and evaluation
├── templates/                # HTML templates for web interface
├── app.py                    # Flask web application
├── run.py                    # Command-line interface
├── requirements.txt          # Dependencies
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Ro5ML.git
   cd Ro5ML
   ```

2. Create a conda environment (recommended) or virtual environment:
   ```bash
   # Using conda
   conda create -n drugpred python=3.9
   conda activate drugpred
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: Installing RDKit might require additional system dependencies. 
   For conda users, you can install it via:
   ```bash
   conda install -c rdkit rdkit
   ```

## Usage

### Command-Line Interface (CLI)

The CLI allows quick prediction from the terminal.

```bash
python run.py --smiles "CCO" --model random_forest
```

Options:
- `--smiles`: SMILES string of the molecule (required)
- `--model`: Model to use (`random_forest`, `gradient_boosting`, `neural_network`) (default: random_forest)
- `--threshold`: Probability threshold for positive class (default: 0.5)

Example output:
```
=== Drug Candidate Prediction ===
SMILES: CCO
Lipinski Pass: True
Lipinski Descriptors:
  MW: 46.07
  logP: -0.31
  HBD: 1
  HBA: 1
  NumRotatableBonds: 0
ML Probability (Active): 0.123
Is Drug Candidate: False
Reason: ML prediction inactive
========================================
```

### Web Interface

Start the Flask web application:
```bash
python app.py
```

Then navigate to `http://localhost:5000` in your web browser.

The web interface provides:
- Input field for SMILES string
- Dropdown to select the model
- Visualization of Lipinski descriptors
- Prediction result with color-coded output (green for candidate, red for non-candidate)

## Data

The project uses the BACE dataset (from Kaggle or similar sources) for training and evaluation. 
The dataset contains molecules with binary activity labels (active/inactive against BACE-1).

Data processing steps:
1. Load raw SMILES and activity labels
2. Standardize SMILES (canonicalization)
3. Remove invalid SMILES and duplicates
4. Calculate molecular descriptors (Lipinski, ECFP4, MACCS)
5. Apply Lipinski rule as a filter
6. Split data into training and test sets (stratified split)

Processed data is stored in `data/processed/`.

## Models

Three models are trained and evaluated:
- Random Forest
- Gradient Boosting
- Multi-layer Perceptron (Neural Network)

Models are saved in the `models/` directory as pickle files.
Model performance metrics are stored in `models/metrics.json`.

### Model Training

To retrain the models, run:
```bash
python src/train_models.py
```

This will:
1. Load processed features from `data/processed/bace_features.csv`
2. Split data into train/test sets
3. Train each model with default hyperparameters
4. Evaluate models and save them to `models/`
5. Save metrics to `models/metrics.json`

## Results

Model performance metrics (from the BACE dataset):

| Model               | ROC-AUC | Accuracy | F1-score | Precision | Recall |
|---------------------|---------|----------|----------|-----------|--------|
| Random Forest       | 0.886   | 0.795    | 0.777    | 0.771     | 0.783  |
| Gradient Boosting   | 0.881   | 0.795    | 0.774    | 0.779     | 0.768  |
| Neural Network      | 0.855   | 0.736    | 0.755    | 0.654     | 0.891  |

Based on these results, Random Forest shows the best balance of metrics and is selected as the default model.

## Notebooks

Jupyter notebooks in the `notebooks/` directory provide:
- Exploratory Data Analysis (EDA)
- Feature engineering visualization
- Model training and hyperparameter tuning
- Evaluation plots (ROC curves, precision-recall, confusion matrices)

To run the notebooks:
```bash
jupyter notebook
```
