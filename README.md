# Drug-Target Interaction Prediction for HIV-1 Protease

This project is a real-data, beginner-friendly bioinformatics pipeline for predicting whether small molecules are active or inactive against HIV-1 protease.

It uses:

- ChEMBL bioactivity data
- RDKit Morgan fingerprints
- scikit-learn classification models

The code is designed to stay practical on an older laptop while still looking like a real cheminformatics workflow.

## Project Goal

Train machine learning models to classify compounds as:

- active against HIV-1 protease
- inactive against HIV-1 protease

using molecular structure information derived from SMILES strings.

## Main Script

- [drug_interaction_ml](c:\Users\Kuldeep Singh\.vscode\extensions\drug_interaction_ml)

Run it from the workspace root with:

```bash
python extensions\drug_interaction_ml
```

## Current Outputs

The pipeline currently writes these files:

- [hiv_protease_chembl_raw.csv](c:\Users\Kuldeep Singh\.vscode\extensions\hiv_protease_chembl_raw.csv): cached raw ChEMBL activity table
- [hiv_protease_smiles_v2.csv](c:\Users\Kuldeep Singh\.vscode\extensions\hiv_protease_smiles_v2.csv): cached molecule-to-SMILES mapping
- [hiv_protease_chembl_dataset_v2.csv](c:\Users\Kuldeep Singh\.vscode\extensions\hiv_protease_chembl_dataset_v2.csv): cleaned curated dataset
- [hiv_protease_train_v2.csv](c:\Users\Kuldeep Singh\.vscode\extensions\hiv_protease_train_v2.csv): train split
- [hiv_protease_test_v2.csv](c:\Users\Kuldeep Singh\.vscode\extensions\hiv_protease_test_v2.csv): test split
- [roc_curve.png](c:\Users\Kuldeep Singh\.vscode\extensions\roc_curve.png): ROC curve plot
- [feature_importance.png](c:\Users\Kuldeep Singh\.vscode\extensions\feature_importance.png): feature importance plot for the best compatible tree model

## Dataset

Source:

- ChEMBL target `CHEMBL243`
- activity types: `IC50`, `Ki`

The script builds a binary classification dataset from real HIV-1 protease measurements.

### Current Dataset Size

From the latest cached run:

- Total labeled molecules: `3913`
- Positive samples: `3231`
- Negative samples: `682`
- Train samples: `3130`
- Test samples: `783`

## Data Processing Pipeline

The script performs these steps:

1. Load cached raw ChEMBL activity data or download it if missing.
2. Keep exact-value measurements only.
3. Convert activity values into pChEMBL-style potency values.
4. Remove noisy, duplicate, or contradictory records where possible.
5. Fetch and cache SMILES strings.
6. Standardize molecules with RDKit before deduplication.
7. Aggregate repeated measurements into one row per standardized molecule.
8. Split the cleaned dataset into train and test sets using stratified sampling.
9. Generate Morgan fingerprints with:
   radius `2`
   `512` bits

### Labeling Rule

- Active: median pChEMBL `>= 7.0`
- Inactive: median pChEMBL `<= 6.0`
- Compounds in the middle range are excluded

### Additional Cleaning

- Exact measurement rows only
- Molecule standardization before deduplication
- Maximum allowed pChEMBL spread: `2.0`

## Models Used

The current script trains:

- Logistic Regression
- Random Forest
- Extra Trees
- HistGradientBoosting

These are all classical models that work well with fingerprint features and remain much lighter than deep learning.

## Latest Results

From the latest successful run in this workspace:

- Logistic Regression: Accuracy `0.866`, ROC-AUC `0.902`
- Random Forest: Accuracy `0.794`, ROC-AUC `0.920`
- Extra Trees: Accuracy `0.898`, ROC-AUC `0.945`
- HistGradientBoosting: Accuracy `0.891`, ROC-AUC `0.935`

At the moment, `Extra Trees` is the strongest overall model in this project.

## GPU Note

The script now detects your AMD GPU correctly.

Detected adapter:

- `AMD Radeon R7 M440`

However, the current pipeline still trains on CPU in this environment.

Reason:

- RDKit fingerprint generation is CPU-based here
- scikit-learn models are CPU-based
- no AMD-compatible ML acceleration backend is installed in this environment

The code includes logic for optional GPU acceleration where supported, but for actual AMD GPU training you would need an additional backend such as a compatible GPU-enabled `lightgbm` build.

## Requirements

Python packages used by the project:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `rdkit`

## Installation

Install the core packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Install RDKit:

Recommended:

```bash
conda install -c conda-forge rdkit
```

Alternative:

```bash
pip install rdkit
```

## Why This Project Is Better Than The Earlier Version

- Uses real ChEMBL data instead of synthetic labels
- Uses a larger cleaned dataset
- Has a clear train/test split
- Applies more realistic cheminformatics preprocessing
- Detects available GPU hardware more honestly
- Compares multiple classical ML algorithms instead of just two simple baselines

## Notes

- First run can be slower because ChEMBL and SMILES data may need to be cached.
- Later runs are faster because the CSV caches are reused.
- The dataset is imbalanced, with many more active than inactive compounds.
- Results can shift slightly if the cached dataset changes or the cleaning rules are updated.

## Reference

- ChEMBL: https://www.ebi.ac.uk/chembl/
- Related HIV protease paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6288788/
