# Drug-Target Interaction Prediction for HIV-1 Protease

An end-to-end cheminformatics / machine-learning pipeline that predicts small-molecule activity against **HIV-1 protease** using real bioactivity data from ChEMBL.

The project covers:

- **Classification** – active vs inactive compound prediction
- **Regression** – continuous pChEMBL potency prediction
- Advanced analyses: applicability-domain scoring, activity-cliff detection, temporal validation

## Project Goal

Train and evaluate ML models that:

1. **Classify** compounds as active or inactive against HIV-1 protease
2. **Predict** continuous potency (pChEMBL values) for drug candidates

using molecular structure information derived from SMILES strings.

---

## Quick Start

### Prerequisites

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | ML models & evaluation |
| `rdkit` | Molecular fingerprints & descriptors |
| `matplotlib` | Static plots |
| `plotly` | Interactive plots |
| `streamlit` | Visualization dashboard |
| `joblib` | Parallel processing |
| `faiss-cpu` | *(optional)* Fast similarity search for applicability domain |

### Installation

```bash
pip install numpy pandas scipy scikit-learn matplotlib plotly streamlit joblib
pip install rdkit          # or: conda install -c conda-forge rdkit
pip install faiss-cpu      # optional, for faster AD computation
```

### Run the Full Pipeline

```bash
python drug_interaction_ml
```

For a quicker verification run, use the reduced profile:

```bash
DTI_PROFILE=fast python drug_interaction_ml
```

### Rerun Classification Only

If the classification step crashed and you've fixed the code, you can skip data fetching / featurization / regression and rerun only classification:

```bash
python rerun_classification.py
```

This script reuses all cached data and checkpoints.

### Visualization Dashboard

```bash
streamlit run visualization_dashboard.py
```

---

## Current Evaluation Protocol

The classification workflow now uses a stricter evaluation setup than the older single-holdout pipeline:

- Nested cross-validation: scaffold-grouped outer folds for performance estimation, grouped inner folds for hyperparameter tuning
- External validation: a scaffold holdout is used as the main unseen evaluation set, with temporal splits saved only as supplementary checks when available
- Molecular representation: ECFP4 (`radius=2`, 2048 bits) plus RDKit descriptors by default
- Preprocessing safety: feature scaling is handled inside sklearn pipelines, so scaler parameters are fit only on training folds
- Hyperparameter optimization: randomized search as the default optimizer, with saved trial artefacts and fixed seeds for reproducibility
- Class imbalance handling: class-weighted baselines are compared against a SMOTE comparator, and SMOTE is applied only inside training folds
- Probability calibration: Platt or isotonic calibration fitted from training-only out-of-fold predictions
- Threshold optimization: model-specific decision thresholds selected from training-only calibrated probabilities instead of a fixed 0.5 cutoff
- Confidence intervals: mean and standard deviation are reported across outer folds
- Leakage checks: exact overlap, scaffold overlap, and near-duplicate Tanimoto checks are written to `classification_leakage_report_v3.json`, and the scaffold external split is rejected if overlap is detected
- Error analysis: false-positive / false-negative breakdowns are written to `classification_error_analysis_v3.csv` and `classification_error_summary_v3.csv`
- Reproducibility artefacts: saved configs, split manifests, nested predictions, tuned parameters, and model metrics are all written to disk

The older regression notes below are still useful for data provenance, but the classification artefacts in the repository now reflect this nested-CV + external-validation workflow.

---

## Pipeline Overview

```
ChEMBL API → Raw Data → Curation → Feature Engineering → Train/Test Split
                                                              │
                     ┌────────────────────────────────────────┤
                     ▼                                        ▼
              Classification                            Regression
              (active/inactive)                         (pChEMBL)
                     │                                        │
              Hyperparameter Tuning                    Hyperparameter Tuning
              (RandomizedSearchCV)                     (RandomizedSearchCV)
                     │                                        │
              Stacking Ensemble                        Evaluation
              + SMOTE baseline                         (scaffold + temporal)
                     │
              Evaluation
              (scaffold + temporal)
                     │
              Applicability Domain
              + Activity Cliffs
                     │
              Plots & Reports
```

---

## Dataset

**Source:** ChEMBL target `CHEMBL243` (HIV-1 protease)

- Activity types: `IC50`, `Ki`
- Units: `nM` (converted to pChEMBL scale)

### Current Dataset Size (v3)

| Metric | Count |
|--------|-------|
| Curated molecules | 4,621 |
| Classification molecules | 4,058 (3,637 active / 421 inactive) |
| Regression molecules | 4,621 |
| Classification train / test | 3,246 / 812 |
| Regression train / test | 3,697 / 924 |

### Labeling Rule

- **Active:** median pChEMBL ≥ 6.5
- **Inactive:** median pChEMBL ≤ 5.5
- Compounds in the ambiguous zone are excluded from classification

### Data Processing Steps

1. Load cached raw ChEMBL activity data (or download if missing)
2. Keep exact-value measurements only
3. Convert activity values to pChEMBL potency
4. Remove noisy, duplicate, or contradictory records (max pChEMBL spread: 2.0)
5. Fetch and cache SMILES strings + publication year metadata
6. Standardize molecules with RDKit (uncharging, canonicalization)
7. Aggregate repeated measurements into one row per molecule
8. **Scaffold-based** train/test split (80/20) using Murcko scaffolds + GroupShuffleSplit

---

## Feature Engineering

The pipeline generates and benchmarks **six** molecular representations:

| Representation | Components |
|----------------|------------|
| `morgan_2048` | Morgan FP (radius 2, 2048 bits) |
| `descriptors` | 18 RDKit physicochemical descriptors |
| `morgan_plus_descriptors` | Morgan FP + descriptors |
| `maccs_plus_descriptors` | MACCS keys + descriptors |
| `atom_pair_torsion_plus_descriptors` | Atom Pair + Topological Torsion FPs + descriptors |
| `all_fingerprints_plus_descriptors` | All fingerprints + descriptors |

The best representation is selected automatically via a quick ExtraTrees benchmark on the training fold.

**Best classification representation (v3):** `morgan_2048` (mean MCC = 0.530)

---

## Models

### Classification

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Bagging ensemble |
| Extra Trees | Randomized splits ensemble |
| HistGradientBoosting | Gradient boosting |
| Stacked Ensemble | Meta-learner over RF + ET + HGB |
| Extra Trees + SMOTE | Oversampling-enhanced ET |

All models are tuned via `RandomizedSearchCV` with **5-fold stratified grouped CV** (grouped by Murcko scaffold).

### Regression

| Model | Type |
|-------|------|
| Random Forest | Bagging |
| Extra Trees | Randomized splits |
| HistGradientBoosting | Gradient boosting |

Tuned via `RandomizedSearchCV` with **5-fold grouped CV**.

---

## Latest Results (v3)

### Classification – Scaffold Holdout

| Model | MCC | PR-AUC | ROC-AUC | Accuracy |
|-------|-----|--------|---------|----------|
| **Logistic Regression** | **0.536** | **0.986** | **0.906** | 0.885 |
| Extra Trees + SMOTE | 0.522 | 0.984 | 0.892 | 0.911 |
| HistGradientBoosting | 0.512 | 0.984 | 0.888 | 0.879 |
| Stacked Ensemble | 0.413 | 0.984 | 0.891 | 0.787 |
| Extra Trees | 0.373 | 0.981 | 0.867 | 0.778 |
| Random Forest | 0.360 | 0.984 | 0.888 | 0.720 |

**Temporal validation:** Logistic Regression MCC = 0.411, PR-AUC = 0.959

### Regression – Scaffold Holdout

Results are saved in `regression_metrics_v3.csv`.

---

## Advanced Analyses

### Applicability Domain

The pipeline estimates an applicability-domain boundary using nearest-neighbour similarity on Morgan fingerprints. If `faiss-cpu` is installed, a fast FAISS inner-product index is used; otherwise it falls back to exhaustive Tanimoto computation via RDKit.

### Activity Cliffs

Pairs of structurally similar compounds (Tanimoto ≥ 0.90) with large potency differences (ΔpChEMBL ≥ 1.50) are identified and saved. **300 activity cliffs** were detected in the v3 dataset.

### Temporal Validation

When publication-year metadata is available, the pipeline builds a temporal train/test split (train ≤ 2003, test > 2003) to simulate prospective prediction.

---

## Output Files

### Data

| File | Description |
|------|-------------|
| `hiv_protease_chembl_raw.csv` | Cached raw ChEMBL activity table |
| `hiv_protease_chembl_dataset_v3.csv` | Cleaned curated dataset |
| `hiv_protease_smiles_v3.csv` | Molecule-to-SMILES mapping |
| `hiv_protease_documents_v3.csv` | Publication year metadata |
| `hiv_protease_classification_train_v3.csv` | Classification train split |
| `hiv_protease_classification_test_v3.csv` | Classification test split |
| `hiv_protease_regression_train_v3.csv` | Regression train split |
| `hiv_protease_regression_test_v3.csv` | Regression test split |
| `hiv_protease_classification_temporal_train_v3.csv` | Temporal classification train |
| `hiv_protease_classification_temporal_test_v3.csv` | Temporal classification test |

### Results & Metrics

| File | Description |
|------|-------------|
| `classification_metrics_v3.csv` | Classification evaluation metrics |
| `classification_cv_results_v3.csv` | Cross-validation results |
| `classification_best_params_v3.json` | Best hyperparameters (checkpoint) |
| `classification_predictions_v3.csv` | Test-set predictions + AD flags |
| `regression_metrics_v3.csv` | Regression evaluation metrics |
| `regression_cv_results_v3.csv` | Regression CV results |
| `regression_best_params_v3.json` | Best regression hyperparameters |
| `regression_predictions_v3.csv` | Regression test-set predictions |
| `feature_benchmarks_v3.csv` | Feature representation benchmark |
| `activity_cliffs_v3.csv` | Detected activity cliff pairs |
| `pipeline_report_v3.json` | Full JSON run report |
| `pipeline_run_v3.log` | Detailed run log |

### Plots

| File | Description |
|------|-------------|
| `roc_curve.png` | ROC curves for all classifiers |
| `precision_recall_curve.png` | PR curves for all classifiers |
| `feature_importance.png` | Top-25 feature importances |
| `regression_predictions.png` | Observed vs predicted pChEMBL scatter |

---

## Environment Profiles

The pipeline supports a `DTI_PROFILE` environment variable:

| Profile | CV Folds | Search Iters | Benchmark Trees |
|---------|----------|--------------|-----------------|
| `full` (default) | 5 | 6 | 160 |
| `fast` | 3 | 3 | 64 |

```bash
set DTI_PROFILE=fast        # Windows
python drug_interaction_ml
```

Additional environment overrides: `DTI_N_JOBS`, `DTI_CLASSIFICATION_CV_FOLDS`, `DTI_REGRESSION_CV_FOLDS`, `DTI_CLASSIFICATION_SEARCH_ITERATIONS`, `DTI_REGRESSION_SEARCH_ITERATIONS`, `DTI_FEATURE_BENCHMARK_TREES`.

---

## GPU Note

The script detects GPU hardware automatically. Detected adapter in this environment:

- **AMD Radeon R7 M440**

However, the pipeline trains on **CPU** because:

- RDKit fingerprint generation is CPU-based
- scikit-learn models are CPU-based
- No AMD-compatible ML acceleration backend (ROCm / OpenCL) is installed

For GPU acceleration, you would need a compatible GPU-enabled build of `lightgbm` or `cuml`.

---

## Project Structure

```
hiv_project/
├── drug_interaction_ml          # Main pipeline script (no .py extension)
├── rerun_classification.py      # Rerun classification only (skips regression)
├── visualization_dashboard.py   # Streamlit interactive dashboard
├── requirements.txt             # Python dependencies
├── README.md
├── *.csv / *.json / *.png       # Generated outputs (see tables above)
└── .venv/                       # Virtual environment
```

---

## References

- ChEMBL: https://www.ebi.ac.uk/chembl/
- HIV-1 protease inhibitors review: https://pmc.ncbi.nlm.nih.gov/articles/PMC6288788/
- RDKit: https://www.rdkit.org/
- FAISS: https://github.com/facebookresearch/faiss
