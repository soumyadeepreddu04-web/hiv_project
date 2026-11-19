# QSAR Activity Prediction for HIV-1 Protease Inhibitors

An end-to-end cheminformatics / machine-learning pipeline that predicts small-molecule activity against **HIV-1 protease** using real bioactivity data from ChEMBL.

## About this project

I built this between April and May 2026, on my own. The question, the data curation rules, the
validation design and all of the code are mine. It was not part of any taught course, it was not
assessed, and I had no supervisor — I was in the final year of a BPharm and wanted to find out
whether the medicinal chemistry I was being taught could be made to predict anything.

Most of the work went into the evaluation rather than the models, and the commit history shows
where the effort went. The first working version (6–7 April) was a single script: it already
grouped the split by scaffold and already used MCC, but it scored each model on one grouped split,
with no probability calibration, no tuned decision threshold, no baseline to compare against and
no check on how similar the holdout really was to the training set. The rewrite on 2 May split it
into the package described below and added all five: nested cross-validation, isotonic/Platt
calibration, MCC-optimised thresholds, a `Dummy Prior` floor, and a leakage report that measures
train/holdout similarity instead of assuming it.

Everything I currently know to be wrong with it is in [Limitations](#limitations), including two
properties of the molecular representation that I found only by reading my own leakage report, and
a reproducibility bug in the split that I found while preparing this write-up and have fixed.

The project covers:

- **Classification** – active vs inactive compound prediction
- Leakage-resistant evaluation: scaffold-grouped nested cross-validation plus a scaffold-held-out external set
- Advanced analyses: probability calibration, applicability-domain scoring, near-duplicate checks, activity-cliff detection

The curation code also assembles a continuous pChEMBL regression dataset (4,621 molecules), but regression models are not trained in the current (v3) release.

## Project Goal

Train and evaluate ML models that **classify** compounds as active or inactive against HIV-1 protease from molecular structure (SMILES), and report performance in a way that holds up on chemical series the model has never seen.

---

## Quick Start

### Prerequisites

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scipy` | Scientific computing |
| `scikit-learn` | ML models & evaluation |
| `rdkit` | Molecular fingerprints & descriptors |
| `matplotlib` | Static plots |
| `plotly` | Interactive plots |
| `streamlit` | Visualization dashboard |
| `joblib` | Parallel processing |

### Installation

```bash
pip install -r requirements.txt
# or install RDKit via conda: conda install -c conda-forge rdkit
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

For a quicker verification run, use the reduced profile:

```bash
DTI_PROFILE=fast python run_pipeline.py
```

Both commands reuse the cached ChEMBL tables in this repository if they are present, and only call
the ChEMBL API when a cache file is missing.

### Visualization Dashboard

```bash
streamlit run visualization_dashboard.py
```

---

## Evaluation Protocol

- **Nested cross-validation:** 3 scaffold-grouped outer folds for performance estimation, 3 grouped inner folds for hyperparameter tuning
- **External validation:** a Murcko-scaffold holdout (812 molecules) that shares no molecules, SMILES or scaffolds with the training set; temporal splits are saved as supplementary artefacts only
- **Molecular representation:** ECFP4 (`radius=2`, 2048 bits) plus 14 RDKit descriptors, fixed in advance — no data-driven representation selection is done before nested CV, so feature choice cannot leak information from the evaluation folds
- **Preprocessing safety:** feature scaling is handled inside sklearn pipelines, so scaler parameters are fit only on training folds
- **Hyperparameter optimization:** randomized search with fixed seeds and saved trial artefacts
- **Class imbalance handling:** class-weighted models are compared against a SMOTE variant, and SMOTE is applied only inside training folds
- **Probability calibration:** Platt or isotonic calibration fitted from training-only out-of-fold predictions (isotonic was selected in v3)
- **Threshold optimization:** model-specific decision thresholds selected from training-only calibrated probabilities instead of a fixed 0.5 cutoff
- **Primary metric:** Matthews correlation coefficient (MCC). 89.6% of the classification set is active, so accuracy and PR-AUC are inflated — on the 812-molecule holdout a model that always predicts "active" scores 0.895 accuracy and 0.895 PR-AUC, at MCC 0.000
- **Leakage checks:** exact overlap, scaffold overlap, and near-duplicate Tanimoto checks are written to `classification_leakage_report_v3.json`, and the scaffold external split is rejected if overlap is detected
- **Error analysis:** false-positive / false-negative breakdowns are written to `classification_error_analysis_v3.csv` and `classification_error_summary_v3.csv`
- **Reproducibility artefacts:** saved configs, split manifests, nested predictions, tuned parameters, and model metrics are all written to disk
- **Pinned split:** the train/holdout split is restored from `classification_saved_splits_v3.csv` when it matches the dataset, because `StratifiedGroupKFold` does not assign folds identically across scikit-learn versions. Recomputing it on scikit-learn 1.6.1 produced a holdout sharing only 162 of its 812 molecules with the published one, so the metrics below were effectively unreproducible before this was pinned. `DTI_RECOMPUTE_SPLIT=1` forces a fresh split

---

## Pipeline Overview

```
ChEMBL API → Raw Data → Curation → Featurization (ECFP4 + descriptors)
                                              │
                                   Scaffold external split
                                   (internal train / external holdout)
                                              │
                             Nested CV on internal train (3 × 3)
                             tuning + calibration + thresholds
                                              │
                             Champion chosen by nested-CV MCC
                                              │
                             Final models refit on internal train
                                              │
                             External holdout evaluation
                                              │
                    Applicability domain + near-duplicates + activity cliffs
                                              │
                                       Plots & reports
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
| Internal train / external holdout | 3,246 / 812 |

### Labeling Rule

- **Active:** median pChEMBL ≥ 6.5
- **Inactive:** median pChEMBL ≤ 5.5
- Compounds in the ambiguous zone are excluded from classification

### Data Processing Steps

1. Load cached raw ChEMBL activity data (or download if missing)
2. Keep exact-value measurements only
3. Convert activity values to pChEMBL potency
4. Drop molecules whose measurements contradict each other (any molecule with a reading
   at or above the active cut-off *and* one at or below the inactive cut-off), then drop
   molecules whose pChEMBL standard deviation across measurements exceeds 2.0 log units
5. Fetch and cache SMILES strings + publication year metadata
6. Standardize molecules with RDKit (uncharging, canonicalization)
7. Aggregate repeated measurements into one row per molecule
8. **Scaffold-based** train/test split (80/20) using Murcko scaffolds + `StratifiedGroupKFold`
   (5 splits, first fold taken as the holdout), so no scaffold appears on both sides and the
   active/inactive ratio is preserved in each side

---

## Models

| Model | Type |
|-------|------|
| Dummy Prior | Always predicts the class prior (floor for every metric) |
| Logistic Regression | Linear baseline (class-weighted) |
| Linear SVC | Linear margin classifier (class-weighted) |
| Random Forest | Bagging ensemble (class-weighted) |
| Extra Trees | Randomized splits ensemble (class-weighted) |
| Extra Trees + SMOTE | Extra Trees with SMOTE oversampling inside training folds (hand-written, `simple_smote` in `training.py`) |
| Gradient Boosting | Gradient-boosted trees |

---

## Latest Results (v3)

The champion model is **Extra Trees + SMOTE**, chosen on nested-CV MCC before the external holdout was scored.

### Nested Cross-Validation (internal train, mean ± SD over 3 outer folds)

| Model | MCC | ROC-AUC | Brier |
|-------|-----|---------|-------|
| **Extra Trees + SMOTE** | **0.584 ± 0.011** | **0.920 ± 0.001** | **0.053** |
| Gradient Boosting | 0.572 ± 0.036 | 0.907 ± 0.006 | 0.056 |
| Logistic Regression | 0.456 ± 0.034 | 0.860 ± 0.016 | 0.069 |
| Random Forest | 0.369 ± 0.055 | 0.857 ± 0.040 | 0.073 |
| Linear SVC | 0.347 ± 0.034 | 0.793 ± 0.022 | 0.078 |
| Extra Trees | 0.343 ± 0.122 | 0.805 ± 0.090 | 0.077 |
| Dummy Prior | 0.000 | 0.500 | 0.093 |

### External Validation (scaffold holdout, 812 molecules)

| Model | MCC | ROC-AUC | Brier |
|-------|-----|---------|-------|
| Extra Trees | 0.577 | 0.873 | 0.063 |
| **Extra Trees + SMOTE** | **0.574** | **0.883** | **0.058** |
| Gradient Boosting | 0.541 | 0.909 | 0.055 |
| Logistic Regression | 0.504 | 0.858 | 0.067 |
| Random Forest | 0.424 | 0.869 | 0.073 |
| Linear SVC | 0.376 | 0.821 | 0.073 |
| Dummy Prior | 0.000 | 0.500 | 0.094 |

Notes:

- The champion's nested-CV MCC (0.584) carried over to unseen scaffolds (0.574), which suggests the internal estimate was not optimistic.
- Plain Extra Trees edged the champion on the holdout (0.577 vs 0.574) but was far less stable across outer folds (SD 0.122 vs 0.011), which is why it was not selected.
- Accuracy, PR-AUC, precision, recall and F1 for every model are in `classification_metrics_v3.csv`.

---

## Advanced Analyses

### Applicability Domain

Each training molecule's nearest-neighbour Tanimoto similarity (ECFP4) to the rest of the training set is computed exhaustively with RDKit. The 5th percentile of those values sets the domain threshold (**0.636** in v3). External molecules whose nearest training neighbour falls below it are flagged as outside the domain; **90.5%** of the external holdout fell inside.

### Near-Duplicate Check

Although no scaffolds are shared between training and external sets, **78** external molecules have a training-set neighbour at Tanimoto ≥ 0.90. These are listed in `classification_leakage_report_v3.json`.

### Activity Cliffs

Pairs of structurally similar compounds (Tanimoto ≥ 0.90) with large potency differences (ΔpChEMBL ≥ 1.50) are identified and saved. **300 activity cliffs** were detected in the v3 dataset.

### Temporal Split

When publication-year metadata is available, the pipeline also builds a temporal train/test split (train ≤ 2003, test > 2003) to simulate prospective prediction. The split files are saved, but models are not scored on them in v3.

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
| `hiv_protease_classification_nested_train_v3.csv` | Internal training set used for nested CV |
| `hiv_protease_classification_external_validation_v3.csv` | Scaffold external holdout (+ `.metadata.json`) |
| `hiv_protease_classification_temporal_train_v3.csv` | Temporal classification train |
| `hiv_protease_classification_temporal_test_v3.csv` | Temporal classification test |

### Results & Metrics

| File | Description |
|------|-------------|
| `classification_metrics_v3.csv` | Nested-CV and external-validation metrics |
| `classification_cv_results_v3.csv` | Per-fold cross-validation results |
| `classification_tuning_results_v3.csv` | Hyperparameter search trials |
| `classification_best_params_v3.json` | Best hyperparameters |
| `classification_nested_predictions_v3.csv` | Out-of-fold nested-CV predictions |
| `classification_predictions_v3.csv` | External-set predictions + AD flags |
| `classification_saved_splits_v3.csv` | Split manifest |
| `classification_leakage_report_v3.json` | Overlap and near-duplicate checks |
| `classification_error_analysis_v3.csv` | Per-molecule error analysis |
| `classification_error_summary_v3.csv` | Error summary by type and AD status |
| `classification_run_config_v3.json` | Run configuration |
| `feature_benchmarks_v3.csv` | Representation used for evaluation |
| `activity_cliffs_v3.csv` | Detected activity cliff pairs |
| `pipeline_report_v3.json` | Full JSON run report |
| `pipeline_run_v3.log` | Detailed run log |

### Plots

| File | Description |
|------|-------------|
| `roc_curve.png` | ROC curves on the external holdout |
| `precision_recall_curve.png` | PR curves on the external holdout |
| `feature_importance.png` | Top-25 Extra Trees feature importances |

---

## Environment Profiles

The pipeline supports a `DTI_PROFILE` environment variable:

| Profile | Outer folds | Inner folds | Search iterations |
|---------|-------------|-------------|-------------------|
| `full` (default) | 3 | 3 | 2 |
| `fast` | 2 | 2 | 1 |

```bash
DTI_PROFILE=fast python run_pipeline.py     # macOS / Linux
set DTI_PROFILE=fast && python run_pipeline.py   # Windows cmd
```

Additional environment overrides: `DTI_N_JOBS`, `DTI_RECOMPUTE_SPLIT`, `DTI_OUTER_CV_FOLDS`, `DTI_INNER_CV_FOLDS`, `DTI_CLASSIFICATION_SEARCH_ITERATIONS`, `DTI_CLASSIFICATION_REPRESENTATION`, `DTI_SMOTE_K_NEIGHBORS`, `DTI_CALIBRATION_AUTO_MIN_SAMPLES`, `DTI_NEAR_DUPLICATE_TANIMOTO_THRESHOLD`, `DTI_NEAR_DUPLICATE_MAX_EXAMPLES`.

---

## Project Structure

```
hiv_project/
├── run_pipeline.py              # Main pipeline entry point
├── dti_pipeline/                # Pipeline package
│   ├── config.py                # Paths, thresholds, run profiles
│   ├── data.py                  # ChEMBL download, curation, splits
│   ├── features.py              # ECFP4 + RDKit descriptors
│   ├── training.py              # Models, nested CV, calibration
│   └── evaluation.py            # Metrics, AD, leakage checks, cliffs, plots
├── visualization_dashboard.py   # Streamlit interactive dashboard
├── requirements.txt             # Python dependencies
├── README.md
└── *.csv / *.json / *.png       # Generated outputs (see tables above)
```

Everything runs on CPU: RDKit fingerprinting and the scikit-learn models used here have no GPU
path, so there is nothing to gain from one. `DTI_N_JOBS` controls worker count and defaults to 1.

---

## References

- ChEMBL: https://www.ebi.ac.uk/chembl/
- HIV-1 protease inhibitors review: https://pmc.ncbi.nlm.nih.gov/articles/PMC6288788/
- RDKit: https://www.rdkit.org/
