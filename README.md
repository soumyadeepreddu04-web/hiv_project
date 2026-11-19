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

## Design decisions

Each of these was a choice between alternatives, so the reasoning matters more than the setting.

**Scaffold-grouped splits, not random ones.** ChEMBL's HIV-1 protease data is largely congeneric
series — many molecules are close analogues from the same paper. On a random split those analogues
land on both sides, the model recognises the series rather than the chemistry, and the score is
inflated. Every split here groups by Bemis–Murcko scaffold: the holdout, the 3 outer folds and the
3 inner tuning folds. `StratifiedGroupKFold` was chosen over `GroupShuffleSplit` so that each fold
keeps the same active/inactive ratio, which matters at a 9:1 imbalance.

**MCC as the primary metric, not accuracy, PR-AUC or ROC-AUC.** 89.6% of the classification set is
active. Measured on the 812-molecule holdout, a model that predicts "active" for everything scores
**accuracy 0.895 and PR-AUC 0.895** while being useless; its MCC is 0.000. PR-AUC is the metric
usually recommended for imbalanced data, but that advice assumes the positive class is the rare
one, and here it is the common one. MCC uses all four cells of the confusion matrix, so it cannot
be won by predicting the majority class. ROC-AUC is reported alongside because it is
threshold-free and comparable to other papers, but it is not what the champion was selected on.

**A `Dummy Prior` model is included in every table.** Not as filler — it is the floor. It makes
the previous point concrete in the results rather than only in prose.

**SMOTE written from scratch, and applied inside training folds only.** `simple_smote` in
`training.py` is about 35 lines of NumPy rather than an `imbalanced-learn` import, because I wanted
to be sure I knew where the synthetic molecules were being created. It is called inside the
per-fold fit, never on data a model is scored against. The SMOTE model also sets
`class_weight=None`: oversampling and class weighting both correct the same imbalance, and
stacking them over-corrects.

**Calibration and the decision threshold are fitted on training data only.** Tree ensembles give
badly scaled probabilities, so each model gets an isotonic calibrator fitted on out-of-fold
predictions from *inside* its training fold, and a decision threshold chosen by maximising MCC on
those same calibrated training-fold values. The outer validation fold and the holdout see only the
already-fixed calibrator and threshold. Isotonic over Platt is selected automatically when the
fold has at least 150 samples and 20 of the minority class; below that it falls back to Platt,
because isotonic regression overfits small samples.

**The molecular representation is fixed in advance.** ECFP4 (2048 bits) plus 14 RDKit descriptors
is set in `config.py` before anything is evaluated. `training.py` does contain a function that
benchmarks five representations against each other, but it is deliberately **not** wired into the
pipeline: choosing the representation by CV score and then reporting nested-CV performance for the
winner lets the evaluation folds inform the feature choice, which makes the headline number
optimistic. The function's docstring says so, and `feature_benchmarks_v3.csv` records that the
representation was fixed rather than selected.

**The champion was picked before the holdout was scored, and not changed afterwards.** Extra Trees
+ SMOTE won on nested-CV MCC. On the holdout, *plain* Extra Trees then scored very slightly higher
(0.577 vs 0.574). Swapping the champion at that point would have been selecting on the test set, so
the tables below still report Extra Trees + SMOTE as champion and state the comparison openly. The
justification is in the fold-level numbers anyway: plain Extra Trees had an outer-fold SD of 0.122
against the champion's 0.011, so its holdout result is not evidence of a better model.

---

## Limitations

These are the problems I know about. Most were found by checking my own outputs rather than by
anything the pipeline flags automatically.

**The fingerprint cannot distinguish homologues that differ only in linker length.** The leakage
report records one external molecule at Tanimoto **1.000** to a training molecule. They are not
duplicates: they are macrocycles differing by a single `-OCH2CH2-` unit, 47 against 44 heavy atoms.
Binary folded ECFP4 records whether a substructure is *present*, not how many times, and at radius
2 every local environment in the longer linker already exists in the shorter one — so the two
molecules have byte-identical feature vectors and the model cannot tell them apart, even though
their Murcko scaffolds differ and the split is therefore technically valid. The same generator's
count-based fingerprint separates them at 0.933. Switching to count fingerprints is the obvious
next step and has not been done here.

**The fingerprint is blind to stereochemistry.** `GetMorganGenerator(radius=2, fpSize=2048)`
leaves `includeChirality` at its default of `False`, so a pair of enantiomers scores Tanimoto
1.000; with chirality enabled the same pair scores 0.714. HIV-1 protease inhibitors are
stereospecific peptidomimetics, so for this target in particular the representation is throwing
away information that determines potency. Fixing it means re-running and re-reporting everything,
which is why the numbers here are left as they are rather than quietly updated.

**"No shared scaffolds" oversells how independent the holdout is.** 78 of the 812 external
molecules have a training-set neighbour at Tanimoto >= 0.90. The split is valid by scaffold, but
those 78 are near-analogues of something the model has seen, so the holdout score is somewhere
between a true prospective estimate and an interpolation.

**Scaffold groups are very uneven.** The 4,058 classification molecules fall into 1,751 scaffold
groups. 1,206 of those are singletons (29.7% of all molecules), while the largest group holds 82
molecules, so grouped folds are balanced by molecule count but not by chemical diversity. Bemis–
Murcko is also a coarse definition: 10 molecules are acyclic and have an empty Murcko scaffold, so
the code falls back to the full SMILES and each becomes its own group.

**The split was not reproducible until now, and the stored metrics predate the fix.** The holdout
was recomputed with `StratifiedGroupKFold` on every run. That function does not promise identical
fold assignment across scikit-learn versions, and it does not deliver it: re-running the same code
with the same seed (42) on scikit-learn 1.6.1 produced a 3,306/752 split instead of the 3,246/812
one the published numbers came from. Only **162 of the original 812 holdout molecules were still in
the holdout** — 1,240 molecules changed side — which also shifted the applicability-domain
threshold (0.636 to 0.642) and the near-duplicate count (78 to 91). So "seed = 42" was never
enough to make this repeatable, and the split manifest that would have fixed it was being written
on every run and never read back. The pipeline now restores it when it matches the dataset, which
I verified reproduces the committed 812-molecule holdout molecule-for-molecule. The metrics in the
tables above were produced before this fix, on the split the manifest pins — they are now
reproducible, but they were not when they were generated.

**The hyperparameter search is small.** 2 random-search trials per model per fold on the default
profile. The models are barely tuned, and the ranking between the mid-table models is not reliable
at that budget.

**The stored threshold is slightly optimistic in-sample.** The isotonic calibrator is fitted on a
fold's out-of-fold probability vector and then applied to that same vector to choose the threshold.
This happens entirely inside the training fold, so the outer-fold and holdout estimates are not
affected, but the threshold and the calibrated training metrics in
`classification_best_params_v3.json` are fitted on the data they are measured on.

**The class imbalance is a property of the literature, not of chemistry.** Inactive compounds are
under-published, so the 9:1 active ratio reflects reporting bias in ChEMBL. A model trained here
predicts "is this the kind of molecule that gets published as an HIV-1 protease hit", which is not
quite the same question as "is this molecule active".

**One target, one dataset version, classification only.** The curation code assembles a continuous
pChEMBL dataset of 4,621 molecules and the pipeline saves temporal splits, but no regression model
is trained and no temporal evaluation is scored in v3.

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
