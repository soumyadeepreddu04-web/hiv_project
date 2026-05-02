from __future__ import annotations

import importlib.util
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


RUN_PROFILE = os.environ.get("DTI_PROFILE", "full").strip().lower()
IS_FAST_PROFILE = RUN_PROFILE == "fast"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CHEMBL_TARGET_IDS = ("CHEMBL243",)
STANDARD_TYPES = ("IC50", "Ki")
STANDARD_UNITS = "nM"
ACTIVE_PCHEMBL_THRESHOLD = 6.5
INACTIVE_PCHEMBL_THRESHOLD = 5.5
MAX_PCHEMBL_SPREAD = 2.0
REQUEST_SLEEP_SECONDS = 0.05
REQUEST_MAX_RETRIES = 4
PAGE_SIZE = 500
MOLECULE_BATCH_SIZE = 20
DOCUMENT_BATCH_SIZE = 50
MIN_CACHED_SMILES_FOR_OFFLINE_BUILD = 3000
DATASET_VERSION = "v3"
OUTER_CV_FOLDS = env_int("DTI_OUTER_CV_FOLDS", 2 if IS_FAST_PROFILE else 3)
INNER_CV_FOLDS = env_int("DTI_INNER_CV_FOLDS", 2 if IS_FAST_PROFILE else 3)
CLASSIFICATION_SEARCH_ITERATIONS = env_int("DTI_CLASSIFICATION_SEARCH_ITERATIONS", 1 if IS_FAST_PROFILE else 2)
FEATURE_BENCHMARK_TREES = env_int("DTI_FEATURE_BENCHMARK_TREES", 48 if IS_FAST_PROFILE else 120)
TOP_FEATURES_TO_PLOT = 25
AD_SIMILARITY_PERCENTILE = 5
ACTIVITY_CLIFF_SIMILARITY = 0.90
ACTIVITY_CLIFF_DELTA_PCHEMBL = 1.50
MAX_PARALLEL_WORKERS = env_int("DTI_N_JOBS", 1)
SMOTE_K_NEIGHBORS = env_int("DTI_SMOTE_K_NEIGHBORS", 5)
CALIBRATION_AUTO_MIN_SAMPLES = env_int("DTI_CALIBRATION_AUTO_MIN_SAMPLES", 150)
NEAR_DUPLICATE_TANIMOTO_THRESHOLD = float(os.environ.get("DTI_NEAR_DUPLICATE_TANIMOTO_THRESHOLD", "0.90"))
NEAR_DUPLICATE_MAX_EXAMPLES = env_int("DTI_NEAR_DUPLICATE_MAX_EXAMPLES", 50)
CLASSIFICATION_REPRESENTATION = os.environ.get("DTI_CLASSIFICATION_REPRESENTATION", "ecfp4_plus_descriptors").strip()

SCRIPT_DIR = Path(__file__).resolve().parent.parent
RAW_CACHE_PATH = SCRIPT_DIR / "hiv_protease_chembl_raw.csv"
DATASET_CACHE_PATH = SCRIPT_DIR / f"hiv_protease_chembl_dataset_{DATASET_VERSION}.csv"
CLASSIFICATION_TRAIN_PATH = SCRIPT_DIR / f"hiv_protease_classification_train_{DATASET_VERSION}.csv"
CLASSIFICATION_TEST_PATH = SCRIPT_DIR / f"hiv_protease_classification_test_{DATASET_VERSION}.csv"
REGRESSION_TRAIN_PATH = SCRIPT_DIR / f"hiv_protease_regression_train_{DATASET_VERSION}.csv"
REGRESSION_TEST_PATH = SCRIPT_DIR / f"hiv_protease_regression_test_{DATASET_VERSION}.csv"
TEMPORAL_CLASSIFICATION_TRAIN_PATH = SCRIPT_DIR / f"hiv_protease_classification_temporal_train_{DATASET_VERSION}.csv"
TEMPORAL_CLASSIFICATION_TEST_PATH = SCRIPT_DIR / f"hiv_protease_classification_temporal_test_{DATASET_VERSION}.csv"
TEMPORAL_REGRESSION_TRAIN_PATH = SCRIPT_DIR / f"hiv_protease_regression_temporal_train_{DATASET_VERSION}.csv"
TEMPORAL_REGRESSION_TEST_PATH = SCRIPT_DIR / f"hiv_protease_regression_temporal_test_{DATASET_VERSION}.csv"
CLASSIFICATION_INTERNAL_TRAIN_PATH = SCRIPT_DIR / f"hiv_protease_classification_nested_train_{DATASET_VERSION}.csv"
CLASSIFICATION_EXTERNAL_PATH = SCRIPT_DIR / f"hiv_protease_classification_external_validation_{DATASET_VERSION}.csv"
CLASSIFICATION_SPLIT_MANIFEST_PATH = SCRIPT_DIR / f"classification_saved_splits_{DATASET_VERSION}.csv"
SMILES_CACHE_PATH = SCRIPT_DIR / f"hiv_protease_smiles_{DATASET_VERSION}.csv"
DOCUMENT_CACHE_PATH = SCRIPT_DIR / f"hiv_protease_documents_{DATASET_VERSION}.csv"
ROC_PLOT_PATH = SCRIPT_DIR / "roc_curve.png"
PR_PLOT_PATH = SCRIPT_DIR / "precision_recall_curve.png"
IMPORTANCE_PLOT_PATH = SCRIPT_DIR / "feature_importance.png"
REGRESSION_PLOT_PATH = SCRIPT_DIR / "regression_predictions.png"
FEATURE_BENCHMARK_PATH = SCRIPT_DIR / f"feature_benchmarks_{DATASET_VERSION}.csv"
CLASSIFICATION_METRICS_PATH = SCRIPT_DIR / f"classification_metrics_{DATASET_VERSION}.csv"
CLASSIFICATION_CV_RESULTS_PATH = SCRIPT_DIR / f"classification_cv_results_{DATASET_VERSION}.csv"
CLASSIFICATION_NESTED_PREDICTIONS_PATH = SCRIPT_DIR / f"classification_nested_predictions_{DATASET_VERSION}.csv"
CLASSIFICATION_PREDICTIONS_PATH = SCRIPT_DIR / f"classification_predictions_{DATASET_VERSION}.csv"
CLASSIFICATION_TUNING_RESULTS_PATH = SCRIPT_DIR / f"classification_tuning_results_{DATASET_VERSION}.csv"
CLASSIFICATION_ERROR_ANALYSIS_PATH = SCRIPT_DIR / f"classification_error_analysis_{DATASET_VERSION}.csv"
CLASSIFICATION_ERROR_SUMMARY_PATH = SCRIPT_DIR / f"classification_error_summary_{DATASET_VERSION}.csv"
CLASSIFICATION_LEAKAGE_REPORT_PATH = SCRIPT_DIR / f"classification_leakage_report_{DATASET_VERSION}.json"
CLASSIFICATION_CONFIG_PATH = SCRIPT_DIR / f"classification_run_config_{DATASET_VERSION}.json"
REGRESSION_METRICS_PATH = SCRIPT_DIR / f"regression_metrics_{DATASET_VERSION}.csv"
REGRESSION_CV_RESULTS_PATH = SCRIPT_DIR / f"regression_cv_results_{DATASET_VERSION}.csv"
REGRESSION_PREDICTIONS_PATH = SCRIPT_DIR / f"regression_predictions_{DATASET_VERSION}.csv"
CLASSIFICATION_BEST_PARAMS_PATH = SCRIPT_DIR / f"classification_best_params_{DATASET_VERSION}.json"
REGRESSION_BEST_PARAMS_PATH = SCRIPT_DIR / f"regression_best_params_{DATASET_VERSION}.json"
ACTIVITY_CLIFF_PATH = SCRIPT_DIR / f"activity_cliffs_{DATASET_VERSION}.csv"
RUN_REPORT_PATH = SCRIPT_DIR / f"pipeline_report_{DATASET_VERSION}.json"
RUN_LOG_PATH = SCRIPT_DIR / f"pipeline_run_{DATASET_VERSION}.log"

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

FEATURE_RECIPES: Mapping[str, Sequence[str]] = {
    "ecfp4_2048": ("ecfp4_2048",),
    "descriptors": ("descriptors",),
    "ecfp4_plus_descriptors": ("ecfp4_2048", "descriptors"),
    "maccs_plus_descriptors": ("maccs", "descriptors"),
    "all_fingerprints_plus_descriptors": ("ecfp4_2048", "maccs", "atom_pair_2048", "torsion_2048", "descriptors"),
}

BENCHMARK_REPRESENTATIONS: Sequence[str] = (
    "ecfp4_2048",
    "descriptors",
    "ecfp4_plus_descriptors",
) if IS_FAST_PROFILE else tuple(FEATURE_RECIPES.keys())


@dataclass
class RuntimeInfo:
    backend: str
    adapters: List[str]
    gpu_name: str
    message: str


@dataclass
class ClassificationResult:
    name: str
    split: str
    accuracy: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    mcc: float
    brier: float
    threshold: float
    calibration_method: str
    y_prob: np.ndarray
    y_pred: np.ndarray


@dataclass
class FeatureBlocks:
    arrays: Dict[str, np.ndarray]
    names: Dict[str, List[str]]
    morgan_fingerprints: List[Any]


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.start = time.perf_counter()
        self.path.write_text("", encoding="utf-8")
        self.log("Pipeline run started.")

    def log(self, message: str) -> None:
        elapsed = time.perf_counter() - self.start
        line = f"[{elapsed:8.1f}s] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    @contextmanager
    def section(self, title: str) -> Iterable[None]:
        started = time.perf_counter()
        self.log(f"{title}...")
        try:
            yield
        except Exception as error:
            self.log(f"{title} failed: {error.__class__.__name__}: {error}")
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.log(f"{title} finished in {elapsed:.1f}s.")


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(inner_value) for inner_value in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
