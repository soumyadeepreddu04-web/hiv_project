from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from dti_pipeline.config import (
    BENCHMARK_REPRESENTATIONS,
    CALIBRATION_AUTO_MIN_SAMPLES,
    CLASSIFICATION_SEARCH_ITERATIONS,
    FEATURE_BENCHMARK_PATH,
    FEATURE_BENCHMARK_TREES,
    INNER_CV_FOLDS,
    MAX_PARALLEL_WORKERS,
    OUTER_CV_FOLDS,
    RANDOM_SEED,
    SMOTE_K_NEIGHBORS,
    ClassificationResult,
    RunLogger,
    to_jsonable,
)
from dti_pipeline.data import scaffold_groups
from dti_pipeline.features import compose_feature_matrix


@dataclass
class FinalModelArtifact:
    name: str
    estimator: Any
    params: Dict[str, Any]
    calibrator: Any
    calibration_method: str
    threshold: float
    oof_prob: np.ndarray
    oof_metrics: Dict[str, float]
    uses_smote: bool


@dataclass(frozen=True)
class ModelSpec:
    estimator: Any
    search_space: Dict[str, Any]
    uses_smote: bool = False


def classification_model_specs() -> Dict[str, ModelSpec]:
    return {
        "Dummy Prior": ModelSpec(
            estimator=DummyClassifier(strategy="prior"),
            search_space={},
        ),
        "Logistic Regression": ModelSpec(
            estimator=Pipeline(
                [
                    ("scale", StandardScaler(with_mean=False)),
                    ("model", LogisticRegression(max_iter=4000, solver="liblinear", random_state=RANDOM_SEED, class_weight="balanced")),
                ]
            ),
            search_space={
                "model__C": loguniform(1e-2, 10.0),
            },
        ),
        "Linear SVC": ModelSpec(
            estimator=Pipeline(
                [
                    ("scale", StandardScaler(with_mean=False)),
                    ("model", LinearSVC(random_state=RANDOM_SEED, class_weight="balanced", max_iter=20000)),
                ]
            ),
            search_space={
                "model__C": loguniform(1e-2, 10.0),
            },
        ),
        "Random Forest": ModelSpec(
            estimator=RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=MAX_PARALLEL_WORKERS, class_weight="balanced_subsample"),
            search_space={
                "n_estimators": randint(100, 251),
                "max_depth": [None, 10, 16, 24],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 5),
                "max_features": ["sqrt", 0.35, 0.50],
            },
        ),
        "Extra Trees": ModelSpec(
            estimator=ExtraTreesClassifier(random_state=RANDOM_SEED, n_jobs=MAX_PARALLEL_WORKERS, class_weight="balanced_subsample"),
            search_space={
                "n_estimators": randint(100, 251),
                "max_depth": [None, 12, 20, 28],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 4),
                "max_features": ["sqrt", 0.35, 0.50],
            },
        ),
        "Extra Trees + SMOTE": ModelSpec(
            estimator=ExtraTreesClassifier(random_state=RANDOM_SEED, n_jobs=MAX_PARALLEL_WORKERS, class_weight=None),
            search_space={
                "n_estimators": randint(100, 251),
                "max_depth": [None, 12, 20, 28],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 4),
                "max_features": ["sqrt", 0.35, 0.50],
            },
            uses_smote=True,
        ),
        "Gradient Boosting": ModelSpec(
            estimator=GradientBoostingClassifier(random_state=RANDOM_SEED),
            search_space={
                "learning_rate": loguniform(0.02, 0.20),
                "n_estimators": randint(80, 181),
                "max_depth": [2, 3, 4, 6],
                "min_samples_leaf": randint(5, 31),
                "subsample": [0.7, 0.85, 1.0],
            },
        ),
    }


def sample_distribution(space: Any, rng: np.random.Generator) -> Any:
    if isinstance(space, (list, tuple)):
        choice = space[int(rng.integers(0, len(space)))]
        return choice.item() if isinstance(choice, np.generic) else choice
    if hasattr(space, "rvs"):
        try:
            value = space.rvs(random_state=rng)
        except TypeError:
            value = space.rvs(random_state=int(rng.integers(0, 2**31 - 1)))
        return value.item() if isinstance(value, np.generic) else value
    return space


def sample_parameter_set(search_space: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    return {name: sample_distribution(space, rng) for name, space in search_space.items()}


def estimator_sample_weight_kwargs(estimator: Any, sample_weight: np.ndarray | None) -> Dict[str, np.ndarray]:
    if sample_weight is None:
        return {}
    if isinstance(estimator, Pipeline):
        last_step_name = estimator.steps[-1][0]
        return {f"{last_step_name}__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def simple_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) != 2:
        return X_train, y_train

    minority_label = labels[np.argmin(counts)]
    majority_count = int(np.max(counts))
    minority_mask = y_train == minority_label
    X_minority = X_train[minority_mask]
    samples_needed = majority_count - len(X_minority)

    if samples_needed <= 0:
        return X_train, y_train
    if len(X_minority) < 2:
        repeated = np.repeat(X_minority, samples_needed, axis=0)
        synthetic_y = np.full(samples_needed, minority_label, dtype=y_train.dtype)
        return np.vstack([X_train, repeated]), np.concatenate([y_train, synthetic_y])

    n_neighbors = min(SMOTE_K_NEIGHBORS, len(X_minority) - 1)
    rng = np.random.default_rng(random_state)
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    neighbor_model.fit(X_minority)
    neighbor_indices = neighbor_model.kneighbors(return_distance=False)

    synthetic = np.zeros((samples_needed, X_train.shape[1]), dtype=np.float32)
    for row_index in range(samples_needed):
        anchor_index = int(rng.integers(0, len(X_minority)))
        choices = neighbor_indices[anchor_index][1:]
        neighbor_index = int(rng.choice(choices)) if len(choices) else anchor_index
        gap = float(rng.random())
        synthetic[row_index] = X_minority[anchor_index] + gap * (X_minority[neighbor_index] - X_minority[anchor_index])

    synthetic_y = np.full(samples_needed, minority_label, dtype=y_train.dtype)
    return np.vstack([X_train, synthetic]), np.concatenate([y_train, synthetic_y])


def fit_classifier(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    use_smote: bool = False,
    random_state: int = RANDOM_SEED,
) -> Any:
    if use_smote:
        X_fit, y_fit = simple_smote(X_train, y_train, random_state=random_state)
        fit_kwargs = {}
    else:
        X_fit, y_fit = X_train, y_train
        fit_kwargs = estimator_sample_weight_kwargs(model, sample_weight)
    try:
        model.fit(X_fit, y_fit, **fit_kwargs)
    except TypeError:
        model.fit(X_fit, y_fit)
    return model


def predict_positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities.reshape(-1)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-decision))

    return np.asarray(model.predict(X)).reshape(-1).astype(float)


def classification_metrics_from_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    clipped_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    y_pred = (clipped_prob >= threshold).astype(int)
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "roc_auc": float(roc_auc_score(y_true, clipped_prob)),
        "pr_auc": float(average_precision_score(y_true, clipped_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, clipped_prob)),
    }


def choose_calibration_method(y_true: np.ndarray, requested_method: str = "auto") -> str:
    if requested_method in {"platt", "isotonic"}:
        return requested_method
    class_counts = np.bincount(y_true.astype(int), minlength=2)
    if len(y_true) >= CALIBRATION_AUTO_MIN_SAMPLES and int(np.min(class_counts)) >= 20:
        return "isotonic"
    return "platt"


def fit_probability_calibrator(raw_prob: np.ndarray, y_true: np.ndarray, method: str = "auto") -> Tuple[Any, str]:
    chosen_method = choose_calibration_method(y_true, requested_method=method)
    clipped_prob = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
    if chosen_method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(clipped_prob, y_true)
        return calibrator, chosen_method

    calibrator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_SEED, class_weight="balanced")
    calibrator.fit(clipped_prob.reshape(-1, 1), y_true)
    return calibrator, chosen_method


def apply_probability_calibrator(calibrator: Any, raw_prob: np.ndarray, method: str) -> np.ndarray:
    clipped_prob = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
    if method == "isotonic":
        return np.clip(np.asarray(calibrator.predict(clipped_prob)).reshape(-1), 1e-6, 1.0 - 1e-6)
    return np.clip(np.asarray(calibrator.predict_proba(clipped_prob.reshape(-1, 1)))[:, 1].reshape(-1), 1e-6, 1.0 - 1e-6)


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique_prob = np.unique(np.clip(y_prob, 1e-6, 1.0 - 1e-6))
    if unique_prob.size > 200:
        quantiles = np.linspace(0.02, 0.98, 197)
        candidate_thresholds = np.unique(np.quantile(unique_prob, quantiles))
    else:
        candidate_thresholds = unique_prob

    candidate_thresholds = np.concatenate(([1e-6], candidate_thresholds, [0.5, 1.0 - 1e-6]))
    best_threshold = 0.5
    best_score = -np.inf
    best_f1 = -np.inf

    for threshold in np.unique(candidate_thresholds):
        y_pred = (y_prob >= threshold).astype(int)
        mcc_value = matthews_corrcoef(y_true, y_pred)
        f1_value = f1_score(y_true, y_pred, zero_division=0)
        if (mcc_value > best_score) or (np.isclose(mcc_value, best_score) and f1_value > best_f1):
            best_score = mcc_value
            best_f1 = f1_value
            best_threshold = float(threshold)

    return best_threshold


def cross_validated_probabilities(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    use_smote: bool = False,
    seed_offset: int = 0,
) -> np.ndarray:
    oof_prob = np.full(len(y), np.nan, dtype=float)
    for fold_number, (fit_idx, valid_idx) in enumerate(cv_splits, start=1):
        fold_model = clone(estimator)
        fold_weights = compute_sample_weight(class_weight="balanced", y=y[fit_idx])
        fit_classifier(
            fold_model,
            X[fit_idx],
            y[fit_idx],
            sample_weight=fold_weights,
            use_smote=use_smote,
            random_state=RANDOM_SEED + seed_offset + fold_number,
        )
        oof_prob[valid_idx] = predict_positive_probability(fold_model, X[valid_idx])

    if np.isnan(oof_prob).any():
        raise RuntimeError("Cross-validated probability generation left missing predictions.")
    return oof_prob


def tune_model_hyperparameters(
    model_name: str,
    model_spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    logger: RunLogger,
    seed_offset: int,
    context_label: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    inner_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED + seed_offset)
    inner_splits = list(inner_cv.split(X, y, groups))
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    n_trials = 1 if not model_spec.search_space else CLASSIFICATION_SEARCH_ITERATIONS
    seen_params: set[Tuple[Tuple[str, Any], ...]] = set()
    trial_rows: List[Dict[str, Any]] = []

    for trial_number in range(1, n_trials + 1):
        params = sample_parameter_set(model_spec.search_space, rng) if model_spec.search_space else {}
        params_key = tuple(sorted((key, repr(value)) for key, value in params.items()))
        if params_key in seen_params:
            continue
        seen_params.add(params_key)

        candidate = clone(model_spec.estimator).set_params(**params)
        oof_prob = cross_validated_probabilities(
            candidate,
            X,
            y,
            groups,
            inner_splits,
            use_smote=model_spec.uses_smote,
            seed_offset=seed_offset + trial_number * 10,
        )
        threshold = optimize_threshold(y, oof_prob)
        metrics = classification_metrics_from_probabilities(y, oof_prob, threshold=threshold)
        trial_rows.append(
            {
                "context": context_label,
                "model": model_name,
                "uses_smote": model_spec.uses_smote,
                "trial": len(trial_rows) + 1,
                "threshold": threshold,
                **metrics,
                "params": to_jsonable(params),
            }
        )

    if not trial_rows:
        raise RuntimeError(f"No tuning trials were completed for model '{model_name}'.")

    trials_df = pd.DataFrame(trial_rows)
    trials_df = trials_df.sort_values(["mcc", "pr_auc", "roc_auc", "brier"], ascending=[False, False, False, True]).reset_index(drop=True)
    best_params = dict(trials_df.iloc[0]["params"])
    logger.log(
        f"{context_label}: best {model_name} MCC={trials_df.iloc[0]['mcc']:.3f}, "
        f"PR-AUC={trials_df.iloc[0]['pr_auc']:.3f}, threshold={trials_df.iloc[0]['threshold']:.3f}"
    )
    return best_params, trials_df


def fit_final_model_artifact(
    model_name: str,
    model_spec: ModelSpec,
    best_params: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed_offset: int,
) -> FinalModelArtifact:
    tuned_estimator = clone(model_spec.estimator).set_params(**best_params)
    inner_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED + seed_offset)
    inner_splits = list(inner_cv.split(X, y, groups))
    oof_prob = cross_validated_probabilities(
        tuned_estimator,
        X,
        y,
        groups,
        inner_splits,
        use_smote=model_spec.uses_smote,
        seed_offset=seed_offset,
    )
    calibrator, calibration_method = fit_probability_calibrator(oof_prob, y, method="auto")
    calibrated_oof = apply_probability_calibrator(calibrator, oof_prob, calibration_method)
    threshold = optimize_threshold(y, calibrated_oof)
    oof_metrics = classification_metrics_from_probabilities(y, calibrated_oof, threshold=threshold)

    final_estimator = clone(model_spec.estimator).set_params(**best_params)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    fit_classifier(
        final_estimator,
        X,
        y,
        sample_weight=sample_weight,
        use_smote=model_spec.uses_smote,
        random_state=RANDOM_SEED + seed_offset + 999,
    )
    return FinalModelArtifact(
        name=model_name,
        estimator=final_estimator,
        params=dict(best_params),
        calibrator=calibrator,
        calibration_method=calibration_method,
        threshold=threshold,
        oof_prob=calibrated_oof,
        oof_metrics=oof_metrics,
        uses_smote=model_spec.uses_smote,
    )


def benchmark_classification_representations(
    train_df: pd.DataFrame,
    blocks: Any,
    logger: RunLogger,
) -> Tuple[pd.DataFrame, str]:
    y_train = train_df["label"].to_numpy(dtype=int)
    groups = scaffold_groups(train_df)
    row_ids = train_df["dataset_index"].to_numpy(dtype=int)
    cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows: List[Dict[str, Any]] = []

    for recipe_name in BENCHMARK_REPRESENTATIONS:
        logger.log(f"Benchmarking classification representation '{recipe_name}'")
        X_train, _ = compose_feature_matrix(blocks, recipe_name, row_ids=row_ids)
        fold_metrics: List[Dict[str, float]] = []
        for fold_number, (fit_idx, valid_idx) in enumerate(cv.split(X_train, y_train, groups), start=1):
            model = ExtraTreesClassifier(
                n_estimators=FEATURE_BENCHMARK_TREES,
                random_state=RANDOM_SEED,
                n_jobs=MAX_PARALLEL_WORKERS,
                class_weight="balanced_subsample",
            )
            fold_weights = compute_sample_weight(class_weight="balanced", y=y_train[fit_idx])
            fit_classifier(model, X_train[fit_idx], y_train[fit_idx], sample_weight=fold_weights)
            y_prob = predict_positive_probability(model, X_train[valid_idx])
            threshold = optimize_threshold(y_train[valid_idx], y_prob)
            metrics = classification_metrics_from_probabilities(y_train[valid_idx], y_prob, threshold)
            fold_metrics.append(metrics)
            logger.log(
                f"  Fold {fold_number}/{INNER_CV_FOLDS}: MCC={metrics['mcc']:.3f}, "
                f"PR-AUC={metrics['pr_auc']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}"
            )

        rows.append(
            {
                "task": "classification",
                "representation": recipe_name,
                "mean_mcc": float(np.mean([metric["mcc"] for metric in fold_metrics])),
                "std_mcc": float(np.std([metric["mcc"] for metric in fold_metrics])),
                "mean_pr_auc": float(np.mean([metric["pr_auc"] for metric in fold_metrics])),
                "mean_roc_auc": float(np.mean([metric["roc_auc"] for metric in fold_metrics])),
                "mean_accuracy": float(np.mean([metric["accuracy"] for metric in fold_metrics])),
                "mean_brier": float(np.mean([metric["brier"] for metric in fold_metrics])),
            }
        )

    benchmark_df = pd.DataFrame(rows).sort_values(["mean_mcc", "mean_pr_auc", "mean_roc_auc"], ascending=False).reset_index(drop=True)
    benchmark_df.to_csv(FEATURE_BENCHMARK_PATH, index=False)
    best_representation = str(benchmark_df.iloc[0]["representation"])
    logger.log(f"Best benchmarked representation: {best_representation} (mean MCC={benchmark_df.iloc[0]['mean_mcc']:.3f})")
    return benchmark_df, best_representation


def validate_group_split_integrity(groups: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, context_label: str) -> int:
    train_groups = set(groups[train_idx].astype(str).tolist())
    valid_groups = set(groups[valid_idx].astype(str).tolist())
    overlap_count = len(train_groups & valid_groups)
    if overlap_count:
        raise RuntimeError(f"{context_label} leaked {overlap_count} group(s) across train/validation.")
    return overlap_count


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    row_ids: np.ndarray,
    logger: RunLogger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_specs = classification_model_specs()
    outer_cv = StratifiedGroupKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    tuning_rows: List[pd.DataFrame] = []

    for fold_number, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        group_overlap_count = validate_group_split_integrity(groups, train_idx, valid_idx, context_label=f"outer_fold_{fold_number}")
        logger.log(f"Nested CV outer fold {fold_number}/{OUTER_CV_FOLDS}: train={len(train_idx)}, valid={len(valid_idx)}")
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]
        group_train_fold = groups[train_idx]

        for model_index, (model_name, model_spec) in enumerate(model_specs.items(), start=1):
            seed_offset = fold_number * 1_000 + model_index * 100
            best_params, trials_df = tune_model_hyperparameters(
                model_name=model_name,
                model_spec=model_spec,
                X=X_train_fold,
                y=y_train_fold,
                groups=group_train_fold,
                logger=logger,
                seed_offset=seed_offset,
                context_label=f"outer_fold_{fold_number}",
            )
            tuning_rows.append(trials_df)
            artifact = fit_final_model_artifact(
                model_name=model_name,
                model_spec=model_spec,
                best_params=best_params,
                X=X_train_fold,
                y=y_train_fold,
                groups=group_train_fold,
                seed_offset=seed_offset + 33,
            )
            raw_valid_prob = predict_positive_probability(artifact.estimator, X_valid_fold)
            valid_prob = apply_probability_calibrator(artifact.calibrator, raw_valid_prob, artifact.calibration_method)
            valid_metrics = classification_metrics_from_probabilities(y_valid_fold, valid_prob, threshold=artifact.threshold)
            y_pred = (valid_prob >= artifact.threshold).astype(int)
            fold_rows.append(
                {
                    "outer_fold": fold_number,
                    "model": model_name,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "group_overlap_count": group_overlap_count,
                    "uses_smote": artifact.uses_smote,
                    "threshold": artifact.threshold,
                    "calibration_method": artifact.calibration_method,
                    **valid_metrics,
                    "params": to_jsonable(best_params),
                }
            )

            for dataset_index, true_label, prob_value, pred_value in zip(row_ids[valid_idx], y_valid_fold, valid_prob, y_pred):
                prediction_rows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "outer_fold": fold_number,
                        "model": model_name,
                        "label": int(true_label),
                        "predicted_probability": float(prob_value),
                        "predicted_label": int(pred_value),
                        "threshold": float(artifact.threshold),
                    }
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows), pd.concat(tuning_rows, ignore_index=True)


def summarize_nested_cv_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1", "mcc", "brier", "threshold"]
    grouped = fold_df.groupby("model", as_index=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = ["model"] + [f"{metric}_{agg}" for metric, agg in summary.columns.tolist()[1:]]
    calibration_mode = (
        fold_df.groupby("model")["calibration_method"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("calibration_method")
        .reset_index()
    )
    summary = summary.merge(calibration_mode, on="model", how="left")
    return summary.sort_values(["mcc_mean", "pr_auc_mean", "roc_auc_mean", "brier_mean"], ascending=[False, False, False, True]).reset_index(drop=True)


def select_champion_model(nested_summary_df: pd.DataFrame) -> str:
    return str(nested_summary_df.iloc[0]["model"])


def fit_final_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    logger: RunLogger,
) -> Tuple[Dict[str, FinalModelArtifact], pd.DataFrame]:
    model_specs = classification_model_specs()
    artifacts: Dict[str, FinalModelArtifact] = {}
    tuning_rows: List[pd.DataFrame] = []

    for model_index, (model_name, model_spec) in enumerate(model_specs.items(), start=1):
        seed_offset = 20_000 + model_index * 100
        best_params, trials_df = tune_model_hyperparameters(
            model_name=model_name,
            model_spec=model_spec,
            X=X,
            y=y,
            groups=groups,
            logger=logger,
            seed_offset=seed_offset,
            context_label="final_internal_train",
        )
        tuning_rows.append(trials_df)
        artifacts[model_name] = fit_final_model_artifact(
            model_name=model_name,
            model_spec=model_spec,
            best_params=best_params,
            X=X,
            y=y,
            groups=groups,
            seed_offset=seed_offset + 33,
        )

    return artifacts, pd.concat(tuning_rows, ignore_index=True)


def evaluate_final_models_on_external(
    artifacts: Mapping[str, FinalModelArtifact],
    X_external: np.ndarray,
    y_external: np.ndarray,
    split_name: str,
    logger: RunLogger | None = None,
) -> List[ClassificationResult]:
    results: List[ClassificationResult] = []

    for model_name, artifact in artifacts.items():
        raw_prob = predict_positive_probability(artifact.estimator, X_external)
        calibrated_prob = apply_probability_calibrator(artifact.calibrator, raw_prob, artifact.calibration_method)
        metrics = classification_metrics_from_probabilities(y_external, calibrated_prob, threshold=artifact.threshold)
        y_pred = (calibrated_prob >= artifact.threshold).astype(int)
        results.append(
            ClassificationResult(
                name=model_name,
                split=split_name,
                accuracy=metrics["accuracy"],
                roc_auc=metrics["roc_auc"],
                pr_auc=metrics["pr_auc"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                mcc=metrics["mcc"],
                brier=metrics["brier"],
                threshold=float(artifact.threshold),
                calibration_method=artifact.calibration_method,
                y_prob=calibrated_prob,
                y_pred=y_pred,
            )
        )
        if logger is not None:
            logger.log(
                f"External evaluation {model_name}: MCC={metrics['mcc']:.3f}, "
                f"PR-AUC={metrics['pr_auc']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}"
            )

    return sorted(results, key=lambda item: (item.mcc, item.pr_auc, item.roc_auc), reverse=True)
