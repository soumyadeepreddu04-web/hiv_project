from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import DataStructs
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from dti_pipeline.config import (
    ACTIVITY_CLIFF_DELTA_PCHEMBL,
    ACTIVITY_CLIFF_PATH,
    ACTIVITY_CLIFF_SIMILARITY,
    AD_SIMILARITY_PERCENTILE,
    CLASSIFICATION_ERROR_ANALYSIS_PATH,
    CLASSIFICATION_ERROR_SUMMARY_PATH,
    CLASSIFICATION_LEAKAGE_REPORT_PATH,
    IMPORTANCE_PLOT_PATH,
    NEAR_DUPLICATE_MAX_EXAMPLES,
    NEAR_DUPLICATE_TANIMOTO_THRESHOLD,
    PR_PLOT_PATH,
    ROC_PLOT_PATH,
    TOP_FEATURES_TO_PLOT,
    ClassificationResult,
    FeatureBlocks,
    RunLogger,
    RuntimeInfo,
    to_jsonable,
)
from dti_pipeline.training import FinalModelArtifact


def plot_roc_curves(results: Sequence[ClassificationResult], y_true: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for result in results:
        fpr, tpr, _ = roc_curve(y_true, result.y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{result.name} (AUC = {result.roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: HIV-1 Protease Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curves_interactive(results: Sequence[ClassificationResult], y_true: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for result in results:
        fpr, tpr, _ = roc_curve(y_true, result.y_prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{result.name} (AUC={result.roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random baseline"))
    fig.update_layout(title="ROC Curve: HIV-1 Protease Classification", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def plot_precision_recall_curves(results: Sequence[ClassificationResult], y_true: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    baseline = float(np.mean(y_true))
    for result in results:
        precision, recall, _ = precision_recall_curve(y_true, result.y_prob)
        plt.plot(recall, precision, linewidth=2, label=f"{result.name} (AP = {result.pr_auc:.3f})")
    plt.axhline(baseline, linestyle="--", color="gray", label=f"Positive rate = {baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: HIV-1 Protease Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_recall_curves_interactive(results: Sequence[ClassificationResult], y_true: np.ndarray) -> go.Figure:
    fig = go.Figure()
    baseline = float(np.mean(y_true))
    for result in results:
        precision, recall, _ = precision_recall_curve(y_true, result.y_prob)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{result.name} (AP={result.pr_auc:.3f})"))
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray", annotation_text=f"Positive rate = {baseline:.3f}")
    fig.update_layout(title="Precision-Recall Curve: HIV-1 Protease Classification", xaxis_title="Recall", yaxis_title="Precision")
    return fig


def unwrap_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "named_steps"):
        last_step_name = list(estimator.named_steps.keys())[-1]
        return estimator.named_steps[last_step_name]
    return estimator


def plot_feature_importance(model: Any, feature_names: Sequence[str], output_path: Path, top_n: int = TOP_FEATURES_TO_PLOT) -> None:
    final_model = unwrap_estimator(model)
    importances = np.asarray(final_model.feature_importances_)
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[index] for index in top_indices]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_indices)), importances[top_indices], color="steelblue")
    plt.xticks(range(len(top_indices)), top_names, rotation=65, ha="right")
    plt.ylabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance_interactive(model: Any, feature_names: Sequence[str], top_n: int = TOP_FEATURES_TO_PLOT) -> go.Figure:
    final_model = unwrap_estimator(model)
    importances = np.asarray(final_model.feature_importances_)
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[index] for index in top_indices]
    top_vals = importances[top_indices]
    fig = go.Figure(go.Bar(x=top_names, y=top_vals, marker_color="steelblue"))
    fig.update_layout(title="Top Feature Importances", xaxis_title="Feature", yaxis_title="Importance")
    return fig


def select_feature_importance_model(
    models: Mapping[str, FinalModelArtifact],
    results: Sequence[ClassificationResult],
) -> Tuple[Any, str]:
    compatible = {
        name: artifact.estimator
        for name, artifact in models.items()
        if hasattr(unwrap_estimator(artifact.estimator), "feature_importances_")
    }
    if not compatible:
        raise ValueError("No fitted classification model exposes feature_importances_.")

    ranked = sorted(results, key=lambda result: (result.mcc, result.pr_auc, result.roc_auc), reverse=True)
    for result in ranked:
        if result.name in compatible:
            return compatible[result.name], result.name

    first_name = next(iter(compatible))
    return compatible[first_name], first_name


def compute_applicability_domain(
    train_row_ids: np.ndarray,
    test_row_ids: np.ndarray,
    morgan_fingerprints: Sequence[Any],
    logger: RunLogger,
) -> Tuple[np.ndarray, float, np.ndarray]:
    train_fps = [morgan_fingerprints[int(index)] for index in train_row_ids]
    test_fps = [morgan_fingerprints[int(index)] for index in test_row_ids]
    train_neighbor_similarity = np.zeros(len(train_fps), dtype=np.float32)

    with logger.section("Estimating applicability-domain threshold"):
        for index, fp in enumerate(train_fps):
            similarities = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
            if len(similarities) > 1:
                similarities[index] = -1.0
                train_neighbor_similarity[index] = max(similarities)
            else:
                train_neighbor_similarity[index] = 1.0
            if index + 1 == len(train_fps) or (index + 1) % 250 == 0:
                logger.log(f"Processed {index + 1}/{len(train_fps)} training fingerprints for AD thresholding.")

    threshold = float(np.percentile(train_neighbor_similarity, AD_SIMILARITY_PERCENTILE))
    logger.log(f"Applicability-domain threshold set at Tanimoto similarity {threshold:.3f}")
    test_similarity = np.zeros(len(test_fps), dtype=np.float32)

    with logger.section("Scoring external molecules against the training chemical space"):
        for index, fp in enumerate(test_fps):
            similarities = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            test_similarity[index] = max(similarities) if similarities else 0.0
            if index + 1 == len(test_fps) or (index + 1) % 250 == 0:
                logger.log(f"Processed {index + 1}/{len(test_fps)} external fingerprints for AD scoring.")

    inside_ad = test_similarity >= threshold
    return train_neighbor_similarity, threshold, inside_ad


def find_activity_cliffs(df: pd.DataFrame, morgan_fingerprints: Sequence[Any], logger: RunLogger) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    y = df["median_pchembl"].to_numpy(dtype=float)
    labels = df["label"].to_numpy()

    with logger.section("Searching for activity cliffs"):
        for left_index in range(len(df) - 1):
            similarities = DataStructs.BulkTanimotoSimilarity(morgan_fingerprints[left_index], morgan_fingerprints[left_index + 1 :])
            for offset, similarity in enumerate(similarities, start=1):
                if similarity < ACTIVITY_CLIFF_SIMILARITY:
                    continue
                right_index = left_index + offset
                delta_pchembl = abs(y[left_index] - y[right_index])
                if delta_pchembl < ACTIVITY_CLIFF_DELTA_PCHEMBL:
                    continue
                records.append(
                    {
                        "left_compound": df.iloc[left_index]["compound_name"],
                        "right_compound": df.iloc[right_index]["compound_name"],
                        "left_smiles": df.iloc[left_index]["smiles"],
                        "right_smiles": df.iloc[right_index]["smiles"],
                        "tanimoto_similarity": float(similarity),
                        "left_pchembl": float(y[left_index]),
                        "right_pchembl": float(y[right_index]),
                        "delta_pchembl": float(delta_pchembl),
                        "left_label": labels[left_index],
                        "right_label": labels[right_index],
                    }
                )
            if left_index + 1 == len(df) - 1 or (left_index + 1) % 250 == 0:
                logger.log(f"Activity-cliff scan progress: {left_index + 1}/{len(df) - 1} anchors processed.")

    if not records:
        logger.log("No activity cliffs passed the configured thresholds.")
        return pd.DataFrame(
            columns=[
                "left_compound",
                "right_compound",
                "left_smiles",
                "right_smiles",
                "tanimoto_similarity",
                "left_pchembl",
                "right_pchembl",
                "delta_pchembl",
                "left_label",
                "right_label",
            ]
        )

    cliffs_df = pd.DataFrame(records).sort_values(["delta_pchembl", "tanimoto_similarity"], ascending=[False, False]).reset_index(drop=True)
    logger.log(f"Detected {len(cliffs_df)} activity cliffs above the similarity and potency-delta thresholds.")
    return cliffs_df


def exact_overlap_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    train_ids = set(train_df["dataset_index"].astype(int).tolist())
    test_ids = set(test_df["dataset_index"].astype(int).tolist())
    train_smiles = set(train_df["canonical_smiles"].astype(str).tolist())
    test_smiles = set(test_df["canonical_smiles"].astype(str).tolist())
    train_scaffolds = set(train_df["scaffold"].fillna(train_df["canonical_smiles"]).astype(str).tolist())
    test_scaffolds = set(test_df["scaffold"].fillna(test_df["canonical_smiles"]).astype(str).tolist())
    return {
        "dataset_index_overlap": sorted(train_ids & test_ids),
        "canonical_smiles_overlap_count": int(len(train_smiles & test_smiles)),
        "scaffold_overlap_count": int(len(train_scaffolds & test_scaffolds)),
    }


def near_duplicate_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_blocks: FeatureBlocks,
    logger: RunLogger,
    threshold: float = NEAR_DUPLICATE_TANIMOTO_THRESHOLD,
    max_examples: int = NEAR_DUPLICATE_MAX_EXAMPLES,
) -> Dict[str, Any]:
    train_ids = train_df["dataset_index"].to_numpy(dtype=int)
    test_ids = test_df["dataset_index"].to_numpy(dtype=int)
    train_fps = [feature_blocks.morgan_fingerprints[int(index)] for index in train_ids]
    test_fps = [feature_blocks.morgan_fingerprints[int(index)] for index in test_ids]

    examples: List[Dict[str, Any]] = []
    count = 0
    max_similarity = 0.0

    with logger.section("Checking train/external near-duplicates"):
        for row_number, (dataset_index, fp) in enumerate(zip(test_ids, test_fps), start=1):
            similarities = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            if not similarities:
                continue
            best_position = int(np.argmax(similarities))
            best_similarity = float(similarities[best_position])
            max_similarity = max(max_similarity, best_similarity)
            if best_similarity >= threshold:
                count += 1
                if len(examples) < max_examples:
                    train_row = train_df.iloc[best_position]
                    test_row = test_df.loc[test_df["dataset_index"] == dataset_index].iloc[0]
                    examples.append(
                        {
                            "test_dataset_index": int(dataset_index),
                            "train_dataset_index": int(train_row["dataset_index"]),
                            "tanimoto_similarity": best_similarity,
                            "test_smiles": test_row["canonical_smiles"],
                            "train_smiles": train_row["canonical_smiles"],
                            "test_scaffold": test_row["scaffold"],
                            "train_scaffold": train_row["scaffold"],
                        }
                    )
            if row_number == len(test_ids) or row_number % 250 == 0:
                logger.log(f"Near-duplicate scan progress: {row_number}/{len(test_ids)} external molecules checked.")

    return {
        "threshold": float(threshold),
        "count_at_or_above_threshold": int(count),
        "max_similarity": float(max_similarity),
        "examples": examples,
    }


def build_leakage_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_blocks: FeatureBlocks,
    logger: RunLogger,
) -> Dict[str, Any]:
    overlap = exact_overlap_report(train_df, test_df)
    near_duplicates = near_duplicate_report(train_df, test_df, feature_blocks, logger)
    report = {
        "scaffold_split_integrity_passed": bool(
            overlap["dataset_index_overlap"] == []
            and overlap["canonical_smiles_overlap_count"] == 0
            and overlap["scaffold_overlap_count"] == 0
        ),
        "exact_overlap": overlap,
        "near_duplicates": near_duplicates,
    }
    return report


def build_prediction_frame(
    external_df: pd.DataFrame,
    external_results: Sequence[ClassificationResult],
    champion_model: str,
    inside_ad: np.ndarray,
) -> pd.DataFrame:
    prediction_df = external_df.copy().reset_index(drop=True)
    prediction_df["inside_applicability_domain"] = inside_ad.astype(bool)

    champion_result = next(result for result in external_results if result.name == champion_model)
    prediction_df["predicted_probability"] = champion_result.y_prob
    prediction_df["predicted_label"] = champion_result.y_pred
    prediction_df["selected_model"] = champion_model
    prediction_df["error_type"] = np.where(
        (prediction_df["label"] == 1) & (prediction_df["predicted_label"] == 0),
        "false_negative",
        np.where(
            (prediction_df["label"] == 0) & (prediction_df["predicted_label"] == 1),
            "false_positive",
            np.where(prediction_df["label"] == 1, "true_positive", "true_negative"),
        ),
    )

    for result in external_results:
        prediction_df[f"prob_{result.name}"] = result.y_prob
        prediction_df[f"pred_{result.name}"] = result.y_pred
        prediction_df[f"threshold_{result.name}"] = result.threshold

    return prediction_df


def build_error_analysis(prediction_df: pd.DataFrame, champion_model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    row_level = prediction_df.copy()
    row_level["confidence"] = np.where(
        row_level["predicted_label"] == 1,
        row_level["predicted_probability"],
        1.0 - row_level["predicted_probability"],
    )
    error_summary = (
        row_level.groupby(["error_type", "inside_applicability_domain"], dropna=False)
        .agg(
            count=("dataset_index", "size"),
            mean_probability=("predicted_probability", "mean"),
            mean_pchembl=("median_pchembl", "mean"),
        )
        .reset_index()
    )
    error_summary["model"] = champion_model
    row_level.to_csv(CLASSIFICATION_ERROR_ANALYSIS_PATH, index=False)
    error_summary.to_csv(CLASSIFICATION_ERROR_SUMMARY_PATH, index=False)
    return row_level, error_summary


def classification_results_to_frame(
    nested_summary_df: pd.DataFrame,
    external_results: Sequence[ClassificationResult],
    champion_model: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in nested_summary_df.iterrows():
        rows.append(
            {
                "stage": "nested_cv_summary",
                "model": row["model"],
                "champion_model": bool(row["model"] == champion_model),
                "primary_metric": "mcc",
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "roc_auc_mean": row["roc_auc_mean"],
                "roc_auc_std": row["roc_auc_std"],
                "pr_auc_mean": row["pr_auc_mean"],
                "pr_auc_std": row["pr_auc_std"],
                "precision_mean": row["precision_mean"],
                "precision_std": row["precision_std"],
                "recall_mean": row["recall_mean"],
                "recall_std": row["recall_std"],
                "f1_mean": row["f1_mean"],
                "f1_std": row["f1_std"],
                "mcc_mean": row["mcc_mean"],
                "mcc_std": row["mcc_std"],
                "brier_mean": row["brier_mean"],
                "brier_std": row["brier_std"],
                "threshold_mean": row["threshold_mean"],
                "threshold_std": row["threshold_std"],
                "calibration_method": row["calibration_method"],
            }
        )

    for result in external_results:
        rows.append(
            {
                "stage": "external_validation",
                "model": result.name,
                "champion_model": bool(result.name == champion_model),
                "primary_metric": "mcc",
                "accuracy": result.accuracy,
                "roc_auc": result.roc_auc,
                "pr_auc": result.pr_auc,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "mcc": result.mcc,
                "brier": result.brier,
                "threshold": result.threshold,
                "calibration_method": result.calibration_method,
                "split": result.split,
            }
        )

    return pd.DataFrame(rows)


def probe_display_adapters() -> List[str]:
    try:
        probe = subprocess.run(
            ["pnputil", "/enum-devices", "/class", "Display"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    if probe.returncode != 0 or not probe.stdout:
        return []

    adapter_names: List[str] = []
    for line in probe.stdout.splitlines():
        if line.strip().startswith("Device Description:"):
            _, _, value = line.partition(":")
            name = value.strip()
            if name:
                adapter_names.append(name)
    return adapter_names


def detect_runtime() -> RuntimeInfo:
    adapters = probe_display_adapters()
    amd_gpu = next((name for name in adapters if ("amd" in name.lower() or "radeon" in name.lower())), "")
    if amd_gpu:
        message = (
            f"AMD GPU detected ({amd_gpu}), but this pipeline runs on CPU-first sklearn/RDKit components. "
            "No GPU-only training path is used."
        )
        return RuntimeInfo(backend="cpu", adapters=adapters, gpu_name=amd_gpu, message=message)

    if adapters:
        return RuntimeInfo(
            backend="cpu",
            adapters=adapters,
            gpu_name="",
            message=f"Display adapters detected ({', '.join(adapters)}). Running on CPU.",
        )

    return RuntimeInfo(
        backend="cpu",
        adapters=[],
        gpu_name="",
        message="No display adapters were detected through pnputil. Running on CPU.",
    )


def print_summary(
    runtime: RuntimeInfo,
    curated_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    internal_train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    representation: str,
    nested_summary_df: pd.DataFrame,
    external_results: Sequence[ClassificationResult],
    champion_model: str,
    external_strategy: str,
    leakage_report: Mapping[str, Any],
    ad_threshold: float,
    inside_ad_fraction: float,
    activity_cliffs_count: int,
) -> None:
    print("\n=== Runtime ===")
    print(runtime.message)

    print("\n=== Dataset Summary ===")
    print(f"Curated molecules: {len(curated_df)}")
    print(f"Classification molecules: {len(classification_df)}")
    print(f"Nested internal train: {len(internal_train_df)}")
    print(f"External validation: {len(external_df)}")
    print(f"External split strategy: {external_strategy}")

    print("\n=== Features ===")
    print(f"Representation used for nested CV + external validation: {representation}")
    print("Primary model-selection metric: MCC")

    print("\n=== Nested CV (mean +/- std) ===")
    for _, row in nested_summary_df.iterrows():
        print(
            f"{row['model']}: MCC={row['mcc_mean']:.3f} +/- {row['mcc_std']:.3f}, "
            f"PR-AUC={row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f}, "
            f"ROC-AUC={row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f}"
        )

    print("\n=== External Validation ===")
    for result in external_results:
        champion_marker = " [champion]" if result.name == champion_model else ""
        print(
            f"{result.name}{champion_marker}: MCC={result.mcc:.3f}, PR-AUC={result.pr_auc:.3f}, "
            f"ROC-AUC={result.roc_auc:.3f}, threshold={result.threshold:.3f}, calibration={result.calibration_method}"
        )

    print("\n=== Leakage Checks ===")
    print(f"Scaffold split integrity passed: {leakage_report['scaffold_split_integrity_passed']}")
    print(f"Exact canonical-SMILES overlap count: {leakage_report['exact_overlap']['canonical_smiles_overlap_count']}")
    print(f"Scaffold overlap count: {leakage_report['exact_overlap']['scaffold_overlap_count']}")
    print(
        f"Near-duplicates >= {leakage_report['near_duplicates']['threshold']:.2f}: "
        f"{leakage_report['near_duplicates']['count_at_or_above_threshold']}"
    )

    print("\n=== Advanced Analyses ===")
    print(f"Applicability-domain similarity threshold: {ad_threshold:.3f}")
    print(f"External molecules inside AD: {inside_ad_fraction:.1%}")
    print(f"Activity cliffs found: {activity_cliffs_count}")

    print("\nSaved files:")
    for path in [
        ROC_PLOT_PATH,
        PR_PLOT_PATH,
        IMPORTANCE_PLOT_PATH,
        CLASSIFICATION_ERROR_ANALYSIS_PATH,
        CLASSIFICATION_ERROR_SUMMARY_PATH,
        CLASSIFICATION_LEAKAGE_REPORT_PATH,
        ACTIVITY_CLIFF_PATH,
    ]:
        print(f"- {path.name}")
