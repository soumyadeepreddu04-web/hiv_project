"""Interactive dashboard for the HIV-1 protease ML pipeline."""

from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


APP_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = APP_DIR / "drug_interaction_ml"
DATA_DIR = APP_DIR

CLASS_METRICS = DATA_DIR / "classification_metrics_v3.csv"
CLASS_CV_RESULTS = DATA_DIR / "classification_cv_results_v3.csv"
CLASS_PRED = DATA_DIR / "classification_predictions_v3.csv"
CLASS_ERROR_SUMMARY = DATA_DIR / "classification_error_summary_v3.csv"
CLASS_LEAKAGE = DATA_DIR / "classification_leakage_report_v3.json"
CLASS_CONFIG = DATA_DIR / "classification_run_config_v3.json"
REGRESSION_METRICS = DATA_DIR / "regression_metrics_v3.csv"
REGRESSION_PLOT_PATH = DATA_DIR / "regression_predictions.png"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.png"


def load_pipeline_module():
    return SourceFileLoader("drug_interaction_ml_runtime", str(PIPELINE_PATH)).load_module()


pipeline = load_pipeline_module()
plot_roc_curves_interactive = pipeline.plot_roc_curves_interactive
plot_precision_recall_curves_interactive = pipeline.plot_precision_recall_curves_interactive


class ClassificationResult:
    def __init__(self, name: str, y_prob, roc_auc: float, pr_auc: float) -> None:
        self.name = name
        self.y_prob = y_prob
        self.roc_auc = roc_auc
        self.pr_auc = pr_auc


def build_classification_results(pred_df: pd.DataFrame):
    if "label" not in pred_df.columns:
        return None, []

    y_true = pred_df["label"].to_numpy()
    probability_columns = [col for col in pred_df.columns if col.startswith("prob_")]
    results = []

    for col in probability_columns:
        probs = pred_df[col].to_numpy()
        results.append(
            ClassificationResult(
                name=col.removeprefix("prob_"),
                y_prob=probs,
                roc_auc=float(roc_auc_score(y_true, probs)),
                pr_auc=float(average_precision_score(y_true, probs)),
            )
        )

    if not results and "predicted_probability" in pred_df.columns:
        probs = pred_df["predicted_probability"].to_numpy()
        results.append(
            ClassificationResult(
                name="Best Model",
                y_prob=probs,
                roc_auc=float(roc_auc_score(y_true, probs)),
                pr_auc=float(average_precision_score(y_true, probs)),
            )
        )

    return y_true, results


st.title("HIV-1 Protease ML Pipeline Dashboard")

if CLASS_METRICS.exists():
    metrics_df = pd.read_csv(CLASS_METRICS)
    nested_df = metrics_df[metrics_df["stage"] == "nested_cv_summary"].copy()
    external_df = metrics_df[metrics_df["stage"] == "external_validation"].copy()

    if not nested_df.empty:
        st.subheader("Classification Nested CV Summary")
        keep_cols = [
            "model",
            "champion_model",
            "primary_metric",
            "mcc_mean",
            "mcc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "roc_auc_mean",
            "roc_auc_std",
            "calibration_method",
        ]
        keep_cols = [col for col in keep_cols if col in nested_df.columns]
        st.dataframe(nested_df[keep_cols].sort_values(["champion_model", "mcc_mean"], ascending=[False, False]), use_container_width=True)

    if not external_df.empty:
        st.subheader("Classification Scaffold External Validation")
        keep_cols = [
            "model",
            "champion_model",
            "primary_metric",
            "mcc",
            "pr_auc",
            "roc_auc",
            "threshold",
            "calibration_method",
            "split",
        ]
        keep_cols = [col for col in keep_cols if col in external_df.columns]
        st.dataframe(external_df[keep_cols].sort_values(["champion_model", "mcc"], ascending=[False, False]), use_container_width=True)
else:
    st.warning("Classification metrics file not found.")

if CLASS_CV_RESULTS.exists():
    st.subheader("Fold-Level Classification Results")
    cv_df = pd.read_csv(CLASS_CV_RESULTS)
    keep_cols = [
        "outer_fold",
        "model",
        "uses_smote",
        "group_overlap_count",
        "mcc",
        "pr_auc",
        "roc_auc",
        "threshold",
    ]
    keep_cols = [col for col in keep_cols if col in cv_df.columns]
    st.dataframe(cv_df[keep_cols].sort_values(["outer_fold", "mcc"], ascending=[True, False]), use_container_width=True)

if CLASS_CONFIG.exists():
    st.subheader("Classification Run Config")
    st.json(json.loads(CLASS_CONFIG.read_text(encoding="utf-8")))

if CLASS_LEAKAGE.exists():
    st.subheader("Leakage Check")
    st.json(json.loads(CLASS_LEAKAGE.read_text(encoding="utf-8")))

if CLASS_ERROR_SUMMARY.exists():
    st.subheader("Error Summary")
    st.dataframe(pd.read_csv(CLASS_ERROR_SUMMARY), use_container_width=True)

if REGRESSION_METRICS.exists():
    st.subheader("Regression Summary")
    st.dataframe(pd.read_csv(REGRESSION_METRICS), use_container_width=True)
else:
    st.warning("Regression metrics file not found.")

if CLASS_PRED.exists():
    pred_df = pd.read_csv(CLASS_PRED)
    y_true, results = build_classification_results(pred_df)
    if y_true is not None and results:
        st.subheader("ROC Curves")
        st.plotly_chart(plot_roc_curves_interactive(results, y_true), use_container_width=True)

        st.subheader("Precision-Recall Curves")
        st.plotly_chart(plot_precision_recall_curves_interactive(results, y_true), use_container_width=True)
    else:
        st.warning("Classification predictions file is missing the probability columns needed for plots.")
else:
    st.warning("Classification predictions file not found.")

if REGRESSION_PLOT_PATH.exists():
    st.subheader("Regression Predictions")
    st.image(Image.open(REGRESSION_PLOT_PATH), caption="Regression Predictions", use_container_width=True)
else:
    st.warning("Regression predictions image not found.")

st.subheader("Feature Importance")
if FEATURE_IMPORTANCE_PATH.exists():
    st.image(Image.open(FEATURE_IMPORTANCE_PATH), caption="Feature Importance", use_container_width=True)
else:
    st.info("Feature importance image not available.")

st.caption("Dashboard powered by Streamlit and Plotly. Generated from scaffold-group CV and scaffold holdout artefacts in the repository.")
