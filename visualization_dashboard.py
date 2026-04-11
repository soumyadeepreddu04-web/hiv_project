"""Interactive dashboard for the HIV-1 protease ML pipeline."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score


APP_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = APP_DIR / "drug_interaction_ml"
DATA_DIR = APP_DIR

CLASS_METRICS = DATA_DIR / "classification_metrics_v3.csv"
REGRESSION_METRICS = DATA_DIR / "regression_metrics_v3.csv"
CLASS_PRED = DATA_DIR / "classification_predictions_v3.csv"
REGRESSION_PLOT_PATH = DATA_DIR / "regression_predictions.png"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.png"


@st.cache_resource
def load_pipeline_module() -> ModuleType:
    loader = SourceFileLoader("drug_interaction_ml_runtime", str(PIPELINE_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader, origin=str(PIPELINE_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pipeline module from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


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
    st.subheader("Classification Summary")
    st.dataframe(load_csv(CLASS_METRICS))
else:
    st.warning("Classification metrics file not found.")

if REGRESSION_METRICS.exists():
    st.subheader("Regression Summary")
    st.dataframe(load_csv(REGRESSION_METRICS))
else:
    st.warning("Regression metrics file not found.")

if CLASS_PRED.exists():
    pred_df = load_csv(CLASS_PRED)
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
    st.image(str(REGRESSION_PLOT_PATH), caption="Regression Predictions", use_container_width=True)
else:
    st.warning("Regression predictions image not found.")

st.subheader("Feature Importance")
if FEATURE_IMPORTANCE_PATH.exists():
    st.image(str(FEATURE_IMPORTANCE_PATH), caption="Feature Importance", use_container_width=True)
else:
    st.info("Feature importance image not available.")

st.caption("Dashboard powered by Streamlit and Plotly. Generated from the pipeline artefacts in the repository.")
