"""
Rerun ONLY the classification predictions part of the pipeline.

This script reuses all cached data (dataset CSV, train/test splits,
feature benchmarks) and reruns:
  - Classification hyperparameter tuning (with checkpoint resume)
  - Stacked ensemble training
  - SMOTE Extra Trees training
  - Scaffold + temporal evaluation
  - ROC / PR / feature-importance plots
  - Applicability-domain scoring
  - Classification predictions CSV
  - Activity cliffs
  - Metrics CSV + report JSON update

It does NOT rerun data fetching, featurization, or regression.
"""
from __future__ import annotations

import json
import traceback
import importlib.machinery
import importlib.util
import pathlib
import sys

import numpy as np

# Import everything we need from the main pipeline module.
# The file has no .py extension, so we import it via importlib.

_file_path = str(pathlib.Path(__file__).resolve().parent / "drug_interaction_ml")
_loader = importlib.machinery.SourceFileLoader("drug_interaction_ml", _file_path)
_spec = importlib.util.spec_from_loader("drug_interaction_ml", _loader, origin=_file_path)
dti = importlib.util.module_from_spec(_spec)
dti.__file__ = _file_path
sys.modules["drug_interaction_ml"] = dti
_spec.loader.exec_module(dti)

# ── Aliases for readability ────────────────────────────────────────────
RunLogger                           = dti.RunLogger
detect_runtime                      = dti.detect_runtime
load_or_build_dataset               = dti.load_or_build_dataset
build_classification_dataset        = dti.build_classification_dataset
featurize_dataset                   = dti.featurize_dataset
split_classification_dataset        = dti.split_classification_dataset
build_temporal_split                = dti.build_temporal_split
benchmark_classification_representations = dti.benchmark_classification_representations
compose_feature_matrix              = dti.compose_feature_matrix
scaffold_groups                     = dti.scaffold_groups
tune_classification_models          = dti.tune_classification_models
train_group_aware_stacking_classifier = dti.train_group_aware_stacking_classifier
fit_classification_model_by_name    = dti.fit_classification_model_by_name
evaluate_classification_models      = dti.evaluate_classification_models
classification_results_to_frame     = dti.classification_results_to_frame
plot_roc_curves                     = dti.plot_roc_curves
plot_precision_recall_curves        = dti.plot_precision_recall_curves
select_feature_importance_model     = dti.select_feature_importance_model
plot_feature_importance             = dti.plot_feature_importance
compute_applicability_domain        = dti.compute_applicability_domain
find_activity_cliffs                = dti.find_activity_cliffs

# Paths
RUN_LOG_PATH                        = dti.RUN_LOG_PATH
CLASSIFICATION_CV_RESULTS_PATH      = dti.CLASSIFICATION_CV_RESULTS_PATH
CLASSIFICATION_METRICS_PATH         = dti.CLASSIFICATION_METRICS_PATH
CLASSIFICATION_PREDICTIONS_PATH     = dti.CLASSIFICATION_PREDICTIONS_PATH
ROC_PLOT_PATH                       = dti.ROC_PLOT_PATH
PR_PLOT_PATH                        = dti.PR_PLOT_PATH
IMPORTANCE_PLOT_PATH                = dti.IMPORTANCE_PLOT_PATH
ACTIVITY_CLIFF_PATH                 = dti.ACTIVITY_CLIFF_PATH
RUN_REPORT_PATH                     = dti.RUN_REPORT_PATH
TEMPORAL_CLASSIFICATION_TRAIN_PATH  = dti.TEMPORAL_CLASSIFICATION_TRAIN_PATH
TEMPORAL_CLASSIFICATION_TEST_PATH   = dti.TEMPORAL_CLASSIFICATION_TEST_PATH


def main() -> None:
    logger = RunLogger(RUN_LOG_PATH)
    runtime = detect_runtime()
    logger.log("=== RERUN: Classification predictions only ===")
    logger.log(runtime.message)

    # ── 1. Reload cached data (no re-fetching) ─────────────────────────
    print("Loading cached dataset …")
    curated_df = load_or_build_dataset(logger)
    classification_df = build_classification_dataset(curated_df, logger)
    feature_blocks = featurize_dataset(curated_df, logger)

    classification_train_df, classification_test_df = split_classification_dataset(
        classification_df, logger
    )
    classification_temporal_split = build_temporal_split(
        classification_df,
        TEMPORAL_CLASSIFICATION_TRAIN_PATH,
        TEMPORAL_CLASSIFICATION_TEST_PATH,
        logger,
        require_binary_labels=True,
    )

    # ── 2. Feature-representation benchmark (uses checkpoint) ──────────
    print("Benchmarking classification representations …")
    _, best_classification_representation = benchmark_classification_representations(
        classification_train_df, feature_blocks, logger
    )

    # ── 3. Build feature matrices ──────────────────────────────────────
    classification_train_row_ids = classification_train_df["dataset_index"].to_numpy(dtype=int)
    classification_test_row_ids  = classification_test_df["dataset_index"].to_numpy(dtype=int)

    classification_X_train, classification_feature_names = compose_feature_matrix(
        feature_blocks, best_classification_representation, row_ids=classification_train_row_ids
    )
    classification_X_test, _ = compose_feature_matrix(
        feature_blocks, best_classification_representation, row_ids=classification_test_row_ids
    )
    y_class_train = classification_train_df["label"].to_numpy(dtype=int)
    y_class_test  = classification_test_df["label"].to_numpy(dtype=int)
    class_groups_train = scaffold_groups(classification_train_df)

    # ── 4. Hyperparameter search (resumes from checkpoint) ─────────────
    print("Tuning classification models …")
    with logger.section("Hyperparameter search for classification models"):
        tuned_classification_models, classification_cv_results = tune_classification_models(
            classification_X_train, y_class_train, class_groups_train, logger
        )
    classification_cv_results.to_csv(CLASSIFICATION_CV_RESULTS_PATH, index=False)

    # ── 5. Stacked ensemble + SMOTE baseline ───────────────────────────
    print("Training stacked ensemble …")
    with logger.section("Training stacked classification ensemble"):
        stacked_classifier = train_group_aware_stacking_classifier(
            {
                name: tuned_classification_models[name]
                for name in ("Random Forest", "Extra Trees", "HistGradientBoosting")
                if name in tuned_classification_models
            },
            classification_X_train,
            y_class_train,
            class_groups_train,
            logger,
        )

    with logger.section("Training SMOTE-enhanced Extra Trees baseline"):
        smote_extra_trees = fit_classification_model_by_name(
            "Extra Trees + SMOTE",
            tuned_classification_models,
            classification_X_train,
            y_class_train,
            class_groups_train,
            logger,
        )

    scaffold_classification_models = dict(tuned_classification_models)
    scaffold_classification_models["Stacked Ensemble"] = stacked_classifier
    scaffold_classification_models["Extra Trees + SMOTE"] = smote_extra_trees

    # ── 6. Evaluate on scaffold holdout ────────────────────────────────
    print("Evaluating on scaffold holdout …")
    with logger.section("Evaluating classification models on scaffold holdout"):
        scaffold_classification_results = evaluate_classification_models(
            scaffold_classification_models,
            classification_X_test,
            y_class_test,
            split_name="scaffold",
            logger=logger,
        )

    classification_results = list(scaffold_classification_results)
    best_scaffold_classifier = max(
        scaffold_classification_results, key=lambda r: (r.mcc, r.pr_auc)
    )
    logger.log(
        f"Best scaffold-holdout classifier: {best_scaffold_classifier.name} "
        f"(MCC={best_scaffold_classifier.mcc:.3f}, PR-AUC={best_scaffold_classifier.pr_auc:.3f})"
    )

    # ── 7. Temporal split evaluation ───────────────────────────────────
    if classification_temporal_split is not None:
        temporal_train_df, temporal_test_df, _ = classification_temporal_split
        temporal_train_ids = temporal_train_df["dataset_index"].to_numpy(dtype=int)
        temporal_test_ids  = temporal_test_df["dataset_index"].to_numpy(dtype=int)
        temporal_X_train, _ = compose_feature_matrix(
            feature_blocks, best_classification_representation, row_ids=temporal_train_ids
        )
        temporal_X_test, _ = compose_feature_matrix(
            feature_blocks, best_classification_representation, row_ids=temporal_test_ids
        )
        temporal_y_train = temporal_train_df["label"].to_numpy(dtype=int)
        temporal_y_test  = temporal_test_df["label"].to_numpy(dtype=int)
        temporal_groups  = scaffold_groups(temporal_train_df)

        with logger.section("Evaluating best classifier on temporal split"):
            temporal_model = fit_classification_model_by_name(
                best_scaffold_classifier.name,
                tuned_classification_models,
                temporal_X_train,
                temporal_y_train,
                temporal_groups,
                logger,
            )
            classification_results.extend(
                evaluate_classification_models(
                    {best_scaffold_classifier.name: temporal_model},
                    temporal_X_test,
                    temporal_y_test,
                    split_name="temporal",
                    logger=logger,
                )
            )

    # ── 8. Metrics CSV ─────────────────────────────────────────────────
    classification_metrics_df = classification_results_to_frame(classification_results)
    classification_metrics_df.to_csv(CLASSIFICATION_METRICS_PATH, index=False)
    print(f"\nClassification metrics saved → {CLASSIFICATION_METRICS_PATH.name}")
    print(classification_metrics_df.to_string(index=False))

    # ── 9. Plots ───────────────────────────────────────────────────────
    plot_roc_curves(scaffold_classification_results, y_class_test, ROC_PLOT_PATH)
    plot_precision_recall_curves(scaffold_classification_results, y_class_test, PR_PLOT_PATH)
    importance_model, importance_model_name = select_feature_importance_model(
        scaffold_classification_models, scaffold_classification_results
    )
    plot_feature_importance(importance_model, classification_feature_names, IMPORTANCE_PLOT_PATH)
    logger.log(f"Feature-importance plot written using {importance_model_name}.")
    print(f"Plots saved → {ROC_PLOT_PATH.name}, {PR_PLOT_PATH.name}, {IMPORTANCE_PLOT_PATH.name}")

    # ── 10. Applicability domain ───────────────────────────────────────
    print("Computing applicability domain …")
    _, ad_threshold, inside_ad = compute_applicability_domain(
        classification_train_row_ids,
        classification_test_row_ids,
        feature_blocks.morgan_fingerprints,
        logger,
    )

    # ── 11. Classification predictions CSV ─────────────────────────────
    best_result = next(
        r for r in scaffold_classification_results
        if r.name == best_scaffold_classifier.name
    )
    predictions_df = classification_test_df.copy()
    predictions_df["predicted_probability"]          = best_result.y_prob
    predictions_df["predicted_label"]                = best_result.y_pred
    predictions_df["inside_applicability_domain"]    = inside_ad
    predictions_df.to_csv(CLASSIFICATION_PREDICTIONS_PATH, index=False)
    print(f"Predictions saved → {CLASSIFICATION_PREDICTIONS_PATH.name}")

    # ── 12. Activity cliffs ────────────────────────────────────────────
    activity_cliffs_df = find_activity_cliffs(
        curated_df, feature_blocks.morgan_fingerprints, logger
    )
    activity_cliffs_df.to_csv(ACTIVITY_CLIFF_PATH, index=False)
    print(f"Activity cliffs saved → {ACTIVITY_CLIFF_PATH.name}  ({len(activity_cliffs_df)} found)")

    # ── 13. Update report JSON (classification section only) ───────────
    if RUN_REPORT_PATH.exists():
        report = json.loads(RUN_REPORT_PATH.read_text(encoding="utf-8"))
    else:
        report = {}
    report["classification"] = classification_metrics_df.to_dict(orient="records")
    report["applicability_domain"] = {
        "threshold": ad_threshold,
        "inside_fraction": float(np.mean(inside_ad)),
    }
    report["activity_cliffs"] = {
        "count": int(len(activity_cliffs_df)),
        "top_examples": activity_cliffs_df.head(5).to_dict(orient="records"),
    }
    RUN_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n✅  Classification rerun complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        try:
            with RUN_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(traceback_text)
                if not traceback_text.endswith("\n"):
                    fh.write("\n")
        except OSError:
            pass
        raise
