"""Aggregate model evaluation, SHAP distributions, and subgroup error analysis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from creditguard.eda import save_figure
from creditguard.explain import explain_pipeline
from creditguard.metrics import classification_metrics, write_json


def selection_plots(y, predictions, directory):
    directory = Path(directory)
    for name, display, file in [
        ("ROC on model selection set", RocCurveDisplay, "roc_curves.png"),
        (
            "Precision–recall on model selection set",
            PrecisionRecallDisplay,
            "precision_recall_curves.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, p in predictions.items():
            display.from_predictions(y, p, name=model, ax=ax)
        ax.set_title(name)
        ax.legend(fontsize=7)
        save_figure(fig, directory / file)
    comparison = pd.DataFrame(
        [{"model": name, **classification_metrics(y, p)} for name, p in predictions.items()]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, score in zip(axes, ["average_precision", "roc_auc"], strict=True):
        ax.barh(comparison.model, comparison[score], color="#7198c4")
        ax.set(title="Model selection comparison", xlabel=score.replace("_", " "))
    save_figure(fig, directory / "model_comparison.png")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (name, p) in zip(axes.ravel(), predictions.items(), strict=False):
        ConfusionMatrixDisplay.from_predictions(y, np.asarray(p) >= 0.5, ax=ax, colorbar=False)
        ax.set_title(name + " (threshold 0.5)", fontsize=9)
    for ax in axes.ravel()[len(predictions) :]:
        ax.set_visible(False)
    save_figure(fig, directory / "selection_confusion_matrices.png")


def calibration_plot(y, original, calibrated, directory):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probabilities in [("Uncalibrated", original), ("Selected calibration", calibrated)]:
        observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(
        title="Calibration on threshold holdout (not calibration training data)",
        xlabel="Mean predicted probability",
        ylabel="Observed event frequency",
    )
    ax.legend()
    save_figure(fig, Path(directory) / "calibration_curve.png")


def threshold_plots(table, threshold, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for column in ["precision", "recall", "f1"]:
        axes[0].plot(table.threshold, table[column], label=column.capitalize())
    axes[1].plot(
        table.threshold,
        table.cost_per_record,
        label="Configured cost",
    )
    for ax in axes:
        ax.axvline(threshold, color="#c77676", linestyle="--", label="Selected threshold")
        ax.set_xlabel("Classification threshold (probability >= threshold)")
        ax.legend(fontsize=8)
    axes[0].set(title="Threshold holdout: classification metrics", ylabel="Score")
    axes[1].set(title="Threshold holdout: conditional cost", ylabel="Cost per record")
    save_figure(fig, Path(directory) / "threshold_optimization.png")


def final_plots(y, probabilities, threshold, directory):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    RocCurveDisplay.from_predictions(y, probabilities, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axes[1])
    ConfusionMatrixDisplay.from_predictions(
        y, np.asarray(probabilities) >= threshold, ax=axes[2], colorbar=False
    )
    axes[0].set_title("Final test: ROC")
    axes[1].set_title("Final test: precision–recall")
    axes[2].set_title(f"Final test: threshold {threshold:.4f}")
    save_figure(fig, Path(directory) / "final_evaluation.png")


def global_explanations(predictor, X, directory, metric_dir):
    explanation, space = explain_pipeline(predictor.pipeline, X, predictor.background)
    importance = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "mean_absolute_shap": np.abs(explanation.values).mean(axis=0),
        }
    )
    importance = importance.sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(Path(metric_dir) / "shap_importance.csv", index=False)
    write_json(
        Path(metric_dir) / "shap_metadata.json",
        {
            "partition": "selection",
            "sample_size": len(X),
            "output_space": space,
            "explains_calibrated_probability": False,
            "additivity_checked": True,
        },
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(15).sort_values("mean_absolute_shap")
    ax.barh(top.feature, top.mean_absolute_shap, color="#7198c4")
    ax.set(
        title="Global SHAP importance on model selection sample",
        xlabel=f"Mean absolute contribution: {space}",
    )
    save_figure(fig, Path(directory) / "shap_importance.png")
    # No raw feature values or individual customer records are exported in public reports.
    shap.summary_plot(
        explanation.values,
        feature_names=explanation.feature_names,
        plot_type="violin",
        max_display=15,
        show=False,
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel(f"SHAP contribution: {space}")
    save_figure(fig, Path(directory) / "shap_summary.png")
    return importance


def error_analysis(X, y, probabilities, threshold, directory, cost_fp=1, cost_fn=5):
    p, y = np.asarray(probabilities), np.asarray(y)
    predicted = p >= threshold
    masks = {
        "false_positive": (y == 0) & predicted,
        "false_negative": (y == 1) & ~predicted,
        "within_0.02_of_threshold": np.abs(p - threshold) <= 0.02,
        "high_confidence_error": ((y == 0) & (p >= 0.8)) | ((y == 1) & (p <= 0.02)),
    }
    summary = {}
    for name, mask in masks.items():
        subset = X.iloc[np.flatnonzero(mask)]
        summary[name] = {
            "n_records": int(mask.sum()),
            "missing_income_rate": float(subset.MonthlyIncome.isna().mean())
            if mask.any()
            else None,
            "median_age": float(subset.age.median()) if mask.any() else None,
            "median_utilization": float(subset.iloc[:, 0].median()) if mask.any() else None,
            "debt_ratio_above_10_rate": float(subset.DebtRatio.gt(10).mean())
            if mask.any()
            else None,
        }
    write_json(Path(directory) / "error_analysis.json", summary)
    groupings = {
        "age": pd.cut(
            X.age,
            [0, 30, 45, 60, np.inf],
            labels=["<=30", "31-45", "46-60", ">60"],
            include_lowest=True,
        ),
        "income": pd.cut(
            X.MonthlyIncome, [-1, 2500, 6000, np.inf], labels=["<=2500", "2500-6000", ">6000"]
        ),
        "debt_ratio": pd.cut(
            X.DebtRatio, [-1, 0.3, 1, 10, np.inf], labels=["<=0.3", "0.3-1", "1-10", ">10"]
        ),
        "missing_income": X.MonthlyIncome.isna().map({True: "Missing", False: "Observed"}),
    }
    rows = []
    for grouping, values in groupings.items():
        values = values.astype(str).fillna("Missing").to_numpy()
        for group in np.unique(values):
            ids = np.flatnonzero(values == group)
            row = {
                "grouping": grouping,
                "group": group,
                "n_records": len(ids),
                "n_positive": int(y[ids].sum()),
                "interpretation": "Descriptive only; no fairness guarantee",
            }
            if (
                len(ids) >= 50
                and len(np.unique(y[ids])) == 2
                and min(y[ids].sum(), (1 - y[ids]).sum()) >= 5
            ):
                row.update(classification_metrics(y[ids], p[ids], threshold, cost_fp, cost_fn))
                row.pop("confusion_matrix")
            else:
                row["interpretation"] = (
                    "Insufficient events or sample size for reliable group metrics"
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(Path(directory) / "subgroup_metrics.csv", index=False)
    return summary
