"""Consistent metrics and JSON output for offline reports and the application."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y, probabilities, threshold=0.5, cost_fp=1, cost_fn=5):
    predicted = np.asarray(probabilities) >= threshold
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": float(cost_fp * fp + cost_fn * fn),
        "cost_per_record": float((cost_fp * fp + cost_fn * fn) / len(y)),
        "n_records": len(y),
        "positive_rate": float(np.mean(y)),
    }


def bootstrap_intervals(y, probabilities, seed=42, samples=300):
    """Stratified row bootstrap; approximate intervals conditional on fitted model/splits."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(probabilities)
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    scores = []
    for _ in range(samples):
        indices = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in classes])
        scores.append(
            [
                roc_auc_score(y[indices], p[indices]),
                average_precision_score(y[indices], p[indices]),
                brier_score_loss(y[indices], p[indices]),
            ]
        )
    intervals = np.quantile(scores, [0.025, 0.975], axis=0)
    return {
        key: {"lower": float(intervals[0, i]), "upper": float(intervals[1, i])}
        for i, key in enumerate(["roc_auc", "average_precision", "brier_score"])
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value)}")
