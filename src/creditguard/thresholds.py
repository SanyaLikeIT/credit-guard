"""Exact threshold sweeps with explicit >= decision semantics."""

import numpy as np
import pandas as pd


def threshold_table(y, probabilities, cost_fp=1.0, cost_fn=5.0):
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Labels and probabilities must be matching, nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    if not np.isfinite([cost_fp, cost_fn]).all() or min(cost_fp, cost_fn) < 0:
        raise ValueError("Costs must be finite and nonnegative")
    order = np.argsort(p, kind="stable")
    ordered = p[order]
    positives = np.r_[0, np.cumsum(y[order])]
    # The last boundary can exceed 1 by one floating-point step, allowing all-negative
    # predictions even when a model returns exactly 1. JSON stores it as a finite number.
    candidates = np.unique(np.r_[0.0, 0.5, ordered, np.nextafter(ordered[-1], np.inf)])
    cut = np.searchsorted(ordered, candidates, side="left")
    fn = positives[cut]
    tp = positives[-1] - fn
    tn = cut - fn
    fp = len(y) - cut - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0
    )
    cost = cost_fp * fp + cost_fn * fn
    return pd.DataFrame(
        {
            "threshold": candidates,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_cost": cost,
            "cost_per_record": cost / len(y),
        }
    )


def select_threshold(table, objective="cost"):
    if objective == "cost":
        best = table.loc[table.total_cost == table.total_cost.min()]
    elif objective == "f1":
        best = table.loc[table.f1 == table.f1.max()]
    else:
        raise ValueError("Threshold objective must be cost or f1")
    # Prefer the highest threshold on ties: fewer positive classifications.
    return float(best.threshold.max())
