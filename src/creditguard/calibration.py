"""Compare calibration methods on an internal holdout, then refit on calibration data."""

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


class ProbabilityCalibrator:
    def __init__(self, method="none"):
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("Unknown calibration method")
        self.method = method

    def fit(self, probabilities, y):
        p = np.asarray(probabilities, dtype=float)
        if self.method == "sigmoid":
            self.estimator_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            self.estimator_.fit(logit(np.clip(p, 1e-8, 1 - 1e-8)).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.estimator_ = IsotonicRegression(out_of_bounds="clip")
            self.estimator_.fit(p, y)
        self.fitted_ = True
        return self

    def predict(self, probabilities):
        if not getattr(self, "fitted_", False):
            raise ValueError("Calibrator has not been fitted")
        p = np.asarray(probabilities, dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid input probabilities")
        if self.method == "sigmoid":
            return self.estimator_.predict_proba(logit(np.clip(p, 1e-8, 1 - 1e-8)).reshape(-1, 1))[
                :, 1
            ]
        if self.method == "isotonic":
            return self.estimator_.predict(p)
        return p.copy()


def choose_calibrator(probabilities, y, seed=42, validation_fraction=0.5, groups=None):
    p, y = np.asarray(probabilities), np.asarray(y)
    if groups is None:
        fit_ids, validation_ids = train_test_split(
            np.arange(len(y)), test_size=validation_fraction, stratify=y, random_state=seed
        )
    else:
        group_series = pd.Series(np.asarray(groups))
        strata = pd.DataFrame({"group": groups, "y": y}).groupby("group").y.mean().ge(0.5)
        left, right = train_test_split(
            strata.index, test_size=validation_fraction, stratify=strata, random_state=seed
        )
        fit_ids = np.flatnonzero(group_series.isin(left))
        validation_ids = np.flatnonzero(group_series.isin(right))
    rows = []
    for method in ("none", "sigmoid", "isotonic"):
        calibrator = ProbabilityCalibrator(method).fit(p[fit_ids], y[fit_ids])
        predicted = calibrator.predict(p[validation_ids])
        rows.append(
            {
                "method": method,
                "brier_score": brier_score_loss(y[validation_ids], predicted),
                "log_loss": log_loss(y[validation_ids], predicted, labels=[0, 1]),
                "roc_auc": roc_auc_score(y[validation_ids], predicted),
                "average_precision": average_precision_score(y[validation_ids], predicted),
                "fit_size": len(fit_ids),
                "validation_size": len(validation_ids),
            }
        )
    comparison = pd.DataFrame(rows)
    method = comparison.sort_values(["brier_score", "log_loss"], kind="stable").iloc[0].method
    return ProbabilityCalibrator(method).fit(p, y), comparison
