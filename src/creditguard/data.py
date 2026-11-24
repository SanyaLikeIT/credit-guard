"""Strict schemas and group-preserving, stratified holdouts."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "SeriousDlqin2yrs"
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
DELINQUENCIES = [FEATURES[2], FEATURES[8], FEATURES[6]]
OPTIONAL_VALUES = {"MonthlyIncome", "NumberOfDependents"}
COUNTS = {FEATURES[i] for i in (1, 2, 5, 6, 7, 8, 9)}
SPLITS = {"train": 0.60, "selection": 0.15, "calibration": 0.10, "threshold": 0.05, "test": 0.10}


def validate_features(frame):
    """Accept missing income/dependents; reject invalid types, infinities, and negatives."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input must be a nonempty table of raw features")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    missing = set(FEATURES) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise ValueError(
            f"Feature schema mismatch; missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    result = frame.loc[:, FEATURES].copy()
    for column in FEATURES:
        series = result[column]
        if column in OPTIONAL_VALUES and series.isna().all():
            series = series.astype(float)
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"{column} must be numeric")
        series = series.astype(float)
        if np.isinf(series.to_numpy()).any() or (series.dropna() < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative values")
        if column not in OPTIONAL_VALUES and series.isna().any():
            raise ValueError(f"{column} cannot be missing")
        if column in COUNTS and not np.allclose(series.dropna() % 1, 0):
            raise ValueError(f"{column} must contain whole numbers")
        result[column] = series
    return result


def load_dataset(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}. See README dataset setup.")
    table = pd.read_csv(source)
    technical = [column for column in table if column.startswith("Unnamed:")]
    if len(technical) > 1:
        raise ValueError("More than one technical CSV index found")
    table = table.drop(columns=technical)
    if TARGET not in table:
        raise ValueError(f"Missing target: {TARGET}")
    y = table.pop(TARGET)
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("Training target must contain both binary classes, without missing labels")
    return validate_features(table), y.astype(int)


def feature_groups(frame):
    """Stable hashes exclude both the label and the technical CSV index."""
    return pd.util.hash_pandas_object(frame.loc[:, FEATURES], index=False).rename("group")


def make_splits(frame, y, seed=42, proportions=None):
    """Stratify feature groups by majority label; retain all conflicting observations.

    Group counts, rather than row counts, determine split sizes, so proportions are approximate.
    Conflicting labels stay together. No resampling or label-dependent deletion is performed.
    """
    proportions = proportions or SPLITS
    if set(proportions) != set(SPLITS) or not np.isclose(sum(proportions.values()), 1):
        raise ValueError("Expected five positive split proportions summing to one")
    if any(value <= 0 for value in proportions.values()):
        raise ValueError("Split proportions must be positive")
    groups = feature_groups(frame)
    summary = pd.DataFrame({"group": groups.to_numpy(), "y": np.asarray(y)})
    aggregated = summary.groupby("group", sort=True).y.agg(["mean", "size", "nunique"])
    # Ties are assigned to the positive stratum. Mixed groups are never split or relabeled.
    strata = (aggregated["mean"] >= 0.5).astype(int)
    remaining = aggregated.index.to_numpy()
    remaining_weight = 1.0
    result = {}
    for offset, name in enumerate(["train", "selection", "calibration", "threshold"]):
        selected, remaining = train_test_split(
            remaining,
            train_size=proportions[name] / remaining_weight,
            random_state=seed + offset,
            stratify=strata.loc[remaining],
        )
        result[name] = np.flatnonzero(groups.isin(selected))
        remaining_weight -= proportions[name]
    result["test"] = np.flatnonzero(groups.isin(remaining))
    row_ids = np.concatenate(list(result.values()))
    assert len(np.unique(row_ids)) == len(frame) == len(row_ids)
    seen = set()
    for indices in result.values():
        current = set(groups.iloc[indices])
        assert not seen.intersection(current), "Duplicate feature groups leaked between splits"
        assert len(np.unique(np.asarray(y)[indices])) == 2, "A holdout is missing a class"
        seen.update(current)
    return result
