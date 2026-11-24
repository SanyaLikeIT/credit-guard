import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from creditguard.config import load_config
from creditguard.data import DELINQUENCIES, FEATURES, feature_groups, make_splits, validate_features
from creditguard.demo import DEMO_PROFILES
from creditguard.features import RiskFeatures
from creditguard.modeling import build_pipeline


def test_schema_order_and_optional_nulls(sample_data):
    X, _ = sample_data
    assert list(validate_features(X[FEATURES[::-1]])) == FEATURES
    record = {**DEMO_PROFILES["lower"], "MonthlyIncome": None, "NumberOfDependents": None}
    assert validate_features(pd.DataFrame([record])).isna().sum().sum() == 2


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "unexpected",
        "text",
        "negative",
        "infinite",
        "fractional",
        "required_null",
        "duplicate_names",
        "boolean",
    ],
)
def test_invalid_schema(sample_data, case):
    X, _ = sample_data
    X = X.head(5).copy()
    if case == "missing":
        X = X.drop(columns="age")
    elif case == "unexpected":
        X["unknown"] = 0
    elif case == "text":
        X["age"] = "not numeric"
    elif case == "negative":
        X["DebtRatio"] = -1
    elif case == "infinite":
        X["DebtRatio"] = np.inf
    elif case == "fractional":
        X["age"] = 21.5
    elif case == "required_null":
        X["age"] = np.nan
    elif case == "boolean":
        X["age"] = True
    else:
        X.columns = ["age"] * len(X.columns)
    with pytest.raises(ValueError):
        validate_features(X)


def test_group_splits_reproducibility_and_conflicts(sample_data):
    X, y = sample_data
    X = pd.concat([X, X.iloc[:20]], ignore_index=True)
    y = pd.concat([y, 1 - y.iloc[:20]], ignore_index=True)
    splits = make_splits(X, y)
    repeated = make_splits(X, y)
    groups = feature_groups(X)
    seen = set()
    for name, indices in splits.items():
        np.testing.assert_array_equal(indices, repeated[name])
        current = set(groups.iloc[indices])
        assert not seen.intersection(current)
        seen |= current
        assert abs(y.iloc[indices].mean() - y.mean()) < 0.1
    assert sum(map(len, splits.values())) == len(X)
    assert not np.array_equal(splits["train"], make_splits(X, y, seed=21)["train"])


def test_engineered_values_and_special_codes():
    record = {
        **DEMO_PROFILES["intermediate"],
        DELINQUENCIES[0]: 2,
        DELINQUENCIES[1]: 1,
        DELINQUENCIES[2]: 3,
    }
    X = pd.DataFrame([record])
    result = RiskFeatures(engineered=True).fit_transform(X)
    assert result.total_past_due.iloc[0] == 6
    assert result.weighted_past_due.iloc[0] == 13
    assert result.has_severe_delinquency.iloc[0] == 1
    assert result.income_per_household_member.iloc[0] == 3500 / 3
    assert result.real_estate_share.iloc[0] == 1 / 6
    for column in DELINQUENCIES:
        X[column] = 98
    result = RiskFeatures(engineered=True).fit_transform(X)
    assert result.special_delinquency_code.iloc[0] == 1
    assert result.total_past_due.isna().all()
    assert result[DELINQUENCIES].isna().all().all()


def test_zero_loans_missing_income_dependents_and_no_delinquencies():
    record = {
        **DEMO_PROFILES["lower"],
        "MonthlyIncome": np.nan,
        "NumberOfDependents": np.nan,
        FEATURES[5]: 0,
        FEATURES[7]: 0,
        "age": 0,
    }
    X = pd.DataFrame([record])
    result = RiskFeatures(engineered=True).fit_transform(X)
    assert result.real_estate_share.iloc[0] == 0
    assert result.total_past_due.iloc[0] == 0
    assert result.has_past_due.iloc[0] == 0
    assert result.missing_income.iloc[0] == result.missing_dependents.iloc[0] == 1
    assert result.income_per_household_member.isna().all()
    assert result.age.isna().all()
    X[FEATURES[7]] = 1
    assert RiskFeatures(engineered=True).fit_transform(X).real_estate_share.isna().all()


@pytest.mark.parametrize("family", ["dummy", "logistic", "forest", "catboost"])
def test_preprocessing_training_statistics_are_frozen(sample_data, family):
    X, y = sample_data
    config = load_config(profile="quick")
    pipeline = build_pipeline(family, config, engineered=True)
    preparation = pipeline[:-1].fit(X, y)
    before = preparation.transform(X.head()).copy()
    names = preparation.get_feature_names_out()
    unusual = X.head().copy()
    unusual["DebtRatio"] = 1e12
    unusual["MonthlyIncome"] = np.nan
    unusual[DELINQUENCIES] = 96
    transformed = preparation.transform(unusual)
    assert transformed.shape == (5, len(names))
    pd.testing.assert_frame_equal(before, preparation.transform(X.head()))
    if family != "catboost":
        assert np.isfinite(transformed.to_numpy()).all()
    assert len(names) == len(set(names))
    clone(pipeline)


def test_clipping_uses_training_only(sample_data):
    X, _ = sample_data
    transformer = RiskFeatures(clip_quantile=0.95).fit(X)
    original = transformer.upper_.copy()
    evaluation = X.head().copy()
    evaluation["DebtRatio"] = 1e15
    assert transformer.transform(evaluation).DebtRatio.max() == original.DebtRatio
    pd.testing.assert_series_equal(original, transformer.upper_)
