"""Training-only clipping and deterministic features shared with inference."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from creditguard.data import DELINQUENCIES, FEATURES


class RiskFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, engineered=False, clip_quantile=None, include_age=True):
        self.engineered = engineered
        self.clip_quantile = clip_quantile
        self.include_age = include_age

    @staticmethod
    def clean(frame):
        result = frame.loc[:, FEATURES].astype(float).copy()
        result.loc[result.age <= 0, "age"] = np.nan
        result.loc[:, DELINQUENCIES] = result.loc[:, DELINQUENCIES].mask(
            result.loc[:, DELINQUENCIES].isin([96, 98])
        )
        return result

    def fit(self, X, y=None):
        if self.clip_quantile is not None and not 0 < self.clip_quantile <= 1:
            raise ValueError("Clipping quantile must be in (0, 1]")
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        clean = self.clean(X)
        columns = ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "MonthlyIncome"]
        self.upper_ = (
            clean[columns].quantile(self.clip_quantile) if self.clip_quantile is not None else None
        )
        self.feature_names_out_ = np.asarray(self._transform(X).columns, dtype=object)
        return self

    def _transform(self, X):
        clean = self.clean(X)
        result = clean.copy()
        if self.upper_ is not None:
            result.loc[:, self.upper_.index] = result[self.upper_.index].clip(
                upper=self.upper_, axis=1
            )
        if self.engineered:
            past_due = clean[DELINQUENCIES]
            result["total_past_due"] = past_due.sum(axis=1, min_count=1)
            result["weighted_past_due"] = past_due.mul([1, 2, 3]).sum(axis=1, min_count=1)
            result["has_past_due"] = (past_due.fillna(0).sum(axis=1) > 0).astype(float)
            result["has_severe_delinquency"] = (clean[DELINQUENCIES[2]] > 0).astype(float)
            result["special_delinquency_code"] = (
                X[DELINQUENCIES].isin([96, 98]).any(axis=1).astype(float)
            )
            for source, name in [
                ("MonthlyIncome", "log_income"),
                ("DebtRatio", "log_debt_ratio"),
                ("RevolvingUtilizationOfUnsecuredLines", "log_utilization"),
            ]:
                result[name] = np.log1p(result[source])
            result["income_per_household_member"] = result.MonthlyIncome / (
                1 + clean.NumberOfDependents
            )
            denominator = clean.NumberOfOpenCreditLinesAndLoans.replace(0, np.nan)
            result["real_estate_share"] = clean.NumberRealEstateLoansOrLines / denominator
            both_zero = (clean.NumberRealEstateLoansOrLines == 0) & denominator.isna()
            result.loc[both_zero, "real_estate_share"] = 0
            result["missing_income"] = clean.MonthlyIncome.isna().astype(float)
            result["missing_dependents"] = clean.NumberOfDependents.isna().astype(float)
        if not self.include_age:
            result = result.drop(columns="age")
        return result

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        return self._transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
