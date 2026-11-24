"""Explain the uncalibrated model; preserve the model output's mathematical units."""

import numpy as np
import shap
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ExplanationUnavailable(ValueError):
    pass


def explain_pipeline(pipeline, frame, background=None):
    transformed = pipeline[:-1].transform(frame)
    names = list(pipeline[:-1].get_feature_names_out())
    model = pipeline[-1]
    if isinstance(model, CatBoostClassifier):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(transformed)
        space = "uncalibrated log-odds (raw model margin)"
    elif isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(transformed)[:, :, 1]
        space = "uncalibrated positive-class probability"
    elif isinstance(model, LogisticRegression):
        if background is None:
            raise ExplanationUnavailable(
                "Logistic explanations require a saved training background"
            )
        explainer = shap.LinearExplainer(model, background)
        explanation = explainer(transformed)
        space = "uncalibrated log-odds (linear model margin)"
    else:
        raise ExplanationUnavailable("SHAP is not available for this model family")
    values = np.asarray(explanation.values)
    if values.shape != (len(frame), len(names)) or not np.isfinite(values).all():
        raise ExplanationUnavailable("SHAP dimensions or values do not match the feature schema")
    explanation.feature_names = names
    if isinstance(model, (CatBoostClassifier, LogisticRegression)):
        expected = (
            model.predict(transformed, prediction_type="RawFormulaVal")
            if isinstance(model, CatBoostClassifier)
            else model.decision_function(transformed)
        )
    else:
        expected = model.predict_proba(transformed)[:, 1]
    if not np.allclose(
        np.asarray(explanation.base_values) + values.sum(axis=1), expected, rtol=1e-4, atol=1e-5
    ):
        raise ExplanationUnavailable("SHAP contributions failed the additivity check")
    return explanation, space
