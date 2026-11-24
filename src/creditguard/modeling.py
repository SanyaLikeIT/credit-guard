"""Small, CPU-bounded searches and model-specific preprocessing."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from creditguard.features import RiskFeatures


def build_pipeline(name, config, engineered=False, include_age=True):
    features = RiskFeatures(
        engineered=engineered,
        include_age=include_age,
        clip_quantile=config["clipping_quantile"] if name == "logistic" else None,
    )
    steps = [("features", features)]
    if name != "catboost":
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ).set_output(transform="pandas"),
            )
        )
    if name == "logistic":
        steps.append(("scaler", StandardScaler().set_output(transform="pandas")))
        model = LogisticRegression(max_iter=2500, random_state=config["seed"])
    elif name == "forest":
        model = RandomForestClassifier(
            n_estimators=config["trees"],
            min_samples_leaf=20,
            max_depth=12,
            n_jobs=config["threads"],
            random_state=config["seed"],
        )
    elif name == "catboost":
        model = CatBoostClassifier(
            iterations=config["boosting_iterations"],
            depth=6,
            learning_rate=0.06,
            loss_function="Logloss",
            eval_metric="AUC",
            thread_count=config["threads"],
            random_seed=config["seed"],
            verbose=False,
            allow_writing_files=False,
        )
    elif name == "dummy":
        model = DummyClassifier(strategy="prior", random_state=config["seed"])
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([*steps, ("model", model)])


def search_space(name, config):
    if name == "logistic":
        return {"model__C": [0.05, 0.5, 5.0], "model__class_weight": [None, "balanced"]}
    if name == "forest":
        return {
            "model__n_estimators": [config["trees"]],
            "model__max_depth": [8, 14, None],
            "model__min_samples_leaf": [10, 30, 60],
            "model__min_samples_split": [2, 10],
            "model__max_features": ["sqrt", 0.7],
            "model__class_weight": [None, "balanced_subsample"],
        }
    if name == "catboost":
        return {
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__l2_leaf_reg": [3, 8, 15],
            "model__auto_class_weights": ["None", "Balanced"],
        }
    raise ValueError(name)


def tune_model(name, X, y, config, cv):
    search = RandomizedSearchCV(
        build_pipeline(name, config),
        search_space(name, config),
        n_iter=config["search_trials"],
        scoring={"ap": "average_precision", "auc": "roc_auc", "brier": "neg_brier_score"},
        refit=False,
        cv=cv,
        n_jobs=1,
        random_state=config["seed"],
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    best = int(np.argmax(results.mean_test_ap.to_numpy()))
    pipeline = build_pipeline(name, config).set_params(**search.cv_results_["params"][best])
    return pipeline, results


def fit_candidate(pipeline, X, y, groups, config, early_stopping=False):
    fitted = clone(pipeline)
    if early_stopping:
        strata = (
            pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y)})
            .groupby("g")
            .y.mean()
            .ge(0.5)
        )
        left, right = train_test_split(
            strata.index, test_size=0.1, stratify=strata, random_state=config["seed"]
        )
        left_ids = np.flatnonzero(np.isin(groups, left))
        right_ids = np.flatnonzero(np.isin(groups, right))
        probe = clone(pipeline)
        transformed = probe[:-1].fit_transform(X.iloc[left_ids], y.iloc[left_ids])
        validation = probe[:-1].transform(X.iloc[right_ids])
        probe[-1].fit(
            transformed,
            y.iloc[left_ids],
            eval_set=(validation, y.iloc[right_ids]),
            early_stopping_rounds=40,
            verbose=False,
        )
        iterations = max(1, probe[-1].get_best_iteration() + 1)
        fitted.set_params(model__iterations=iterations)
    return fitted.fit(X, y)
