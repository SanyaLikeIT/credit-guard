"""Reproducible, leakage-aware training with a final holdout evaluated only after freezing."""

import argparse
import hashlib
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from creditguard.calibration import choose_calibrator
from creditguard.config import load_config
from creditguard.data import feature_groups, load_dataset, make_splits
from creditguard.demo import DEMO_PROFILES
from creditguard.eda import create_eda
from creditguard.metrics import bootstrap_intervals, classification_metrics, write_json
from creditguard.modeling import build_pipeline, fit_candidate, tune_model
from creditguard.predict import RiskPredictor, save_predictor
from creditguard.reporting import (
    calibration_plot,
    error_analysis,
    final_plots,
    global_explanations,
    selection_plots,
    threshold_plots,
)
from creditguard.thresholds import select_threshold, threshold_table


def run_experiment(config):
    started = time.perf_counter()
    root = Path(config["report_dir"])
    metric_dir, figures, private = root / "metrics", root / "figures", root / "private"
    if (metric_dir / "final_test_metrics.json").exists() and (
        Path(config["model_dir"]) / "pipeline.joblib"
    ).exists():
        raise ValueError(
            "This run already has a final evaluation. Use a distinct output directory "
            "for a new experiment; do not tune on the same final holdout."
        )
    for path in [metric_dir, figures, private]:
        path.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(config["dataset"])
    fingerprint = hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest()
    splits = make_splits(X, y, config["seed"], config["split_proportions"])
    groups = feature_groups(X)
    manifest = pd.DataFrame(
        {"row_position": np.arange(len(X)), "feature_group": groups.astype(str), "split": ""}
    )
    for name, ids in splits.items():
        manifest.loc[ids, "split"] = name
    manifest.to_csv(private / "split_manifest.csv", index=False)
    write_json(
        metric_dir / "split_summary.json",
        {
            "seed": config["seed"],
            "dataset_sha256": fingerprint,
            "strategy": (
                "Stratified feature-group holdouts; majority-label group strata; "
                "conflicts retained together"
            ),
            "partitions": {
                name: {"n_records": len(ids), "positive_rate": float(y.iloc[ids].mean())}
                for name, ids in splits.items()
            },
            "n_records": len(X),
            "duplicate_feature_rows": int(X.duplicated().sum()),
            "conflicting_feature_groups": int(
                pd.DataFrame({"g": groups, "y": y}).groupby("g").y.nunique().gt(1).sum()
            ),
            "overlap_checks_passed": True,
        },
    )
    train_ids, selection_ids = splits["train"], splits["selection"]
    X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
    X_selection, y_selection = X.iloc[selection_ids], y.iloc[selection_ids]
    train_groups = groups.iloc[train_ids].to_numpy()
    create_eda(X_train, y_train, root)
    cv = list(
        StratifiedGroupKFold(
            n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"]
        ).split(X_train, y_train, train_groups)
    )
    for left, right in cv:
        assert not set(train_groups[left]).intersection(train_groups[right])
    models, predictions, comparison, ablation = {}, {}, [], []
    with threadpool_limits(limits=config["threads"]):
        dummy_start = time.perf_counter()
        dummy = build_pipeline("dummy", config).fit(X_train, y_train)
        models["dummy"] = dummy
        predictions["dummy"] = dummy.predict_proba(X_selection)[:, 1]
        comparison.append(
            {
                "model": "dummy",
                "family": "dummy",
                "engineered": False,
                "training_seconds": time.perf_counter() - dummy_start,
                **classification_metrics(y_selection, predictions["dummy"]),
            }
        )
        for family in ["logistic", "forest", "catboost"]:
            family_start = time.perf_counter()
            print(
                f"Searching {family}: {config['search_trials']} candidates, "
                f"{config['cv_folds']} group-aware folds",
                flush=True,
            )
            candidate, results = tune_model(family, X_train, y_train, config, cv)
            results.to_csv(metric_dir / f"search_{family}.csv", index=False)
            raw = fit_candidate(
                candidate,
                X_train,
                y_train,
                train_groups,
                config,
                early_stopping=family == "catboost",
            )
            family_seconds = time.perf_counter() - family_start
            for engineered in [False, True]:
                begin = time.perf_counter()
                fitted = (
                    clone(raw).set_params(features__engineered=True).fit(X_train, y_train)
                    if engineered
                    else raw
                )
                name = family + ("_engineered" if engineered else "_raw")
                models[name] = fitted
                p = predictions[name] = fitted.predict_proba(X_selection)[:, 1]
                measured = classification_metrics(y_selection, p)
                train_ap = classification_metrics(y_train, fitted.predict_proba(X_train)[:, 1])[
                    "average_precision"
                ]
                row = {
                    "model": name,
                    "family": family,
                    "engineered": engineered,
                    "training_seconds": family_seconds + time.perf_counter() - begin,
                    "training_average_precision": train_ap,
                    **measured,
                }
                comparison.append(row)
                ablation.append(
                    {
                        "model": name,
                        "family": family,
                        "engineered": engineered,
                        "average_precision": measured["average_precision"],
                        "roc_auc": measured["roc_auc"],
                        "brier_score": measured["brier_score"],
                        "matched_model_parameters": True,
                    }
                )
                print(
                    f"{name}: selection AP={measured['average_precision']:.6f}, "
                    f"ROC-AUC={measured['roc_auc']:.6f}",
                    flush=True,
                )
    comparison = pd.DataFrame(comparison).sort_values(
        ["average_precision", "brier_score"], ascending=[False, True]
    )
    comparison.drop(columns="confusion_matrix").to_csv(
        metric_dir / "model_comparison.csv", index=False
    )
    pd.DataFrame(ablation).to_csv(metric_dir / "feature_ablation.csv", index=False)
    selection_plots(y_selection, predictions, figures)
    winner_name = comparison.iloc[0].model
    if winner_name == "dummy":
        raise ValueError(
            "No trained model beat the dummy; inspect training without consulting final test"
        )
    winner = models[winner_name]
    # Matched-parameter weighting study on selection data; never on final test.
    family = winner_name.split("_")[0]
    weight_parameter = "auto_class_weights" if family == "catboost" else "class_weight"
    weights = ["None", "Balanced"] if family == "catboost" else [None, "balanced"]
    imbalance_rows = []
    for weight in weights:
        if winner[-1].get_params().get(weight_parameter) == weight:
            alternative = winner
        else:
            alternative = clone(winner).set_params(**{f"model__{weight_parameter}": weight})
            with threadpool_limits(limits=config["threads"]):
                alternative.fit(X_train, y_train)
        p = alternative.predict_proba(X_selection)[:, 1]
        for boundary in [0.5, config["cost_fp"] / (config["cost_fp"] + config["cost_fn"])]:
            imbalance_rows.append(
                {
                    "weighting": str(weight),
                    **classification_metrics(
                        y_selection, p, boundary, config["cost_fp"], config["cost_fn"]
                    ),
                }
            )
    pd.DataFrame(imbalance_rows).drop(columns="confusion_matrix").to_csv(
        metric_dir / "imbalance_comparison.csv", index=False
    )
    # Selection-set-only age ablation is descriptive; it does not change the frozen winner.
    no_age = clone(winner).set_params(features__include_age=False).fit(X_train, y_train)
    age_rows = [
        {"variant": "with_age", **classification_metrics(y_selection, predictions[winner_name])},
        {
            "variant": "without_age",
            **classification_metrics(y_selection, no_age.predict_proba(X_selection)[:, 1]),
        },
    ]
    pd.DataFrame(age_rows).drop(columns="confusion_matrix").to_csv(
        metric_dir / "age_ablation.csv", index=False
    )
    # Save logistic coefficients even when another family wins.
    logistic = models["logistic_engineered"]
    pd.DataFrame(
        {
            "feature": logistic[:-1].get_feature_names_out(),
            "standardized_coefficient": logistic[-1].coef_[0],
        }
    ).to_csv(metric_dir / "logistic_coefficients.csv", index=False)
    calibration_ids = splits["calibration"]
    calibration_p = winner.predict_proba(X.iloc[calibration_ids])[:, 1]
    calibrator, calibration_results = choose_calibrator(
        calibration_p,
        y.iloc[calibration_ids],
        config["seed"],
        config["calibration_validation_fraction"],
        groups.iloc[calibration_ids].to_numpy(),
    )
    calibration_results.to_csv(metric_dir / "calibration_comparison.csv", index=False)
    threshold_ids = splits["threshold"]
    raw_threshold_p = winner.predict_proba(X.iloc[threshold_ids])[:, 1]
    p_threshold = calibrator.predict(raw_threshold_p)
    y_threshold = y.iloc[threshold_ids]
    sweep = threshold_table(y_threshold, p_threshold, config["cost_fp"], config["cost_fn"])
    sweep.to_csv(metric_dir / "threshold_analysis.csv", index=False)
    threshold = select_threshold(sweep)
    f1_threshold = select_threshold(sweep, "f1")
    sensitivity = []
    for ratio in config["cost_ratios"]:
        table = threshold_table(y_threshold, p_threshold, 1, ratio)
        chosen = select_threshold(table)
        sensitivity.append(
            {
                "cost_fp": 1,
                "cost_fn": ratio,
                **classification_metrics(y_threshold, p_threshold, chosen, 1, ratio),
            }
        )
    pd.DataFrame(sensitivity).drop(columns="confusion_matrix").to_csv(
        metric_dir / "cost_sensitivity.csv", index=False
    )
    strategies = []
    for name, boundary in [
        ("default_0.5", 0.5),
        ("f1_optimized", f1_threshold),
        ("cost_optimized", threshold),
    ]:
        strategies.append(
            {
                "strategy": name,
                **classification_metrics(
                    y_threshold, p_threshold, boundary, config["cost_fp"], config["cost_fn"]
                ),
            }
        )
    pd.DataFrame(strategies).drop(columns="confusion_matrix").to_csv(
        metric_dir / "threshold_comparison.csv", index=False
    )
    calibration_plot(y_threshold, raw_threshold_p, p_threshold, figures)
    threshold_plots(sweep, threshold, figures)
    pd.DataFrame({"target": y_threshold.to_numpy(), "probability": p_threshold}).to_csv(
        private / "threshold_predictions.csv", index=False
    )
    background = winner[:-1].transform(
        X_train.sample(min(100, len(X_train)), random_state=config["seed"])
    )
    predictor = RiskPredictor(winner, calibrator, threshold, winner_name, background)
    shap_sample = X_selection.sample(
        min(config["shap_sample"], len(X_selection)), random_state=config["seed"]
    )
    global_explanations(predictor, shap_sample, figures, metric_dir)
    demo = predictor.predict(pd.DataFrame(list(DEMO_PROFILES.values())))
    if not np.all(np.diff(demo.risk_probability) > 0):
        raise ValueError(
            "Synthetic demonstration profile risk order is inconsistent; "
            "review before final evaluation"
        )
    write_json(
        metric_dir / "demo_predictions.json",
        dict(zip(DEMO_PROFILES, demo.risk_probability, strict=True)),
    )
    portable_config = {
        k: (str(v.relative_to(Path.cwd())) if v.is_relative_to(Path.cwd()) else v.name)
        if isinstance(v, Path)
        else v
        for k, v in config.items()
    }
    metadata = {
        "model_name": winner_name,
        "seed": config["seed"],
        "profile": config["profile"],
        "dataset_sha256": fingerprint,
        "threshold": threshold,
        "calibration_method": calibrator.method,
        "cost_fp": config["cost_fp"],
        "cost_fn": config["cost_fn"],
        "config": portable_config,
        "python_version": platform.python_version(),
        "library_versions": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "catboost",
                "shap",
                "streamlit",
                "joblib",
            ]
        },
        "selected_parameters": winner[-1].get_params(),
        "feature_engineering": bool(winner[0].engineered),
        "transformed_feature_names": list(winner[:-1].get_feature_names_out()),
        "selection_rule": "Highest model-selection Average Precision; Brier score breaks ties",
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "training_seconds_before_test": time.perf_counter() - started,
        "synthetic_test_run": config.get("synthetic_test_run", False),
    }
    metadata = save_predictor(predictor, config["model_dir"], metadata)
    write_json(metric_dir / "selection_frozen.json", metadata)
    print(
        f"Frozen {winner_name}; calibration={calibrator.method}; threshold={threshold:.8f}. "
        "Evaluating final holdout once.",
        flush=True,
    )
    # No modeling, calibration, or threshold decisions follow this point.
    test_ids = splits["test"]
    p_test = predictor.predict(X.iloc[test_ids]).risk_probability.to_numpy()
    y_test = y.iloc[test_ids]
    pd.DataFrame({"target": y_test.to_numpy(), "probability": p_test}).to_csv(
        private / "final_test_predictions.csv", index=False
    )
    final = classification_metrics(y_test, p_test, threshold, config["cost_fp"], config["cost_fn"])
    final["confidence_intervals_95"] = bootstrap_intervals(
        y_test, p_test, config["seed"], config["bootstrap_samples"]
    )
    final["artifact_sha256"] = metadata["artifact_sha256"]
    final["dataset_sha256"] = fingerprint
    write_json(metric_dir / "final_test_metrics.json", final)
    final_plots(y_test, p_test, threshold, figures)
    error_analysis(
        X.iloc[test_ids],
        y_test,
        p_test,
        threshold,
        metric_dir,
        config["cost_fp"],
        config["cost_fn"],
    )
    write_json(
        metric_dir / "run_summary.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "model": winner_name,
            "profile": config["profile"],
        },
    )
    print(
        f"Complete. Final AP={final['average_precision']:.6f}; "
        f"ROC-AUC={final['roc_auc']:.6f}; Brier={final['brier_score']:.6f}",
        flush=True,
    )
    return predictor, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, help="Isolated run directory; recommended for quick experiments"
    )
    args = parser.parse_args()
    config = load_config(args.config, args.profile)
    if args.dataset:
        config["dataset"] = args.dataset.resolve()
    if args.output_dir or args.profile == "quick":
        output = (args.output_dir or Path("outputs/quick")).resolve()
        config["model_dir"], config["report_dir"] = output / "models", output / "reports"
    try:
        with threadpool_limits(limits=config["threads"]):
            run_experiment(config)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
