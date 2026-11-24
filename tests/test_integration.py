import json

import numpy as np
import pytest

from creditguard.config import load_config
from creditguard.data import TARGET
from creditguard.predict import load_predictor
from creditguard.train import run_experiment


@pytest.mark.integration
def test_end_to_end_synthetic_training(sample_data, tmp_path):
    X, y = sample_data
    dataset = tmp_path / "synthetic.csv"
    X.assign(**{TARGET: y}).to_csv(dataset, index=False)
    config = load_config(profile="quick")
    config.update(
        dataset=dataset,
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        trees=8,
        boosting_iterations=20,
        shap_sample=10,
        bootstrap_samples=5,
        threads=2,
        synthetic_test_run=True,
    )
    predictor, final = run_experiment(config)
    assert 0 <= final["average_precision"] <= 1
    loaded, metadata = load_predictor(config["model_dir"])
    np.testing.assert_allclose(
        loaded.predict(X.head()).risk_probability, predictor.predict(X.head()).risk_probability
    )
    assert metadata["synthetic_test_run"]
    frozen = json.loads((config["report_dir"] / "metrics/selection_frozen.json").read_text())
    assert frozen["artifact_sha256"] == final["artifact_sha256"]
    for report in [
        "model_comparison.csv",
        "feature_ablation.csv",
        "calibration_comparison.csv",
        "cost_sensitivity.csv",
        "shap_importance.csv",
        "subgroup_metrics.csv",
    ]:
        assert (config["report_dir"] / "metrics" / report).is_file()
    with pytest.raises(ValueError, match="already has a final evaluation"):
        run_experiment(config)
