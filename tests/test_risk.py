import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from creditguard.calibration import ProbabilityCalibrator, choose_calibrator
from creditguard.explain import ExplanationUnavailable, explain_pipeline
from creditguard.metrics import classification_metrics
from creditguard.predict import load_predictor, save_predictor
from creditguard.thresholds import select_threshold, threshold_table


@pytest.mark.parametrize("cost_fp,cost_fn", [(1, 1), (1, 2), (1, 5), (1, 20), (0, 1), (1, 0)])
def test_threshold_sweep_matches_brute_force(cost_fp, cost_fn):
    y = np.array([0, 1, 0, 1, 0, 1, 0])
    p = np.array([0, 0.1, 0.1, 0.5, 0.7, 1, 1])
    table = threshold_table(y, p, cost_fp, cost_fn)
    for row in table.itertuples():
        tn, fp, fn, tp = confusion_matrix(y, p >= row.threshold, labels=[0, 1]).ravel()
        assert [row.tn, row.fp, row.fn, row.tp] == [tn, fp, fn, tp]
        assert row.total_cost == cost_fp * fp + cost_fn * fn
    best = select_threshold(table)
    assert table.loc[table.threshold == best, "total_cost"].iloc[0] == table.total_cost.min()
    assert best == table.loc[table.total_cost == table.total_cost.min(), "threshold"].max()
    assert (
        table.loc[table.threshold == select_threshold(table, "f1"), "f1"].iloc[0] == table.f1.max()
    )


def test_known_optimal_threshold():
    table = threshold_table([0, 0, 1, 1], [0.1, 0.2, 0.4, 0.8], 1, 5)
    assert select_threshold(table) == 0.4
    assert table.total_cost.min() == 0
    assert select_threshold(threshold_table([0, 0], [1, 1], 1, 5)) > 1
    assert select_threshold(threshold_table([1, 1], [0, 0], 1, 5)) == 0


@pytest.mark.parametrize("p,cost", [([np.nan], 1), ([1.1], 1), ([0.5], -1)])
def test_invalid_threshold_inputs(p, cost):
    with pytest.raises(ValueError):
        threshold_table([0], p, cost, 5)


@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_calibrator(method, sample_data):
    _, y = sample_data
    p = np.linspace(0.01, 0.99, len(y))
    calibrator = ProbabilityCalibrator(method).fit(p, y)
    calibrated = calibrator.predict([0, 0.5, 1])
    assert np.isfinite(calibrated).all() and ((calibrated >= 0) & (calibrated <= 1)).all()


def test_calibration_selection_uses_internal_validation(sample_data):
    X, y = sample_data
    p = np.clip(X.iloc[:, 0].to_numpy() / 1.5, 0, 1)
    calibrator, results = choose_calibrator(p, y, groups=np.arange(len(y)))
    assert set(results.method) == {"none", "sigmoid", "isotonic"}
    assert (results.fit_size + results.validation_size == len(y)).all()
    assert calibrator.method == results.sort_values(["brier_score", "log_loss"]).iloc[0].method


def test_inference_single_batch_and_calibration(trained_predictor, sample_data):
    X, _ = sample_data
    batch = trained_predictor.predict(X.head(3))
    single = trained_predictor.predict(X.head(1))
    assert single.risk_probability.iloc[0] == pytest.approx(
        batch.risk_probability.iloc[0], abs=1e-14
    )
    expected = trained_predictor.calibrator.predict(
        trained_predictor.pipeline.predict_proba(X.head(3))[:, 1]
    )
    np.testing.assert_allclose(batch.risk_probability, expected)
    assert batch.risk_probability.between(0, 1).all()
    assert (
        (batch.risk_probability >= trained_predictor.threshold)
        == (batch.risk_class == "above_threshold")
    ).all()
    changed = trained_predictor.predict(X.head(3), threshold=1)
    np.testing.assert_array_equal(batch.risk_probability, changed.risk_probability)
    with pytest.raises(ValueError):
        trained_predictor.predict(X.drop(columns="age"))


def test_serialization_checksum_and_schema(trained_predictor, sample_data, tmp_path):
    X, _ = sample_data
    save_predictor(trained_predictor, tmp_path, {"model_name": "test"})
    loaded, metadata = load_predictor(tmp_path)
    pd.testing.assert_frame_equal(loaded.predict(X.head()), trained_predictor.predict(X.head()))
    assert metadata["artifact_version"] == 1
    (tmp_path / "pipeline.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_predictor(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_predictor(tmp_path / "missing")


def test_shap_names_dimensions_and_additivity(trained_predictor, sample_data):
    X, _ = sample_data
    explanation, space = explain_pipeline(
        trained_predictor.pipeline, X.head(3), trained_predictor.background
    )
    assert explanation.values.shape == (3, len(explanation.feature_names))
    assert "log-odds" in space
    expected = trained_predictor.pipeline.decision_function(X.head(3))
    np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), expected)
    with pytest.raises(ExplanationUnavailable):
        explain_pipeline(trained_predictor.pipeline, X.head())


def test_confusion_and_cost_definitions():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], threshold=0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["total_cost"] == 6
    assert result["cost_per_record"] == 1.5
