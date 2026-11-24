from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from creditguard.metrics import classification_metrics, write_json
from creditguard.predict import save_predictor

APP = Path(__file__).resolve().parents[1] / "app/main.py"


def test_app_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("CREDITGUARD_MODEL_DIR", str(tmp_path / "missing"))
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert any("not available" in element.value for element in app.info)


@pytest.fixture
def app_with_artifacts(monkeypatch, tmp_path, trained_predictor, sample_data):
    models, reports = tmp_path / "models", tmp_path / "reports"
    save_predictor(trained_predictor, models, {"calibration_method": "sigmoid"})
    X, y = sample_data
    final = classification_metrics(y, trained_predictor.predict(X).risk_probability, 0.25)
    write_json(reports / "metrics/final_test_metrics.json", final)
    write_json(
        reports / "metrics/selection_frozen.json",
        {
            "model_name": "synthetic_test_model",
            "calibration_method": "sigmoid",
            "threshold": 0.25,
            "cost_fp": 1,
            "cost_fn": 5,
        },
    )
    pd.DataFrame([{"model": "synthetic_test_model", **final}]).to_csv(
        reports / "metrics/model_comparison.csv", index=False
    )
    monkeypatch.setenv("CREDITGUARD_MODEL_DIR", str(models))
    monkeypatch.setenv("CREDITGUARD_REPORT_DIR", str(reports))
    return AppTest.from_file(str(APP), default_timeout=15).run()


def test_prediction_form_and_threshold_do_not_retrain(app_with_artifacts):
    app = app_with_artifacts
    assert not app.exception
    app.button(key="FormSubmitter:risk_form-Calculate Risk").click().run()
    assert not app.exception
    probability = app.metric[0].value
    app.slider(key="decision_threshold").set_value(0.8).run()
    assert not app.exception
    assert app.metric[0].value == probability
    app.button[2].click().run()
    assert not app.exception


@pytest.mark.parametrize(
    "page,title",
    [("views/model_performance.py", "Model Performance"), ("views/about.py", "About CreditGuard")],
)
def test_other_pages(app_with_artifacts, page, title):
    app = app_with_artifacts.switch_page(page).run()
    assert not app.exception
    assert any(element.value == title for element in app.header)
