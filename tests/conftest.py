import numpy as np
import pandas as pd
import pytest

from creditguard.data import FEATURES


@pytest.fixture
def sample_data():
    """Synthetic fixtures only; never used as portfolio experiment results."""
    rng = np.random.default_rng(17)
    n = 700
    X = pd.DataFrame(
        {
            FEATURES[0]: rng.uniform(0, 1.5, n),
            "age": rng.integers(21, 85, n),
            FEATURES[2]: rng.poisson(0.5, n),
            "DebtRatio": rng.lognormal(-1, 1, n),
            "MonthlyIncome": rng.lognormal(8, 0.5, n),
            FEATURES[5]: rng.integers(0, 15, n),
            FEATURES[6]: rng.poisson(0.15, n),
            FEATURES[7]: rng.integers(0, 4, n),
            FEATURES[8]: rng.poisson(0.2, n),
            FEATURES[9]: rng.integers(0, 4, n),
        }
    )
    score = -3.5 + 2.5 * X[FEATURES[0]] + X[FEATURES[6]] + 0.5 * X[FEATURES[2]]
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-score))), name="SeriousDlqin2yrs")
    X.loc[rng.choice(n, 60, replace=False), "MonthlyIncome"] = np.nan
    X.loc[rng.choice(n, 25, replace=False), "NumberOfDependents"] = np.nan
    return X, y


@pytest.fixture
def trained_predictor(sample_data):
    from creditguard.calibration import ProbabilityCalibrator
    from creditguard.config import load_config
    from creditguard.modeling import build_pipeline
    from creditguard.predict import RiskPredictor

    X, y = sample_data
    config = load_config(profile="quick")
    pipeline = build_pipeline("logistic", config, engineered=True).fit(X, y)
    calibrator = ProbabilityCalibrator("sigmoid").fit(pipeline.predict_proba(X)[:, 1], y)
    return RiskPredictor(
        pipeline, calibrator, 0.25, "test_logistic", pipeline[:-1].transform(X.iloc[:40])
    )
