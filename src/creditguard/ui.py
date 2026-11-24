"""Local Streamlit pages using saved artifacts and the shared inference interface."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from creditguard.data import FEATURES
from creditguard.demo import DEMO_PROFILES
from creditguard.explain import ExplanationUnavailable, explain_pipeline
from creditguard.predict import load_predictor

LABELS = dict(
    zip(
        FEATURES,
        [
            "Credit utilization",
            "Age",
            "30–59 days past due",
            "Debt ratio",
            "Monthly income",
            "Open credit lines and loans",
            "90+ days past due",
            "Real estate loans or lines",
            "60–89 days past due",
            "Dependents",
        ],
        strict=True,
    )
)
LABELS.update(
    {
        "total_past_due": "Total past-due count",
        "weighted_past_due": "Severity-weighted past-due count",
        "has_past_due": "Any past-due payments",
        "has_severe_delinquency": "Any 90+ day delinquency",
        "special_delinquency_code": "Unresolved delinquency code",
        "log_income": "Log monthly income",
        "log_debt_ratio": "Log debt ratio",
        "log_utilization": "Log utilization",
        "income_per_household_member": "Income per household member",
        "real_estate_share": "Real estate share of credit lines",
        "missing_income": "Income unavailable",
        "missing_dependents": "Dependents unavailable",
    }
)


def readable_name(name):
    if name.startswith("missingindicator_"):
        return "Missing: " + LABELS.get(name.removeprefix("missingindicator_"), name)
    return LABELS.get(name, name.replace("_", " "))


def paths():
    return Path(os.environ.get("CREDITGUARD_MODEL_DIR", "models/final")), Path(
        os.environ.get("CREDITGUARD_REPORT_DIR", "reports")
    )


@st.cache_resource(show_spinner="Loading trained model…")
def cached_predictor(directory, timestamp):
    return load_predictor(directory)


def get_model():
    directory, _ = paths()
    try:
        timestamp = (directory / "metadata.json").stat().st_mtime_ns
        return cached_predictor(str(directory), timestamp)
    except (FileNotFoundError, ValueError) as error:
        st.info(
            str(error)
            if isinstance(error, ValueError)
            else "Trained model artifacts are not available yet."
        )
        st.code("python -m creditguard.train --profile full", language="bash")
        return None


def risk_page():
    st.header("Risk Assessment")
    st.write(
        "Estimate serious delinquency risk over the next two years "
        "using historical credit characteristics."
    )
    loaded = get_model()
    if loaded is None:
        return
    predictor, metadata = loaded
    for feature, value in DEMO_PROFILES["intermediate"].items():
        st.session_state.setdefault(feature, value)
    examples = {
        "Lower-risk example": "lower",
        "Intermediate-risk example": "intermediate",
        "Higher-risk example": "higher",
    }
    example = st.selectbox("Synthetic demonstration profile", list(examples))
    if st.button("Load example"):
        for feature, value in DEMO_PROFILES[examples[example]].items():
            st.session_state[feature] = value
        st.session_state["missing_income"] = False
        st.session_state["missing_dependents"] = False
        st.session_state.pop("risk_result", None)
    st.caption(
        "Examples are synthetic and were checked for relative risk order using the saved model."
    )
    with st.form("risk_form"):
        personal, credit, history = st.columns(3)
        with personal:
            st.subheader("Personal information")
            st.number_input("Age", min_value=18, max_value=120, step=1, key="age")
            st.number_input(
                "Monthly income",
                min_value=0,
                step=100,
                key="MonthlyIncome",
                help="Dataset units; the data dictionary does not specify a currency.",
            )
            missing_income = st.checkbox("Income unavailable", key="missing_income")
            st.number_input("Dependents", min_value=0, step=1, key="NumberOfDependents")
            missing_dependents = st.checkbox("Dependents unavailable", key="missing_dependents")
        with credit:
            st.subheader("Credit information")
            st.number_input(
                "Credit utilization",
                min_value=0.0,
                step=0.05,
                key=FEATURES[0],
                help="Balance divided by credit limits: 0.25 represents 25%. "
                "Values above 1 are supported.",
            )
            st.number_input(
                "Debt ratio",
                min_value=0.0,
                step=0.05,
                key="DebtRatio",
                help="Monthly debt payments, alimony, and living costs divided by "
                "monthly gross income, as defined in the dataset.",
            )
            st.number_input("Open credit lines and loans", min_value=0, step=1, key=FEATURES[5])
            st.number_input("Real estate loans or lines", min_value=0, step=1, key=FEATURES[7])
        with history:
            st.subheader("Payment history")
            for feature in [FEATURES[2], FEATURES[8], FEATURES[6]]:
                st.number_input(LABELS[feature], min_value=0, step=1, key=feature)
            st.caption(
                "Delinquency counts follow the dataset definitions. Codes 96 and 98 "
                "are treated as unresolved values by preprocessing."
            )
        submitted = st.form_submit_button("Calculate Risk", type="primary")
    if submitted:
        record = {feature: st.session_state[feature] for feature in FEATURES}
        if missing_income:
            record["MonthlyIncome"] = np.nan
        if missing_dependents:
            record["NumberOfDependents"] = np.nan
        frame = pd.DataFrame([record])
        try:
            p = float(predictor.predict(frame).risk_probability.iloc[0])
            result = {"probability": p}
            try:
                explanation, space = explain_pipeline(
                    predictor.pipeline, frame, predictor.background
                )
                result.update(
                    {
                        "shap": explanation.values[0],
                        "base": float(np.asarray(explanation.base_values).reshape(-1)[0]),
                        "features": explanation.feature_names,
                        "space": space,
                    }
                )
            except (ExplanationUnavailable, ValueError) as error:
                result["explanation_error"] = str(error)
            st.session_state["risk_result"] = result
        except ValueError as error:
            st.error(str(error))
    if "risk_result" not in st.session_state:
        return
    result = st.session_state["risk_result"]
    st.divider()
    left, right = st.columns(2)
    left.metric("Estimated risk probability", f"{result['probability']:.2%}")
    right.metric("Recommended threshold", f"{predictor.threshold:.4f}")
    st.caption(
        f"Model: {predictor.model_name} · Calibration: "
        f"{metadata.get('calibration_method', predictor.calibrator.method)}"
    )
    if st.button("Reset to recommended threshold"):
        st.session_state["decision_threshold"] = predictor.threshold
    decision = st.slider(
        "Classification threshold",
        0.0,
        max(1.0, predictor.threshold),
        value=predictor.threshold,
        step=0.001,
        format="%.4f",
        key="decision_threshold",
    )
    st.progress(result["probability"])
    st.write("Above threshold" if result["probability"] >= decision else "Below threshold")
    fig, ax = plt.subplots(figsize=(9, 1.4))
    ax.barh(["Risk"], [result["probability"]], color="#7198c4")
    ax.axvline(decision, color="#c77676", linestyle="--", label=f"Threshold {decision:.4f}")
    ax.set(xlim=(0, max(1, decision)), xlabel="Predicted probability")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    if "shap" in result:
        st.subheader("Factors behind this prediction")
        st.caption(
            f"SHAP explains {result['space']}. It does not explain the calibrated "
            f"probability or establish causation. Base value: {result['base']:.4f}."
        )
        table = pd.DataFrame(
            {
                "Feature": [readable_name(name) for name in result["features"]],
                "Contribution": result["shap"],
            }
        )
        table = table.iloc[np.argsort(np.abs(table.Contribution))[-10:]]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(
            table.Feature,
            table.Contribution,
            color=["#c77676" if value > 0 else "#7198c4" for value in table.Contribution],
        )
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel(result["space"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(result.get("explanation_error", "Explanation unavailable"))
    st.caption(
        "Changing the threshold changes only classification. Inputs stay in this "
        "local session and are not saved or sent to an external service."
    )


def performance_page():
    st.header("Model Performance")
    _, report_dir = paths()
    metric_dir = report_dir / "metrics"
    required = [
        metric_dir / name
        for name in ["final_test_metrics.json", "selection_frozen.json", "model_comparison.csv"]
    ]
    if not all(path.is_file() for path in required):
        st.info("Experiment reports are not available yet. Run the full training command first.")
        st.code("python -m creditguard.train --profile full", language="bash")
        return
    final, metadata = [json.loads(path.read_text()) for path in required[:2]]
    st.write(
        f"Selected model: **{metadata['model_name']}** · "
        f"Calibration: **{metadata['calibration_method']}**"
    )
    columns = st.columns(3)
    for column, label, key in zip(
        columns,
        ["ROC-AUC", "Average Precision", "Brier score"],
        ["roc_auc", "average_precision", "brier_score"],
        strict=True,
    ):
        column.metric(label, f"{final[key]:.4f}")
    st.caption(
        f"Independent final test: {final['n_records']:,} records · "
        f"Frozen threshold: {metadata['threshold']:.4f} · "
        f"Conditional C_FP:C_FN = {metadata['cost_fp']:g}:{metadata['cost_fn']:g}"
    )
    st.subheader("Model selection comparison")
    comparison = pd.read_csv(required[2])
    st.dataframe(
        comparison[
            [
                "model",
                "roc_auc",
                "average_precision",
                "brier_score",
                "precision",
                "recall",
                "f1",
                "threshold",
            ]
        ],
        hide_index=True,
    )
    st.caption(
        "Comparison uses the model-selection set at threshold 0.5; "
        "final test metrics above evaluate only the frozen selected system."
    )
    tabs = st.tabs(["Final evaluation", "Calibration and threshold", "Global explanations"])
    for tab, images in zip(
        tabs,
        [
            ["final_evaluation.png"],
            ["calibration_curve.png", "threshold_optimization.png"],
            ["shap_importance.png", "shap_summary.png"],
        ],
        strict=True,
    ):
        with tab:
            for name in images:
                image = report_dir / "figures" / name
                if image.is_file():
                    st.image(str(image), width="stretch")
    st.caption(
        "Threshold and calibration plots use their stated development holdouts. "
        "This page does not select a threshold on the final test."
    )


def about_page():
    st.header("About CreditGuard")
    st.write(
        "CreditGuard studies serious delinquency risk within two years using the "
        "historical Give Me Some Credit dataset. The target is 90-day delinquency "
        "or worse, following the dataset definition."
    )
    st.markdown(
        "[Dataset and competition terms](https://www.kaggle.com/c/GiveMeSomeCredit/data) · "
        "[Source code](https://github.com/SanyaLikeIT/credit-guard)"
    )
    st.subheader("Method")
    st.write(
        "Feature-group holdouts allocate approximately 60% to training, 15% to model "
        "selection, 10% to calibration, 5% to threshold optimization, and 10% to final "
        "evaluation. Duplicate feature vectors and conflicting labels stay in one "
        "partition. Searches use group-aware cross-validation inside training data."
    )
    st.write(
        "Dummy, Logistic Regression, Random Forest, and CatBoost are compared by "
        "Average Precision. Calibration is compared on an internal calibration holdout. "
        "A separate threshold holdout minimizes C_FP × FP + C_FN × FN. "
        "The default 1:5 costs are hypothetical research assumptions."
    )
    st.write(
        "SHAP explains the uncalibrated model output in the units shown beside each "
        "explanation. The interface loads saved models and reports without training."
    )
    st.subheader("Limitations")
    st.write(
        "Historical observations, no external or temporal validation, possible "
        "distribution shift, and limited demographic information prevent conclusions "
        "about present-day lending or fairness. Removing age alone cannot establish "
        "fairness. Subgroup results are descriptive and depend on sample size."
    )
    st.caption("Python · pandas · scikit-learn · CatBoost · SHAP · Streamlit")
    st.info(
        "This project is intended for educational and research purposes. "
        "It is not a production credit decision system or financial advice."
    )
