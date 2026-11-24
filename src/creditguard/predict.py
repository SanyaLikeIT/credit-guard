"""Trusted local artifacts and a shared inference interface for CLI and Streamlit."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from creditguard.data import FEATURES, TARGET, validate_features
from creditguard.metrics import write_json

ARTIFACT_VERSION = 1


class RiskPredictor:
    def __init__(self, pipeline, calibrator, threshold, model_name, background=None):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.model_name = model_name
        self.background = background

    def predict(self, frame, threshold=None):
        frame = validate_features(frame)
        threshold = self.threshold if threshold is None else float(threshold)
        if not 0 <= threshold <= 1.000000000000001:
            raise ValueError("Threshold must be in [0, 1] (or the all-negative floating boundary)")
        p = self.calibrator.predict(self.pipeline.predict_proba(frame)[:, 1])
        return pd.DataFrame(
            {
                "risk_probability": p,
                "classification_threshold": threshold,
                "risk_class": [
                    "above_threshold" if value >= threshold else "below_threshold" for value in p
                ],
                "model_name": self.model_name,
                "calibrated": self.calibrator.method != "none",
            },
            index=frame.index,
        )


def save_predictor(predictor, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pipeline.joblib"
    joblib.dump(predictor, path, compress=3)
    metadata = {
        **metadata,
        "artifact_version": ARTIFACT_VERSION,
        "feature_schema": FEATURES,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scikit_learn_version": version("scikit-learn"),
    }
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_predictor(directory="models/final"):
    """Load only artifacts from a trusted local training run. Never accept uploaded joblib files."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file() or not (directory / "pipeline.joblib").is_file():
        raise FileNotFoundError(
            "Trained artifacts are missing. Run: python -m creditguard.train --profile full"
        )
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("artifact_version") != ARTIFACT_VERSION
        or metadata.get("feature_schema") != FEATURES
    ):
        raise ValueError("Incompatible model artifact schema; retrain with this package")
    current = version("scikit-learn").split(".")[:2]
    if metadata["scikit_learn_version"].split(".")[:2] != current:
        raise ValueError("scikit-learn version differs from training; use the recorded environment")
    path = directory / "pipeline.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
        raise ValueError("Artifact checksum mismatch; refusing to load")
    predictor = joblib.load(path)
    if not isinstance(predictor, RiskPredictor):
        raise ValueError("Unexpected artifact type")
    return predictor, metadata


def main():
    from creditguard.demo import DEMO_PROFILES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/final")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--json", type=Path, help="Path to a JSON object of raw features")
    inputs.add_argument("--csv", type=Path, help="Path to a CSV of raw features")
    inputs.add_argument("--demo", choices=list(DEMO_PROFILES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        predictor, _ = load_predictor(args.model_dir)
        if args.csv:
            frame = pd.read_csv(args.csv)
            frame = frame.drop(
                columns=[column for column in frame if column.startswith("Unnamed:")]
            )
            if TARGET in frame and frame[TARGET].isna().all():
                frame = frame.drop(columns=TARGET)
        else:
            record = DEMO_PROFILES[args.demo] if args.demo else json.loads(args.json.read_text())
            frame = pd.DataFrame([record])
        predictions = predictor.predict(frame)
        if args.output:
            predictions.to_csv(args.output, index=False)
            print(f"Wrote {len(predictions)} predictions to {args.output}")
        else:
            print(predictions.to_json(orient="records", indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
