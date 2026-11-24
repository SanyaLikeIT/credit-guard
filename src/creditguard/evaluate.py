"""Read and verify the frozen final evaluation without retraining or retuning."""

import argparse
import json
from pathlib import Path

from creditguard.predict import load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/final")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    try:
        _, metadata = load_predictor(args.model_dir)
        metrics = json.loads(
            (Path(args.report_dir) / "metrics/final_test_metrics.json").read_text()
        )
        if (
            metrics["artifact_sha256"] != metadata["artifact_sha256"]
            or metrics["dataset_sha256"] != metadata["dataset_sha256"]
        ):
            raise ValueError("Evaluation does not match the saved model and dataset")
        print(json.dumps({"model": metadata["model_name"], "evaluation": metrics}, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
