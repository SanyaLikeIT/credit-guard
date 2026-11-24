"""Experiment configuration and portable paths."""

from pathlib import Path

import yaml


def load_config(path="configs/default.yaml", profile="full"):
    source = Path(path).resolve()
    with source.open() as handle:
        config = yaml.safe_load(handle)
    if profile not in config["profiles"]:
        raise ValueError(f"Unknown training profile: {profile}")
    config.update(config.pop("profiles")[profile])
    config["profile"] = profile
    root = source.parent.parent
    for key in ("dataset", "model_dir", "report_dir"):
        value = Path(config[key])
        config[key] = root / value if not value.is_absolute() else value
    return config
