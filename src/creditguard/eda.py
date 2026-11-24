"""Training-only exploratory analysis, with aggregate plots and no customer exports."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from creditguard.config import load_config
from creditguard.data import DELINQUENCIES, FEATURES, feature_groups, load_dataset, make_splits
from creditguard.metrics import write_json


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def create_eda(X, y, report_dir):
    """X and y must be the training partition, never the final holdout."""
    root = Path(report_dir)
    figures, metrics = root / "figures", root / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    grouped = pd.DataFrame({"g": feature_groups(X).to_numpy(), "y": np.asarray(y)})
    summary = {
        "partition": "train",
        "n_records": len(X),
        "n_features": len(FEATURES),
        "positive_rate": float(y.mean()),
        "dtypes": X.dtypes.astype(str).to_dict(),
        "missing": X.isna().sum().to_dict(),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "conflicting_feature_groups": int(grouped.groupby("g").y.nunique().gt(1).sum()),
        "zero_age": int(X.age.eq(0).sum()),
        "utilization_above_one": int(X[FEATURES[0]].gt(1).sum()),
        "special_delinquency_codes": {c: int(X[c].isin([96, 98]).sum()) for c in DELINQUENCIES},
        "all_three_codes_cooccur": int(X[DELINQUENCIES].isin([96, 98]).all(axis=1).sum()),
    }
    write_json(metrics / "training_eda.json", summary)
    X.describe(percentiles=[0.01, 0.5, 0.95, 0.995]).T.to_csv(
        metrics / "training_descriptive_statistics.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["No event", "Serious delinquency"],
        [(y == 0).sum(), (y == 1).sum()],
        color=["#7198c4", "#c77676"],
    )
    axes[0].set(title="Training class distribution", ylabel="Number of records")
    missing = X.isna().mean().mul(100).sort_values(ascending=False)
    axes[1].barh(missing.index, missing.values, color="#7198c4")
    axes[1].set(title="Training missing values", xlabel="Missing records (%)")
    axes[1].tick_params(axis="y", labelsize=7)
    save_figure(fig, figures / "eda_class_and_missing.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlations = X.corr(method="spearman")
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=7)
    ax.set_title("Training Spearman correlations (including original special codes)")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    save_figure(fig, figures / "eda_correlations.png")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for column, ax in zip(FEATURES, axes.ravel(), strict=True):
        logarithmic = column in {"MonthlyIncome", "DebtRatio", FEATURES[0]}
        values = np.log1p(X[column]) if logarithmic else X[column]
        for label, color in [(0, "#7198c4"), (1, "#c77676")]:
            ax.hist(
                values.loc[y == label].dropna(),
                bins=35,
                alpha=0.45,
                density=True,
                label=f"Target {label}",
                color=color,
            )
        ax.set(title=column, xlabel="log(1 + value)" if logarithmic else "Value", ylabel="Density")
        ax.title.set_fontsize(7)
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, figures / "eda_target_distributions.png")
    rate_rows = []
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for column, ax in zip(FEATURES, axes.ravel(), strict=True):
        values = X[column]
        if values.nunique() > 10:
            bins = pd.qcut(values, 8, duplicates="drop").astype(str)
        else:
            bins = values.astype(str)
        bins = bins.mask(values.isna(), "Missing")
        table = (
            pd.DataFrame({"bin": bins, "target": np.asarray(y)})
            .groupby("bin", observed=True)
            .target.agg(["mean", "size"])
        )
        # Preserve numerical quantile order rather than lexicographic interval order.
        ordered = (
            pd.DataFrame({"bin": bins, "value": values})
            .groupby("bin", observed=True)
            .value.mean()
            .sort_values()
            .index
        )
        table = table.reindex(ordered)
        ax.bar(range(len(table)), table["mean"], color="#7198c4")
        ax.set(title=column, xlabel="Bins ordered by mean value; missing last", ylabel="Event rate")
        ax.title.set_fontsize(7)
        ax.set_xticks(range(len(table)), range(1, len(table) + 1), fontsize=7)
        for bin_name, row in table.iterrows():
            rate_rows.append(
                {
                    "feature": column,
                    "bin": bin_name,
                    "n_records": int(row["size"]),
                    "event_rate": row["mean"],
                }
            )
    save_figure(fig, figures / "eda_feature_event_rates.png")
    pd.DataFrame(rate_rows).to_csv(metrics / "training_feature_event_rates.csv", index=False)
    patterns = pd.Series("No delinquencies", index=X.index)
    patterns.loc[X[DELINQUENCIES].sum(axis=1) > 0] = "Any delinquency"
    patterns.loc[X[DELINQUENCIES[2]] > 0] = "90+ days"
    patterns.loc[X[DELINQUENCIES].isin([96, 98]).any(axis=1)] = "Special code 96/98"
    table = (
        pd.DataFrame({"pattern": patterns, "target": y})
        .groupby("pattern")
        .target.agg(["size", "mean"])
    )
    table.to_csv(metrics / "training_delinquency_patterns.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(table.index, table["size"], color="#7198c4")
    axes[1].bar(table.index, table["mean"], color="#c77676")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    axes[0].set(title="Training delinquency patterns", ylabel="Number of records")
    axes[1].set(title="Event frequency by delinquency pattern", ylabel="Event rate")
    save_figure(fig, figures / "eda_delinquency_patterns.png")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    X, y = load_dataset(config["dataset"])
    splits = make_splits(X, y, config["seed"], config["split_proportions"])
    print(create_eda(X.iloc[splits["train"]], y.iloc[splits["train"]], config["report_dir"]))


if __name__ == "__main__":
    main()
