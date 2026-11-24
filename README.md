# CreditGuard

CreditGuard is a research project for predicting serious delinquency within two years using the
historical **Give Me Some Credit** dataset. Implementation and real-data experiments are in progress.
No model quality results have been measured yet.

## Dataset access

1. Sign in to [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data).
2. Review and accept the applicable competition terms when prompted.
3. Download `cs-training.csv`, `cs-test.csv`, and the data dictionary.
4. Place the files in `data/raw/` (create this directory if necessary).

The dataset license is subject to the [competition rules](https://www.kaggle.com/competitions/GiveMeSomeCredit/rules).
This repository does not redistribute the source data. Do not upload credentials or raw records.
The original `cs-test.csv` has no public labels and is not the project's final evaluation set.

## Development environment

Use a local environment; no system packages are required:

```fish
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -e '.[dev]'
```

Training and application commands will be documented with their verified behavior as implementation
progresses. Main remains unchanged until validation succeeds.

This project is intended for educational and research purposes. It is not a production credit
decision system or financial advice.
