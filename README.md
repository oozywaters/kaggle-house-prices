# Kaggle House Prices

Boilerplate for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) competition.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Get the data

Set up [Kaggle API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`), accept the competition rules on the Kaggle website, then:

```sh
uv run python src/download_data.py
```

This downloads `train.csv`, `test.csv`, `data_description.txt`, and `sample_submission.csv` into `data/` (git-ignored).

## Train

```sh
uv run python main.py
```

Trains the model from `configs/config.yaml` (CatBoost by default) on `log1p(SalePrice)`, prints K-fold CV RMSE on the log scale (the leaderboard metric), and writes `submissions/<model>_submission.csv`.

Any config field can be overridden from the CLI:

```sh
uv run python main.py model=ridge
uv run python main.py model=lightgbm params.lightgbm.n_estimators=800 cv.n_folds=10
```

Available models: `linreg`, `ridge`, `lasso`, `elasticnet`, `knn`, `decision_tree`, `random_forest`, `catboost`, `lightgbm`, `xgboost`.

## Submit

```sh
uv run kaggle competitions submit -c house-prices-advanced-regression-techniques -f submissions/catboost_submission.csv -m "baseline catboost"
```

## Project layout

```
├── main.py                # entry point: load config, train with CV, write submission
├── configs/
│   └── config.yaml        # experiment settings (model, hyperparams, CV)
├── src/
│   ├── config.py          # config schema + loader (OmegaConf)
│   ├── download_data.py   # fetch competition data via Kaggle API
│   ├── features.py        # missing-value cleaning, ordinal encoding, feature engineering
│   └── pipeline.py        # preprocessing + model pipeline
├── notebooks/             # exploratory analysis
├── data/                  # raw data (git-ignored)
└── submissions/           # generated submissions (git-ignored)
```
