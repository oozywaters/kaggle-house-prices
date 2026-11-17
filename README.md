# Kaggle House Prices

ML pipeline for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) competition.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Get the data

The data is committed to the repo, so this step is optional — only needed to refresh it. Set up [Kaggle API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`), accept the competition rules on the Kaggle website, then:

```sh
uv run python src/download_data.py
```

This downloads `train.csv`, `test.csv`, `data_description.txt`, and `sample_submission.csv` into `data/`.

## Train

```sh
uv run python main.py
```

Trains the model from `configs/config.yaml` (CatBoost by default) on `log1p(SalePrice)`, prints stratified K-fold CV RMSE on the log scale (the leaderboard metric), and writes `submissions/<model>_submission.csv`. Folds are stratified on binned price so each fold has a similar price mix.

Any config field can be overridden from the CLI:

```sh
uv run python main.py model=ridge
uv run python main.py model=lightgbm params.lightgbm.n_estimators=800 cv.n_folds=10
```

Available models: `linreg` (baseline), `ridge`, `lasso`, `elasticnet`, `knn`, `decision_tree`, `random_forest`, `catboost`, `lightgbm`, `xgboost`, `mlp` (PyTorch), and ensembles `voting`, `stacking_linreg`, `stacking_ridge`.

## Compare models

```sh
uv run python compare.py
```

Cross-validates every model and prints a ranked table. Best so far: **catboost, CV RMSE 0.1223**. Full analysis in `notebooks/eda.ipynb`.

## Submit

```sh
uv run kaggle competitions submit -c house-prices-advanced-regression-techniques -f submissions/catboost_submission.csv -m "baseline catboost"
```

## Project layout

```
├── main.py                # entry point: load config, train with CV, write submission
├── compare.py             # cross-validate all models, print ranked table
├── configs/
│   └── config.yaml        # experiment settings (model, hyperparams, CV)
├── src/
│   ├── config.py          # config schema + loader (OmegaConf)
│   ├── download_data.py   # fetch competition data via Kaggle API
│   ├── features.py        # missing-value cleaning, ordinal encoding, feature engineering
│   ├── nn.py              # PyTorch MLP wrapped as a sklearn regressor
│   └── pipeline.py        # preprocessing + model pipeline
├── notebooks/             # EDA + model comparison (eda.ipynb)
├── data/                  # raw competition data
└── submissions/           # generated submissions (git-ignored)
```
