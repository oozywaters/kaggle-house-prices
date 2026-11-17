import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    RandomForestRegressor, StackingRegressor, VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.features import HousePricesFeatures
from src.nn import TorchMLPRegressor

MODELS = {
    "linreg": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "knn": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "catboost": CatBoostRegressor,
    "lightgbm": LGBMRegressor,
    "xgboost": XGBRegressor,
    "mlp": TorchMLPRegressor,
}

def _ensemble_bases(seed):
    return [
        ("ridge", Ridge(alpha=10.0)),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=seed)),
        ("lgbm", LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1,
                               random_state=seed)),
    ]

# voting = average the base models; stacking = a meta-model learns how to combine them
ENSEMBLES = {
    "voting": lambda seed: VotingRegressor(_ensemble_bases(seed)),
    "stacking_linreg": lambda seed: StackingRegressor(
        _ensemble_bases(seed), final_estimator=LinearRegression()),
    "stacking_ridge": lambda seed: StackingRegressor(
        _ensemble_bases(seed), final_estimator=Ridge(alpha=10.0)),
}

TARGET = "SalePrice"

def make_folds(X, y, cfg):
    """Stratified folds on binned SalePrice (so each fold has a similar price mix)."""
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.seed)
    return list(cv.split(X, bins))

def build_preprocessor() -> ColumnTransformer:
    num_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer([
        ("num", Pipeline(num_steps), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline(cat_steps), make_column_selector(dtype_include=object)),
    ])

def build_model(cfg):
    if cfg.model in ENSEMBLES:
        return ENSEMBLES[cfg.model](cfg.seed)
    if cfg.model not in MODELS:
        raise ValueError(f"unknown model {cfg.model!r}, "
                         f"choose from {sorted(MODELS) + sorted(ENSEMBLES)}")
    params = OmegaConf.to_container(cfg.params, resolve=True).get(cfg.model, {})
    return MODELS[cfg.model](**params)

def build_pipeline(cfg) -> Pipeline:
    return Pipeline([
        ("features", HousePricesFeatures()),
        ("preprocess", build_preprocessor()),
        ("model", build_model(cfg)),
    ])
