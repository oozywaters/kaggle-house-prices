import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
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

TARGET = "SalePrice"

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
    if cfg.model not in MODELS:
        raise ValueError(f"unknown model {cfg.model!r}, choose from {sorted(MODELS)}")
    params = OmegaConf.to_container(cfg.params, resolve=True).get(cfg.model, {})
    return MODELS[cfg.model](**params)

def build_pipeline(cfg) -> Pipeline:
    return Pipeline([
        ("features", HousePricesFeatures()),
        ("preprocess", build_preprocessor()),
        ("model", build_model(cfg)),
    ])
