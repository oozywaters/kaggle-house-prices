import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# NA here means "no such feature", not missing -> fill with "None"
NONE_CATEGORICAL = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
    "MiscFeature",
]

# NA here means "none of it" -> fill with 0
ZERO_NUMERIC = [
    "MasVnrArea", "GarageYrBlt", "GarageCars", "GarageArea", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
]

# quality grades as ordered numbers: Po < Fa < TA < Gd < Ex
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

# MSSubClass is a number but really a category code
AS_STRING = ["MSSubClass"]

def fill_none_and_zero(df: pd.DataFrame) -> pd.DataFrame:
    none_fill = {col: "None" for col in NONE_CATEGORICAL if col in df}
    zero_fill = {col: 0 for col in ZERO_NUMERIC if col in df}
    return df.fillna({**none_fill, **zero_fill})

def encode_quality(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in QUALITY_COLUMNS if c in df]
    mapped = {c: df[c].map(QUALITY_MAP).fillna(0).astype(int) for c in cols}
    return df.assign(**mapped)

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    total_sf = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    total_bath = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    porch_sf = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
        + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    return df.assign(
        TotalSF=total_sf,
        TotalBath=total_bath,
        TotalPorchSF=porch_sf,
        HouseAge=df["YrSold"] - df["YearBuilt"],
        RemodAge=df["YrSold"] - df["YearRemodAdd"],
        HasPool=(df["PoolArea"] > 0).astype(int),
        Has2ndFloor=(df["2ndFlrSF"] > 0).astype(int),
        HasGarage=(df["GarageArea"] > 0).astype(int),
        HasBsmt=(df["TotalBsmtSF"] > 0).astype(int),
        HasFireplace=(df["Fireplaces"] > 0).astype(int),
    )

class HousePricesFeatures(BaseEstimator, TransformerMixin):
    """Clean NaNs, turn quality grades into numbers, add a few new features.

    Most NaNs mean "no such feature" (PoolQC NA = no pool), so they're filled
    with "None"/0, not guessed. LotFrontage is filled with the median of its
    Neighborhood, learned per CV fold so no data leaks. See notebooks/eda.ipynb.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.lot_frontage_by_neighborhood_ = (
            X.groupby("Neighborhood")["LotFrontage"].median()
        )
        self.lot_frontage_global_ = X["LotFrontage"].median()
        return self

    def _fill_lot_frontage(self, df: pd.DataFrame) -> pd.Series:
        by_neighborhood = df["Neighborhood"].map(self.lot_frontage_by_neighborhood_)
        return df["LotFrontage"].fillna(by_neighborhood).fillna(self.lot_frontage_global_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.drop(columns=["Id"], errors="ignore").copy()
        df["LotFrontage"] = self._fill_lot_frontage(df)
        df[AS_STRING] = df[AS_STRING].astype(str)
        df = fill_none_and_zero(df)
        df = encode_quality(df)
        return add_engineered(df)
