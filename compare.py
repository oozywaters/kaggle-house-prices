"""Cross-validate every model and print a ranked comparison table.

Usage: uv run python compare.py [cv.n_folds=...] [seed=...]
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.config import load_config
from src.pipeline import ENSEMBLES, MODELS, TARGET, build_pipeline, make_folds

def main() -> None:
    cfg = load_config()
    names = list(MODELS) + list(ENSEMBLES)
    print(f"comparing {len(names)} models | cv={cfg.cv.n_folds} folds, seed={cfg.seed}")

    train = pd.read_csv(Path(cfg.data_dir) / "train.csv")
    X = train.drop(columns=[TARGET])
    y = np.log1p(train[TARGET])
    folds = make_folds(X, y, cfg)

    rows = []
    for name in names:
        model_cfg = cfg.copy()
        model_cfg.model = name
        rmse = -cross_val_score(
            build_pipeline(model_cfg), X, y, cv=folds,
            scoring="neg_root_mean_squared_error",
        )
        rows.append({"model": name, "cv_rmse": rmse.mean(), "std": rmse.std()})

    results = (
        pd.DataFrame(rows).sort_values("cv_rmse").round(4).reset_index(drop=True)
    )
    results[""] = ""
    results.loc[0, ""] = "<- best"
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
