"""Train one model with cross-validation and write a Kaggle submission.

We train on log1p(SalePrice) because that's what the leaderboard scores, then
turn predictions back into dollars with expm1. So the CV RMSE here matches the
leaderboard metric.

Usage: uv run python main.py [model=...] [params.<model>.<param>=...] [cv.n_folds=...]
"""

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import KFold, cross_val_score

from src.config import load_config
from src.pipeline import TARGET, build_pipeline

def main() -> None:
    cfg = load_config()
    params = OmegaConf.to_container(cfg.params, resolve=True).get(cfg.model, {})
    details = " ".join(f"{k}={v}" for k, v in params.items())
    print(f"model={cfg.model} | cv={cfg.cv.n_folds} folds, seed={cfg.seed} | {details}")

    data_dir = Path(cfg.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X = train.drop(columns=[TARGET])
    y = np.log1p(train[TARGET])

    pipeline = build_pipeline(cfg)

    cv = KFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    rmse = -scores
    print(f"folds: {' '.join(f'{s:.4f}' for s in rmse)}")
    print(f"CV RMSE (log scale): {rmse.mean():.4f} ± {rmse.std():.4f}")

    pipeline.fit(X, y)
    predictions = np.expm1(pipeline.predict(test))

    out = Path(cfg.submission_path)
    out.parent.mkdir(exist_ok=True)
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})
    submission.to_csv(out, index=False)
    print(f"Submission written to {out}")


if __name__ == "__main__":
    main()
