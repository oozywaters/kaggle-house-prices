"""PyTorch MLP wrapped as a scikit-learn regressor (``main.py model=mlp``)."""

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn

# single-threaded torch avoids a segfault from the duplicate OpenMP runtimes
# that torch and catboost/lightgbm/xgboost each bundle on macOS
torch.set_num_threads(1)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

class MLP(nn.Module):
    """Stack of Linear -> [BatchNorm] -> activation -> [Dropout] blocks."""

    def __init__(self, in_dim, hidden_sizes, activation, batchnorm, dropout):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers, prev = [], in_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act())
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn-compatible MLP regressor backed by PyTorch."""

    def __init__(self, hidden_sizes=(256, 128), activation="relu", batchnorm=True,
                 dropout=0.2, optimizer="adamw", lr=1e-3, weight_decay=1e-4,
                 batch_size=64, n_epochs=100, scheduler="cosine", seed=42):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.scheduler = scheduler
        self.seed = seed

    def _to_tensor(self, X):
        if sparse.issparse(X):
            X = X.toarray()
        return torch.as_tensor(np.asarray(X, dtype=np.float32))

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        features = self._to_tensor(X)
        # center target: net learns residuals, not a big offset
        self.y_mean_ = float(np.mean(y))
        targets = torch.as_tensor(np.asarray(y, dtype=np.float32) - self.y_mean_)

        self.model_ = MLP(features.shape[1], list(self.hidden_sizes),
                          self.activation, self.batchnorm, self.dropout)
        optimizer = OPTIMIZERS[self.optimizer](
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs)
            if self.scheduler == "cosine" else None
        )
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(features, targets)
        # drop_last: no size-1 batches (break BatchNorm)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.model_.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(self._to_tensor(X)).numpy() + self.y_mean_
