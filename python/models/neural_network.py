"""Model B: small PyTorch networks (optional dependency).

Architecture (kept deliberately small — the dataset is small too):

    Input (n_features)
        -> Dense(hidden[0]) + ReLU + Dropout
        -> Dense(hidden[1]) + ReLU
        -> Dense(1)

* Presence network ends in a logit; ``BCEWithLogitsLoss`` is applied and
  ``sigmoid`` is used at inference to report P(person).
* Position network ends in a linear value; ``SmoothL1Loss`` (Huber) makes
  training robust to outlier positions.  Predictions are clipped to [0, 1].

PyTorch is an optional dependency: importing this module without torch
raises a clear, actionable RuntimeError (the baselines keep working).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as _exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is not installed. Model B requires it: `pip install torch` "
        "(CPU-only wheel is sufficient: see requirements.txt). "
        "The KNN / Random Forest baselines work without it."
    ) from _exc


class PresenceNet(nn.Module):
    """Small MLP for binary presence classification."""

    def __init__(self, n_features: int, hidden: Optional[List[int]] = None,
                 dropout: float = 0.2) -> None:
        super().__init__()
        hidden = hidden or [64, 32]
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: UP037
        return self.net(x).squeeze(-1)


class PositionNet(nn.Module):
    """Small MLP for normalized position regression."""

    def __init__(self, n_features: int, hidden: Optional[List[int]] = None,
                 dropout: float = 0.2) -> None:
        super().__init__()
        hidden = hidden or [64, 32]
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: UP037
        return self.net(x).squeeze(-1)



def train_torch_model(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> nn.Module:
    """Train a small MLP for ``task`` in {"presence", "position"}.

    Early stopping on validation loss; returns the best model (eval mode).
    """
    train_cfg = cfg.get("train", {})
    hidden = [int(h) for h in train_cfg.get("nn_hidden", [64, 32])]
    dropout = float(train_cfg.get("nn_dropout", 0.2))
    epochs = int(train_cfg.get("nn_epochs", 60))
    batch = int(train_cfg.get("nn_batch_size", 64))
    lr = float(train_cfg.get("nn_learning_rate", 1e-3))
    patience = int(train_cfg.get("nn_patience", 8))
    seed = int(train_cfg.get("seed", 42))

    torch.manual_seed(seed)
    n_features = x_train.shape[1]
    if task == "presence":
        model: nn.Module = PresenceNet(n_features, hidden, dropout)
        criterion = nn.BCEWithLogitsLoss()
    elif task == "position":
        model = PositionNet(n_features, hidden, dropout)
        criterion = nn.SmoothL1Loss()  # Huber loss
    else:
        raise ValueError(f"unknown task: {task!r}")

    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(np.asarray(y_train, dtype=np.float32))
    x_va = torch.tensor(x_val, dtype=torch.float32)
    y_va = torch.tensor(np.asarray(y_val, dtype=np.float32))

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    n = len(x_tr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for lo in range(0, n, batch):
            idx = perm[lo:lo + batch]
            optim.zero_grad()
            loss = criterion(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_va), y_va))
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def save_torch_model(
    model: nn.Module,
    path: str,
    meta: Dict,
) -> None:
    """Save model weights + metadata json next to them."""
    torch.save(model.state_dict(), path)
    Path(path).with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


def load_torch_model(task: str, n_features: int, path: str,
                     cfg: dict) -> nn.Module:
    """Rebuild the network and load saved weights (eval mode)."""
    train_cfg = cfg.get("train", {})
    hidden = [int(h) for h in train_cfg.get("nn_hidden", [64, 32])]
    dropout = float(train_cfg.get("nn_dropout", 0.2))
    model = PresenceNet(n_features, hidden, dropout) if task == "presence" \
        else PositionNet(n_features, hidden, dropout)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class TorchPresenceModel:
    """Uniform inference wrapper (same interface as the sklearn wrapper)."""

    def __init__(self, model: nn.Module, name: str = "neural_network") -> None:
        self.model = model
        self.name = name
        self.kind = "torch"

    def predict(self, features: np.ndarray) -> float:
        x = torch.tensor(np.asarray(features, dtype=np.float64)
                         .reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            logit = float(self.model(x).item())
        return 1.0 / (1.0 + float(np.exp(-logit)))


class TorchPositionModel:
    """Uniform inference wrapper producing positions clipped to [0, 1]."""

    def __init__(self, model: nn.Module, name: str = "neural_network") -> None:
        self.model = model
        self.name = name
        self.kind = "torch"

    def predict(self, features: np.ndarray) -> float:
        x = torch.tensor(np.asarray(features, dtype=np.float64)
                         .reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            raw = float(self.model(x).item())
        return min(max(raw, 0.0), 1.0)
