"""Sequence model: a small two-branch 1D-CNN over cycle curves.

Branch 1 — voltage-grid curves (3 channels × 1000 points):
    Qdlin(cycle 10), Qdlin(cycle 100), ΔQ = late − early.
Branch 2 — per-cycle summaries (3 channels × 100 cycles):
    capacity-fade curve, internal resistance, charge time.
Global-average-pooled embeddings from both branches feed a small MLP head
that predicts log10(cycle life).

Wrapped in an sklearn-compatible estimator (fit/predict on the project X
matrix) so it can sit inside the ensemble and under MAPIE. Trained on CPU
(deterministic; MPS deliberately avoided — see BUILD_LOG).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.feature_matrix import SEQ_LAYOUT


def _seq_blocks(X: np.ndarray, n_scalar: int) -> tuple[np.ndarray, np.ndarray]:
    """→ (curves [n,3,1000], summaries [n,3,100]) from the project matrix."""
    block = np.asarray(X)[:, n_scalar:]
    ofs, views = 0, {}
    for name, size in SEQ_LAYOUT:
        views[name] = block[:, ofs:ofs + size]
        ofs += size
    early, late = views["qdlin_early"], views["qdlin_late"]
    curves = np.stack([early, late, late - early], axis=1)
    summaries = np.stack([views["fade"], views["ir"], views["charge_time"]],
                         axis=1)
    return curves.astype(np.float32), summaries.astype(np.float32)


class TwoBranchCNN(nn.Module):
    def __init__(self, width: int = 16, dropout: float = 0.2):
        super().__init__()
        def block(cin, cout, k, s):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s,
                                           padding=k // 2),
                                 nn.BatchNorm1d(cout), nn.ReLU())
        self.curve_net = nn.Sequential(
            block(3, width, 9, 4), block(width, 2 * width, 7, 4),
            block(2 * width, 2 * width, 5, 2), nn.AdaptiveAvgPool1d(1))
        self.summary_net = nn.Sequential(
            block(3, width, 5, 2), block(width, 2 * width, 3, 2),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(
            nn.Linear(4 * width, 4 * width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * width, 1))

    def forward(self, curves, summaries):
        a = self.curve_net(curves).squeeze(-1)
        b = self.summary_net(summaries).squeeze(-1)
        return self.head(torch.cat([a, b], dim=1)).squeeze(-1)


class SeqCNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_scalar: int, width: int = 16, dropout: float = 0.2,
                 lr: float = 3e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 400, patience: int = 40,
                 batch_size: int = 16, val_fraction: float = 0.15,
                 random_state: int = 42):
        self.n_scalar = n_scalar
        self.width = width
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.random_state = random_state

    # --- channel-wise standardisation fitted on training data only -------
    def _fit_scaler(self, curves, summaries):
        def stats(a):
            mean = np.nanmean(a, axis=(0, 2), keepdims=True)
            std = np.nanstd(a, axis=(0, 2), keepdims=True) + 1e-8
            return mean.astype(np.float32), std.astype(np.float32)
        self.curve_stats_ = stats(curves)
        self.summary_stats_ = stats(summaries)

    def _transform(self, curves, summaries):
        cm, cs = self.curve_stats_
        sm, ss = self.summary_stats_
        c = np.nan_to_num((curves - cm) / cs, nan=0.0)
        s = np.nan_to_num((summaries - sm) / ss, nan=0.0)
        return torch.from_numpy(c), torch.from_numpy(s)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        curves, summaries = _seq_blocks(X, self.n_scalar)
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-8)
        yt = (y - self.y_mean_) / self.y_std_

        self._fit_scaler(curves, summaries)
        c_all, s_all = self._transform(curves, summaries)
        yt = torch.from_numpy(yt)

        n = len(yt)
        n_val = max(3, int(round(n * self.val_fraction)))
        perm = rng.permutation(n)
        va_idx, tr_idx = perm[:n_val], perm[n_val:]

        model = TwoBranchCNN(self.width, self.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=self.patience // 4)
        loss_fn = nn.SmoothL1Loss()

        best_val, best_state, since = np.inf, None, 0
        for _ in range(self.max_epochs):
            model.train()
            order = rng.permutation(len(tr_idx))
            for start in range(0, len(order), self.batch_size):
                idx = tr_idx[order[start:start + self.batch_size]]
                opt.zero_grad()
                out = model(c_all[idx], s_all[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = float(loss_fn(model(c_all[va_idx], s_all[va_idx]),
                                    yt[va_idx]))
            sched.step(val)
            if val < best_val - 1e-5:
                best_val, since = val, 0
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        return self

    def predict(self, X):
        curves, summaries = _seq_blocks(X, self.n_scalar)
        c, s = self._transform(curves, summaries)
        with torch.no_grad():
            out = self.model_(c, s).numpy()
        return out * self.y_std_ + self.y_mean_
