"""Ensemble of the elastic-net baseline, LightGBM, and the sequence CNN.

Stacking weights are learned from out-of-fold predictions on the training
data (non-negative least squares, weights sum to 1) — no test data involved.
The whole thing is an sklearn-compatible regressor over the project X
matrix, so MAPIE can clone/fit it inside cross-conformal CV.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold


class StackedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, components: list | None = None, stack_folds: int = 5,
                 random_state: int = 42):
        self.components = components
        self.stack_folds = stack_folds
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        comps = [clone(c) for c in self.components]

        # out-of-fold predictions → stacking weights
        kf = KFold(n_splits=min(self.stack_folds, len(y) - 1), shuffle=True,
                   random_state=self.random_state)
        oof = np.full((len(y), len(comps)), np.nan)
        for tr, va in kf.split(X):
            for j, comp in enumerate(comps):
                m = clone(comp).fit(X[tr], y[tr])
                oof[va, j] = m.predict(X[va])
        w, _ = nnls(oof, y)
        if w.sum() <= 1e-12:
            w = np.ones(len(comps))
        self.weights_ = w / w.sum()
        self.oof_ = oof

        # final components trained on the full training data
        self.fitted_ = [clone(c).fit(X, y) for c in self.components]
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.fitted_])
        return preds @ self.weights_

    def component_predictions(self, X) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.fitted_])
