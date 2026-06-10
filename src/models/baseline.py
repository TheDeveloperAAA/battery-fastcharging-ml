"""Severson-style regularised linear baselines (elastic net) on named
feature subsets. All take the full project X matrix and slice internally,
so they are interchangeable with the other models under MAPIE."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_matrix import FEATURE_SETS, feature_indices


class SubsetElasticNet(BaseEstimator, RegressorMixin):
    """Elastic net on a named scalar-feature subset of the project matrix.

    Hyperparameters (l1_ratio grid, alphas) follow Severson's elastic-net
    setup, selected by internal cross-validation on the training data only.
    """

    def __init__(self, feature_set: str = "variance", cv: int = 5,
                 random_state: int = 42):
        self.feature_set = feature_set
        self.cv = cv
        self.random_state = random_state

    def _cols(self):
        return feature_indices(FEATURE_SETS[self.feature_set])

    def fit(self, X, y):
        cols = self._cols()
        Xs = np.asarray(X)[:, cols]
        n_splits = min(self.cv, len(y) - 1)
        self.model_ = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                n_alphas=100, cv=n_splits, max_iter=50000,
                random_state=self.random_state),
        )
        self.model_.fit(Xs, y)
        return self

    def predict(self, X):
        cols = self._cols()
        return self.model_.predict(np.asarray(X)[:, cols])

    def clone_unfitted(self):
        return clone(self)
