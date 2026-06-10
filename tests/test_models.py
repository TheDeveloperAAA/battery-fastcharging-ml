"""Smoke tests for the model stack on synthetic data (fast, no real data)."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from src.models.feature_matrix import SCALAR_COLS, SEQ_DIM


def synthetic_xy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    n_scalar = len(SCALAR_COLS)
    X_scalar = rng.normal(size=(n, n_scalar))
    X_seq = rng.normal(size=(n, SEQ_DIM)).astype(np.float32)
    X = np.hstack([X_scalar, X_seq])
    # target driven by the dq_var column (as in reality)
    j = SCALAR_COLS.index("dq_var")
    y = 3.0 - 0.4 * X_scalar[:, j] + rng.normal(0, 0.05, n)
    return X, y, n_scalar


def test_baseline_learns_variance_signal():
    from src.models.baseline import SubsetElasticNet
    X, y, _ = synthetic_xy()
    m = SubsetElasticNet("variance").fit(X[:45], y[:45])
    pred = m.predict(X[45:])
    rmse = np.sqrt(np.mean((pred - y[45:]) ** 2))
    assert rmse < 0.15


def test_lgbm_fit_predict():
    from src.models.gbm import ScalarLGBM
    X, y, n_scalar = synthetic_xy()
    m = ScalarLGBM(n_scalar).fit(X[:45], y[:45])
    assert m.predict(X[45:]).shape == (15,)


def test_cnn_fit_predict_deterministic():
    from src.models.sequence import SeqCNNRegressor
    X, y, n_scalar = synthetic_xy(n=30)
    kw = dict(n_scalar=n_scalar, max_epochs=10, patience=5, random_state=1)
    p1 = SeqCNNRegressor(**kw).fit(X, y).predict(X)
    p2 = SeqCNNRegressor(**kw).fit(X, y).predict(X)
    assert p1.shape == (30,)
    np.testing.assert_allclose(p1, p2, rtol=1e-5)


def test_ensemble_weights_sum_to_one():
    from src.models.baseline import SubsetElasticNet
    from src.models.ensemble import StackedEnsemble
    from src.models.gbm import ScalarLGBM
    X, y, n_scalar = synthetic_xy()
    ens = StackedEnsemble(components=[
        SubsetElasticNet("variance"), ScalarLGBM(n_scalar)]).fit(X, y)
    assert ens.weights_.sum() == pytest.approx(1.0)
    assert (ens.weights_ >= 0).all()
    assert ens.predict(X).shape == y.shape


def test_split_conformal_coverage_sane():
    from src.models.baseline import SubsetElasticNet
    from src.models.conformal import intervals, split_conformal
    X, y, _ = synthetic_xy(n=200, seed=3)
    fit, cal, test = X[:100], X[100:160], X[160:]
    yf, yc, yt = y[:100], y[100:160], y[160:]
    model = SubsetElasticNet("variance").fit(fit, yf)
    mapie = split_conformal(model, cal, yc, confidence_level=0.9)
    pred, lo, hi = intervals(mapie, test)
    cover = np.mean((yt >= lo) & (yt <= hi))
    assert cover >= 0.75          # small-sample slack around 0.9
    assert np.all(lo <= hi)
