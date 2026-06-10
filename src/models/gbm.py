"""LightGBM model on the scalar feature block, with Optuna tuning.

Notes on small-n discipline (41 training cells):
- Tuning objective is 5-fold CV RMSE on the training split only.
- Search space is kept conservative (shallow trees, strong regularisation,
  small num_leaves) — anything else overfits 41 points instantly.
- The final model is refit on the full training split with the median of the
  per-fold early-stopped iteration counts.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

DEFAULT_PARAMS = dict(
    objective="regression",
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=7,
    max_depth=3,
    min_child_samples=5,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbose=-1,
)


class ScalarLGBM(BaseEstimator, RegressorMixin):
    """LGBMRegressor over the scalar block of the project X matrix."""

    def __init__(self, n_scalar: int, params: dict | None = None,
                 random_state: int = 42):
        self.n_scalar = n_scalar
        self.params = params
        self.random_state = random_state

    def fit(self, X, y):
        p = dict(DEFAULT_PARAMS)
        if self.params:
            p.update(self.params)
        self.model_ = LGBMRegressor(random_state=self.random_state,
                                    deterministic=True, n_jobs=4, **p)
        self.model_.fit(np.asarray(X)[:, :self.n_scalar], y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X)[:, :self.n_scalar])


def tune_lgbm(X_scalar: np.ndarray, y: np.ndarray, n_trials: int = 120,
              seed: int = 42, n_folds: int = 5) -> dict:
    """Optuna TPE search; returns best params (incl. tuned n_estimators)."""
    import optuna

    def cv_rmse(params: dict) -> tuple[float, int]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        errs, iters = [], []
        for tr, va in kf.split(X_scalar):
            model = LGBMRegressor(random_state=seed, deterministic=True,
                                  n_jobs=4, verbose=-1, n_estimators=2000,
                                  **params)
            model.fit(X_scalar[tr], y[tr],
                      eval_set=[(X_scalar[va], y[va])],
                      eval_metric="rmse",
                      callbacks=[early_stopping(100, verbose=False)])
            pred = model.predict(X_scalar[va])
            errs.append(np.sqrt(np.mean((pred - y[va]) ** 2)))
            iters.append(model.best_iteration_ or 2000)
        return float(np.mean(errs)), int(np.median(iters))

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2,
                                              log=True),
            num_leaves=trial.suggest_int("num_leaves", 3, 15),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            min_child_samples=trial.suggest_int("min_child_samples", 3, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
        rmse, best_iter = cv_rmse(params)
        trial.set_user_attr("best_iter", best_iter)
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["best_iter"]
    best["subsample_freq"] = 1
    return {"params": best, "cv_rmse_log": study.best_value,
            "n_trials": n_trials}
