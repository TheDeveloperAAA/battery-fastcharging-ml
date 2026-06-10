"""Conformal wrappers (MAPIE v1 API — verified against v1.4.1 source).

Two flavours, both leakage-free:
- split: prefit model + held-out calibration cells (carved from TRAIN only).
- cross: CrossConformalRegressor ("plus") refits the model in 5-fold CV over
  all training cells — better data efficiency at n=41.

Intervals are computed on the log10(cycle-life) scale and exponentiated by
the caller.
"""

from __future__ import annotations

import numpy as np
from mapie.regression import CrossConformalRegressor, SplitConformalRegressor


def split_conformal(fitted_model, X_cal, y_cal,
                    confidence_level: float = 0.9) -> SplitConformalRegressor:
    scr = SplitConformalRegressor(estimator=fitted_model,
                                  confidence_level=confidence_level,
                                  conformity_score="absolute", prefit=True)
    scr.conformalize(np.asarray(X_cal), np.asarray(y_cal))
    return scr


def cross_conformal(unfitted_model, X_train, y_train,
                    confidence_level: float = 0.9, cv: int = 5,
                    ) -> CrossConformalRegressor:
    ccr = CrossConformalRegressor(estimator=unfitted_model,
                                  confidence_level=confidence_level,
                                  conformity_score="absolute",
                                  method="plus", cv=cv)
    ccr.fit_conformalize(np.asarray(X_train), np.asarray(y_train))
    return ccr


def intervals(mapie_model, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (point, lower, upper) on the log scale."""
    pred, pis = mapie_model.predict_interval(np.asarray(X))
    return pred, pis[:, 0, 0], pis[:, 1, 0]
