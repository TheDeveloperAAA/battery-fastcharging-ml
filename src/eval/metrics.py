"""Evaluation metrics: point accuracy (cycles), interval calibration, sharpness.

Models are trained on log10(cycle life); every metric here converts back to
cycles first so numbers are comparable with Severson et al. 2019 Table 1.
"""

from __future__ import annotations

import numpy as np


def to_cycles(y_log: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(y_log, dtype=float))


def point_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y, yhat = to_cycles(y_true_log), to_cycles(y_pred_log)
    err = yhat - y
    return {
        "rmse_cycles": float(np.sqrt(np.mean(err ** 2))),
        "mae_cycles": float(np.mean(np.abs(err))),
        "mape_pct": float(np.mean(np.abs(err) / y) * 100.0),
        "n": int(len(y)),
    }


def interval_metrics(y_true_log: np.ndarray, lower_log: np.ndarray,
                     upper_log: np.ndarray) -> dict:
    """PICP and MPIW in cycle units (intervals computed on the log scale)."""
    y = to_cycles(y_true_log)
    lo, hi = to_cycles(lower_log), to_cycles(upper_log)
    covered = (y >= lo) & (y <= hi)
    return {
        "picp": float(np.mean(covered)),
        "mpiw_cycles": float(np.mean(hi - lo)),
        "mpiw_relative": float(np.mean((hi - lo) / y)),
        "n": int(len(y)),
    }


def reliability_curve(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                      conformity_scores: np.ndarray,
                      levels: np.ndarray | None = None) -> dict:
    """Empirical coverage vs nominal level for a split-conformal predictor.

    Given absolute-residual conformity scores from the calibration set,
    rebuild the interval at each nominal level and measure test coverage.
    """
    if levels is None:
        levels = np.arange(0.5, 1.0, 0.05)
    scores = np.sort(np.asarray(conformity_scores))
    n = len(scores)
    empirical = []
    for cl in levels:
        k = int(np.ceil((n + 1) * cl))
        q = scores[min(k, n) - 1] if k <= n else np.inf
        covered = np.abs(y_true_log - y_pred_log) <= q
        empirical.append(float(np.mean(covered)))
    return {"nominal": [float(x) for x in levels], "empirical": empirical}
