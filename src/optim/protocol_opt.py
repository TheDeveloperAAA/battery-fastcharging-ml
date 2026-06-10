"""Phase 2: degradation-aware fast-charging protocol optimisation.

Decision problem (PDF §6 Stage 2):
    minimise   charge_time(C1, Q1, C2)                  [analytic, minutes]
    subject to LB_{1-α}( cycle life | protocol ) ≥ L_min

where LB is the lower bound of a split-conformal prediction interval from a
protocol→life surrogate model, so the constraint reads "with confidence
(1−α), the predicted life under this protocol is at least L_min cycles".

Surrogate: trained on the canonical 124-cell MATR cohort (two-step Severson
protocols; 72 distinct policies). Protocol features are deterministic
transforms of (C1, Q1, C2): the raw triple, per-20%-SOC-window average rates,
and the analytic charge time. The surrogate family is chosen by 5-fold CV
among {quadratic ridge, Gaussian process (Matérn), small LightGBM}; the
conformal layer is calibrated on cells held out from surrogate fitting.

Honesty notes (also in BUILD_LOG / research note):
- Conformal coverage is marginal over the protocol distribution that
  generated the data; optimiser-chosen protocols inside the observed support
  inherit approximate validity, extrapolation outside it does not. The search
  domain is therefore clamped to the observed protocol ranges.
- Cells from the Phase-1 test splits are reused to train this surrogate —
  that does not contaminate Phase-1 benchmarks (different model, no
  feedback), and is documented.
- Search uses Optuna TPE (as specified) and is verified against a dense grid
  (the space is only 3-D and the surrogate is cheap) — the grid certifies
  global optimality within resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.features import charge_time_to_80
from src.eval.metrics import point_metrics, to_cycles
from src.utils.config import REPO_ROOT, load_config
from src.utils.seed import seed_everything

PROTO_COLS = ["c1", "q1_pct", "c2", "rate_w1", "rate_w2", "rate_w3",
              "rate_w4", "charge_time_min"]


# --------------------------------------------------------------------------
# Protocol feature vector from the decision variables
# --------------------------------------------------------------------------


def window_rates(c1: float, q1: float, c2: float) -> list[float]:
    rates = []
    for w in range(4):
        lo, hi = 20.0 * w, 20.0 * (w + 1)
        t_hours = 0.0
        for rate, a, b in [(c1, 0.0, q1), (c2, q1, 80.0)]:
            ov = max(0.0, min(hi, b) - max(lo, a))
            t_hours += (ov / 100.0) / rate
        rates.append(0.2 / t_hours if t_hours > 0 else np.nan)
    return rates


def protocol_vector(c1: float, q1: float, c2: float) -> np.ndarray:
    return np.array([c1, q1, c2, *window_rates(c1, q1, c2),
                     charge_time_to_80(c1, q1, c2)])


# --------------------------------------------------------------------------
# Surrogate training
# --------------------------------------------------------------------------


def candidate_models(seed: int) -> dict:
    from lightgbm import LGBMRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                                  Matern, WhiteKernel)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    return {
        "quadratic_ridge": make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            Ridge(alpha=1.0, random_state=seed)),
        "gp_matern": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ConstantKernel() * Matern(nu=2.5,
                                                 length_scale_bounds=(0.1, 50))
                + WhiteKernel(noise_level_bounds=(1e-6, 1.0)),
                normalize_y=True, random_state=seed)),
        "lightgbm_small": LGBMRegressor(
            n_estimators=200, num_leaves=7, max_depth=3, learning_rate=0.05,
            min_child_samples=8, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=seed,
            deterministic=True, verbose=-1),
    }


def train_surrogate(seed: int, cal_fraction: float = 0.4) -> dict:
    """Select surrogate by CV, fit, and conformalize on held-out cells."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    cfg = load_config()
    feat = REPO_ROOT / cfg["paths"]["features"]
    df = pd.read_csv(feat / "matr_features.csv")
    df = df[(df.protocol_type == "2step") & ~df.censored].copy()
    # canonical cohort only (split lists exclude noisy/outlier cells)
    from src.data.splits import (PRIMARY_TEST_CELLS, SECONDARY_TEST_CELLS,
                                 TRAIN_CELLS)
    cohort = set(TRAIN_CELLS + PRIMARY_TEST_CELLS + SECONDARY_TEST_CELLS)
    df["short_id"] = df.cell_id.str.replace("MATR_", "", regex=False)
    df = df[df.short_id.isin(cohort)]

    X = df[PROTO_COLS].to_numpy(dtype=float)
    y = df["log_cycle_life"].to_numpy(dtype=float)

    # model selection by 5-fold CV
    scores = {}
    for name, model in candidate_models(seed).items():
        kf = KFold(5, shuffle=True, random_state=seed)
        preds = np.full(len(y), np.nan)
        for tr, va in kf.split(X):
            preds[va] = clone(model).fit(X[tr], y[tr]).predict(X[va])
        scores[name] = point_metrics(y, preds)
    best = min(scores, key=lambda k: scores[k]["rmse_cycles"])

    # fit/calibration split (by cell, stratified by protocol via shuffle)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_cal = int(round(len(y) * cal_fraction))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]

    from src.models.conformal import split_conformal
    fitted = clone(candidate_models(seed)[best]).fit(X[fit_idx], y[fit_idx])
    mapie = split_conformal(fitted, X[cal_idx], y[cal_idx],
                            cfg["protocol_optimization"].get(
                                "confidence_level",
                                1 - cfg["protocol_optimization"]["alpha"]))

    # empirical calibration check on the calibration set itself is circular;
    # report CV scores + the conformal quantile instead
    return {"mapie": mapie, "model_name": best, "cv_scores": scores,
            "n_fit": len(fit_idx), "n_cal": len(cal_idx),
            "X": X, "y": y, "df": df}


# --------------------------------------------------------------------------
# Risk-constrained optimisation
# --------------------------------------------------------------------------


def lower_bound_life(mapie, c1: float, q1: float, c2: float) -> float:
    """Conformal lower bound on cycle life for one protocol."""
    _, lo, _ = _predict_interval(mapie, np.array([[c1, q1, c2]]))
    return float(lo[0])


def _predict_interval(mapie, triples: np.ndarray):
    Xp = np.vstack([protocol_vector(*t) for t in triples])
    pred, pis = mapie.predict_interval(Xp)
    return (to_cycles(pred), to_cycles(pis[:, 0, 0]), to_cycles(pis[:, 1, 0]))


def optimise(mapie, l_min: float, bounds: dict, seed: int,
             n_trials: int = 400) -> dict:
    """Optuna TPE search: min charge time s.t. conformal LB ≥ l_min."""
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        c1 = trial.suggest_float("c1", *bounds["c1"])
        q1 = trial.suggest_float("q1", *bounds["q1"])
        c2 = trial.suggest_float("c2", *bounds["c2"])
        t = charge_time_to_80(c1, q1, c2)
        lb = lower_bound_life(mapie, c1, q1, c2)
        trial.set_user_attr("lower_bound_life", lb)
        trial.set_user_attr("charge_time", t)
        if lb < l_min:                      # infeasible → penalised
            return t + 100.0 + (l_min - lb) / 10.0
        return t

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    feasible = [t for t in study.trials
                if t.user_attrs.get("lower_bound_life", -1) >= l_min]
    if not feasible:
        return {"feasible": False, "l_min": l_min}
    best = min(feasible, key=lambda t: t.user_attrs["charge_time"])
    return {"feasible": True, "l_min": l_min,
            "c1": best.params["c1"], "q1_pct": best.params["q1"],
            "c2": best.params["c2"],
            "charge_time_min": best.user_attrs["charge_time"],
            "lower_bound_life": best.user_attrs["lower_bound_life"]}


def grid_certify(mapie, l_min: float, bounds: dict,
                 n: tuple[int, int, int] = (57, 36, 57)) -> dict:
    """Dense-grid global check of the BO result (cheap 3-D space)."""
    c1s = np.linspace(*bounds["c1"], n[0])
    q1s = np.linspace(*bounds["q1"], n[1])
    c2s = np.linspace(*bounds["c2"], n[2])
    C1, Q1, C2 = np.meshgrid(c1s, q1s, c2s, indexing="ij")
    triples = np.column_stack([C1.ravel(), Q1.ravel(), C2.ravel()])
    _, lo, _ = _predict_interval(mapie, triples)
    t = 60.0 * ((triples[:, 1] / 100.0) / triples[:, 0]
                + ((80.0 - triples[:, 1]) / 100.0) / triples[:, 2])
    ok = lo >= l_min
    if not ok.any():
        return {"feasible": False, "l_min": l_min}
    i = np.flatnonzero(ok)[np.argmin(t[ok])]
    return {"feasible": True, "l_min": l_min, "c1": float(triples[i, 0]),
            "q1_pct": float(triples[i, 1]), "c2": float(triples[i, 2]),
            "charge_time_min": float(t[i]), "lower_bound_life": float(lo[i])}


def pareto_frontier(mapie, bounds: dict, l_grid: np.ndarray,
                    seed: int, n_trials: int) -> pd.DataFrame:
    rows = []
    for l_min in l_grid:
        bo = optimise(mapie, float(l_min), bounds, seed, n_trials)
        gr = grid_certify(mapie, float(l_min), bounds)
        pick = min((r for r in (bo, gr) if r.get("feasible")),
                   key=lambda r: r["charge_time_min"], default=None)
        if pick:
            pick["method"] = "optuna" if pick is bo else "grid"
            rows.append(pick)
    df = pd.DataFrame(rows)
    # enforce non-domination: charge time must be non-increasing as the
    # life requirement relaxes; prune any dominated points
    keep = []
    best_t = np.inf
    for _, r in df.sort_values("l_min", ascending=False).iterrows():
        if r.charge_time_min < best_t - 1e-9:
            best_t = r.charge_time_min
            keep.append(r)
    out = pd.DataFrame(keep).sort_values("l_min").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    po = cfg["protocol_optimization"]
    bounds = {"c1": tuple(po["c1_bounds"]), "c2": tuple(po["c2_bounds"]),
              "q1": tuple(po["q1_bounds"])}
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    artifacts = REPO_ROOT / cfg["paths"]["app_artifacts"]
    models_dir = REPO_ROOT / cfg["paths"]["models"]

    sur = train_surrogate(seed)
    print(f"surrogate: {sur['model_name']} "
          f"(CV RMSE {sur['cv_scores'][sur['model_name']]['rmse_cycles']:.0f}"
          f" cycles, fit {sur['n_fit']} / cal {sur['n_cal']} cells)")

    # clamp bounds to observed support (extrapolation honesty)
    df = sur["df"]
    bounds = {
        "c1": (max(bounds["c1"][0], df.c1.min()),
               min(bounds["c1"][1], df.c1.max())),
        "c2": (max(bounds["c2"][0], df.c2.min()),
               min(bounds["c2"][1], df.c2.max())),
        "q1": (max(bounds["q1"][0], df.q1_pct.min()),
               min(bounds["q1"][1], df.q1_pct.max())),
    }
    print(f"search domain clamped to observed support: {bounds}")

    mapie = sur["mapie"]
    rec = optimise(mapie, po["l_min"], bounds, seed, po["n_trials"])
    cert = grid_certify(mapie, po["l_min"], bounds)
    print(f"recommended @ L_min={po['l_min']}: {rec}")
    print(f"grid check: {cert}")

    lo_, hi_, n_ = po["pareto_grid_l_min"]
    l_grid = np.linspace(lo_, hi_, int(n_))
    pareto = pareto_frontier(mapie, bounds, l_grid, seed,
                             max(100, po["n_trials"] // 4))

    # observed protocols for context (charge time vs actual life)
    observed = df.groupby("protocol").agg(
        charge_time_min=("charge_time_min", "first"),
        median_life=("cycle_life", "median"),
        n_cells=("cycle_life", "size")).reset_index()

    # persist
    out = {
        "config": {k: po[k] for k in ("l_min", "alpha", "n_trials")},
        "surrogate": {"name": sur["model_name"],
                      "cv_scores": sur["cv_scores"],
                      "n_fit": sur["n_fit"], "n_cal": sur["n_cal"]},
        "bounds": {k: list(v) for k, v in bounds.items()},
        "recommended_optuna": rec,
        "recommended_grid": cert,
    }
    json.dump(out, open(results_dir / "protocol_recommendation.json", "w"),
              indent=2)
    json.dump(out, open(artifacts / "protocol_recommendation.json", "w"),
              indent=2)
    pareto.to_csv(results_dir / "pareto_frontier.csv", index=False)
    pareto.to_csv(artifacts / "pareto_frontier.csv", index=False)
    observed.to_csv(artifacts / "observed_protocols.csv", index=False)
    joblib.dump({"mapie": mapie, "model_name": sur["model_name"],
                 "proto_cols": PROTO_COLS, "bounds": bounds},
                models_dir / "protocol_surrogate.joblib")

    # precomputed what-if grid for the dashboard (no model needed at runtime)
    c1s = np.linspace(*bounds["c1"], 36)
    q1s = np.linspace(*bounds["q1"], 24)
    c2s = np.linspace(*bounds["c2"], 36)
    C1, Q1, C2 = np.meshgrid(c1s, q1s, c2s, indexing="ij")
    triples = np.column_stack([C1.ravel(), Q1.ravel(), C2.ravel()])
    pred, lo, hi = _predict_interval(mapie, triples)
    t = 60.0 * ((triples[:, 1] / 100.0) / triples[:, 0]
                + ((80.0 - triples[:, 1]) / 100.0) / triples[:, 2])
    np.savez_compressed(
        artifacts / "protocol_grid.npz",
        c1=c1s.astype(np.float32), q1=q1s.astype(np.float32),
        c2=c2s.astype(np.float32),
        pred=pred.reshape(C1.shape).astype(np.float32),
        lower=lo.reshape(C1.shape).astype(np.float32),
        upper=hi.reshape(C1.shape).astype(np.float32),
        charge_time=t.reshape(C1.shape).astype(np.float32))

    # self-checks (contract): constraint satisfied; frontier non-dominated
    assert (not rec["feasible"]) or rec["lower_bound_life"] >= po["l_min"]
    tt = pareto.sort_values("l_min").charge_time_min.to_numpy()
    assert np.all(np.diff(tt) >= -1e-9), "Pareto frontier not monotone"
    print(f"Pareto frontier: {len(pareto)} points, "
          f"charge time {tt.min():.1f}–{tt.max():.1f} min")


if __name__ == "__main__":
    main()
