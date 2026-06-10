"""Phase 1 driver: train the probabilistic cycle-life predictor.

Steps (all settings from config.yaml):
1. Assemble train (41) / primary-test (42) / secondary-test (40) matrices.
2. Train baselines (elastic net on variance/discharge/full sets), tuned
   LightGBM (Optuna, training split only), sequence CNN, stacked ensemble.
3. Point metrics on both test sets + 5-fold CV on train; compared against the
   published Severson Table 1 numbers (cited, not invented).
4. Conformal intervals at 90%: split-conformal (prefit, calibration cells
   carved from train) AND cross-conformal (5-fold plus). PICP / MPIW /
   reliability curve.
5. Early-prediction error vs observation horizon (20..100 cycles).
6. Cross-dataset transfer: MATR-trained discharge-feature model applied to
   CALCE / NASA via capacity-normalized features.
7. Compact on-controller estimator (small LightGBM) exported to models/ and
   app/artifacts/.

Run:  PYTHONPATH=. .venv/bin/python -m src.models.train
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from src.data.splits import (PRIMARY_TEST_CELLS, SECONDARY_TEST_CELLS,
                             TRAIN_CELLS, train_calibration_split)
from src.eval.metrics import (interval_metrics, point_metrics,
                              reliability_curve, to_cycles)
from src.models.baseline import SubsetElasticNet
from src.models.conformal import cross_conformal, intervals, split_conformal
from src.models.ensemble import StackedEnsemble
from src.models.feature_matrix import (FEATURE_SETS, SCALAR_COLS,
                                       build_matrix, feature_indices)
from src.models.gbm import ScalarLGBM, tune_lgbm
from src.models.sequence import SeqCNNRegressor
from src.utils.config import REPO_ROOT, load_config
from src.utils.seed import seed_everything

# Severson et al. 2019, Table 1 (primary-test values in parentheses exclude
# the b2c1 outlier — the comparable column since our split excludes it).
SEVERSON_BENCHMARK = {
    "source": "Severson et al., Nature Energy 4, 383-391 (2019), Table 1; "
              "doi:10.1038/s41560-019-0356-8",
    "rmse_cycles": {
        "variance": {"train": 103, "primary_test": 138,
                     "primary_test_excl_outlier": 138, "secondary_test": 196},
        "discharge": {"train": 76, "primary_test": 91,
                      "primary_test_excl_outlier": 86, "secondary_test": 173},
        "full": {"train": 51, "primary_test": 118,
                 "primary_test_excl_outlier": 100, "secondary_test": 214},
    },
    "mape_pct": {
        "variance": {"train": 14.1, "primary_test": 14.7,
                     "primary_test_excl_outlier": 13.2, "secondary_test": 11.4},
        "discharge": {"train": 9.8, "primary_test": 13.0,
                      "primary_test_excl_outlier": 10.1, "secondary_test": 8.6},
        "full": {"train": 5.6, "primary_test": 14.1,
                 "primary_test_excl_outlier": 7.5, "secondary_test": 10.7},
    },
}


def assemble(cfg) -> dict:
    feat = REPO_ROOT / cfg["paths"]["features"]
    seq_dir = feat / "sequences"
    csv = feat / "matr_features.csv"
    data = {}
    for name, cells in [("train", TRAIN_CELLS),
                        ("primary_test", PRIMARY_TEST_CELLS),
                        ("secondary_test", SECONDARY_TEST_CELLS)]:
        data[name] = build_matrix(csv, seq_dir, cells)
    return data


def cv_point_metrics(model, X, y, seed: int, folds: int = 5) -> dict:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.full(len(y), np.nan)
    for tr, va in kf.split(X):
        m = clone(model).fit(X[tr], y[tr])
        preds[va] = m.predict(X[va])
    return point_metrics(y, preds)


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    models_dir = REPO_ROOT / cfg["paths"]["models"]
    artifacts = REPO_ROOT / cfg["paths"]["app_artifacts"]
    for d in (results_dir, models_dir, artifacts):
        d.mkdir(parents=True, exist_ok=True)

    data = assemble(cfg)
    X_tr, y_tr = data["train"]["X"], data["train"]["y"]
    n_scalar = data["train"]["n_scalar"]
    thorough = cfg["training"]["mode"] == "thorough"

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    print("[1/7] tuning LightGBM" if thorough else "[1/7] LightGBM defaults")
    if thorough:
        tune = tune_lgbm(X_tr[:, :n_scalar], y_tr,
                         n_trials=cfg["training"]["optuna_trials_gbm"],
                         seed=seed)
        lgbm_params = tune["params"]
        json.dump(tune, open(results_dir / "optuna_lgbm.json", "w"), indent=2)
    else:
        lgbm_params = None

    models = {
        "elastic_net_variance": SubsetElasticNet("variance",
                                                 random_state=seed),
        "elastic_net_discharge": SubsetElasticNet("discharge",
                                                  random_state=seed),
        "elastic_net_full": SubsetElasticNet("full", random_state=seed),
        "lightgbm": ScalarLGBM(n_scalar, lgbm_params, random_state=seed),
        "sequence_cnn": SeqCNNRegressor(n_scalar, random_state=seed),
    }
    ensemble = StackedEnsemble(
        components=[SubsetElasticNet("full", random_state=seed),
                    ScalarLGBM(n_scalar, lgbm_params, random_state=seed),
                    SeqCNNRegressor(n_scalar, random_state=seed)],
        random_state=seed)
    models["ensemble"] = ensemble

    # ------------------------------------------------------------------ #
    # Point metrics
    # ------------------------------------------------------------------ #
    print("[2/7] point metrics")
    metrics: dict = {"benchmark": SEVERSON_BENCHMARK, "models": {}}
    preds_rows = []
    for name, model in models.items():
        fitted = clone(model).fit(X_tr, y_tr)
        entry = {"train_cv": cv_point_metrics(model, X_tr, y_tr, seed)}
        for split in ("primary_test", "secondary_test"):
            yp = fitted.predict(data[split]["X"])
            entry[split] = point_metrics(data[split]["y"], yp)
            for cid, yt, yh in zip(data[split]["cell_ids"],
                                   data[split]["y"], yp):
                preds_rows.append({
                    "model": name, "split": split, "cell_id": cid,
                    "true_cycle_life": float(to_cycles(yt)),
                    "pred_cycle_life": float(to_cycles(yh))})
        metrics["models"][name] = entry
        if name == "ensemble":
            metrics["ensemble_weights"] = {
                c: float(w) for c, w in zip(
                    ["elastic_net_full", "lightgbm", "sequence_cnn"],
                    fitted.weights_)}
            models["ensemble_fitted"] = fitted
        print(f"  {name}: primary RMSE "
              f"{entry['primary_test']['rmse_cycles']:.0f} cyc, "
              f"MAPE {entry['primary_test']['mape_pct']:.1f}%")

    # ------------------------------------------------------------------ #
    # Conformal prediction
    # ------------------------------------------------------------------ #
    print("[3/7] conformal intervals")
    cl = cfg["uncertainty"]["confidence_level"]
    fit_cells, cal_cells = train_calibration_split(
        seed, cfg["splits"]["calibration_fraction"])
    feat_csv = REPO_ROOT / cfg["paths"]["features"] / "matr_features.csv"
    seq_dir = REPO_ROOT / cfg["paths"]["features"] / "sequences"
    d_fit = build_matrix(feat_csv, seq_dir, fit_cells)
    d_cal = build_matrix(feat_csv, seq_dir, cal_cells)

    ens_for_split = clone(ensemble).fit(d_fit["X"], d_fit["y"])
    mapie_split = split_conformal(ens_for_split, d_cal["X"], d_cal["y"], cl)
    mapie_cross = cross_conformal(clone(ensemble), X_tr, y_tr, cl,
                                  cv=cfg["splits"]["cv_folds"])

    calibration: dict = {
        "confidence_level": cl,
        "calibration_cells": cal_cells,
        "n_calibration": len(cal_cells),
        "methods": {},
    }
    interval_rows = []
    for method, mapie in [("split", mapie_split), ("cross", mapie_cross)]:
        m_entry = {}
        for split in ("primary_test", "secondary_test"):
            pred, lo, hi = intervals(mapie, data[split]["X"])
            m_entry[split] = {
                **interval_metrics(data[split]["y"], lo, hi),
                **point_metrics(data[split]["y"], pred)}
            for cid, yt, yh, l_, h_ in zip(data[split]["cell_ids"],
                                           data[split]["y"], pred, lo, hi):
                interval_rows.append({
                    "method": method, "split": split, "cell_id": cid,
                    "true_cycle_life": float(to_cycles(yt)),
                    "pred_cycle_life": float(to_cycles(yh)),
                    "lower": float(to_cycles(l_)),
                    "upper": float(to_cycles(h_))})
        calibration["methods"][method] = m_entry
        print(f"  {method}: primary PICP {m_entry['primary_test']['picp']:.2f}"
              f" (nominal {cl}), MPIW "
              f"{m_entry['primary_test']['mpiw_cycles']:.0f} cyc")

    # reliability curve from split-conformal residuals
    y_both = np.concatenate([data["primary_test"]["y"],
                             data["secondary_test"]["y"]])
    pred_both = ens_for_split.predict(
        np.vstack([data["primary_test"]["X"], data["secondary_test"]["X"]]))
    calibration["reliability"] = reliability_curve(
        y_both, pred_both, mapie_split.conformity_scores)

    # ------------------------------------------------------------------ #
    # Early prediction vs horizon
    # ------------------------------------------------------------------ #
    print("[4/7] early-prediction curve")
    hz_csv = REPO_ROOT / cfg["paths"]["features"] / \
        "matr_features_horizons.csv"
    df_h = pd.read_csv(hz_csv)
    df_h["short_id"] = df_h.cell_id.str.replace("MATR_", "", regex=False)
    early = []
    for h in sorted(df_h.horizon.unique()):
        sub = df_h[df_h.horizon == h].set_index("short_id")
        tr_ok = [c for c in TRAIN_CELLS if c in sub.index]
        te_ok = [c for c in PRIMARY_TEST_CELLS if c in sub.index]
        Xh_tr = sub.loc[tr_ok, SCALAR_COLS].to_numpy()
        Xh_te = sub.loc[te_ok, SCALAR_COLS].to_numpy()
        yh_tr = sub.loc[tr_ok, "log_cycle_life"].to_numpy()
        yh_te = sub.loc[te_ok, "log_cycle_life"].to_numpy()
        for mname, m in [("elastic_net_variance",
                          SubsetElasticNet("variance", random_state=seed)),
                         ("lightgbm",
                          ScalarLGBM(len(SCALAR_COLS), lgbm_params,
                                     random_state=seed))]:
            fitted = m.fit(Xh_tr, yh_tr)
            pm = point_metrics(yh_te, fitted.predict(Xh_te))
            early.append({"horizon_cycles": int(h), "model": mname, **pm})
    metrics["early_prediction"] = early

    # ------------------------------------------------------------------ #
    # Cross-dataset transfer (capacity-normalized discharge features)
    # ------------------------------------------------------------------ #
    print("[5/7] cross-dataset transfer")
    metrics["cross_dataset"] = cross_dataset_eval(cfg, seed)

    # ------------------------------------------------------------------ #
    # Compact estimator
    # ------------------------------------------------------------------ #
    print("[6/7] compact estimator")
    metrics["compact_model"] = export_compact(
        X_tr[:, :n_scalar], y_tr, data, n_scalar, seed,
        models_dir, artifacts)

    # ------------------------------------------------------------------ #
    # Persist everything
    # ------------------------------------------------------------------ #
    print("[7/7] saving artifacts")
    pd.DataFrame(preds_rows).to_csv(results_dir / "predictions_point.csv",
                                    index=False)
    df_int = pd.DataFrame(interval_rows)
    df_int.to_csv(results_dir / "predictions_intervals.csv", index=False)
    df_int.to_csv(artifacts / "predictions_intervals.csv", index=False)
    json.dump(metrics, open(results_dir / "metrics.json", "w"), indent=2)
    json.dump(calibration, open(results_dir / "calibration_report.json", "w"),
              indent=2)
    json.dump(calibration, open(artifacts / "calibration_report.json", "w"),
              indent=2)
    json.dump(metrics, open(artifacts / "metrics.json", "w"), indent=2)

    joblib.dump({"ensemble": models["ensemble_fitted"],
                 "mapie_split": mapie_split,
                 "mapie_cross": mapie_cross,
                 "scalar_cols": list(SCALAR_COLS),
                 "n_scalar": n_scalar},
                models_dir / "ensemble_conformal.joblib")

    metrics["wall_time_s"] = round(time.time() - t0, 1)
    json.dump(metrics, open(results_dir / "metrics.json", "w"), indent=2)
    print(f"done in {metrics['wall_time_s']}s")


def cross_dataset_eval(cfg, seed: int) -> dict:
    """Train discharge-feature model on MATR, test on CALCE / NASA.

    Capacity-derived features are normalized to the MATR 1.1 Ah scale so the
    model sees comparable magnitudes (log-variance shifts by 2·log10(c),
    log-min by log10(c), capacities scale by 1/c with c = Cnom/1.1).
    """
    feat = REPO_ROOT / cfg["paths"]["features"]
    cols = FEATURE_SETS["discharge"]

    def norm_matrix(df: pd.DataFrame) -> np.ndarray:
        c = df["nominal_capacity_ah"].to_numpy() / 1.1
        X = df[cols].to_numpy(dtype=float).copy()
        for j, name in enumerate(cols):
            if name == "dq_var":
                X[:, j] -= 2 * np.log10(c)
            elif name in ("dq_min", "dq_mean"):
                X[:, j] -= np.log10(c)
            elif name in ("qd_cycle2", "qd_max_minus_2"):
                X[:, j] /= c
        return X

    df_tr = pd.read_csv(feat / "matr_features.csv")
    if "nominal_capacity_ah" not in df_tr:
        df_tr["nominal_capacity_ah"] = 1.1
    df_tr["short_id"] = df_tr.cell_id.str.replace("MATR_", "", regex=False)
    df_tr = df_tr.set_index("short_id").loc[TRAIN_CELLS].reset_index()
    Xn_tr, y_tr = norm_matrix(df_tr), df_tr["log_cycle_life"].to_numpy()

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], n_alphas=50, cv=5,
                     max_iter=50000, random_state=seed)).fit(Xn_tr, y_tr)

    out = {"features": cols, "note": "capacity-normalized; censored cells "
           "excluded; expect degradation across chemistry (LFP→LCO)"}
    for ds in ("calce", "nasa"):
        path = feat / f"{ds}_features.csv"
        if not path.exists():
            out[ds] = "features not built"
            continue
        df = pd.read_csv(path)
        df = df[~df.censored & df[cols].notna().all(axis=1)]
        if df.empty:
            out[ds] = "no eligible cells"
            continue
        yp = model.predict(norm_matrix(df))
        out[ds] = {**point_metrics(df["log_cycle_life"].to_numpy(), yp),
                   "cells": df.cell_id.tolist(),
                   "pred_cycle_life": [float(v) for v in to_cycles(yp)],
                   "true_cycle_life": df.cycle_life.tolist()}
    return out


def export_compact(X_scalar_tr, y_tr, data, n_scalar, seed,
                   models_dir: Path, artifacts: Path) -> dict:
    """Small LightGBM suitable for an on-controller SoH/RUL estimator."""
    from lightgbm import LGBMRegressor
    compact = LGBMRegressor(
        objective="regression", n_estimators=120, num_leaves=7, max_depth=3,
        learning_rate=0.05, min_child_samples=5, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, deterministic=True, verbose=-1)
    compact.fit(X_scalar_tr, y_tr, feature_name=list(SCALAR_COLS))
    entry = {}
    for split in ("primary_test", "secondary_test"):
        yp = compact.predict(data[split]["X"][:, :n_scalar])
        entry[split] = point_metrics(data[split]["y"], yp)
    txt_path = models_dir / "compact_soh_estimator.txt"
    compact.booster_.save_model(str(txt_path))
    (artifacts / "compact_soh_estimator.txt").write_text(txt_path.read_text())
    meta = {
        "format": "LightGBM booster text dump (pure C inference via "
                  "lightgbm/treelite or hand-rolled tree walk)",
        "n_trees": compact.booster_.num_trees(),
        "num_leaves_max": 7, "max_depth": 3,
        "n_features": len(SCALAR_COLS),
        "feature_names": list(SCALAR_COLS),
        "target": "log10(cycle life)",
        "size_bytes": txt_path.stat().st_size,
        "metrics": entry,
    }
    json.dump(meta, open(models_dir / "compact_soh_estimator.json", "w"),
              indent=2)
    json.dump(meta, open(artifacts / "compact_soh_estimator.json", "w"),
              indent=2)
    return meta


if __name__ == "__main__":
    main()
