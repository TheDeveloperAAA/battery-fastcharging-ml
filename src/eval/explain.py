"""SHAP explainability for the LightGBM component (exact TreeSHAP).

Precomputes everything the dashboard needs as static files (the deployed app
never runs SHAP itself):
- ``shap_values.csv``  — per test cell × feature attribution (log10-cycles)
- ``shap_meta.json``   — base value, feature order by global importance
- a beeswarm + bar figure for results/figures

Sanity gate (contract): the top global feature must be physically sensible —
one of the ΔQ(V) statistics (capacity-curve degradation signal).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.splits import PRIMARY_TEST_CELLS, SECONDARY_TEST_CELLS
from src.models.feature_matrix import SCALAR_COLS, build_matrix
from src.utils.config import REPO_ROOT, load_config

PHYSICALLY_SENSIBLE_TOP = {"dq_var", "dq_min", "dq_skew", "dq_kurt",
                           "dq_mean", "fade_slope_2_h", "fade_slope_last10"}


def main() -> None:
    cfg = load_config()
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    artifacts = REPO_ROOT / cfg["paths"]["app_artifacts"]
    feat = REPO_ROOT / cfg["paths"]["features"]

    bundle = joblib.load(REPO_ROOT / cfg["paths"]["models"]
                         / "ensemble_conformal.joblib")
    ensemble = bundle["ensemble"]
    # LightGBM component (index 1 in the ensemble component list)
    lgbm = ensemble.fitted_[1].model_
    n_scalar = bundle["n_scalar"]

    cells = PRIMARY_TEST_CELLS + SECONDARY_TEST_CELLS
    data = build_matrix(feat / "matr_features.csv", feat / "sequences", cells)
    X = data["X"][:, :n_scalar]

    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(X)
    base = float(np.ravel(explainer.expected_value)[0])

    df_sv = pd.DataFrame(sv, columns=SCALAR_COLS)
    df_sv.insert(0, "cell_id", data["cell_ids"])
    df_x = pd.DataFrame(X, columns=SCALAR_COLS)
    df_x.insert(0, "cell_id", data["cell_ids"])

    importance = df_sv[list(SCALAR_COLS)].abs().mean().sort_values(
        ascending=False)
    top = importance.index[0]
    assert top in PHYSICALLY_SENSIBLE_TOP, (
        f"top SHAP feature '{top}' is not a degradation signal — "
        f"investigate before shipping")
    print("global importance (top 8):")
    print(importance.head(8).to_string())

    for out in (results_dir, artifacts):
        df_sv.to_csv(out / "shap_values.csv", index=False)
        df_x.to_csv(out / "shap_feature_values.csv", index=False)
        json.dump({"base_value_log10_cycles": base,
                   "global_importance": importance.to_dict(),
                   "model": "lightgbm component of the stacked ensemble",
                   "explainer": "shap.TreeExplainer (exact TreeSHAP)"},
                  open(out / "shap_meta.json", "w"), indent=2)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    exp = shap.Explanation(values=sv, base_values=base, data=X,
                           feature_names=list(SCALAR_COLS))
    plt.figure()
    shap.plots.beeswarm(exp, max_display=12, show=False)
    plt.title("SHAP attributions — LightGBM cycle-life model (test cells)")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=300)
    plt.close("all")
    print(f"saved SHAP artifacts; top feature: {top}")


if __name__ == "__main__":
    main()
