"""Battery cycle-life prediction & fast-charging optimisation dashboard.

Serves precomputed artifacts only (results of the training pipeline in the
main repository) plus the compact LightGBM SoH estimator for live what-if
interaction. No raw datasets, no heavy models — designed for the free
Hugging Face Spaces tier.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Battery Fast-Charging ML", page_icon="🔋",
                   layout="wide")


# --------------------------------------------------------------------------
# cached loaders
# --------------------------------------------------------------------------

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


@st.cache_data
def load_json(name: str) -> dict:
    import json
    return json.loads((ART / name).read_text())


@st.cache_data
def load_grid():
    z = np.load(ART / "protocol_grid.npz")
    return {k: z[k] for k in z.files}


@st.cache_resource
def load_compact_model():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(ART / "compact_soh_estimator.txt"))


def charge_time_to_80(c1: float, q1: float, c2: float) -> float:
    return 60.0 * ((q1 / 100.0) / c1 + ((80.0 - q1) / 100.0) / c2)


def grid_interp(grid: dict, key: str, c1: float, q1: float,
                c2: float) -> float:
    """Trilinear interpolation on the precomputed protocol grid."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (grid["c1"], grid["q1"], grid["c2"]), grid[key],
        bounds_error=False, fill_value=None)
    return float(interp([[c1, q1, c2]])[0])


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

st.title("🔋 Probabilistic Battery Cycle-Life Prediction & "
         "Degradation-Aware Fast Charging")
st.markdown(
    "A machine-learning system that predicts how long a lithium-ion cell "
    "will last from its **first 100 charge–discharge cycles** — with "
    "calibrated uncertainty — and recommends fast-charging current profiles "
    "that charge quickly **without violating a battery-lifetime guarantee**. "
    "Trained on the Toyota Research Institute / Severson 2019 dataset "
    "(124 LFP cells, 72 fast-charging protocols).")

view = st.sidebar.radio(
    "View", ["Cycle-life predictions", "Charging-protocol advisor",
             "Speed-vs-life Pareto frontier", "Explainability (SHAP)",
             "Live compact model"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to read the numbers**\n\n"
    "- *Cycle life*: cycles until the cell falls to 80% of its rated "
    "capacity.\n"
    "- *90% interval*: a conformal-prediction band calibrated so that about "
    "90% of unseen cells fall inside it.\n"
    "- *C-rate*: charging current relative to capacity (1C ≈ 1.1 A here; "
    "4C charges in ~15 min).")


# --------------------------------------------------------------------------
if view == "Cycle-life predictions":
    df = load_csv("predictions_intervals.csv")
    cal = load_json("calibration_report.json")

    method = st.radio("Conformal method", ["cross", "split"], horizontal=True,
                      help="'cross' = 5-fold cross-conformal (uses all "
                           "training cells); 'split' = held-out calibration "
                           "set. Both are leakage-free.")
    split = st.selectbox("Test set", ["primary_test", "secondary_test"],
                         format_func=lambda s: {
                             "primary_test": "Primary test (42 cells, same "
                                             "batches as training)",
                             "secondary_test": "Secondary test (40 cells, "
                                               "later batch — harder)"}[s])
    sub = df[(df.method == method) & (df.split == split)].sort_values(
        "true_cycle_life").reset_index(drop=True)

    m = cal["methods"][method][split]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{m['rmse_cycles']:.0f} cycles")
    c2.metric("MAPE", f"{m['mape_pct']:.1f} %")
    c3.metric("Coverage (PICP)", f"{m['picp']*100:.0f} %",
              help=f"Target: {cal['confidence_level']*100:.0f}%")
    c4.metric("Mean interval width", f"{m['mpiw_cycles']:.0f} cycles")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub.true_cycle_life, y=sub.pred_cycle_life, mode="markers",
        name="prediction",
        error_y=dict(type="data", symmetric=False,
                     array=sub.upper - sub.pred_cycle_life,
                     arrayminus=sub.pred_cycle_life - sub.lower,
                     thickness=1),
        text=sub.cell_id, hovertemplate="%{text}<br>true %{x:.0f} | "
        "pred %{y:.0f}<extra></extra>"))
    lim = [0, max(sub.true_cycle_life.max(), sub.upper.max()) * 1.05]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title="True cycle life (cycles)",
                      yaxis_title="Predicted (with 90% interval)",
                      height=560)
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        "Each point is one battery cell, predicted **after observing only "
        "its first 100 cycles** (~13% of the median cell's life). "
        "Vertical bars are 90% "
        "conformal intervals — calibrated, not eyeballed: across held-out "
        "cells, ~90% of the bars cross the diagonal.")

    with st.expander("Reliability: does 'X% confidence' really mean X%?"):
        rel = cal["reliability"]
        figr = go.Figure()
        figr.add_trace(go.Scatter(x=rel["nominal"], y=rel["empirical"],
                                  mode="lines+markers", name="empirical"))
        figr.add_trace(go.Scatter(x=[0.5, 1], y=[0.5, 1], mode="lines",
                                  name="ideal",
                                  line=dict(dash="dash", color="gray")))
        figr.update_layout(xaxis_title="Nominal confidence level",
                           yaxis_title="Observed coverage on test cells",
                           height=420)
        st.plotly_chart(figr, width="stretch")


# --------------------------------------------------------------------------
elif view == "Charging-protocol advisor":
    rec = load_json("protocol_recommendation.json")
    grid = load_grid()
    b = rec["bounds"]

    st.markdown(
        "TRI fast-charging protocols have the form **C1(Q1%)–C2**: charge at "
        "C1 until the battery reaches Q1% state-of-charge, then at C2 up to "
        "80%, then a fixed gentle 1C CC-CV step to full. Pick a candidate "
        "profile and see what it costs in battery life.")

    col1, col2, col3 = st.columns(3)
    c1 = col1.slider("C1 — first-step current (C-rate)",
                     float(b["c1"][0]), float(b["c1"][1]), 5.0, 0.05)
    q1 = col2.slider("Q1 — switch point (% SOC)",
                     float(b["q1"][0]), float(b["q1"][1]), 40.0, 1.0)
    c2 = col3.slider("C2 — second-step current (C-rate)",
                     float(b["c2"][0]), float(b["c2"][1]), 4.0, 0.05)

    t = charge_time_to_80(c1, q1, c2)
    pred = grid_interp(grid, "pred", c1, q1, c2)
    lo = grid_interp(grid, "lower", c1, q1, c2)
    hi = grid_interp(grid, "upper", c1, q1, c2)
    l_min = rec["config"]["l_min"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Charge time 0→80%", f"{t:.1f} min")
    k2.metric("Predicted cycle life", f"{pred:.0f} cycles",
              help="Median prediction of the protocol→life surrogate")
    k3.metric("90%-confidence lower bound", f"{lo:.0f} cycles")

    if lo >= l_min:
        st.success(f"✅ Meets the lifetime guarantee: ≥ {l_min} cycles with "
                   f"90% confidence (lower bound {lo:.0f}).")
    else:
        st.error(f"❌ Violates the lifetime guarantee of {l_min} cycles at "
                 f"90% confidence (lower bound {lo:.0f}). Charge slower or "
                 f"rebalance the two steps.")

    best = rec["recommended_grid"] if rec["recommended_grid"]["feasible"] \
        else rec["recommended_optuna"]
    st.markdown("---")
    st.markdown(
        f"**Optimiser recommendation @ L_min = {l_min} cycles:** "
        f"`{best['c1']:.2f}C({best['q1_pct']:.0f}%)-{best['c2']:.2f}C` — "
        f"{best['charge_time_min']:.1f} min to 80%, guaranteed "
        f"≥ {best['lower_bound_life']:.0f} cycles at 90% confidence. "
        f"(Risk-constrained Bayesian optimisation, verified by dense grid "
        f"search.)")


# --------------------------------------------------------------------------
elif view == "Speed-vs-life Pareto frontier":
    pareto = load_csv("pareto_frontier.csv")
    obs = load_csv("observed_protocols.csv")

    st.markdown(
        "**The fundamental trade-off:** charging faster degrades the cell "
        "faster. Each blue point is the *fastest protocol that still "
        "guarantees* a given cycle life at 90% confidence; grey points are "
        "the 72 protocols actually tested in the lab (median observed life "
        "per protocol).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs.median_life, y=obs.charge_time_min, mode="markers",
        name="observed protocols (median life)",
        marker=dict(color="lightgray", size=7),
        text=obs.protocol,
        hovertemplate="%{text}<br>life %{x:.0f} | %{y:.1f} min"
                      "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=pareto.lower_bound_life, y=pareto.charge_time_min,
        mode="lines+markers", name="Pareto frontier (90% guaranteed life)",
        marker=dict(size=8),
        hovertemplate="guaranteed ≥ %{x:.0f} cycles<br>%{y:.1f} min"
                      "<extra></extra>"))
    fig.update_layout(xaxis_title="Cycle life (cycles)",
                      yaxis_title="Charge time 0→80% SOC (minutes)",
                      height=560)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        pareto[["l_min", "c1", "q1_pct", "c2", "charge_time_min",
                "lower_bound_life"]].rename(columns={
                    "l_min": "required life (cycles)",
                    "c1": "C1", "q1_pct": "Q1 (%)", "c2": "C2",
                    "charge_time_min": "charge time (min)",
                    "lower_bound_life": "guaranteed life (90%)"}),
        width="stretch", hide_index=True)


# --------------------------------------------------------------------------
elif view == "Explainability (SHAP)":
    sv = load_csv("shap_values.csv")
    fv = load_csv("shap_feature_values.csv")
    meta = load_json("shap_meta.json")

    st.markdown(
        "**Which measurements drive each prediction?** SHAP attributes every "
        "prediction to the input features (units: log₁₀ cycles; positive = "
        "longer predicted life). The dominant feature is the *variance of "
        "ΔQ(V)* — how much the discharge curve has already shifted between "
        "cycle 10 and cycle 100 — exactly the physics-backed signal "
        "identified by Severson et al. (2019).")

    imp = pd.Series(meta["global_importance"]).head(12)
    figb = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1],
                            orientation="h"))
    figb.update_layout(title="Global importance — mean |SHAP|",
                       xaxis_title="mean |SHAP| (log₁₀ cycles)", height=420)
    st.plotly_chart(figb, width="stretch")

    feats = imp.index[:8].tolist()
    rows = []
    for f in feats:
        x = fv[f].to_numpy()
        rng = np.nanmax(x) - np.nanmin(x)
        norm = (x - np.nanmin(x)) / (rng if rng > 0 else 1.0)
        rows.append(pd.DataFrame({"feature": f, "shap": sv[f],
                                  "feature_value": norm}))
    dd = pd.concat(rows)
    figsw = go.Figure(go.Scatter(
        x=dd.shap, y=dd.feature, mode="markers",
        marker=dict(color=dd.feature_value, colorscale="RdBu_r", size=6,
                    colorbar=dict(title="feature value<br>(low→high)")),
        hovertemplate="SHAP %{x:.3f}<extra></extra>"))
    figsw.update_layout(title="Beeswarm — per-cell attributions (test sets)",
                        xaxis_title="SHAP value (log₁₀ cycles)", height=480)
    st.plotly_chart(figsw, width="stretch")

    st.markdown("**Single-cell explanation**")
    cell = st.selectbox("Cell", sv.cell_id.tolist())
    row = sv[sv.cell_id == cell].drop(columns="cell_id").iloc[0]
    top = row.abs().sort_values(ascending=False).head(10).index
    contrib = row[top]
    figw = go.Figure(go.Bar(
        x=contrib.values[::-1], y=list(top)[::-1], orientation="h",
        marker_color=["#d62728" if v < 0 else "#2ca02c"
                      for v in contrib.values[::-1]]))
    base = meta["base_value_log10_cycles"]
    pred = base + row.sum()
    figw.update_layout(
        title=f"{cell}: base {10**base:.0f} cycles → predicted "
              f"{10**pred:.0f} cycles",
        xaxis_title="contribution (log₁₀ cycles)", height=420)
    st.plotly_chart(figw, width="stretch")


# --------------------------------------------------------------------------
elif view == "Live compact model":
    meta = load_json("compact_soh_estimator.json")
    booster = load_compact_model()
    fv = load_csv("shap_feature_values.csv")

    st.markdown(
        f"This is the **deployable estimator**: a {meta['n_trees']}-tree "
        f"LightGBM model, {meta['size_bytes']/1024:.0f} kB on disk, "
        f"depth ≤ {meta['num_leaves_max']} leaves per tree — small enough "
        f"for a battery-management controller. It runs **live in this app**: "
        f"pick a real test cell, then perturb its degradation signals and "
        f"watch the prediction respond.")

    cell = st.selectbox("Start from cell", fv.cell_id.tolist())
    x0 = fv[fv.cell_id == cell].drop(columns="cell_id").iloc[0]
    names = meta["feature_names"]

    adjustable = ["dq_var", "dq_min", "fade_slope_2_h", "qd_cycle2",
                  "charge_time_min"]
    cols = st.columns(len(adjustable))
    x = x0.copy()
    for c, f in zip(cols, adjustable):
        if f in x.index and np.isfinite(x[f]):
            lo_v, hi_v = float(fv[f].min()), float(fv[f].max())
            if lo_v < hi_v:
                x[f] = c.slider(f, lo_v, hi_v, float(x[f]),
                                (hi_v - lo_v) / 100)

    X = np.array([[x.get(n, np.nan) for n in names]])
    pred = float(booster.predict(X)[0])
    base_pred = float(booster.predict(
        np.array([[x0.get(n, np.nan) for n in names]]))[0])
    d1, d2 = st.columns(2)
    d1.metric("Compact-model prediction", f"{10**pred:.0f} cycles",
              delta=f"{10**pred - 10**base_pred:+.0f} vs the real cell")
    d2.metric("Cell's original prediction", f"{10**base_pred:.0f} cycles")

    st.caption(
        "Predictions are point estimates from the compact model "
        f"(test-set RMSE ≈ {meta['metrics']['primary_test']['rmse_cycles']:.0f}"
        " cycles); the full dashboard intervals come from the conformal "
        "ensemble in the main pipeline.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Severson et al., Nat. Energy (2019); Attia et al., Nature (2020) "
    "— TRI/MATR public dataset. Validation: CALCE, NASA PCoE. "
    "Code: see the GitHub repository.")
