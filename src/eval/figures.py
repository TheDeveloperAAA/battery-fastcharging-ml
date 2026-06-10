"""Publication-style figures from the saved results (matplotlib, 300 dpi).

Run after train.py / protocol_opt.py:
    PYTHONPATH=. .venv/bin/python -m src.eval.figures
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import REPO_ROOT, load_config  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def fig_pred_vs_true(results, fig_dir):
    df = pd.read_csv(results / "predictions_intervals.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, split, title in [
            (axes[0], "primary_test", "Primary test (42 cells)"),
            (axes[1], "secondary_test", "Secondary test (40 cells)")]:
        sub = df[(df.method == "cross") & (df.split == split)]
        ax.errorbar(sub.true_cycle_life, sub.pred_cycle_life,
                    yerr=[sub.pred_cycle_life - sub.lower,
                          sub.upper - sub.pred_cycle_life],
                    fmt="o", ms=4, lw=0.8, capsize=2, alpha=0.85,
                    color="#1f77b4", ecolor="#9ecae1")
        lim = [100, max(2400, sub.upper.max() * 1.05)]
        ax.plot(lim, lim, "--", color="gray", lw=1)
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
               xlabel="Observed cycle life", title=title)
    axes[0].set_ylabel("Predicted cycle life (90% conformal interval)")
    fig.suptitle("Early prediction from the first 100 cycles "
                 "(ensemble + cross-conformal)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "pred_vs_true.png", bbox_inches="tight")
    plt.close(fig)


def fig_reliability(results, fig_dir):
    cal = json.load(open(results / "calibration_report.json"))
    rel = cal["reliability"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0.45, 1], [0.45, 1], "--", color="gray", lw=1, label="ideal")
    ax.plot(rel["nominal"], rel["empirical"], "o-", color="#d62728",
            label="ensemble + split conformal")
    picp = cal["methods"]["cross"]["primary_test"]["picp"]
    ax.scatter([cal["confidence_level"]], [picp], marker="*", s=180,
               color="#2ca02c", zorder=5,
               label=f"cross-conformal @ 90% (PICP {picp:.2f})")
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage "
           "(both test sets)", title="Calibration of prediction intervals")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", bbox_inches="tight")
    plt.close(fig)


def fig_early_prediction(results, fig_dir):
    metrics = json.load(open(results / "metrics.json"))
    df = pd.DataFrame(metrics["early_prediction"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, label, color in [
            ("elastic_net_variance", "variance model (elastic net)",
             "#1f77b4"),
            ("lightgbm", "LightGBM (all features)", "#d62728")]:
        sub = df[df.model == name]
        ax.plot(sub.horizon_cycles, sub.mape_pct, "o-", label=label,
                color=color)
    ax.set(xlabel="Cycles observed before predicting",
           ylabel="MAPE on primary test (%)",
           title="Prediction error vs early-data horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "early_prediction.png", bbox_inches="tight")
    plt.close(fig)


def fig_pareto(results, fig_dir, artifacts):
    pareto = pd.read_csv(results / "pareto_frontier.csv")
    obs = pd.read_csv(artifacts / "observed_protocols.csv")
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.scatter(obs.median_life, obs.charge_time_min, s=22, color="lightgray",
               edgecolor="gray", lw=0.4,
               label="observed protocols (median life)")
    ax.plot(pareto.lower_bound_life, pareto.charge_time_min, "o-",
            color="#1f77b4", label="Pareto frontier "
            "(life guaranteed at 90% conf.)")
    rec = json.load(open(results / "protocol_recommendation.json"))
    best = rec["recommended_grid"] if rec["recommended_grid"].get("feasible") \
        else rec["recommended_optuna"]
    if best.get("feasible"):
        ax.scatter([best["lower_bound_life"]], [best["charge_time_min"]],
                   marker="*", s=240, color="#d62728", zorder=5,
                   label=f"recommended @ L_min={rec['config']['l_min']}")
    ax.set(xlabel="Cycle life (cycles)",
           ylabel="Charge time 0→80% SOC (min)",
           title="Speed-versus-life trade-off")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)


def fig_cross_dataset(results, fig_dir):
    metrics = json.load(open(results / "metrics.json"))
    cd = metrics.get("cross_dataset", {})
    rows = []
    for ds in ("calce", "nasa"):
        e = cd.get(ds)
        if isinstance(e, dict):
            for c, t, p in zip(e["cells"], e["true_cycle_life"],
                               e["pred_cycle_life"]):
                rows.append({"dataset": ds.upper(), "cell": c, "true": t,
                             "pred": p})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for ds, color in [("CALCE", "#1f77b4"), ("NASA", "#d62728")]:
        sub = df[df.dataset == ds]
        if len(sub):
            ax.scatter(sub.true, sub.pred, s=30, color=color, label=ds)
    lim = [df[["true", "pred"]].min().min() * 0.8,
           df[["true", "pred"]].max().max() * 1.2]
    ax.plot(lim, lim, "--", color="gray", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="Observed cycle life",
           ylabel="Predicted (MATR-trained model)",
           title="Cross-dataset transfer (honest view)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_dataset.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    results = REPO_ROOT / cfg["paths"]["results"]
    artifacts = REPO_ROOT / cfg["paths"]["app_artifacts"]
    fig_dir = results / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_pred_vs_true(results, fig_dir)
    fig_reliability(results, fig_dir)
    fig_early_prediction(results, fig_dir)
    if (results / "pareto_frontier.csv").exists():
        fig_pareto(results, fig_dir, artifacts)
    fig_cross_dataset(results, fig_dir)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
