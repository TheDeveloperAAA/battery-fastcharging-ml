"""Generate notebooks/walkthrough.ipynb — the narrating interview notebook.

The notebook tells the project story and loads REAL saved results (no
fabricated outputs): metrics.json, calibration_report.json, the Pareto
frontier, SHAP values, and the figures. Regenerate with:

    PYTHONPATH=. .venv/bin/python notebooks/build_walkthrough.py
"""

import nbformat as nbf


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                          "name": "python3"}

cells = [
    md("""# Probabilistic Battery Cycle-Life Prediction & Degradation-Aware Fast Charging

**A narrated walkthrough.** This notebook reads the *actual saved artifacts*
of the pipeline (no numbers are re-typed by hand) and explains each stage:

1. **Data** — TRI/MATR fast-charging dataset (Severson 2019 + Attia 2020),
   124-cell canonical cohort + 45 CLO cells; CALCE & NASA for transfer.
2. **Early-cycle features** — why the variance of ΔQ₁₀₀₋₁₀(V) predicts life.
3. **Probabilistic predictor** — elastic net → LightGBM → 1D-CNN → stacked
   ensemble, wrapped in conformal prediction for calibrated intervals.
4. **Decision layer** — fastest charging protocol subject to
   P(life ≥ L_min) ≥ 90%.
5. **Explainability** — SHAP, and whether it matches the physics.

> Run top-to-bottom after `python -m src.models.train` and
> `python -m src.optim.protocol_opt` have produced `results/`.
"""),
    code("""import json, pandas as pd, numpy as np
from pathlib import Path
ROOT = Path("..") if Path.cwd().name == "notebooks" else Path(".")
RES = ROOT / "results"
metrics = json.loads((RES / "metrics.json").read_text())
cal = json.loads((RES / "calibration_report.json").read_text())
print("artifacts loaded:", sorted(p.name for p in RES.glob('*.json')))"""),
    md("""## 1. The dataset and the splits

124 A123 LFP cells (1.1 Ah), each fast-charged with one of 72 two-step
policies `C1(Q1%)-C2` and discharged at 4C. Cycle life spans **148–2237
cycles** despite identical chemistry — the protocol matters enormously.
We use the canonical Severson splits (41 train / 42 primary test / 40
secondary test, outlier b2c1 excluded) so our numbers are directly
comparable with the published Table 1."""),
    code("""df = pd.read_csv(ROOT/'data/features/matr_features.csv') \\
    if (ROOT/'data/features/matr_features.csv').exists() else None
if df is not None:
    ax = df.cycle_life.hist(bins=40, figsize=(7,3))
    ax.set(xlabel='cycle life (cycles)', ylabel='# cells',
           title='MATR cycle-life distribution (180 cells incl. CLO batch)')
else:
    print('feature table not present (data/ is gitignored) — '
          'run the Phase 0 pipeline to regenerate')"""),
    md("""## 2. Point accuracy vs the published benchmark

`metrics.json` stores our results next to Severson's Table 1 (cited, not
re-derived). RMSE in cycles is dominated by the longest-lived cells; MAPE is
the fairer cross-paper comparison."""),
    code("""rows = []
for name, entry in metrics['models'].items():
    rows.append({'model': name,
                 'primary RMSE': entry['primary_test']['rmse_cycles'],
                 'primary MAPE %': entry['primary_test']['mape_pct'],
                 'secondary RMSE': entry['secondary_test']['rmse_cycles'],
                 'secondary MAPE %': entry['secondary_test']['mape_pct']})
ours = pd.DataFrame(rows).round(1)
bench = metrics['benchmark']
print('Severson 2019 Table 1 (primary test, excl. outlier): '
      f"variance RMSE {bench['rmse_cycles']['variance']['primary_test_excl_outlier']}, "
      f"full RMSE {bench['rmse_cycles']['full']['primary_test_excl_outlier']}")
ours"""),
    md("""## 3. Calibrated uncertainty — the heart of the project

A point estimate is not enough to make charging decisions. We wrap the
ensemble in **conformal prediction**: split-conformal (held-out calibration
cells) and cross-conformal (5-fold, uses all training cells). The check that
matters: a *90% interval should contain ~90% of unseen cells* (PICP), and be
as narrow as possible (MPIW)."""),
    code("""for method, entry in cal['methods'].items():
    p = entry['primary_test']
    print(f"{method:6s} primary: PICP {p['picp']:.2f} "
          f"(nominal {cal['confidence_level']}), MPIW {p['mpiw_cycles']:.0f} cycles")
rel = pd.DataFrame(cal['reliability'])
rel.plot(x='nominal', y='empirical', marker='o', figsize=(5,4),
         title='Reliability: nominal vs empirical coverage', legend=False)"""),
    md("""## 4. Early prediction — how soon can we know?

The same pipeline run with fewer observed cycles. Error rises gracefully as
the horizon shrinks — at 100 cycles (≈5–8% of life) prediction is already
actionable."""),
    code("""ep = pd.DataFrame(metrics['early_prediction'])
ep.pivot_table(index='horizon_cycles', columns='model',
               values='mape_pct').plot(marker='o', figsize=(6,4),
               ylabel='MAPE on primary test (%)',
               title='Error vs cycles observed')"""),
    md("""## 5. The decision layer: risk-constrained fast charging

minimise charge time(C1, Q1, C2) subject to
**conformal lower bound on life ≥ L_min at 90% confidence** — a
CVaR-style constraint. Search: Optuna TPE, certified against a dense grid.
The Pareto frontier makes the speed-vs-life price explicit."""),
    code("""rec = json.loads((RES/'protocol_recommendation.json').read_text())
pareto = pd.read_csv(RES/'pareto_frontier.csv')
best = rec['recommended_grid'] if rec['recommended_grid'].get('feasible') \\
    else rec['recommended_optuna']
print(f"Recommended @ L_min={rec['config']['l_min']}: "
      f"{best['c1']:.2f}C({best['q1_pct']:.0f}%)-{best['c2']:.2f}C  "
      f"→ {best['charge_time_min']:.1f} min to 80% SOC, "
      f"guaranteed ≥{best['lower_bound_life']:.0f} cycles @ 90%")
pareto.plot(x='lower_bound_life', y='charge_time_min', marker='o',
            figsize=(6,4), legend=False,
            xlabel='guaranteed cycle life (90% conf.)',
            ylabel='charge time 0→80% (min)',
            title='Speed-versus-life Pareto frontier')"""),
    md("""## 6. Explainability — does the model believe the physics?

SHAP on the LightGBM component. The dominant feature should be (and is) the
**variance of ΔQ(V)** — the same quantity Severson et al. identified, which
tracks loss of active material visible in how the discharge curve deforms."""),
    code("""shap_meta = json.loads((RES/'shap_meta.json').read_text())
imp = pd.Series(shap_meta['global_importance']).head(10)
imp[::-1].plot.barh(figsize=(6,4), title='Global SHAP importance')
print('top feature:', imp.index[0])"""),
    md("""## 7. Cross-dataset honesty + the compact estimator

- **Transfer (MATR→CALCE/NASA)**: different chemistry (LFP→LCO), different
  protocols — we report degradation rather than hide it.
- **Compact model**: a ~100 kB LightGBM dump (the artifact a BMS/DSP would
  run); accuracy next to the full ensemble."""),
    code("""cd = metrics['cross_dataset']
for ds in ('calce','nasa'):
    e = cd.get(ds)
    if isinstance(e, dict):
        print(f"{ds.upper():6s} zero-shot MAPE {e['zero_shot']['mape_pct']:.0f}%  | "
              f"LOO-intercept-corrected {e['loo_intercept_corrected']['mape_pct']:.1f}%  | "
              f"Spearman ρ {e['spearman_rank_corr']:.2f}  (n={e['zero_shot']['n']})")
cm = metrics['compact_model']
print(f"\\ncompact: {cm['n_trees']} trees, {cm['size_bytes']/1024:.0f} kB, "
      f"primary MAPE {cm['metrics']['primary_test']['mape_pct']:.1f}%")"""),
    md("""## Takeaways

1. **Accuracy competitive with the published benchmark** from 100 cycles.
2. **Intervals that mean what they say** (PICP ≈ nominal) — the property the
   decision layer stands on.
3. **A defensible charging recommendation**: fastest protocol whose 90%
   lower life bound clears the requirement, plus the full Pareto curve.
4. Compact, deployable estimator + dashboard for stakeholders.

*Built fully from public data; reproducible end-to-end via the repository
README.*"""),
]

nb.cells = cells
out = "notebooks/walkthrough.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
