# Probabilistic Battery Cycle-Life Prediction & Degradation-Aware Fast-Charging Optimization

Predicts the cycle life of lithium-ion cells from their **first 100
charge–discharge cycles** with **calibrated conformal uncertainty**, then uses
the probabilistic predictions to recommend fast-charging protocols that
**minimise charging time subject to a battery-life guarantee**
(`P(life ≥ L_min) ≥ 90%`).

**Live dashboard:** _will be filled at deployment_ <!-- SPACE_URL -->

Built on public data only: the Toyota Research Institute / Stanford / MIT
fast-charging dataset (Severson et al. 2019; Attia et al. 2020), validated
cross-dataset on CALCE (Univ. of Maryland) and NASA PCoE cells.

## Headline results

<!-- HEADLINE_RESULTS -->

## What's in the box

| Stage | Deliverable | Where |
|---|---|---|
| 0 | Robust data pipeline (download with mirrors/retries, BatteryML preprocessing, Severson features, canonical splits) | `src/data/` |
| 1 | Probabilistic predictor: elastic-net baselines, tuned LightGBM, two-branch 1D-CNN, stacked ensemble, MAPIE conformal intervals (split + cross), calibration report, compact on-controller estimator | `src/models/`, `results/`, `models/` |
| 2 | Risk-constrained charging-protocol optimiser + speed-vs-life Pareto frontier | `src/optim/` |
| 3 | Streamlit dashboard (predictions with bands, protocol advisor, Pareto explorer, SHAP) deployed on HF Spaces (Docker) | `app/` |
| 4 | Tests, narrated notebook, research note, publication-style figures | `tests/`, `notebooks/`, `RESEARCH_NOTE.md`, `results/figures/` |

## Reproduce from scratch

Requires Python 3.11, ~15 GB disk, internet. On macOS: `brew install libomp`
first (LightGBM links it).

```bash
git clone <THIS_REPO> && cd <THIS_REPO>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# BatteryML is not on PyPI — install from source:
git clone https://github.com/microsoft/BatteryML third_party/BatteryML
pip install ./third_party/BatteryML

# 1. Download (~11 GB; resumable, verified sizes, HF mirror fallback)
python -m src.data.download --dataset all --out data/raw

# 2. Preprocess + engineer features (MATR is memory-bounded for 8 GB machines)
PYTHONPATH=. python -m src.data.build_dataset --dataset MATR
PYTHONPATH=. python -m src.data.build_dataset --dataset CALCE
PYTHONPATH=. python -m src.data.build_dataset --dataset NASA

# 3. Train + conformalize + evaluate (TRAINING_MODE=fast skips Optuna search)
OMP_NUM_THREADS=1 PYTHONPATH=. python -m src.models.train

# 4. Protocol optimisation + Pareto frontier
OMP_NUM_THREADS=1 PYTHONPATH=. python -m src.optim.protocol_opt

# 5. Explainability + figures
PYTHONPATH=. python -m src.eval.explain
PYTHONPATH=. python -m src.eval.figures

# 6. Dashboard (locally)
streamlit run app/app.py

# Tests
PYTHONPATH=. python -m pytest tests/ -q
```

Every stage is config-driven (`config.yaml`: seeds, α, L_min, protocol-space
bounds, paths) and idempotent (cached downloads/preprocessing are skipped).

## Method in one paragraph

Early-cycle features are engineered per Severson et al. — most importantly
`log₁₀ var(ΔQ₁₀₀₋₁₀(V))`, the variance across a 1000-point voltage grid of
the difference between the cycle-100 and cycle-10 discharge-capacity curves —
plus capacity-fade slope, internal-resistance trend, temperature integral,
charge time, and the charging-protocol parameters. Models are trained on
`log₁₀(cycle life)` with the canonical 41/42/40 splits. A stacked ensemble
(elastic net + LightGBM + 1D-CNN over the Q(V) curves, NNLS stacking weights
from out-of-fold predictions) is wrapped with MAPIE conformal prediction
(split and 5-fold cross variants) to produce intervals whose coverage is
verified on held-out cells. The decision layer fits a protocol→life surrogate
with its own conformal calibration and solves: minimise analytic charge time
over two-step policies `C1(Q1%)→C2` subject to the 90%-confidence lower life
bound clearing `L_min` (Optuna TPE search, certified by dense grid), tracing
the full speed-vs-life Pareto frontier.

## Honesty notes

- Benchmarks against Severson Table 1 are cited from the paper, not re-run.
- Cross-dataset transfer (LFP→LCO chemistry) degrades, and we report it.
- Conformal validity is marginal w.r.t. the protocol distribution that
  generated the data; the optimiser is clamped to the observed support.
- This is battery-side prediction and charging-reference logic — not
  power-electronic converter design.

## Citations

- K. A. Severson et al., "Data-driven prediction of battery cycle life
  before capacity degradation," *Nature Energy* 4, 383–391 (2019).
  doi:10.1038/s41560-019-0356-8
- P. M. Attia et al., "Closed-loop optimization of fast-charging protocols
  for batteries with machine learning," *Nature* 578, 397–402 (2020).
  doi:10.1038/s41586-020-1994-5
- BatteryML (Microsoft): https://github.com/microsoft/BatteryML
- CALCE battery data: https://calce.umd.edu/battery-data ·
  NASA PCoE: Prognostics Center of Excellence Battery Data Set
