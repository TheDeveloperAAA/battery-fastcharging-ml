---
title: Battery Fast-Charging ML
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Probabilistic battery life prediction + charging advisor
---

# Probabilistic Battery Cycle-Life Prediction & Degradation-Aware Fast Charging

Interactive dashboard for an ML system that predicts lithium-ion cycle life
from the first 100 cycles **with calibrated conformal uncertainty**, and
recommends fast-charging protocols that minimise charge time subject to a
probabilistic battery-life constraint.

- **Data**: Toyota Research Institute / Severson et al. (Nature Energy 2019)
  + Attia et al. (Nature 2020) public fast-charging dataset; CALCE and NASA
  PCoE for cross-dataset validation.
- **Everything heavy is precomputed** — the Space serves static artifacts
  plus a compact (~100 kB) LightGBM SoH estimator for live what-if
  interaction.

Source code, training pipeline, and reproduction steps: see the GitHub
repository linked from the main project README.
