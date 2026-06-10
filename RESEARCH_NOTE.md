# Research note — A battery-health prediction and charging-reference layer for EV fast-charging converters

*Prepared for the power-electronics group of Dr. S. K. Patro (multiport EV
charging stations, CC-CV control of dual-active-bridge converters, hybrid
Si/SiC chargers). One page; numbers refer to artifacts in `results/`.*

## What this delivers to the group

A converter delivers whatever current reference its control loop is given.
This project supplies the *battery-side intelligence that should set that
reference*: (i) a data-driven estimator of cell cycle life / state-of-health
from early cycling data, with **calibrated confidence bounds** rather than a
bare point estimate, and (ii) a **charging-current recommendation layer**
that turns those bounds into CC-stage current setpoints which minimise
0→80%-SOC charge time while honouring an explicit lifetime constraint
(`life ≥ L_min with 90% confidence`). Both are pure software, trained on the
public Toyota Research Institute fast-charging dataset (124 LFP 18650 cells,
72 two-step CC fast-charge policies, 4C discharge, 30 °C) and validated on
CALCE and NASA cells.

## How it maps onto the group's work

- **CC-CV / DAB control (M. Dhayapule):** the recommended two-step policy
  `C1(Q1%)→C2` is exactly a pair of CC current references plus a switchover
  SOC — directly usable as the outer-loop setpoint schedule for a DAB CC-CV
  controller. The compact estimator (a ~100 kB gradient-boosted-tree dump;
  pure integer/float tree-walk, no matrix algebra) is sized for the same DSP
  class that runs the converter loops (e.g. EEN-635 platforms), enabling an
  SoH-adaptive current reference as the pack ages.
- **Multiport stations (J. Kumar):** with per-port life predictions and
  uncertainty, limited converter capacity can be allocated health-aware
  (e.g. prioritise ports whose cells are far from their life constraint);
  the Pareto frontier quantifies the watt-minutes a port gives up per cycle
  of guaranteed life gained.
- **Hybrid Si/SiC charger (Infineon project):** the per-protocol charging
  stress profiles (current windows per SOC band, charge-time distribution)
  bound the operating points and thermal duty the power stage must support.

## What was actually built (and verified)

1. **Predictor.** Severson-style early-cycle features (the variance of the
   ΔQ(V) curve between cycles 10 and 100 is the dominant signal) feed an
   ensemble (elastic net, LightGBM, 1D-CNN on the Q(V) curves). Accuracy on
   the canonical held-out splits is competitive with the published benchmark
   (Nature Energy 2019), and prediction intervals are **conformally
   calibrated**: a 90% interval empirically covers ≈90% of unseen cells —
   verified, not assumed. <!-- NOTE_NUMBERS -->
2. **Decision layer.** Risk-constrained optimisation: minimise analytic
   charge time over the two-step CC policy space subject to the
   90%-confidence lower life bound ≥ L_min; solved with Bayesian
   optimisation and certified by dense grid search; outputs recommended
   setpoints and the full speed-vs-life Pareto frontier.
3. **Deployment artifacts.** Compact on-controller estimator, an interactive
   dashboard (predictions with bands, protocol advisor, Pareto explorer,
   SHAP attributions showing the physics-consistent drivers), reproducible
   pipeline, and tests.

## Scope and ceiling (stated plainly)

This is battery-side data science and charging-reference logic; it does not
design converters. Its value is as the complementary decision layer the
hardware delivers against. Limits: protocols outside the dataset's observed
support are extrapolation (the optimiser is clamped to the support);
chemistry transfer (LFP→LCO) degrades accuracy and is reported honestly;
conformal guarantees are marginal, not per-cell. A short co-authored
SEFET/ECCE-style paper on "predict-then-decide fast charging with calibrated
uncertainty" is a realistic target.
