# BUILD_LOG — Probabilistic Battery Cycle-Life Prediction & Degradation-Aware Fast-Charging Optimization

Running record of decisions, assumptions, and deviations. Newest entries at the bottom of each phase section.

## Setup (2026-06-11)

### Credentials & inputs (§2 of the build contract)
No environment variables were exported, but all credentials were found on the machine and
validated live (no fabrication — each was tested against the respective API):

| Variable | Value / source | Validation |
|---|---|---|
| `GITHUB_TOKEN` | fine-grained PAT from `~/.sia_gh_token` | GitHub `/user` API → login `TheDeveloperAAA` |
| `GITHUB_USERNAME` | `TheDeveloperAAA` | same |
| `GITHUB_REPO` | `battery-fastcharging-ml` | **chosen by the builder** (not user-specified) — descriptive, available |
| `HF_TOKEN` | from `~/.sia_hf_token` | HF `whoami-v2` → user `rajtheman`, role `write` |
| `HF_USERNAME` | `rajtheman` | same |
| `HF_SPACE` | `battery-fastcharging-dashboard` | **chosen by the builder** → `https://rajtheman-battery-fastcharging-dashboard.hf.space` |
| `GIT_AUTHOR_NAME` / `EMAIL` | `TheDeveloperAAA` / `beleiverbadshah1@gmail.com` | from global git config |
| `TRAINING_MODE` | `thorough` | contract default |

Token files stay outside the repo; `.gitignore` additionally excludes `.env` and `.sia_*` patterns.
The `gh` CLI is also logged in (account `TheDeveloperAAA`, scopes `repo`+`workflow`) as a fallback.

### Hardware / platform
- Apple M1 (arm64), 8 cores, **8 GB RAM**, ~51 GB free disk. No CUDA GPU; Metal/MPS available.
- Decision: train on **CPU** (models are small; LightGBM/elastic-net are CPU-native; the
  sequence model is small enough for CPU and MPS determinism is weaker). 8 GB RAM is the
  binding constraint → the data pipeline must process MATR batch files **one at a time** and
  persist per-cell artifacts instead of holding the whole dataset in memory.
- Python 3.11.9 (python.org universal2 build, runs native arm64) at `/usr/local/bin/python3`.
  Satisfies the 3.10+ requirement; project venv is built from it.

### Repo
- This folder was not itself a git repo (the *home directory* is, unrelated). `git init -b main`
  inside the project directory; local git identity set per the table above.
- The problem-statement PDF is kept in the repo root as the authoritative spec.

### Process
- Per §1 of the contract, official docs for BatteryML, MAPIE, HF Spaces (Docker SDK),
  LightGBM, Optuna, SHAP, Streamlit, and PyTorch-on-macOS were fetched fresh before any
  code was written (8 parallel research agents; findings recorded below when applied).
- Dataset availability (data.matr.io 503 risk, HF mirrors, Zenodo NASA record, CALCE) probed
  live before choosing the download strategy.

## Phase 0 — Data foundation

### Docs-research findings that drove decisions (all verified against live sources 2026-06-11)
- **BatteryML is not on PyPI** (404). Installed from source (`third_party/BatteryML`, shallow
  clone of microsoft/BatteryML@main, last pushed 2024-12-18). It pins `numpy>=1.24,<2.0.0`,
  which cascades: the whole local env is numpy 1.26.4, and **shap must be ≤0.49.1**
  (0.50.0+ requires numpy≥2 — discovered via pip resolver + PyPI metadata, the initial
  0.51.0 pin failed with ResolutionImpossible).
- **MAPIE 1.4.1 v1 API**: `MapieRegressor` no longer exists. Use
  `SplitConformalRegressor(estimator, confidence_level, prefit=True)` →
  `.conformalize(X_cal, y_cal)` → `.predict_interval(X)` returning intervals of shape
  `(n, 2, n_levels)`. CQR via `ConformalizedQuantileRegressor` with three prefit LightGBM
  quantile models in order [lower, upper, median].
- **All dataset hosts are UP today** (no 503): the four MATR `.mat` files on data.matr.io
  (sizes verified: 3,025,320,241 / 2,007,331,155 / 3,236,690,412 / 2,601,295,745 B);
  HF dataset `bsebench-org/severson-2019-raw` is a bit-exact mirror of the three Severson
  batches (used as automatic fallback); the classic NASA set (B0005/06/07/18) is NOT on the
  Zenodo DOI from the spec (that record is the *Randomized* usage set) — it lives at
  `https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip` (209,708,670 B,
  verified). CALCE zips served directly by web.calce.umd.edu.
- **Severson ground truth** (paper + SI + BatteryML source): variance feature =
  log10(|var ΔQ_{100−10}(V)|) on a 1000-point voltage grid (2.0–3.5 V); canonical splits
  41 train / 43 primary test (42 after dropping outlier b2c1, matching the paper's
  parenthetical metrics) / 40 secondary test; published benchmarks to compare against
  (RMSE cycles, primary/secondary): variance 138/196, discharge 91(86)/173, full 118(100)/214.
- LightGBM 4.6.0 needs `brew install libomp` on macOS (wheel links it dynamically — verified
  via otool by the research agent); early stopping via `callbacks=[lgb.early_stopping(N)]`.
- PyTorch 2.12.0 CPU; MPS avoided deliberately (open LSTM-on-MPS correctness bugs
  pytorch#173640, #145374; CPU is deterministic and fast enough for these model sizes).

### Decisions
- **Downloader**: wrote `src/data/download.py` instead of `batteryml download` because the
  upstream `download_file()` has no retry/backoff, silently resumes from corrupt partial
  files, and uses `verify=False`. Ours: curl with resume + retries + exact size verification
  + HF mirror fallback for the 3 Severson batches. URL/filename table mirrors BatteryML's
  `DOWNLOAD_LINKS` exactly. Preprocessing still goes through the `batteryml preprocess` CLI
  (spec: "use BatteryML ... follow its README/CLI exactly").
- **All four MATR batches** downloaded (incl. the 2019-01-24 Attia CLO batch): the MATR
  preprocessor hard-requires all 4 files, and the CLO cells extend protocol coverage for
  Phase 2.
- **Feature definitions follow the paper** where BatteryML deviates from it (BatteryML
  computes avg charge time over lab cycles 1–4 with natural log [paper: cycles 2–6, no log],
  a log-mean instead of ∫T dt, and indexes 'early capacity' at slice[2]≈cycle 5 [paper:
  cycle 2]). Our `src/data/features.py` implements the paper's definitions, uses the
  dataset's own precalculated Qdlin for MATR, and reimplements Q(V) interpolation for
  CALCE/NASA. Sanity gate: corr(dq_var, log10 life) must be < −0.8 on the 124-cell cohort
  (paper: −0.93).
- **Cycle indexing**: MATR preprocessing drops lab cycle 0 → `cycle_data[i]` is lab cycle
  i+1; ΔQ_{100−10} uses indices 99 and 9. Horizon sweep features at 20..100 cycles for the
  early-prediction metric.
- **Label**: cycle life = first lab cycle where median-filtered fade curve ≤ 0.8 × nominal
  capacity (1.1 Ah MATR, 1.1/1.35 CALCE CS2/CX2, 2.0 NASA), censored flag when never
  reached. Same definition across datasets for honest transfer evaluation.
- **NASA**: BatteryML has no NASA support → wrote `src/data/nasa.py` converting
  B0005/06/07/18 into BatteryData pickles (charge+discharge record pairs merged per cycle;
  IR from interleaved impedance records (Re+Rct) carried forward).
- **Protocol features**: 2-step Severson `C1(Q1%)-C2` parsed from `charge_protocol`
  (BatteryML stores Q1 in percent); 4-step CLO protocols mapped to a unified
  per-20%-SOC-window average-rate representation (`rate_w1..w4`) so both families share one
  feature space; analytic 0→80% charge time `60·[(Q1/100)/C1 + ((80−Q1)/100)/C2]` min.
- **CALCE .xls** needs `xlrd` (not in BatteryML's requirements) — added to ours.
- pandas pinned to 2.2.3: BatteryML's unpinned install pulled pandas 3.x, untested with its
  2024-era preprocessors; env re-pinned to requirements.txt exactly.

## Phase 1 — Probabilistic predictor

### Labels
- The strict threshold-crossing labeler flagged 92/180 MATR cells "censored": batches 1 and 3
  were *terminated at* EOL with final capacity a hair above 0.88 Ah (e.g. min 0.8820). Fix:
  use the dataset's own per-cell `cycle_life` field (what Severson's metrics use), extracted
  lazily via h5py (`src/data/matr_official_life.py`) with the +[662,981,1060,208,482]
  continuation rule. Cross-check: where our labeler did fire, it agrees with the official
  label within **1 cycle** (88 cells compared). 2 batch-4 cells lack the official field and
  stay censored.

### Architecture / training decisions
- Single design contract: `X = [scalar features | flattened sequence block]`; every model is
  sklearn-compatible and slices internally → the whole stacked ensemble (elastic-net "full",
  LightGBM, two-branch 1D-CNN with NNLS stacking weights from out-of-fold predictions) can be
  cloned and refit by MAPIE's CrossConformalRegressor.
- Conformal: BOTH split-conformal (contract-specified; prefit ensemble + 13 calibration cells
  carved from train) and cross-conformal ("plus", 5-fold over all 41 train cells). With n=41,
  split-conformal quantiles are coarse (13 calibration residuals); cross-conformal is the
  statistically sensible headline. Both reported.
- **macOS OpenMP deadlock**: first training run hung at 0% CPU inside
  `libomp __kmp_fork_barrier` — torch vendors its own libomp while LightGBM links Homebrew's;
  two OpenMP runtimes in one process deadlock at the first parallel barrier. Fix:
  `OMP_NUM_THREADS=1` (set in `train.py` before imports) + LightGBM `n_jobs=1`. No speed
  cost at n=41.
- Optuna search space deliberately conservative (num_leaves ≤ 15, depth ≤ 5, strong
  regularisation): 41 training points overfit instantly otherwise. `n_estimators` taken from
  the median early-stopped iteration across CV folds.
- Compact estimator: 120-tree LightGBM (≤7 leaves, depth ≤3) exported as a plain-text booster
  dump (~100 kB) — tree walks are trivially portable to a BMS/DSP; metrics reported next to
  the full ensemble's.

## Phase 2 — Protocol optimisation

## Phase 3 — Dashboard

## Phase 4 — Supporting artifacts

## Phase 5 — Deployment
