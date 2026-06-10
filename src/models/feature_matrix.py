"""Assemble model inputs from the Phase-0 artifacts.

Column-layout contract used by every model in this project:
``X = [scalar features (SCALAR_COLS order) | flattened sequence block]``
where the sequence block is ``[qdlin_early(1000) | qdlin_late(1000) |
fade(100) | ir(100) | charge_time(100)]`` (float32, NaN allowed — models
impute). The ensemble slices this matrix internally: feature models read the
scalar part, the CNN reads the sequence part.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import (DISCHARGE_FEATURES, FULL_FEATURES,
                               PROTOCOL_FEATURES, VARIANCE_FEATURES)

SEQ_LAYOUT = [("qdlin_early", 1000), ("qdlin_late", 1000),
              ("fade", 100), ("ir", 100), ("charge_time", 100)]
SEQ_DIM = sum(n for _, n in SEQ_LAYOUT)

# Scalar features fed to the GBM (superset incl. protocol parameters);
# baselines use the named Severson subsets.
SCALAR_COLS = sorted(set(VARIANCE_FEATURES + DISCHARGE_FEATURES
                         + FULL_FEATURES
                         + ["dq_mean", "fade_slope_last10",
                            "fade_intercept_last10", "qd_max_minus_2",
                            "ir_cycle2"]
                         + PROTOCOL_FEATURES))

FEATURE_SETS = {
    "variance": VARIANCE_FEATURES,
    "discharge": DISCHARGE_FEATURES,
    "full": FULL_FEATURES,
    "scalar_all": SCALAR_COLS,
}


def load_sequences(seq_dir: Path, cell_ids: list[str]) -> np.ndarray:
    rows = []
    for cid in cell_ids:
        with np.load(seq_dir / f"{cid}.npz") as z:
            parts = []
            for name, size in SEQ_LAYOUT:
                arr = np.asarray(z[name], dtype=np.float32)
                out = np.full(size, np.nan, dtype=np.float32)
                out[:min(size, len(arr))] = arr[:size]
                parts.append(out)
            rows.append(np.concatenate(parts))
    return np.vstack(rows)


def build_matrix(features_csv: Path, seq_dir: Path | None,
                 cells: list[str] | None = None,
                 prefix: str = "MATR_") -> dict:
    """Returns dict with X, y (log10 life), cell ids, and column metadata."""
    df = pd.read_csv(features_csv)
    df["short_id"] = df.cell_id.str.replace(prefix, "", regex=False)
    if cells is not None:
        df = df.set_index("short_id").loc[cells].reset_index()
    X_scalar = df[SCALAR_COLS].to_numpy(dtype=np.float64)
    y = df["log_cycle_life"].to_numpy(dtype=np.float64)
    out = {
        "cell_ids": df.cell_id.tolist(),
        "scalar_cols": list(SCALAR_COLS),
        "y": y,
        "cycle_life": df["cycle_life"].to_numpy(),
        "censored": df["censored"].to_numpy(dtype=bool),
        "df": df,
    }
    if seq_dir is not None:
        X_seq = load_sequences(seq_dir, df.cell_id.tolist())
        out["X"] = np.hstack([X_scalar, X_seq])
        out["n_scalar"] = X_scalar.shape[1]
    else:
        out["X"] = X_scalar
        out["n_scalar"] = X_scalar.shape[1]
    return out


def scalar_view(X: np.ndarray, n_scalar: int) -> np.ndarray:
    return X[:, :n_scalar]


def seq_view(X: np.ndarray, n_scalar: int) -> dict[str, np.ndarray]:
    block = X[:, n_scalar:]
    views, ofs = {}, 0
    for name, size in SEQ_LAYOUT:
        views[name] = block[:, ofs:ofs + size]
        ofs += size
    return views


def feature_indices(names: list[str]) -> list[int]:
    return [SCALAR_COLS.index(n) for n in names]
