"""Global seeding for full reproducibility (Python, NumPy, PyTorch).

LightGBM is seeded per-model via ``random_state`` (plus ``deterministic=True``
where it matters); Optuna via ``TPESampler(seed=...)`` with ``n_jobs=1``.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
