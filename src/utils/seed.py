"""Global seeding for full reproducibility (Python, NumPy, PyTorch).

LightGBM is seeded per-model via ``random_state`` (plus ``deterministic=True``
where it matters); Optuna via ``TPESampler(seed=...)`` with ``n_jobs=1``.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, torch_too: bool = True) -> None:
    """Seed Python/NumPy (and torch when requested).

    ``torch_too=False`` exists because initialising torch alongside
    LightGBM + Accelerate-backed scipy/sklearn in the same process segfaults
    on this macOS/arm64 stack (three native runtimes fighting); processes
    that never use the neural model (e.g. the protocol optimiser) skip the
    torch import entirely.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if not torch_too:
        return
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
