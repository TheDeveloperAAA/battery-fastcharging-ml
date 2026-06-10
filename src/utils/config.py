"""Project configuration loader. Single source of truth: config.yaml at repo root."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # TRAINING_MODE env var overrides config (per the build contract)
    mode = os.environ.get("TRAINING_MODE")
    if mode in ("thorough", "fast"):
        cfg["training"]["mode"] = mode
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repo root."""
    return REPO_ROOT / cfg["paths"][key]
