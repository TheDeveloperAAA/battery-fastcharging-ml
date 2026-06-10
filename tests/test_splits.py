"""Split integrity: canonical lists match BatteryML source; no leakage."""

import ast
import re
from pathlib import Path

from src.data.splits import (PRIMARY_TEST_CELLS, SECONDARY_TEST_CELLS,
                             TRAIN_CELLS, train_calibration_split)

BATTERYML = Path(__file__).resolve().parents[1] / "third_party" / "BatteryML"
SPLIT_SRC = BATTERYML / "batteryml" / "train_test_split" / "MATR_split.py"


def _lists_from_source(class_name: str) -> tuple[list, list]:
    src = SPLIT_SRC.read_text()
    block = src.split(f"class {class_name}")[1].split("MATRTrainTestSplitter.__init__")[0]
    found = re.findall(r"(train_ids|test_ids) = (\[[^\]]*\])", block, re.S)
    out = {}
    for name, lit in found:
        out[name] = ast.literal_eval(lit)
    return out["train_ids"], out["test_ids"]


def test_train_and_primary_match_batteryml():
    if not SPLIT_SRC.exists():
        import pytest
        pytest.skip("BatteryML source not present")
    train_ids, test_ids = _lists_from_source("MATRPrimaryTestTrainTestSplitter")
    assert TRAIN_CELLS == train_ids
    # BatteryML pops b2c1 at runtime; our list excludes it statically
    test_ids.remove("b2c1")
    assert PRIMARY_TEST_CELLS == test_ids


def test_secondary_matches_batteryml():
    if not SPLIT_SRC.exists():
        import pytest
        pytest.skip("BatteryML source not present")
    train_ids, test_ids = _lists_from_source("MATRSecondaryTestTrainTestSplitter")
    assert TRAIN_CELLS == train_ids
    assert SECONDARY_TEST_CELLS == test_ids


def test_split_sizes_and_disjointness():
    assert len(TRAIN_CELLS) == 41
    assert len(PRIMARY_TEST_CELLS) == 42      # 43 minus outlier b2c1
    assert len(SECONDARY_TEST_CELLS) == 40
    assert "b2c1" not in PRIMARY_TEST_CELLS
    assert not set(TRAIN_CELLS) & set(PRIMARY_TEST_CELLS)
    assert not set(TRAIN_CELLS) & set(SECONDARY_TEST_CELLS)
    assert not set(PRIMARY_TEST_CELLS) & set(SECONDARY_TEST_CELLS)


def test_calibration_split_deterministic_and_disjoint():
    fit1, cal1 = train_calibration_split(seed=42, calibration_fraction=0.35)
    fit2, cal2 = train_calibration_split(seed=42, calibration_fraction=0.35)
    assert fit1 == fit2 and cal1 == cal2
    assert not set(fit1) & set(cal1)
    assert sorted(fit1 + cal1) == sorted(TRAIN_CELLS)
    # calibration cells never overlap any test set
    assert not set(cal1) & set(PRIMARY_TEST_CELLS)
    assert not set(cal1) & set(SECONDARY_TEST_CELLS)
