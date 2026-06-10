"""Feature-engineering unit tests on synthetic BatteryData cells."""

import numpy as np
import pytest

from batteryml import BatteryData, CycleData, CyclingProtocol

from src.data import features as F


def synthetic_cell(n_cycles: int = 120, life: int = 90,
                   nominal: float = 1.1) -> BatteryData:
    """Cell whose capacity fades linearly through 80% SoH at ``life``."""
    cycles = []
    rng = np.random.default_rng(0)
    for i in range(1, n_cycles + 1):
        qmax = nominal * (1.0 - 0.2 * i / life)
        n = 200
        t = np.linspace(0, 3600, n)
        # charge half then discharge half
        current = np.concatenate([np.full(n // 2, 1.1),
                                  np.full(n - n // 2, -4.4)])
        v = np.concatenate([np.linspace(2.0, 3.5, n // 2),
                            np.linspace(3.5, 2.0, n - n // 2)])
        qd = np.concatenate([np.zeros(n // 2),
                             np.linspace(0, qmax, n - n // 2)])
        cycles.append(CycleData(
            cycle_number=i,
            voltage_in_V=v.tolist(),
            current_in_A=(current + rng.normal(0, 0.001, n)).tolist(),
            discharge_capacity_in_Ah=qd.tolist(),
            time_in_s=t.tolist(),
            temperature_in_C=np.full(n, 30.0).tolist(),
            internal_resistance_in_ohm=0.016 + 1e-5 * i,
        ))
    return BatteryData(
        cell_id="TEST_cell",
        cycle_data=cycles,
        nominal_capacity_in_Ah=nominal,
        min_voltage_limit_in_V=2.0,
        max_voltage_limit_in_V=3.5,
        charge_protocol=[
            CyclingProtocol(rate_in_C=5.4, start_soc=0.0, end_soc=40.0),
            CyclingProtocol(rate_in_C=3.6, start_soc=40.0, end_soc=1.0),
        ],
    )


@pytest.fixture(scope="module")
def cell():
    return synthetic_cell()


def test_cycle_life_linear_fade(cell):
    life, censored = F.cycle_life(cell, eol_soh=0.8)
    assert not censored
    assert abs(life - 90) <= 3  # median filter shifts at most a few cycles


def test_cycle_life_censored():
    young = synthetic_cell(n_cycles=50, life=500)
    life, censored = F.cycle_life(young, eol_soh=0.8)
    assert censored and life == 50


def test_features_no_nan(cell):
    f = F.extract_features(cell, horizon=100)
    for key in ["dq_var", "dq_min", "qd_cycle2", "fade_slope_2_h",
                "avg_charge_time_2_6", "ir_min_2_h"]:
        assert np.isfinite(f[key]), key


def test_fade_slope_negative(cell):
    f = F.extract_features(cell, horizon=100)
    assert f["fade_slope_2_h"] < 0


def test_horizon_uses_only_early_cycles(cell):
    """Features at horizon h must not change when later cycles are removed."""
    f_full = F.extract_features(cell, horizon=60)
    truncated = synthetic_cell()
    truncated.cycle_data = truncated.cycle_data[:60]
    f_trunc = F.extract_features(truncated, horizon=60)
    for k in f_full:
        a, b = f_full[k], f_trunc[k]
        if np.isfinite(a) and np.isfinite(b):
            assert a == pytest.approx(b, rel=1e-6), k


def test_protocol_params_two_step(cell):
    p = F.protocol_params(cell)
    assert p["protocol_type"] == "2step"
    assert p["c1"] == 5.4 and p["c2"] == 3.6 and p["q1_pct"] == 40.0
    # charge time: 60*(0.4/5.4 + 0.4/3.6) = 11.11 min
    assert p["charge_time_min"] == pytest.approx(11.111, abs=0.01)
    # SOC-window rates: w1 (0-20%) and w2 (20-40%) fully at 5.4C
    assert p["rate_w1"] == pytest.approx(5.4)
    assert p["rate_w2"] == pytest.approx(5.4)
    assert p["rate_w3"] == pytest.approx(3.6)
    assert p["rate_w4"] == pytest.approx(3.6)


def test_charge_time_formula():
    # one-step 4C: 60*(0.8/4) = 12 min
    assert F.charge_time_to_80(4.0, 80.0, 4.0) == pytest.approx(12.0)
    # Severson's stated range for the 72 protocols is 9–13.3 min
    assert 9.0 < F.charge_time_to_80(8.0, 20.0, 3.6) < 13.4


def test_qdlin_interpolation_monotone(cell):
    q = F.qdlin_from_cycle(cell.cycle_data[50], 2.0, 3.5)
    assert np.isfinite(q).all()
    # discharge capacity decreases as voltage increases (Q(V) inverse)
    assert q[0] >= q[-1]
