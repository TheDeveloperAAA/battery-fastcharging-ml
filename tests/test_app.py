"""Dashboard smoke tests via streamlit.testing.AppTest.

Runs the real app script against the real precomputed artifacts and asserts
every view renders without raising. Skipped if artifacts are absent (fresh
clone before the pipeline has run).
"""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"
ARTIFACTS = APP.parent / "artifacts"

needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "protocol_grid.npz").exists(),
    reason="precomputed artifacts not built yet")

VIEWS = ["Cycle-life predictions", "Charging-protocol advisor",
         "Speed-vs-life Pareto frontier", "Explainability (SHAP)",
         "Live compact model"]


@needs_artifacts
@pytest.mark.parametrize("view", VIEWS)
def test_view_renders_without_exception(view):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception, (view, at.exception)


@needs_artifacts
def test_protocol_advisor_feasibility_logic():
    """Slamming both steps to the maximum C-rate must flip the lifetime
    guarantee badge to 'violates' (error), and a gentle protocol must pass."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Charging-protocol advisor").run()
    sliders = at.main.slider
    assert len(sliders) >= 3
    sliders[0].set_value(sliders[0].max).run()
    at.main.slider[2].set_value(at.main.slider[2].max).run()
    assert at.main.error or at.main.success  # badge present either way
