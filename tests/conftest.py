"""Shared pytest fixtures and import path setup.

``src`` is a source directory rather than an installed package, so it is
prepended to the import path here once instead of in every test module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physics.thermal_model import tmax_envelope  # noqa: E402

#: Block label for a fired condition. The never-flashed reference row carries
#: "R", which the results loader fills in for any blank.
DESIGN_BLOCK = "S"

#: Conditions the plan and results fixtures are built from. These are fixture
#: data, not a design: they span the settable box on instrument resolution so
#: the assertions exercise real coordinates, and they are deliberately
#: independent of whatever conditions the campaign actually fires.
FIXTURE_VOLTAGE_V = np.array([510.0, 561.0, 611.0, 662.0, 716.0])
FIXTURE_PULSE_WIDTH_MS = np.array([0.1, 0.4, 1.2, 3.9, 8.5])


@pytest.fixture(scope="session")
def design_plan() -> dict[str, np.ndarray]:
    """Return fixture conditions in the shape the results sheet is built from.

    Simulated peak temperature is attached as a band across the available
    thermal models, for reporting only. No thermal quantity takes part in
    choosing the conditions.

    :return: Plan columns keyed by name.
    """
    tmax_low_c, tmax_high_c = tmax_envelope(FIXTURE_VOLTAGE_V, FIXTURE_PULSE_WIDTH_MS)

    return {
        "voltage_v": FIXTURE_VOLTAGE_V,
        "pulse_width_ms": FIXTURE_PULSE_WIDTH_MS,
        "tmax_low_c": tmax_low_c,
        "tmax_high_c": tmax_high_c,
        "block": np.full(FIXTURE_VOLTAGE_V.size, DESIGN_BLOCK),
        "note": np.full(FIXTURE_VOLTAGE_V.size, ""),
    }
