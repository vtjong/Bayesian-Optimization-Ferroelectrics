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

import seed  # noqa: E402
from physics.thermal_model import tmax_envelope  # noqa: E402

#: Block label for a fired seed condition. The never-flashed reference row
#: carries "R", which the results loader fills in for any blank.
SEED_BLOCK = "S"


@pytest.fixture(scope="session")
def seed_plan() -> dict[str, np.ndarray]:
    """Return the committed seed in the shape the results sheet is built from.

    Simulated peak temperature is attached as a band across the available
    thermal models, for reporting only. No thermal quantity takes part in
    choosing the conditions.

    :return: Plan columns keyed by name.
    """
    voltage_v, pulse_width_ms = seed.make_seed()
    tmax_low_c, tmax_high_c = tmax_envelope(voltage_v, pulse_width_ms)

    return {
        "voltage_v": voltage_v,
        "pulse_width_ms": pulse_width_ms,
        "tmax_low_c": tmax_low_c,
        "tmax_high_c": tmax_high_c,
        "block": np.full(voltage_v.size, SEED_BLOCK),
        "note": np.full(voltage_v.size, ""),
    }
