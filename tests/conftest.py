"""Shared pytest fixtures and import path setup.

``src`` is a source directory rather than an installed package, so it is prepended to the import
path here once instead of in every test module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import seed as seed_module  # noqa: E402
from physics.thermal_model import FLASH  # noqa: E402

SEED_BLOCK = "S"  # a seed condition; the never-flashed reference row is "R"


@pytest.fixture(scope="session")
def seed_plan() -> dict:
    """The committed seed, in the dict shape the results sheet is built from.

    Peak temperature is attached for reporting only. It plays no part in choosing the conditions --
    see the module docstring of ``src/seed.py``.
    """
    voltage, time_ms = seed_module.make_seed()
    return {
        "V": voltage,
        "t": time_ms,
        "tmax": FLASH.tmax(voltage, time_ms),
        "block": np.full(voltage.size, SEED_BLOCK),
        "note": np.full(voltage.size, ""),
    }
