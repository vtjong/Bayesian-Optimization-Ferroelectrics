"""Shared pytest fixtures and import path setup.

``src`` is a source directory rather than an installed package, so it is prepended to the import
path here once instead of in every test module.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from discovery.kinetics import build_ensemble  # noqa: E402
from run_flash_plan import make_plan  # noqa: E402

# The pair of flash times the boundary tilt is quoted over: the shortest and longest table rows
# that the campaign actually uses. Kept here so every test reports tilt on the same interval.
LADDER_LO_MS, LADDER_HI_MS = 2.6, 10.1


@pytest.fixture(scope="session")
def ensemble() -> dict:
    """The four boundary hypotheses, built once for the whole session."""
    return build_ensemble()


@pytest.fixture(scope="session")
def seed_plan() -> dict:
    """The committed seed design, generated with the defaults the script ships with."""
    return make_plan(n_core=5, seed=7)


@pytest.fixture(scope="session")
def seed_power(seed_plan) -> dict:
    """Tilt-identification power of the committed design, scored once for the whole session."""
    from run_seed_power import run

    return run(trials=400, seed=0, plan=seed_plan)
