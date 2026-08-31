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
from design_space import T_HI, T_LO  # noqa: E402
from physics.constants import T_TRANSITION_REF_C  # noqa: E402
from physics.kinetics import build_ensemble  # noqa: E402
from physics.thermal_model import FLASH  # noqa: E402

# The dwell interval the tilt is quoted over. Taken from the instrument's full settable range
# rather than from a design constant: the previous version read it from the seed generator, which
# had restricted it to 2.6 ms, so the tests silently asserted a span the tool exceeds.
DWELL_LO_MS, DWELL_HI_MS = T_LO, T_HI

SEED_BLOCK = "S"  # a seed condition; the never-flashed reference row is "R"


def _reachable_window(target_c: float) -> tuple:
    """Shortest and longest pulse width at which ``target_c`` is reachable at some voltage.

    Computed rather than hard-coded. The reachable ceiling falls with pulse width, so a fixed pair
    would silently become unreachable if the reference temperature or the table changed -- and a
    test that asks for a condition AT an unreachable temperature gets a NaN voltage rather than a
    failure.

    :param target_c: peak temperature the condition must reach.
    """
    times = np.geomspace(T_LO, T_HI, 400)
    ok = [float(t) for t in times if FLASH.tmax_range(float(t))[1] > target_c]
    return min(ok), max(ok)


# Pulse widths at which the reference transition temperature can actually be produced. Roughly
# 1.3-7.1 ms: shorter pulses cannot reach it at any voltage, and neither can the longest ones.
ISO_LO_MS, ISO_HI_MS = _reachable_window(T_TRANSITION_REF_C)

# The closed-form tilt theta*ln(t2/t1) is a LOCAL linearization -- theta = kB*T^2/Ea, so it holds
# only while the boundary moves little compared with T itself. Across the full 101x dwell range the
# boundary moves ~89 C and theta varies ~28%, and the identity misses by 6.6 C. It is therefore
# checked over the reachable window, a 5.6x ratio, where the linearization is sound.
LINEAR_LO_MS, LINEAR_HI_MS = ISO_LO_MS, ISO_HI_MS


@pytest.fixture(scope="session")
def ensemble() -> dict:
    """The boundary hypotheses, built once for the whole session."""
    return build_ensemble()


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
