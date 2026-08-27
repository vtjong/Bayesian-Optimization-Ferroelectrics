"""The design box: which conditions the tool can be set to, and how to measure distance in it.

This module holds NO physics. It knows which (voltage, time) pairs are settable and characterized,
how to snap a real-valued condition onto the tool's resolution, and how to map the box onto the
unit square. It does not know what any of those conditions do to a film.

That separation is load-bearing rather than tidiness. The surrogate and the acquisition fit and
search a scalar field over this box; if they could reach the thermal model, the boundary they
report would partly be an echo of that model rather than independent evidence about the film.
Keeping the box here lets both sides depend on it while neither depends on the other, and
``tests/test_layering.py`` enforces that.

The measured grid is read from the table CSV because its axes ARE the box -- the campaign is
deliberately never extrapolated past the conditions the tool was characterized at. The peak
temperatures in that file are loaded here too, since re-parsing the same CSV in two places is how
two readers drift apart, but nothing in this module interprets them; that is ``synthetic.py``'s
job.
"""

from typing import Tuple

import numpy as np

from paths import FLASH_TABLE_CSV

# What the tool can actually be set to. Conditions are snapped to this before they reach a plan,
# so a design can never ask for a voltage or a pulse width the operator cannot dial in.
V_STEP_V = 1.0
T_STEP_MS = 0.1
_T_DECIMALS = int(round(-np.log10(T_STEP_MS)))


def load_flash_table(path=FLASH_TABLE_CSV) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the measured peak-temperature table; returns ``(voltages, times, tmax_table)``.

    The CSV is the single source of truth for this measurement -- it is deliberately NOT mirrored
    as a literal in the source, so the two cannot drift apart. Header cells are ``V=<volts>`` and
    row labels ``t=<milliseconds>``.

    :param path: path to the table CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"measured flash table not found at {path}")
    rows = [line.split(",") for line in path.read_text().strip().splitlines() if line.strip()]
    voltages = np.array([float(c.strip().removeprefix("V=")) for c in rows[0][1:]], float)
    times = np.array([float(r[0].strip().removeprefix("t=")) for r in rows[1:]], float)
    table = np.array([[float(c) for c in r[1:]] for r in rows[1:]], float)
    if table.shape != (times.size, voltages.size):
        raise ValueError(f"table shape {table.shape} does not match axes in {path}")
    return voltages, times, table


GRID_V, GRID_T, GRID_TMAX = load_flash_table()

# Design box = the extent of the measured grid; the model is never extrapolated beyond it.
V_LO, V_HI = float(GRID_V[0]), float(GRID_V[-1])
T_LO, T_HI = float(GRID_T[0]), float(GRID_T[-1])

_LOG_T_LO, _LOG_T_HI = np.log10(T_LO), np.log10(T_HI)


def snap(v: float, t: float) -> Tuple[int, float]:
    """Round a condition to what the tool can actually be set to: whole volts, 0.1 ms.

    :param v: flash voltage.
    :param t: flash time (ms).
    """
    return int(round(v / V_STEP_V) * V_STEP_V), round(round(t / T_STEP_MS) * T_STEP_MS, _T_DECIMALS)


def snap_all(v: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Snap arrays of conditions onto the tool's resolution.

    :param v: flash voltages.
    :param t: flash times (ms).
    """
    v = np.round(np.asarray(v, float) / V_STEP_V) * V_STEP_V
    t = np.round(np.round(np.asarray(t, float) / T_STEP_MS) * T_STEP_MS, _T_DECIMALS)
    return v, t


def is_snapped(v: np.ndarray, t: np.ndarray) -> bool:
    """Whether every condition already sits on the tool's resolution.

    :param v: flash voltages.
    :param t: flash times (ms).
    """
    v = np.asarray(v, float)
    t = np.asarray(t, float)
    return bool(
        np.allclose(v, np.round(v / V_STEP_V) * V_STEP_V)
        and np.allclose(t, np.round(t / T_STEP_MS) * T_STEP_MS)
    )


def in_box(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Whether each condition lies inside the characterized grid.

    :param v: flash voltages.
    :param t: flash times (ms).
    """
    v = np.asarray(v, float)
    t = np.asarray(t, float)
    return (v >= V_LO) & (v <= V_HI) & (t >= T_LO) & (t <= T_HI)


def normalize(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map ``(V, t)`` onto the unit box, with time in log10.

    Any kernel or maximin criterion over this box measures distance, and raw units are unusable:
    voltage spans 210 while flash time spans two decades, so a raw metric would be ~99% voltage.
    Log time additionally makes the boundary geometry closer to isotropic.

    :param v: flash voltages.
    :param t: flash times (ms).
    """
    x = (np.asarray(v, float) - V_LO) / (V_HI - V_LO)
    y = (np.log10(np.asarray(t, float)) - _LOG_T_LO) / (_LOG_T_HI - _LOG_T_LO)
    return np.column_stack([x, y])


def min_separation(v: np.ndarray, t: np.ndarray) -> float:
    """Smallest pairwise distance in the normalized box -- the maximin design criterion.

    :param v: flash voltages.
    :param t: flash times (ms).
    """
    p = normalize(v, t)
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())
