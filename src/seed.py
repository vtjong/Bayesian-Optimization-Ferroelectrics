"""Seed conditions for the flash-anneal campaign.

A Latin hypercube over the instrument's full settable range, and nothing else. No bound in this
module is derived from a thermal model: the only inputs are the voltage and pulse-width limits of
the tool and its setting resolution.

That is deliberate. The previous seed carried two bounds that came from the thermal simulation
rather than the instrument -- a 350 C lower bound on simulated peak temperature, and a 2.6 ms
minimum pulse width inherited from the interval where the temperature table interpolates poorly.
Each was a claim about where the answer is not, made before looking. Between them they removed the
region the transition turned out to occupy, and all ten specimens came back on one side. Measured
against a full-range draw, the pulse-width restriction cost a factor of eleven in the lowest
delivered fluence the design could reach and the temperature floor cost a further 13%; which
coordinate the hypercube was stratified in changed the result by about 2%.

TIME IS DRAWN LOG-UNIFORMLY. The window spans two decades, so a linear draw places about 90% of
conditions above 1 ms and leaves the short-pulse region -- the only part of the box that reaches low
fluence -- effectively unsampled. Measured on a ten-point draw: linear scaling puts one condition
below 1 ms, log scaling puts four.

THE DRAW IS SELECTED, NOT RE-ROLLED. A single hypercube realization can leave two conditions close
together. The realization used is the most separated of a fixed number of draws, under a fixed
criterion, both declared here before drawing. That makes the selection a rule rather than a
preference. What must never happen is re-running with a new seed because a draw looks unappealing;
that is the same error as the floor, applied to whole designs instead of one bound.

Delivered fluence and simulated peak temperature are computed elsewhere, AFTER the draw, to describe
what it covers. Neither influences it. ``tests/test_seed_independence.py`` asserts that this module
imports nothing but numpy and scipy.
"""

from typing import Tuple

import numpy as np
from scipy.stats import qmc

V_LO, V_HI = 506.0, 716.0  # flash voltage limits (V)
T_LO, T_HI = 0.1, 10.1  # flash time limits (ms)
V_STEP = 1.0  # tool setting resolution, volts
T_STEP = 0.1  # tool setting resolution, milliseconds

SEED_SIZE = 10
RNG_SEED = 20260831  # fixed before the draw; changing it requires a new record
MAXIMIN_DRAWS = 400  # realizations scored; the most separated is kept


def to_conditions(unit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map unit-square points onto settable ``(voltage, pulse width)``.

    Voltage scales linearly. Time scales in log10 and is then exponentiated, so both decades of the
    window receive equal sampling density. Both are snapped to what the tool can be set to, so a
    design can never ask for a value the operator cannot dial in.

    :param unit: points in the unit square, shape ``(n, 2)``.
    """
    scaled = qmc.scale(unit, [V_LO, np.log10(T_LO)], [V_HI, np.log10(T_HI)])
    voltage = np.round(scaled[:, 0] / V_STEP) * V_STEP
    time_ms = np.round(10.0**scaled[:, 1] / T_STEP) * T_STEP
    return voltage, np.clip(np.round(time_ms, 1), T_LO, T_HI)


def separation(voltage: np.ndarray, time_ms: np.ndarray) -> float:
    """Smallest pairwise distance in normalized ``(V, log t)`` -- the maximin criterion.

    Distances are taken in the coordinates the surrogate itself uses, so a design that looks well
    spread by this measure is well spread to the kernel. Raw units would be unusable: voltage spans
    210 while time spans two decades, so a raw metric would be almost entirely voltage.

    :param voltage: flash voltages.
    :param time_ms: flash times (ms).
    """
    x = np.column_stack(
        [
            (voltage - V_LO) / (V_HI - V_LO),
            (np.log10(time_ms) - np.log10(T_LO)) / (np.log10(T_HI) - np.log10(T_LO)),
        ]
    )
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def make_seed(n: int = SEED_SIZE, rng_seed: int = RNG_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Draw the seed, returned sorted by pulse width.

    Realizations that snap two conditions onto one another are skipped -- the only rejection this
    module performs. There is no floor, no band and no informative range.

    :param n: number of conditions.
    :param rng_seed: base RNG seed; realization ``k`` uses ``rng_seed + k``.
    """
    best = None
    for k in range(MAXIMIN_DRAWS):
        voltage, time_ms = to_conditions(qmc.LatinHypercube(d=2, seed=rng_seed + k).random(n))
        if len(set(zip(voltage.tolist(), time_ms.tolist()))) < n:
            continue
        score = separation(voltage, time_ms)
        if best is None or score > best[0]:
            best = (score, voltage, time_ms)
    if best is None:
        raise RuntimeError(f"no realization of {n} distinct settable conditions in {MAXIMIN_DRAWS}")
    _, voltage, time_ms = best
    order = np.argsort(time_ms)
    return voltage[order], time_ms[order]
