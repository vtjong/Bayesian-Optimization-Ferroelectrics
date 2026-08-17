"""Ground truths the seed design did NOT come from, for stress-testing it.

The seed's blocks are placed using a five-member ensemble, and block B is restricted further to
where that ensemble predicts a fraction between 0.03 and 0.97. There is a circularity in that: the
design decides where to look using the same models it is meant to test, so if every member is wrong
in the same way, the batch can systematically avoid the region that would reveal it.

Scoring the design against those five members cannot detect that. This module supplies truths that
sit OUTSIDE the family in ways the family cannot represent -- a response that falls at high
temperature, a second crystallization branch at long dwell, a boundary that curves in log t, a
direct dependence on voltage. Some are physically motivated (the repo's own RTA series is
non-monotone in temperature); others exist only to be hostile.

Every truth returns a fraction in [0, 1] on the SAME measured Tmax table, so a design is never
penalised for a thermal-model disagreement. Only the crystallization response differs.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .constants import (
    CELSIUS_TO_KELVIN,
    EA_EV,
    KB_EV,
    T_REF_MS,
    T_TRANSITION_REF_C,
    T_TRANSITION_SIGMA_C,
)
from .kinetics import logistic_sharpness
from .synthetic import FLASH

# Multipliers defining how hostile each variant is. They are deliberately large: a stress test that
# only perturbs within the prior tests nothing the minimax placement has not already handled.
SHARPNESS_BROAD = 1.0 / 3.0
SHARPNESS_SHARP = 3.0
TILT_LARGE_MULTIPLIER = 3.0
CURVATURE_C_PER_LOG2 = 12.0  # boundary curvature in log-dwell, deg C
ROLLOVER_ABOVE_C = 40.0  # response starts falling this far above the transition
ROLLOVER_DEPTH = 0.55  # how far it falls, as a fraction
SECOND_BRANCH_T_C = 385.0  # a low-temperature crystallization branch...
SECOND_BRANCH_MIN_MS = 7.0  # ...that only opens at long dwell
VOLTAGE_CHANNEL_PER_100V = 0.35  # non-thermal dependence on voltage


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _theta(t_c: float) -> float:
    """Arrhenius temperature scale kB*T^2/Ea at ``t_c`` (deg C)."""
    return KB_EV * (t_c + CELSIUS_TO_KELVIN) ** 2 / EA_EV


@dataclass(frozen=True)
class Truth:
    """A ground-truth response surface over (V, t).

    :param name: short identifier used in reports.
    :param fn: maps ``(V, t)`` to crystalline fraction in [0, 1].
    :param outside_family: whether the seed's five-member ensemble can represent this at all.
    :param why: one line on what this variant is probing.
    """

    name: str
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    outside_family: bool
    why: str

    def __call__(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.clip(self.fn(np.asarray(v, float), np.asarray(t, float)), 0.0, 1.0)


def _tilted(t0: float, sharp: float, tilt: float):
    """The family the design assumes: a logistic in Tmax, shifted linearly in log dwell."""

    def f(v, t):
        tm = FLASH.tmax(v, t)
        return _sigmoid(sharp * (tm - t0 + tilt * np.log(t / T_REF_MS)))

    return f


def build_truths() -> Dict[str, Truth]:
    """The adversarial suite, keyed by name.

    The first few are inside the design's own family and vary only its parameters; they test
    robustness to being wrong about WHERE. The rest are outside the family and test robustness to
    being wrong about WHAT SHAPE.
    """
    ref = T_TRANSITION_REF_C
    s = logistic_sharpness(ref)
    tilt = _theta(ref)
    out: Dict[str, Truth] = {}

    def add(name, fn, outside, why):
        out[name] = Truth(name=name, fn=fn, outside_family=outside, why=why)

    # --- inside the family: the transition is somewhere else, or has a different shape ---
    add("baseline", _tilted(ref, s, tilt), False, "the design's own central hypothesis")
    add(
        "cold_transition",
        _tilted(ref - T_TRANSITION_SIGMA_C, s, tilt),
        False,
        "transition one sigma BELOW the bracket midpoint",
    )
    add(
        "hot_transition",
        _tilted(ref + T_TRANSITION_SIGMA_C, s, tilt),
        False,
        "transition one sigma ABOVE the bracket midpoint",
    )
    add("no_tilt", _tilted(ref, s, 0.0), False, "a pure peak-temperature threshold")
    add(
        "large_tilt",
        _tilted(ref, s, TILT_LARGE_MULTIPLIER * tilt),
        False,
        "dwell matters far more than any ensemble member allows",
    )
    add(
        "broad",
        _tilted(ref, s * SHARPNESS_BROAD, tilt),
        False,
        "transition three times wider than assumed",
    )
    add(
        "sharp",
        _tilted(ref, s * SHARPNESS_SHARP, tilt),
        False,
        "transition three times sharper than assumed",
    )

    # --- outside the family: shapes no member can express ---
    def curved(v, t):
        u = np.log(t / T_REF_MS)
        tm = FLASH.tmax(v, t)
        return _sigmoid(s * (tm - ref + tilt * u - CURVATURE_C_PER_LOG2 * u**2))

    add(
        "curved_in_dwell",
        curved,
        True,
        "boundary curves in log t; every member is linear in it",
    )

    def rollover(v, t):
        """Rises through the transition, then FALLS -- as the repo's own RTA 2Pr series does."""
        tm = FLASH.tmax(v, t)
        up = _sigmoid(s * (tm - ref + tilt * np.log(t / T_REF_MS)))
        down = _sigmoid(s * (tm - (ref + ROLLOVER_ABOVE_C)))
        return up * (1.0 - ROLLOVER_DEPTH * down)

    add(
        "rollover",
        rollover,
        True,
        "response peaks then declines; the readout is not monotone in thermal severity",
    )

    def second_branch(v, t):
        """A cold branch that only opens at long dwell -- the topology the seed could miss."""
        tm = FLASH.tmax(v, t)
        main = _sigmoid(s * (tm - ref + tilt * np.log(t / T_REF_MS)))
        gate = _sigmoid(2.0 * (t - SECOND_BRANCH_MIN_MS))
        cold = _sigmoid(s * (tm - SECOND_BRANCH_T_C)) * gate
        return np.maximum(main, cold)

    add(
        "long_dwell_branch",
        second_branch,
        True,
        "crystallizes at 385 C if held beyond 7 ms; a region the ensemble says is dead",
    )

    def voltage_channel(v, t):
        """Part of the response is driven by voltage directly, not through temperature."""
        tm = FLASH.tmax(v, t)
        extra = VOLTAGE_CHANNEL_PER_100V * (v - 611.0) / 100.0
        return _sigmoid(s * (tm - ref + tilt * np.log(t / T_REF_MS)) + extra * 4.0)

    add(
        "voltage_channel",
        voltage_channel,
        True,
        "a non-thermal dependence on voltage; breaks the design's central assumption",
    )

    def step(v, t):
        """A hard threshold with no graded region at all."""
        tm = FLASH.tmax(v, t)
        return (tm + tilt * np.log(t / T_REF_MS) > ref).astype(float)

    add("hard_step", step, True, "no graded transition; the readout is effectively binary")
    return out
