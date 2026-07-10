"""Synthetic FLA crystallization data for the chart-comparison power study.

Analytic forward model used for the synthetic study:

    Tmax(V,t) = T_room + (V - V0) * t / (a + b t^2),   V0=28.61, a=1.278, b=0.184

so Tmax peaks in t at t = sqrt(a/b) ~ 2.6 ms (the boundary is RE-ENTRANT in (V,t):
two pulse widths reach the same Tmax). Each shot's temperature trace is asymmetric:
a half-sine rise to the peak at s = t/2, then exponential cooling with tau = 4 ms.

From the trace we compute the descriptors the charts use: peak temperature Tmax, thermal
budget TB = integral of (T - T_room) dt, and the activated thermal budget
TBac(Ea) = integral of exp(-Ea / k T) dt.

The crystallization outcome is governed by a CONTROLLING quantity phi (the "truth"):
    P(crystallize) = sigmoid(k_sharp * z(phi))
where z(.) is the z-score over the design so the boundary sits at the median of phi.

READOUTS (the thing the power study varies):
  - "binary"      : y ~ Bernoulli(P)                       (pass/fail)
  - "continuous"  : y = clip(P + Normal(0, sigma), 0, 1)   (XRD / optical crystalline fraction)
The continuous readout's sigma encodes the metrology quality (XRD low, optical proxies higher).
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

# --- constants of the peak-temperature model (stand-in units for the method testbed) ---
# The REAL thermal anchor, fit to flash IR data, is Tmax-25 = 55.6 * V^2.30 * t^0.13
# (R2=0.83, MAE=54 C; V in kV, t in ms; see src/run_calibration.py). This synthetic keeps a
# self-consistent re-entrant stand-in so the (V,t) testbed geometry is fixed across the study.
T_ROOM = 25.0      # deg C
V0 = 28.61         # V
A_FIT = 1.278      # ms
B_FIT = 0.184      # 1/ms
TAU_COOL = 4.0     # ms, cooling time constant
KB_EV = 8.617e-5   # Boltzmann constant, eV/K
T_ONSET_C = 390.0  # measured crystallization onset (flash T50=388 C, RTA 357 C); run_calibration.py

# design box over (voltage, pulse width)
V_LO, V_HI = 350.0, 750.0   # volts
T_LO, T_HI = 0.1, 10.0      # ms


def tmax(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Peak temperature (deg C) for voltage v and pulse width t."""
    return T_ROOM + (v - V0) * t / (A_FIT + B_FIT * t ** 2)


def _trace(v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """Asymmetric trace T(s): half-sine rise to peak at s=t/2, then exp cooling."""
    peak = tmax(np.asarray(v), np.asarray(t))
    s = np.linspace(0.0, t + 6.0 * TAU_COOL, n)
    rise = s <= t / 2.0
    T = np.empty_like(s)
    # half-sine from T_ROOM at s=0 to peak at s=t/2
    T[rise] = T_ROOM + (peak - T_ROOM) * np.sin(np.pi * s[rise] / t)
    # exponential cooling back toward T_ROOM after the peak
    T[~rise] = T_ROOM + (peak - T_ROOM) * np.exp(-(s[~rise] - t / 2.0) / TAU_COOL)
    return s, T


# --- controlling quantities (the planted "truth") ------------------------------------

def phi_tbac(V, t, ea):
    """phi = activated thermal budget at activation energy ea (vectorized over shots)."""
    out = np.empty(len(V))
    for i, (vi, ti) in enumerate(zip(V, t)):
        s, T = _trace(float(vi), float(ti))
        out[i] = np.trapezoid(np.exp(-ea / (KB_EV * (T + 273.15))), s)
    return out


def phi_tmax(V, t):
    return tmax(V, t)


def phi_dwell(V, t, t_star=600.0):
    """phi = dwell: time the trace spends above T* (deg C). Near-peak threshold (600C)
    makes this rank-DECORRELATED from Tmax (|corr|~0.32, vs ~0.82 at 450C): it measures
    how LONG near the peak, not how HOT. Pairing the two gives a genuine 2-coord boundary."""
    out = np.empty(len(V))
    for i, (vi, ti) in enumerate(zip(V, t)):
        s, T = _trace(float(vi), float(ti))
        out[i] = np.trapezoid((T > t_star).astype(float), s)
    return out


@dataclass(frozen=True)
class Scenario:
    """A planted ground-truth crystallization rule."""
    name: str
    phi: Callable          # phi(V, t) -> controlling-quantity array, OR None for two-mechanism
    ea_true: float = None  # the planted activation energy, if applicable
    two_mechanism: bool = False


SCENARIOS = {
    # A: single order parameter, Ea ON the chart grid
    "A": Scenario("A: single phi (Ea=2.5 on grid)",
                  lambda V, t: phi_tbac(V, t, 2.5), ea_true=2.5),
    # B: single order parameter, Ea BETWEEN grid points (needs the warp for precision)
    "B": Scenario("B: single phi (Ea=2.25 off grid)",
                  lambda V, t: phi_tbac(V, t, 2.25), ea_true=2.25),
    # C: two mechanisms -- crystallize only if hot enough (peak T) AND long enough
    # (dwell). Tmax and dwell are rank-decorrelated, so NO single chart collapses it.
    "C": Scenario("C: two-mechanism (Tmax AND dwell)", None, two_mechanism=True),
}

# readout noise assumptions (crystalline-fraction units; LABELLED ASSUMPTIONS, calibrate later)
READOUT_SIGMA = {
    "binary": None,        # Bernoulli pass/fail, no continuous noise
    "xrd": 0.03,           # direct structural, low noise
    "raman": 0.07,         # phase-sensitive, moderate
    "optical": 0.12,       # ellipsometry/reflectance proxy, higher / more indirect
    "permittivity": 0.06,  # dielectric-constant proxy; ~6% within-sample repeatability
                           # from large-signal P-V loops (see src/run_calibration.py)
}


def sample_design(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """n shots uniformly over the (V, t) design box."""
    V = rng.uniform(V_LO, V_HI, n)
    t = rng.uniform(T_LO, T_HI, n)
    return V, t


def _prob_crystallize(V, t, scenario: Scenario, k_sharp: float = 40.0) -> np.ndarray:
    """P(crystallize) for each shot under the scenario's controlling quantity.

    Works in QUANTILE-RANK space: TBac spans many orders of magnitude, so a
    linear-space threshold collapses to "Tmax > cutoff". Ranking makes the boundary sit at
    the median rank of the controlling quantity and the transition sharpness well-defined
    (k_sharp=40 -> 5-95% transition spans ~15% of the rank range).
    """
    if scenario.two_mechanism:
        # need BOTH high peak T (nucleation) AND long dwell (growth); neither alone
        # suffices -> the boundary needs 2 coordinates, so no 1-D chart collapses it
        r = np.minimum(_rank(phi_tmax(V, t)), _rank(phi_dwell(V, t)))
        return 1.0 / (1.0 + np.exp(-k_sharp * (r - 0.5)))
    r = _rank(scenario.phi(V, t))
    return 1.0 / (1.0 + np.exp(-k_sharp * (r - 0.5)))


def _rank(x: np.ndarray) -> np.ndarray:
    """Quantile rank of each entry in [0, 1] (robust to the scale of x)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1 + 1e-12)


def make_dataset(n: int, scenario: Scenario, readout: str, rng: np.random.Generator):
    """Return (V, t, y) for n shots under a scenario and readout.

    y is binary {0,1} for readout='binary', else a continuous crystalline fraction in [0,1].
    """
    V, t = sample_design(n, rng)
    p = _prob_crystallize(V, t, scenario)
    sigma = READOUT_SIGMA[readout]
    if sigma is None:
        y = (rng.uniform(size=n) < p).astype(float)
    else:
        y = np.clip(p + rng.normal(0.0, sigma, n), 0.0, 1.0)
    return V, t, y
