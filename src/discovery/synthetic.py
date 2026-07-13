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
from scipy.interpolate import RectBivariateSpline

# --- MEASURED flash-lamp peak-temperature table (deg C) -------------------------------
# Rows = flash time (ms), columns = flash voltage (V). See data/flash_temp_table.csv.
T_ROOM = 25.0  # deg C
KB_EV = 8.617e-5  # Boltzmann constant, eV/K
T_ONSET_C = 380.0  # crystallization onset temperature (deg C)

FLASH_V = np.array([506.0, 548.0, 590.0, 632.0, 674.0, 716.0])  # flash voltage (V)
FLASH_T = np.array([0.1, 2.6, 5.1, 7.6, 10.1])  # flash time (ms)
FLASH_TMAX = np.array(
    [  # rows = t, cols = V
        [81.6, 97.26, 118.9, 141.2, 165.5, 189.2],
        [286.3, 336.71, 387.6, 444.8, 504.77, 563.335],
        [270.4, 314.5, 360.45, 411.28, 461.252, 514.998],
        [248.4, 288.1, 328.3, 370.8, 414.97, 438.1],
        [230.4, 267.1, 302.206, 340.835, 380.9, 421.551],
    ]
)
_SPLINE = RectBivariateSpline(FLASH_T, FLASH_V, FLASH_TMAX, kx=3, ky=3)

# universal normalized pulse shape T(tau)/Tmax (same for all conditions): rise then exp decay
PLATEAU, TAU_DECAY, T_RISE, TRACE_MS = 0.15, 35.0, 2.0, 250.0

# design box over (flash voltage, flash time) = the measured grid extent
V_LO, V_HI = 506.0, 716.0  # volts
T_LO, T_HI = 0.1, 10.1  # ms


def tmax(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Peak temperature (deg C) at voltage v, flash time t (smooth spline over the table)."""
    v = np.clip(np.asarray(v, float), FLASH_V[0], FLASH_V[-1])
    t = np.clip(np.asarray(t, float), FLASH_T[0], FLASH_T[-1])
    return np.asarray(_SPLINE.ev(t, v))


def _shape(tau: np.ndarray) -> np.ndarray:
    """Universal normalized temperature T(tau)/Tmax: sin rise over T_RISE, exp decay to PLATEAU."""
    tau = np.asarray(tau, float)
    s = np.zeros_like(tau)
    r = (tau >= 0) & (tau < T_RISE)
    s[r] = np.sin(np.pi * tau[r] / (2.0 * T_RISE))
    d = tau >= T_RISE
    s[d] = PLATEAU + (1.0 - PLATEAU) * np.exp(-(tau[d] - T_RISE) / TAU_DECAY)
    return s


def _trace(v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """Trace T(tau) = T_room + (Tmax - T_room) * universal_shape(tau); tau in ms since flash."""
    peak = float(tmax(v, t))
    tau = np.linspace(0.0, TRACE_MS, n)
    return tau, T_ROOM + (peak - T_ROOM) * _shape(tau)


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
    phi: Callable  # phi(V, t) -> controlling-quantity array, OR None for two-mechanism
    ea_true: float = None  # the planted activation energy, if applicable
    two_mechanism: bool = False


SCENARIOS = {
    # A: single order parameter, Ea ON the chart grid
    "A": Scenario("A: single phi (Ea=2.5 on grid)", lambda V, t: phi_tbac(V, t, 2.5), ea_true=2.5),
    # B: single order parameter, Ea BETWEEN grid points (needs the warp for precision)
    "B": Scenario(
        "B: single phi (Ea=2.25 off grid)", lambda V, t: phi_tbac(V, t, 2.25), ea_true=2.25
    ),
    # C: two mechanisms -- crystallize only if hot enough (peak T) AND long enough
    # (dwell). Tmax and dwell are rank-decorrelated, so NO single chart collapses it.
    "C": Scenario("C: two-mechanism (Tmax AND dwell)", None, two_mechanism=True),
}

# readout noise assumptions (crystalline-fraction units; LABELLED ASSUMPTIONS, calibrate later)
READOUT_SIGMA = {
    "binary": None,  # Bernoulli pass/fail, no continuous noise
    "xrd": 0.03,  # direct structural, low noise
    "raman": 0.07,  # phase-sensitive, moderate
    "optical": 0.12,  # ellipsometry/reflectance proxy, higher / more indirect
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
