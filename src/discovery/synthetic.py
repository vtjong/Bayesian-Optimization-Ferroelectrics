"""Synthetic FLA crystallization data, calibrated to the measured flash-lamp thermal response.

Peak temperature is a MEASURED table Tmax(V, t) (flash voltage x flash time, deg C); a bicubic
spline interpolates between the tabulated values. Every shot's temperature trace is the UNIVERSAL
normalized pulse shape T(tau)/Tmax -- which the measurements show is the same for all flash
conditions -- scaled by that shot's Tmax. A consequence: with a universal shape every thermal
descriptor (dwell, thermal budget, activated budget) is a deterministic function of Tmax, so the
single-pulse crystallization boundary is exactly the Tmax = onset level set. Temperature peaks in t
near t ~ 2.6-3 ms, so the boundary is RE-ENTRANT in (V, t).

The forward model is a ``ThermalModel`` (Protocol); ``TableThermalModel`` is the measured
implementation and ``FLASH`` the default instance. Module-level ``tmax`` / ``_trace`` wrap it, so
swapping in another ``ThermalModel`` does not touch the callers.

The crystallization outcome is governed by a CONTROLLING quantity phi (the "truth"):
    P(crystallize) = sigmoid(k_sharp * z(phi))
where z(.) is the quantile rank over the design so the boundary sits at the median of phi. From
the trace we derive the descriptor charts use: peak Tmax, thermal budget, activated budget.

READOUTS (the thing the power study varies):
  - "binary"      : y ~ Bernoulli(P)                       (pass/fail)
  - "continuous"  : y = clip(P + Normal(0, sigma), 0, 1)   (crystalline-fraction readout)
sigma encodes the metrology quality (permittivity / XRD / Raman / optical).
"""

from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, runtime_checkable

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
# design box over (flash voltage, flash time) = the measured grid extent
V_LO, V_HI = 506.0, 716.0  # volts
T_LO, T_HI = 0.1, 10.1  # ms


@dataclass(frozen=True)
class PulseShape:
    """Universal normalized pulse shape T(tau)/Tmax: sin rise then exp decay to a warm plateau.

    :param plateau: plateau level the trace settles to, as a fraction of Tmax.
    :param tau_decay: exponential decay time constant (ms).
    :param t_rise: rise time to the peak (ms).
    :param duration_ms: total trace length simulated (ms).
    """

    plateau: float = 0.15
    tau_decay: float = 35.0
    t_rise: float = 2.0
    duration_ms: float = 250.0

    def __call__(self, tau: np.ndarray) -> np.ndarray:
        """Normalized temperature T(tau)/Tmax at times tau (ms since the flash)."""
        tau = np.asarray(tau, float)
        s = np.zeros_like(tau)
        r = (tau >= 0) & (tau < self.t_rise)
        s[r] = np.sin(np.pi * tau[r] / (2.0 * self.t_rise))
        d = tau >= self.t_rise
        s[d] = self.plateau + (1.0 - self.plateau) * np.exp(
            -(tau[d] - self.t_rise) / self.tau_decay
        )
        return s


@runtime_checkable
class ThermalModel(Protocol):
    """Forward model: (voltage, time) -> peak temperature and the full temperature trace."""

    def tmax(self, V: np.ndarray, t: np.ndarray) -> np.ndarray: ...

    def trace(self, v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]: ...


class TableThermalModel:
    """Peak temperature from a MEASURED (voltage x time) table, scaled by a universal pulse shape.

    Smoothly interpolates the tabulated peak temperatures (bicubic spline) and applies the single
    universal trace shape to every condition -- so all thermal descriptors collapse to a function
    of Tmax and the crystallization boundary is exactly the Tmax = onset level set.

    :param voltages: measured flash voltages (grid columns).
    :param times: measured flash times (grid rows).
    :param tmax_table: peak temperatures (deg C), shape ``(len(times), len(voltages))``.
    :param shape: the universal normalized pulse shape (defaults to ``PulseShape()``).
    :param t_room: baseline (room) temperature (deg C).
    """

    def __init__(
        self, voltages, times, tmax_table, shape: PulseShape = None, t_room: float = T_ROOM
    ):
        self.voltages = np.asarray(voltages, float)
        self.times = np.asarray(times, float)
        self.tmax_table = np.asarray(tmax_table, float)
        self.shape = shape or PulseShape()
        self.t_room = t_room
        self._spline = RectBivariateSpline(self.times, self.voltages, self.tmax_table, kx=3, ky=3)

    def tmax(self, V: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Peak temperature (deg C) at voltage V, flash time t (smooth spline over the table)."""
        V = np.clip(np.asarray(V, float), self.voltages[0], self.voltages[-1])
        t = np.clip(np.asarray(t, float), self.times[0], self.times[-1])
        return np.asarray(self._spline.ev(t, V))

    def trace(self, v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
        """Temperature trace ``T(tau) = t_room + (Tmax - t_room) * shape(tau)``; tau in ms.

        :param v: flash voltage.
        :param t: flash time.
        :param n: number of trace samples.
        """
        peak = float(self.tmax(v, t))
        tau = np.linspace(0.0, self.shape.duration_ms, n)
        return tau, self.t_room + (peak - self.t_room) * self.shape(tau)


# Default forward model = the measured flash-lamp table. Module-level tmax/_trace wrap it so
# existing callers (charts, functional, picker) keep a stable import.
FLASH = TableThermalModel(FLASH_V, FLASH_T, FLASH_TMAX)


def tmax(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Peak temperature (deg C) at voltage V, flash time t, from the default ``FLASH`` model."""
    return FLASH.tmax(V, t)


def _trace(v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """Temperature trace from the default ``FLASH`` model (see ``TableThermalModel.trace``)."""
    return FLASH.trace(v, t, n)


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
    """n shots uniformly over the (V, t) design box.

    :param n: number of shots to draw.
    :param rng: random generator.
    """
    V = rng.uniform(V_LO, V_HI, n)
    t = rng.uniform(T_LO, T_HI, n)
    return V, t


def sample_readout(p: np.ndarray, readout: str, rng: np.random.Generator) -> np.ndarray:
    """Draw an observed outcome from a per-shot crystallization probability.

    Single source of truth for the readout-noise model shared by every dataset builder:
    "binary" is a Bernoulli pass/fail draw; any other readout is a continuous crystalline
    fraction with that readout's Gaussian metrology noise (READOUT_SIGMA), clipped to [0, 1].

    :param p: per-shot crystallization probability in [0, 1].
    :param readout: metrology key into READOUT_SIGMA.
    :param rng: random generator supplying the measurement noise.
    """
    sigma = READOUT_SIGMA[readout]
    if sigma is None:
        return (rng.uniform(size=len(p)) < p).astype(float)
    return np.clip(p + rng.normal(0.0, sigma, len(p)), 0.0, 1.0)


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

    :param n: number of shots to simulate.
    :param scenario: the planted ground-truth crystallization rule.
    :param readout: metrology key selecting the readout-noise model.
    :param rng: random generator for the design draw and the readout noise.
    """
    V, t = sample_design(n, rng)
    p = _prob_crystallize(V, t, scenario)
    y = sample_readout(p, readout, rng)
    return V, t, y
