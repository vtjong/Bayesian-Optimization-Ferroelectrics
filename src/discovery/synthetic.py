"""Flash-lamp thermal model for FLA HZO: (voltage, time) -> peak temperature and trace.

Peak temperature is a MEASURED table Tmax(V, t) read from ``data/flash_temp_table.csv``, bicubic
spline interpolated. Temperature peaks in t near 2.6-3 ms, so Tmax is RE-ENTRANT in (V, t) and a
Tmax level set folds -- it cannot be written t = f(V).

The temperature TRACE is not measured. It is asserted by a normalized pulse shape, and because
activated kinetics integrate the trace, that assertion decides the crystallization boundary's
geometry. ``SHAPES`` holds the candidate cooling laws; see ``PulseShape``.

The forward model is a ``ThermalModel`` (Protocol); ``TableThermalModel`` is the table-backed
implementation and ``FLASH`` the default instance, so swapping the model does not touch callers.
All numeric parameters come from ``constants``.
"""

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .constants import (
    BISECT_ITERS,
    FLASH_TABLE_CSV,
    KB_EV,
    LEGACY_DURATION_MS,
    LEGACY_PLATEAU_FRAC,
    LEGACY_RISE_MS,
    LEGACY_TAU_DECAY_MS,
    RAMP_A_FAST,
    RAMP_A_SLOW,
    RAMP_PLATEAU_FRAC,
    RAMP_TAU_FAST_MS,
    RAMP_TAU_SLOW_MS,
    T_ONSET_C,
    T_ROOM_C,
    TRACE_DURATION_MS,
)

T_ROOM = T_ROOM_C  # backwards-compatible alias for existing callers

__all__ = [
    "FLASH",
    "FLASH_ISOT",
    "FLASH_T",
    "FLASH_TMAX",
    "FLASH_V",
    "SHAPES",
    "DiffusionPulse",
    "FrozenPulse",
    "PulseShape",
    "RampPulse",
    "RectangularPulse",
    "TableThermalModel",
    "ThermalModel",
    "T_HI",
    "T_LO",
    "T_ONSET_C",
    "T_ROOM",
    "V_HI",
    "V_LO",
    "KB_EV",
    "thermal_model",
    "tmax",
]


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
    voltages = np.array([float(c.strip().lstrip("V=")) for c in rows[0][1:]], float)
    times = np.array([float(r[0].strip().lstrip("t=")) for r in rows[1:]], float)
    table = np.array([[float(c) for c in r[1:]] for r in rows[1:]], float)
    if table.shape != (times.size, voltages.size):
        raise ValueError(f"table shape {table.shape} does not match axes in {path}")
    return voltages, times, table


FLASH_V, FLASH_T, FLASH_TMAX = load_flash_table()

# Design box = the extent of the measured grid; the model is never extrapolated beyond it.
V_LO, V_HI = float(FLASH_V[0]), float(FLASH_V[-1])
T_LO, T_HI = float(FLASH_T[0]), float(FLASH_T[-1])


@runtime_checkable
class PulseShape(Protocol):
    """Normalized pulse shape g(tau; t_pulse) in [0, 1], peaking at 1.

    The trace is ``T(tau) = T_room + (Tmax - T_room) * g(tau; t_pulse)``. Implementations differ in
    ONE respect that decides the whole crystallization-boundary geometry: whether the commanded
    pulse width reaches the shape at all. If it does not, every thermal descriptor reduces to a
    function of Tmax and the boundary is forced to be a plain Tmax level set.
    """

    duration_ms: float

    def __call__(self, tau: np.ndarray, t_pulse: float) -> np.ndarray: ...


@dataclass(frozen=True)
class FrozenPulse:
    """THE NULL HYPOTHESIS: sin rise then exp decay to a warm plateau, INDEPENDENT of pulse width.

    DO NOT DELETE THIS AS DEAD CODE. It is not merely the historical model; it is the zero-tilt
    hypothesis that the campaign's seed design exists to test. The iso-Tmax ladder (block A of the
    seed plan) is built specifically to distinguish this shape's prediction -- a flat response
    across pulse width at fixed peak temperature -- from every other member of the ensemble. Remove
    it and the experiment has nothing to falsify.

    The commanded pulse width is ignored, so every shot has the same effective dwell and the
    Arrhenius budget collapses to a function of Tmax alone -- i.e. the crystallization boundary is
    forced to be exactly the Tmax level set, with zero kinetic tilt. It is also what reproduces the
    earlier campaign's numbers. It is NOT the physics default (see ``DiffusionPulse``).

    :param plateau: level the trace settles to, as a fraction of Tmax (a wrong boundary condition:
        the substrate is a heat sink, so the film must return to room temperature).
    :param tau_decay: exponential decay time constant (ms).
    :param t_rise: rise time to the peak (ms) -- fixed, so a 0.1 ms shot still gets a 2 ms rise.
    :param duration_ms: total trace length simulated (ms).
    """

    plateau: float = LEGACY_PLATEAU_FRAC
    tau_decay: float = LEGACY_TAU_DECAY_MS
    t_rise: float = LEGACY_RISE_MS
    duration_ms: float = LEGACY_DURATION_MS

    def __call__(self, tau: np.ndarray, t_pulse: float = 0.0) -> np.ndarray:
        """Normalized temperature at times tau (ms since the flash); ``t_pulse`` is ignored."""
        tau = np.asarray(tau, float)
        s = np.zeros_like(tau)
        r = (tau >= 0) & (tau < self.t_rise)
        s[r] = np.sin(np.pi * tau[r] / (2.0 * self.t_rise))
        d = tau >= self.t_rise
        s[d] = self.plateau + (1.0 - self.plateau) * np.exp(
            -(tau[d] - self.t_rise) / self.tau_decay
        )
        return s


@dataclass(frozen=True)
class RampPulse:
    """Linear ramp over the commanded pulse width, then a two-exponential decay to a plateau.

    An empirical parameterization of the measured normalized transient. The decay constants below
    are eyeball fits to a normalized figure, NOT measurements, and they dominate the result:
    ``a_fast / tau_fast`` contributes ~12 ms of effective dwell, more than the commanded pulse
    width over most of the design box, which holds the kinetic tilt down to ~15 C. Treat the tilt
    it predicts as a consequence of those four numbers, not as a measured property of the tool.

    :param plateau: level the trace settles to, as a fraction of Tmax (see ``FrozenPulse``).
    :param a_fast: amplitude of the fast decay component.
    :param tau_fast: fast decay time constant (ms).
    :param a_slow: amplitude of the slow decay component.
    :param tau_slow: slow decay time constant (ms).
    :param duration_ms: total trace length simulated (ms).
    """

    plateau: float = RAMP_PLATEAU_FRAC
    a_fast: float = RAMP_A_FAST
    tau_fast: float = RAMP_TAU_FAST_MS
    a_slow: float = RAMP_A_SLOW
    tau_slow: float = RAMP_TAU_SLOW_MS
    duration_ms: float = TRACE_DURATION_MS

    def __call__(self, tau: np.ndarray, t_pulse: float) -> np.ndarray:
        """Normalized temperature at times tau (ms), ramping to 1 over ``t_pulse``."""
        tau = np.asarray(tau, float)
        g = np.zeros_like(tau)
        r = (tau >= 0) & (tau < t_pulse)
        g[r] = tau[r] / t_pulse
        d = tau >= t_pulse
        s = tau[d] - t_pulse
        g[d] = (
            self.plateau
            + self.a_fast * np.exp(-s / self.tau_fast)
            + self.a_slow * np.exp(-s / self.tau_slow)
        )
        return g


@dataclass(frozen=True)
class DiffusionPulse:
    """PHYSICS DEFAULT: constant surface flux for ``t_pulse`` into a semi-infinite substrate.

    A 30 nm stack has negligible heat capacity, so its temperature is slaved to heat diffusing into
    the thick fused-silica substrate. The exact Carslaw & Jaeger surface solution for a constant
    flux applied over [0, t_pulse] and then removed, normalized to its peak at tau = t_pulse, is

        g(tau) = sqrt(tau / t_pulse)                                  tau <= t_pulse
        g(tau) = (sqrt(tau) - sqrt(tau - t_pulse)) / sqrt(t_pulse)    tau >  t_pulse

    It has ZERO fitted parameters and decays as tau^(-1/2) to room temperature -- no plateau. Note
    g depends on tau only through tau / t_pulse: diffusion has no intrinsic timescale, so the pulse
    width is the ONLY timescale in the problem and the effective dwell is EXACTLY proportional to
    it. The fixed 8/35/40 ms decay constants in the other shapes assert an intrinsic timescale that
    conduction does not have; that assertion is what sets their tilt to 0-15 C instead of ~51 C.

    Idealizations (stated, not hidden): a semi-infinite substrate with no radiative or convective
    loss, a top-hat lamp pulse, and temperature-independent properties.

    :param duration_ms: total trace length simulated (ms).
    """

    duration_ms: float = TRACE_DURATION_MS

    def __call__(self, tau: np.ndarray, t_pulse: float) -> np.ndarray:
        """Normalized temperature at times tau (ms) for a flux applied over ``t_pulse``."""
        tau = np.asarray(tau, float)
        g = np.zeros_like(tau)
        on = (tau >= 0) & (tau <= t_pulse)
        g[on] = np.sqrt(tau[on] / t_pulse)
        off = tau > t_pulse
        g[off] = (np.sqrt(tau[off]) - np.sqrt(tau[off] - t_pulse)) / np.sqrt(t_pulse)
        return g


@dataclass(frozen=True)
class RectangularPulse:
    """Bounding case: held at Tmax for the pulse width, then instantly cold.

    Not physical (no substrate can quench that fast); included because it is the extreme of the
    "effective dwell tracks the commanded width" family and so bounds the kinetic tilt.

    :param duration_ms: total trace length simulated (ms).
    """

    duration_ms: float = TRACE_DURATION_MS

    def __call__(self, tau: np.ndarray, t_pulse: float) -> np.ndarray:
        """Normalized temperature at times tau (ms): 1 while the pulse is on, 0 after."""
        tau = np.asarray(tau, float)
        return np.where((tau >= 0) & (tau <= t_pulse), 1.0, 0.0)


@runtime_checkable
class ThermalModel(Protocol):
    """Forward model: (voltage, time) -> peak temperature and the full temperature trace."""

    def tmax(self, V: np.ndarray, t: np.ndarray) -> np.ndarray: ...

    def trace(self, v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]: ...


class TableThermalModel:
    """Peak temperature from a MEASURED (voltage x time) table, scaled by a normalized pulse shape.

    Smoothly interpolates the tabulated peak temperatures (bicubic spline) and applies the supplied
    normalized trace shape, scaled per shot by that shot's Tmax. Whether the crystallization
    boundary is a plain Tmax level set or carries a kinetic tilt is decided entirely by ``shape``:
    a pulse-width-independent shape (``FrozenPulse``) makes every thermal descriptor a function of
    Tmax alone and forces zero tilt; a width-tracking shape (``DiffusionPulse``) does not.

    :param voltages: measured flash voltages (grid columns).
    :param times: measured flash times (grid rows).
    :param tmax_table: peak temperatures (deg C), shape ``(len(times), len(voltages))``.
    :param shape: normalized pulse shape (defaults to ``DiffusionPulse()``, the physics default).
    :param t_room: baseline (room) temperature (deg C).
    """

    def __init__(
        self, voltages, times, tmax_table, shape: PulseShape = None, t_room: float = T_ROOM
    ):
        self.voltages = np.asarray(voltages, float)
        self.times = np.asarray(times, float)
        self.tmax_table = np.asarray(tmax_table, float)
        self.shape = shape if shape is not None else DiffusionPulse()
        self.t_room = t_room
        self._spline = RectBivariateSpline(self.times, self.voltages, self.tmax_table, kx=3, ky=3)

    def tmax(self, V: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Peak temperature (deg C) at voltage V, flash time t (smooth spline over the table)."""
        V = np.clip(np.asarray(V, float), self.voltages[0], self.voltages[-1])
        t = np.clip(np.asarray(t, float), self.times[0], self.times[-1])
        return np.asarray(self._spline.ev(t, V))

    def voltages_for_tmax(self, targets_c: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Flash voltages reaching each peak temperature in ``targets_c``; NaN where unreachable.

        Tmax is monotone in V at fixed t (every table row increases left to right), so bisection on
        the spline inverts it exactly. The whole batch is bisected at once -- inverting points one
        at a time dominates the runtime of the seed-design search, which evaluates many candidate
        designs.

        :param targets_c: desired peak temperatures (deg C).
        :param t: flash times (ms), broadcast against ``targets_c``.
        """
        targets_c, t = np.broadcast_arrays(
            np.atleast_1d(np.asarray(targets_c, float)), np.atleast_1d(np.asarray(t, float))
        )
        lo = np.full(targets_c.shape, float(self.voltages[0]))
        hi = np.full(targets_c.shape, float(self.voltages[-1]))
        reachable = (self.tmax(lo, t) <= targets_c) & (targets_c <= self.tmax(hi, t))
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            below = self.tmax(mid, t) < targets_c
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return np.where(reachable, 0.5 * (lo + hi), np.nan)

    def voltage_for_tmax(self, target_c: float, t: float) -> float:
        """Flash voltage reaching peak temperature ``target_c`` at flash time ``t``, or NaN.

        :param target_c: desired peak temperature (deg C).
        :param t: flash time (ms).
        """
        return float(self.voltages_for_tmax(np.array([target_c]), np.array([t]))[0])

    def tmax_range(self, t: float) -> Tuple[float, float]:
        """Reachable ``(min, max)`` peak temperature at flash time ``t`` across the voltage axis."""
        return float(self.tmax(self.voltages[0], t)), float(self.tmax(self.voltages[-1], t))

    def trace(self, v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
        """Temperature trace ``T(tau) = t_room + (Tmax - t_room) * shape(tau; t)``; tau in ms.

        The sample grid explicitly includes the pulse width and refines around it. The diffusion
        shape peaks exactly at ``tau = t`` with a square-root cusp, so a plain uniform grid misses
        the peak and under-reports it by tens of degrees.

        :param v: flash voltage.
        :param t: flash time (also the pulse width handed to the shape).
        :param n: approximate number of trace samples.
        """
        peak = float(self.tmax(v, t))
        t = float(t)
        tau = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, self.shape.duration_ms, n),
                    np.linspace(max(0.0, 0.8 * t), min(1.2 * t, self.shape.duration_ms), n // 4),
                    [t],
                ]
            )
        )
        return tau, self.t_room + (peak - self.t_room) * self.shape(tau, t)


def thermal_model(shape: PulseShape) -> TableThermalModel:
    """Measured-table thermal model carrying ``shape`` -- one ensemble member per pulse shape."""
    return TableThermalModel(FLASH_V, FLASH_T, FLASH_TMAX, shape=shape)


# Named ensemble members. They share the MEASURED Tmax table and differ only in the trace shape,
# so comparing them isolates the one thing we have never measured: the cooling law.
SHAPES: Dict[str, PulseShape] = {
    "isoT": FrozenPulse(),  # legacy: width-independent -> zero tilt
    "ramp": RampPulse(),  # collaborator's eyeball-fit decay -> ~15 C tilt
    "diffusion": DiffusionPulse(),  # derived substrate conduction -> ~51 C tilt (DEFAULT)
    "rect": RectangularPulse(),  # bounding case -> ~51 C tilt
}

# Default forward model = the measured flash-lamp table with the PHYSICS-default cooling law.
# Module-level tmax/_trace wrap it so existing callers keep a stable import.
FLASH = thermal_model(SHAPES["diffusion"])

# The historical model, kept so the earlier iso-Tmax campaign numbers stay reproducible.
FLASH_ISOT = thermal_model(SHAPES["isoT"])


def tmax(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Peak temperature (deg C) at voltage V, flash time t, from the default ``FLASH`` model."""
    return FLASH.tmax(V, t)


def _trace(v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """Temperature trace from the default ``FLASH`` model (see ``TableThermalModel.trace``)."""
    return FLASH.trace(v, t, n)
