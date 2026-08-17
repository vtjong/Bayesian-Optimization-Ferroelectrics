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
from functools import lru_cache
from typing import Dict, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .constants import (
    BISECT_ITERS,
    FLASH_TABLE_CSV,
    KB_EV,
    LAMP_A,
    LAMP_QUAD_NODES,
    LAMP_TAU_FAST_MS,
    LAMP_TAU_SLOW_MS,
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
    V_SCAN_POINTS,
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
    "LampDrivenPulse",
    "FrozenPulse",
    "PulseShape",
    "RampPulse",
    "RectangularPulse",
    "TableThermalModel",
    "DEFAULT_SHAPE",
    "default_shape",
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
    voltages = np.array([float(c.strip().removeprefix("V=")) for c in rows[0][1:]], float)
    times = np.array([float(r[0].strip().removeprefix("t=")) for r in rows[1:]], float)
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

    DO NOT DELETE THIS AS DEAD CODE. It is the zero-tilt hypothesis that the campaign's seed
    design exists to test. The iso-Tmax ladder (block A of the
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
    the slow component's 40 ms constant keeps the trace warm long after the pulse ends, which
    compresses the RATIO of effective dwells across the design box and so holds the kinetic tilt
    well below every conduction member's. Treat the tilt it predicts as a consequence of those four
    numbers, not as a measured property of the tool.

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

    A 10 nm stack has negligible heat capacity, so its temperature is slaved to heat diffusing into
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
class LampDrivenPulse:
    """PHYSICS DEFAULT: measured lamp irradiance driving semi-infinite substrate conduction.

    Same conduction physics as ``DiffusionPulse``, but the source term is the MEASURED lamp
    envelope rather than a top hat. Surface temperature follows Duhamel's integral,

        T(tau) - T_room  ~  INT_0^min(tau, t) q(s) / sqrt(tau - s) ds

    with q(s) a two-exponential fit to the delivered-fluence data. The sqrt singularity is removed
    by substituting u = sqrt(tau - s), giving 2 * INT q(tau - u^2) du over u in
    [sqrt(tau - min(tau,t)), sqrt(tau)].

    WHY THIS AND NOT ``DiffusionPulse``: conduction alone has no intrinsic timescale, so a top-hat
    drive makes the whole transient self-similar in tau/t and the effective dwell exactly
    proportional to pulse width. But the lamp is not a top hat. Measured fluence rises only
    sublinearly with commanded width -- the local exponent d lnE/d lnt falls from 0.42 at 2.6 ms to
    0.01 at 10.1 ms under the fitted saturating law -- i.e. the irradiance droops with a ~2 ms time
    constant that sits squarely inside the 2.6-10.1 ms design range. That intrinsic timescale breaks
    the self-similarity and reduces the predicted tilt relative to a top-hat drive. The top-hat
    idealization, not any of the conduction idealizations, is what makes the difference.

    LIMITATION. This shape truncates a FIXED-AMPLITUDE envelope at the commanded width, so its
    unnormalized peak is monotone non-decreasing in t and saturates rather than turning over. It
    therefore cannot itself generate the re-entrance of the measured Tmax table, and must not be
    cited as evidence for it: re-entrance requires the drive AMPLITUDE to fall with commanded
    width. This shape supplies a normalized trace only; Tmax always comes from the measured table.

    :param a_fast: weight of the fast decay component of the lamp envelope.
    :param tau_fast: fast decay constant of the lamp envelope (ms).
    :param tau_slow: slow decay constant of the lamp envelope (ms).
    :param nodes: quadrature nodes for the Duhamel integral.
    :param duration_ms: total trace length simulated (ms).
    """

    a_fast: float = LAMP_A
    tau_fast: float = LAMP_TAU_FAST_MS
    tau_slow: float = LAMP_TAU_SLOW_MS
    nodes: int = LAMP_QUAD_NODES
    duration_ms: float = TRACE_DURATION_MS

    def irradiance(self, s: np.ndarray) -> np.ndarray:
        """Lamp irradiance envelope at time ``s`` (ms) after firing, normalized to 1 at s = 0."""
        s = np.asarray(s, float)
        return self.a_fast * np.exp(-s / self.tau_fast) + (1.0 - self.a_fast) * np.exp(
            -s / self.tau_slow
        )

    def _duhamel(self, tau: np.ndarray, t_pulse: float) -> np.ndarray:
        """Unnormalized surface temperature rise: 2 * INT q(tau - u^2) du."""
        tau = np.asarray(tau, float)
        lo = np.sqrt(np.clip(tau - np.minimum(tau, t_pulse), 0.0, None))
        hi = np.sqrt(np.clip(tau, 0.0, None))
        w = np.linspace(0.0, 1.0, self.nodes)
        u = lo[..., None] + (hi - lo)[..., None] * w
        return 2.0 * np.trapezoid(self.irradiance(tau[..., None] - u**2), u, axis=-1)

    def peak_time(self, t_pulse: float) -> float:
        """Time of maximum surface temperature (ms).

        NOT ``t_pulse``. Because the lamp irradiance droops with a ~2 ms constant, the surface
        stops gaining once the remaining flux no longer offsets conduction into the substrate, so
        for pulses longer than a few ms the peak sits near 2.3 ms no matter how long the pulse is
        commanded. That saturation is exactly why the measured Tmax table is re-entrant in t.
        """
        return float(_lamp_peak(self, float(t_pulse))[0])

    def __call__(self, tau: np.ndarray, t_pulse: float) -> np.ndarray:
        """Normalized temperature at times tau (ms) for a pulse of width ``t_pulse``."""
        tau = np.asarray(tau, float)
        out = np.where(tau >= 0.0, self._duhamel(np.clip(tau, 0.0, None), float(t_pulse)), 0.0)
        peak = _lamp_peak(self, float(t_pulse))[1]
        return out / peak if peak > 0 else out


@lru_cache(maxsize=4096)
def _lamp_peak(shape: "LampDrivenPulse", t_pulse: float) -> Tuple[float, float]:
    """``(peak_time, peak_value)`` of the unnormalized lamp-driven rise, by dense scan + refine.

    Cached because normalizing every call would otherwise rescan; the shape is frozen and hashable
    so the cache key is well defined.
    """
    coarse = np.linspace(0.0, min(4.0 * max(t_pulse, shape.tau_fast), shape.duration_ms), 2000)
    d = shape._duhamel(coarse, t_pulse)
    i = int(np.argmax(d))
    lo = coarse[max(i - 1, 0)]
    hi = coarse[min(i + 1, coarse.size - 1)]
    fine = np.linspace(lo, hi, 2000)
    df = shape._duhamel(fine, t_pulse)
    j = int(np.argmax(df))
    return float(fine[j]), float(df[j])


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
    :param shape: normalized pulse shape; defaults to ``default_shape()``, the same shape the
        module-level ``FLASH`` uses, so there is exactly one default in the codebase.
    :param t_room: baseline (room) temperature (deg C).
    """

    def __init__(
        self, voltages, times, tmax_table, shape: PulseShape = None, t_room: float = T_ROOM
    ):
        self.voltages = np.asarray(voltages, float)
        self.times = np.asarray(times, float)
        self.tmax_table = np.asarray(tmax_table, float)
        self.shape = shape if shape is not None else default_shape()
        self.t_room = t_room
        self._spline = RectBivariateSpline(self.times, self.voltages, self.tmax_table, kx=3, ky=3)

    def tmax(self, V: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Peak temperature (deg C) at voltage V, flash time t (smooth spline over the table)."""
        V = np.clip(np.asarray(V, float), self.voltages[0], self.voltages[-1])
        t = np.clip(np.asarray(t, float), self.times[0], self.times[-1])
        return np.asarray(self._spline.ev(t, V))

    def voltages_for_tmax(self, targets_c: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Flash voltages reaching each peak temperature in ``targets_c``; NaN where unreachable.

        Every measured table row increases left to right, but the bicubic spline through them is
        NOT monotone in V everywhere: it overshoots near the hot end for flash times around 8-9 ms.
        The reachable ceiling is therefore taken from a dense scan rather than from the endpoint,
        and the bisection bracket stops at that maximum so the inverse is single-valued. The whole
        batch is bisected at once -- inverting points one at a time dominates the runtime of the
        seed-design search, which evaluates many candidate designs.

        :param targets_c: desired peak temperatures (deg C).
        :param t: flash times (ms), broadcast against ``targets_c``.
        """
        targets_c, t = np.broadcast_arrays(
            np.atleast_1d(np.asarray(targets_c, float)), np.atleast_1d(np.asarray(t, float))
        )
        # The TABLE rows increase left to right, but the bicubic spline through them does not:
        # it overshoots near the top of the voltage axis for some flash times, so tmax(V_HI, t) is
        # not the reachable ceiling there. Locate the true maximum on a dense scan and bisect only
        # on the rising branch below it.
        scan_v = np.linspace(self.voltages[0], self.voltages[-1], V_SCAN_POINTS)
        scan_t = np.broadcast_to(t[..., None], t.shape + scan_v.shape)
        scan_tmax = self.tmax(np.broadcast_to(scan_v, scan_t.shape), scan_t)
        peak_idx = np.argmax(scan_tmax, axis=-1)
        ceiling = np.take_along_axis(scan_tmax, peak_idx[..., None], axis=-1)[..., 0]
        floor = scan_tmax[..., 0]
        reachable = (floor <= targets_c) & (targets_c <= ceiling)

        lo = np.full(targets_c.shape, float(self.voltages[0]))
        hi = scan_v[peak_idx]  # bracket ends at the true maximum, not at the axis endpoint
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
        """Reachable ``(min, max)`` peak temperature at flash time ``t`` across the voltage axis.

        The ceiling comes from the same dense scan ``voltages_for_tmax`` uses, not from the
        voltage-axis endpoint: the spline is not monotone in V for flash times around 8-9 ms, so
        the endpoint understates what is reachable. Using the endpoint here while the inversion
        used the scan would let a design ask for a temperature the inversion can reach but the
        feasibility check has already rejected.

        :param t: flash time (ms).
        """
        scan_v = np.linspace(self.voltages[0], self.voltages[-1], V_SCAN_POINTS)
        scan = self.tmax(scan_v, np.full_like(scan_v, float(t)))
        return float(self.tmax(self.voltages[0], t)), float(scan.max())

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
        base = np.linspace(0.0, self.shape.duration_ms, n)
        # Refine around wherever THIS shape actually peaks. It is not always the pulse edge: the
        # lamp-driven shape peaks where the drooping irradiance stops outrunning conduction, which
        # for long pulses is well before t_pulse. A grid that straddles the peak under-reports it.
        coarse = np.linspace(0.0, min(4.0 * max(t, 1.0), self.shape.duration_ms), 4000)
        tpk = float(coarse[np.argmax(self.shape(coarse, t))])
        span = max(0.05 * max(t, 1.0), 2.0 * (coarse[1] - coarse[0]))
        fine = np.linspace(
            max(0.0, tpk - span), min(tpk + span, self.shape.duration_ms), max(n // 2, 64)
        )
        tau = np.unique(np.concatenate([base, fine, [t, tpk]]))
        return tau, self.t_room + (peak - self.t_room) * self.shape(tau, t)


def thermal_model(shape: PulseShape) -> TableThermalModel:
    """Measured-table thermal model carrying ``shape`` -- one ensemble member per pulse shape."""
    return TableThermalModel(FLASH_V, FLASH_T, FLASH_TMAX, shape=shape)


# Named ensemble members. They share the MEASURED Tmax table and differ only in the trace shape,
# so comparing them isolates the one thing we have never measured: the cooling law.
SHAPES: Dict[str, PulseShape] = {
    "isoT": FrozenPulse(),  # width-independent -> zero tilt (the null hypothesis)
    "ramp": RampPulse(),  # empirical two-exponential transient -> ~13 C tilt
    "lamp": LampDrivenPulse(),  # measured lamp + conduction -> ~38 C tilt (DEFAULT)
    "diffusion": DiffusionPulse(),  # conduction under a TOP-HAT drive -> ~50 C tilt
    "rect": RectangularPulse(),  # bounding case; degenerate with diffusion by construction
}

DEFAULT_SHAPE = "lamp"  # the single default; used by TableThermalModel and by FLASH alike


def default_shape() -> PulseShape:
    """The campaign's default pulse shape. One definition, referenced everywhere."""
    return SHAPES[DEFAULT_SHAPE]


# Default forward model = the measured Tmax table driven by the MEASURED lamp envelope.
# Module-level tmax/_trace wrap it so existing callers keep a stable import.
FLASH = thermal_model(default_shape())

# The zero-tilt null, bound to a thermal model so it can be scored alongside the kinetic members.
FLASH_ISOT = thermal_model(SHAPES["isoT"])


def tmax(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Peak temperature (deg C) at voltage V, flash time t, from the default ``FLASH`` model."""
    return FLASH.tmax(V, t)


def _trace(v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """Temperature trace from the default ``FLASH`` model (see ``TableThermalModel.trace``)."""
    return FLASH.trace(v, t, n)
