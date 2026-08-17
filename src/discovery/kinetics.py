"""Crystallization-boundary models over the (flash voltage, flash time) design box.

The campaign target is TOTAL CRYSTALLINE FRACTION (not phase), so the map is monotone in thermal
severity and there is a single boundary to find. What is NOT known is where that boundary sits in
(V, t), and the uncertainty is dominated by one unmeasured thing: the cooling law.

Peak temperature Tmax(V, t) is a measured table. The temperature TRACE is not measured; it is
asserted by a pulse shape. Activated kinetics integrate the trace, so the shape decides whether the
boundary is a plain Tmax level set or carries a kinetic tilt:

    Phi(V, t) = INT exp(-Ea / (kB * T(tau))) dtau         Arrhenius (thermal) budget
    X(V, t)   = 1 - exp(-(nu * Phi)^n)                    Avrami / JMAK transformed fraction

Because the Arrhenius weight collapses onto a narrow cap below the peak, Phi factorizes as
``Phi = t_eff * exp(-Ea / kB Tmax)`` where t_eff is an effective dwell. Holding Phi at its critical
value then gives the boundary tilt in closed form,

    Tb(t2) - Tb(t1) = -theta * ln( t_eff(t2) / t_eff(t1) ),    theta = kB * Tmax^2 / Ea

so a shape whose t_eff ignores the commanded pulse width forces zero tilt (the boundary is exactly
the Tmax level set), while a shape whose t_eff tracks it does not. Over t in [2.6, 10.1] ms the
shapes in ``synthetic.SHAPES`` span 0 to 51 C of tilt. This module builds one boundary model per
shape so a design can be scored against the whole ensemble instead of one assumed member.

All members are anchored to the same (T_ONSET_C, T_REF_MS) so they differ ONLY in tilt; otherwise a
comparison confounds the cooling law with the onset temperature.
"""

from dataclasses import dataclass
from typing import Dict, Protocol, runtime_checkable

import numpy as np

from .constants import (
    AVRAMI_N,
    BISECT_ITERS,
    BISECT_TMAX_HI_C,
    BISECT_TMAX_LO_C,
    CELSIUS_TO_KELVIN,
    EA_EV,
    KB_EV,
    QUAD_FAR_POINTS,
    QUAD_NEAR_POINTS,
    QUAD_NEAR_WINDOW_PULSES,
    QUAD_TAU_MIN_MS,
    T_ONSET_C,
    T_REF_MS,
    T_ROOM_C,
)
from .synthetic import FLASH_T, SHAPES, TableThermalModel, thermal_model


def theta_kelvin(tmax_c: float, ea_ev: float = EA_EV) -> float:
    """Temperature drop below the peak that divides the Arrhenius rate by e, ``kB*T^2/Ea`` (K).

    :param tmax_c: peak temperature (deg C).
    :param ea_ev: activation energy (eV).
    """
    return KB_EV * (tmax_c + CELSIUS_TO_KELVIN) ** 2 / ea_ev


def logistic_sharpness(tmax_c: float, ea_ev: float = EA_EV, n: float = AVRAMI_N) -> float:
    """Logistic slope (per deg C) matching JMAK at the half-transformed point.

    The exact JMAK profile ``X = 1 - exp(-ln2 * exp(n*(T-Tc)/theta))`` is a Gompertz, not a
    logistic: it is asymmetric about X = 1/2, with a longer tail on the cold side. Matching the
    midpoint slope of a logistic to it gives ``s = 2*ln2 * n * Ea / (kB * Tc^2)``. Use this instead
    of a hardcoded sharpness so the transition width is a stated consequence of (Ea, n).

    :param tmax_c: temperature at the half-transformed point (deg C).
    :param ea_ev: activation energy (eV).
    :param n: Avrami exponent.
    """
    return 2.0 * np.log(2.0) * n / theta_kelvin(tmax_c, ea_ev)


@runtime_checkable
class BoundaryModel(Protocol):
    """Forward model: (voltage, time) -> crystalline fraction in [0, 1]."""

    name: str

    def fraction(self, V: np.ndarray, t: np.ndarray) -> np.ndarray: ...

    def fraction_grid(self, v_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray: ...


def _trace_times(t_pulse: float, duration_ms: float) -> np.ndarray:
    """Integration grid dense near the peak, sparse in the tail.

    The Arrhenius integrand is concentrated within ``theta`` of the peak -- for the diffusion shape
    that cap is a fraction of a millisecond -- so a uniform grid over the full 320 ms trace would
    under-resolve the only part that contributes.

    :param t_pulse: commanded pulse width (ms).
    :param duration_ms: total trace length (ms).
    """
    window = QUAD_NEAR_WINDOW_PULSES * t_pulse
    near = np.linspace(0.0, min(window, duration_ms), QUAD_NEAR_POINTS)
    far = np.geomspace(max(window, QUAD_TAU_MIN_MS), duration_ms, QUAD_FAR_POINTS)
    return np.unique(np.concatenate([near, far]))


@dataclass
class KineticBoundary:
    """JMAK transformed fraction driven by the Arrhenius budget of a thermal model's trace.

    ``nu`` is calibrated so X = 1/2 at ``(t_star, t_ref)``, which pins every ensemble member to the
    same onset and leaves the pulse shape as the only difference between them.

    :param thermal: thermal forward model supplying Tmax and the temperature trace.
    :param name: short identifier used in outputs.
    :param ea_ev: activation energy (eV).
    :param n: Avrami exponent.
    :param t_star: peak temperature at the anchor (deg C).
    :param t_ref: flash time at the anchor (ms).
    """

    thermal: TableThermalModel
    name: str = "kinetic"
    ea_ev: float = EA_EV
    n: float = AVRAMI_N
    t_star: float = T_ONSET_C
    t_ref: float = T_REF_MS

    def __post_init__(self) -> None:
        phi_ref = float(self.budget(np.array([self.t_star]), self.t_ref)[0])
        self.nu = (-np.log(0.5)) ** (1.0 / self.n) / phi_ref

    def budget(self, tmax_c: np.ndarray, t: float) -> np.ndarray:
        """Arrhenius budget Phi for peak temperatures ``tmax_c`` at a single flash time ``t``.

        :param tmax_c: peak temperatures (deg C), any shape.
        :param t: flash time / pulse width (ms).
        """
        tmax_c = np.atleast_1d(np.asarray(tmax_c, float))
        tau = _trace_times(float(t), self.thermal.shape.duration_ms)
        g = self.thermal.shape(tau, float(t))[:, None]
        t_kelvin = T_ROOM_C + (tmax_c[None, :] - T_ROOM_C) * g + CELSIUS_TO_KELVIN
        return np.trapezoid(np.exp(-self.ea_ev / (KB_EV * t_kelvin)), tau, axis=0)

    def fraction_from_tmax(self, tmax_c: np.ndarray, t: float) -> np.ndarray:
        """Crystalline fraction from peak temperature and flash time.

        :param tmax_c: peak temperatures (deg C).
        :param t: flash time (ms).
        """
        return 1.0 - np.exp(-((self.nu * self.budget(tmax_c, t)) ** self.n))

    def fraction(self, V: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Crystalline fraction at scattered conditions; grouped by flash time for vectorization.

        :param V: flash voltages.
        :param t: flash times (ms), broadcast against ``V``.
        """
        V, t = np.broadcast_arrays(np.atleast_1d(np.asarray(V, float)), np.atleast_1d(t))
        out = np.empty(V.shape, float)
        for t_val in np.unique(t):
            m = t == t_val
            out[m] = self.fraction_from_tmax(self.thermal.tmax(V[m], t_val), float(t_val))
        return out

    def fraction_grid(self, v_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Crystalline fraction on the outer grid, shape ``(len(t_grid), len(v_grid))``.

        :param v_grid: flash voltages (columns).
        :param t_grid: flash times (rows).
        """
        out = np.empty((len(t_grid), len(v_grid)), float)
        for i, t_val in enumerate(t_grid):
            out[i] = self.fraction_from_tmax(self.thermal.tmax(v_grid, t_val), float(t_val))
        return out

    def boundary_tmax(self, t: float) -> float:
        """Peak temperature at which X = 1/2 for flash time ``t`` (bisection on Phi).

        :param t: flash time (ms).
        """
        lo, hi = BISECT_TMAX_LO_C, BISECT_TMAX_HI_C
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if float(self.fraction_from_tmax(np.array([mid]), t)[0]) < 0.5:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def tilt_c(self, t_lo: float = FLASH_T[1], t_hi: float = FLASH_T[-1]) -> float:
        """Boundary tilt: drop in boundary peak temperature from ``t_lo`` to ``t_hi`` (deg C).

        Zero means the boundary is exactly a Tmax level set.

        :param t_lo: short flash time (ms).
        :param t_hi: long flash time (ms).
        """
        return self.boundary_tmax(t_lo) - self.boundary_tmax(t_hi)

    def sharpness(self) -> float:
        """Logistic-equivalent transition sharpness (per deg C) implied by ``(ea_ev, n)``."""
        return logistic_sharpness(self.t_star, self.ea_ev, self.n)


@dataclass
class IsoTmaxBoundary:
    """The historical model: a logistic in peak temperature, with NO time-at-temperature term.

    Retained as an explicit ensemble member -- the zero-tilt hypothesis -- rather than as the
    default assumption. Its sharpness is derived from (Ea, n) via ``logistic_sharpness`` so it is
    directly comparable to the kinetic members instead of being an independent free constant.

    :param thermal: thermal forward model supplying Tmax.
    :param name: short identifier used in outputs.
    :param t_star: onset peak temperature (deg C).
    :param sharp: logistic slope per deg C; derived from (Ea, n) when omitted.
    :param ea_ev: activation energy used to derive ``sharp`` (eV).
    :param n: Avrami exponent used to derive ``sharp``.
    """

    thermal: TableThermalModel
    name: str = "isoT"
    t_star: float = T_ONSET_C
    sharp: float = None
    ea_ev: float = EA_EV
    n: float = AVRAMI_N

    def __post_init__(self) -> None:
        if self.sharp is None:
            self.sharp = logistic_sharpness(self.t_star, self.ea_ev, self.n)

    def fraction(self, V: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Crystalline fraction: a logistic of ``(Tmax - t_star)``; re-entrant in (V, t).

        :param V: flash voltages.
        :param t: flash times (ms).
        """
        return 1.0 / (1.0 + np.exp(-self.sharp * (self.thermal.tmax(V, t) - self.t_star)))

    def fraction_grid(self, v_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Crystalline fraction on the outer grid, shape ``(len(t_grid), len(v_grid))``.

        :param v_grid: flash voltages (columns).
        :param t_grid: flash times (rows).
        """
        vv, tt = np.meshgrid(v_grid, t_grid)
        return self.fraction(vv, tt)

    def boundary_tmax(self, t: float) -> float:
        """Peak temperature at which X = 1/2 -- by construction ``t_star``, for any ``t``."""
        return self.t_star

    def tilt_c(self, t_lo: float = FLASH_T[1], t_hi: float = FLASH_T[-1]) -> float:
        """Boundary tilt, identically zero for this model."""
        return 0.0

    def sharpness(self) -> float:
        """Logistic transition sharpness (per deg C)."""
        return self.sharp


def build_ensemble(
    t_star: float = T_ONSET_C, ea_ev: float = EA_EV, n: float = AVRAMI_N
) -> Dict[str, BoundaryModel]:
    """The four boundary hypotheses, sharing the measured Tmax table and one onset anchor.

    Members, by the tilt they imply over t in [2.6, 10.1] ms:
      ``isoT``      0 C   -- historical; the pulse width never reaches the trace
      ``ramp``     ~14 C  -- empirical two-exponential decay with eyeball-fit constants
      ``diffusion``~51 C  -- derived semi-infinite substrate conduction (PHYSICS DEFAULT)
      ``rect``     ~51 C  -- bounding case, held at peak for the pulse width

    :param t_star: onset peak temperature at the anchor (deg C).
    :param ea_ev: activation energy (eV).
    :param n: Avrami exponent.
    """
    iso = thermal_model(SHAPES["isoT"])
    models: Dict[str, BoundaryModel] = {
        "isoT": IsoTmaxBoundary(iso, name="isoT", t_star=t_star, ea_ev=ea_ev, n=n)
    }
    for key in ("ramp", "diffusion", "rect"):
        models[key] = KineticBoundary(
            thermal_model(SHAPES[key]), name=key, ea_ev=ea_ev, n=n, t_star=t_star
        )
    return models


DEFAULT_MODEL = "diffusion"  # the only cooling law that is derived rather than fitted


def disagreement(models: Dict[str, BoundaryModel], V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spread (max - min) of predicted crystalline fraction across the ensemble.

    This is what a seed condition is worth as a discriminator: conditions where the hypotheses
    agree cost a shot and settle nothing.

    :param models: ensemble of boundary models.
    :param V: flash voltages.
    :param t: flash times (ms).
    """
    preds = np.stack([m.fraction(V, t) for m in models.values()])
    return preds.max(axis=0) - preds.min(axis=0)
