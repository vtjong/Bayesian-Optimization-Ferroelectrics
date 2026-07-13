"""Calibrated flash-lamp thermal model for FLA HZO: (voltage, time) -> peak T and trace.

Peak temperature is a MEASURED table Tmax(V, t) (flash voltage x flash time, deg C); a bicubic
spline interpolates between the tabulated values. Every shot's temperature trace is the UNIVERSAL
normalized pulse shape T(tau)/Tmax -- which the measurements show is (approximately) the same for
all flash conditions -- scaled by that shot's Tmax. Temperature peaks in t near t ~ 2.6-3 ms, so
Tmax(V, t) is RE-ENTRANT in (V, t) and a Tmax level set (e.g. the crystallization onset) folds.

The forward model is a ``ThermalModel`` (Protocol); ``TableThermalModel`` is the measured
implementation and ``FLASH`` the default instance. Module-level ``tmax`` / ``_trace`` wrap it, so
swapping in another ``ThermalModel`` does not touch the callers.
"""

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

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
# existing callers (picker, run scripts) keep a stable import.
FLASH = TableThermalModel(FLASH_V, FLASH_T, FLASH_TMAX)


def tmax(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Peak temperature (deg C) at voltage V, flash time t, from the default ``FLASH`` model."""
    return FLASH.tmax(V, t)


def _trace(v: float, t: float, n: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """Temperature trace from the default ``FLASH`` model (see ``TableThermalModel.trace``)."""
    return FLASH.trace(v, t, n)
