"""Peak-temperature model as a physics-informed Gaussian process, with uncertainty.

The campaign's peak temperature comes from 30 measured nodes on a 5 x 6 (V, t) grid. Interpolating
them with a bicubic spline has three defects that matter for experiment design:

  * it reports no uncertainty at all, so a temperature quoted between nodes looks exactly as
    trustworthy as one quoted at a node;
  * it is not monotone in voltage -- it overshoots near the hot end for flash times around 8-9 ms,
    which the inversion has to work around with a dense scan;
  * it has nothing to say about the 0.1-2.6 ms interval, where there are no nodes at all and peak
    temperature climbs 374 C. The spline fills that gap with an interpolant that is pure artefact,
    and the campaign currently handles this with a hard exclusion rule.

This module replaces the interpolation with

    T_max(V, t) - T_room  ~  GP( m(V, t),  k )

where the MEAN FUNCTION carries the physics and the GP models only what the physics misses:

    m(V, t) = c * E(V, t) * t^(-p)

A thin film on a thick substrate reaches a surface temperature set by the delivered energy spread
over the depth heat has time to reach. Ideal conduction into a semi-infinite solid gives a depth
growing as sqrt(alpha t), i.e. p = 1/2. Fitting p instead of fixing it lands at p = 0.40 and cuts
the residual scatter from 30 C to 12 C (94% to 99% of the variance explained), so the exponent is
left free and its departure from 1/2 is reported -- a finite substrate, a non-top-hat drive, or
temperature-dependent properties would all push it that way.

E(V, t) is the MEASURED delivered fluence, fitted from the bolometer readings with a saturating
form ``E = a V^b (1 - exp(-t/tau))``. The constant ``c`` absorbs absorptance and substrate
properties.

What this buys:

  * an honest error bar everywhere, in particular a LARGE one across the un-noded gap. The
    exclusion of t < 2.6 ms stops being a hard rule and becomes a consequence of the uncertainty;
  * sensible behaviour between and beyond nodes, because the mean function already knows that
    temperature rises with voltage and turns over in time;
  * a residual field that is small and smooth, so the GP is interpolating a correction rather than
    inventing the whole surface.

Only PulseForge-era bolometer rows are used (the workbook also holds rows from an earlier
kV-scale tool, which are not commensurate with this voltage axis).
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .constants import DATA_DIR, T_ROOM_C
from .synthetic import FLASH_T, FLASH_TMAX, FLASH_V, T_HI, T_LO, V_HI, V_LO

BOLOMETER_XLSX = DATA_DIR / "Bolometer_readings_PulseForge.xlsx"
BOLOMETER_SHEET = "Combined"
FLUENCE_COLUMN = "Energy density new cone (J/cm^2)"  # new cone only; the old cone is superseded
CAMPAIGN_V_MAX = 1200.0  # excludes the earlier kV-scale tool's rows

GP_RESTARTS = 12
# The table is the campaign's best estimate of peak temperature and carries no stated measurement
# uncertainty, so the nugget is held small and the GP is made to pass through the nodes. The
# uncertainty it then reports is INTERPOLATION uncertainty -- how much the surface is pinned
# between measurements -- which is the quantity the design actually needs. A loose nugget lets the
# GP absorb the residual's real structure as noise and stop honouring the measurements.
NUGGET_BOUNDS = (1e-8, 1e-2)  # C^2
LENGTHSCALE0 = (0.5, 0.5)
LENGTHSCALE_BOUNDS = (0.05, 10.0)
AMPLITUDE_BOUNDS = (1e-2, 1e6)


def load_fluence(path=BOLOMETER_XLSX) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured delivered fluence on the campaign tool: ``(V, t_ms, E_J_cm2)``.

    :param path: bolometer workbook.
    """
    d = pd.read_excel(path, sheet_name=BOLOMETER_SHEET)
    d.columns = [str(c).strip() for c in d.columns]
    v = d["Voltage (V)"].to_numpy(float)
    t = d["Time (ms)"].to_numpy(float)
    e = d[FLUENCE_COLUMN].to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(t) & np.isfinite(e) & (e > 0) & (v < CAMPAIGN_V_MAX)
    return v[ok], t[ok], e[ok]


def fit_fluence(v: np.ndarray, t: np.ndarray, e: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit ``E = a V^b (1 - exp(-t/tau))``; returns ``(a, b, tau, mean_rel_err)``.

    The saturating form is used rather than a plain power law because the lamp's delivered energy
    levels off with commanded width, and it is that saturation -- not conduction -- that makes peak
    temperature turn over in time.

    :param v: measured voltages.
    :param t: measured flash times (ms).
    :param e: measured fluence (J/cm^2).
    """

    def model(x, a, b, tau):
        return a * x[0] ** b * (1.0 - np.exp(-x[1] / tau))

    popt, _ = curve_fit(model, np.vstack([v, t]), e, p0=[1e-3, 1.3, 2.0], maxfev=60000)
    rel = float(np.mean(np.abs(model(np.vstack([v, t]), *popt) - e) / e))
    return float(popt[0]), float(popt[1]), float(popt[2]), rel


@dataclass
class ThermalGP:
    """Peak temperature as a GP over a physics mean, fitted to the measured table.

    :param a: fluence prefactor.
    :param b: fluence voltage exponent.
    :param tau: fluence saturation time (ms).
    :param scale: fitted constant relating ``E * t^-p`` to temperature rise.
    :param time_exponent: fitted ``p``; 0.5 is ideal semi-infinite conduction.
    :param fluence_rel_err: mean relative residual of the fluence fit, for reporting.
    """

    a: float
    b: float
    tau: float
    scale: float = 1.0
    time_exponent: float = 0.5
    fluence_rel_err: float = 0.0
    _gp: GaussianProcessRegressor = field(default=None, repr=False)

    def fluence(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Delivered fluence (J/cm^2) at ``(v, t)`` from the fitted saturating form."""
        v = np.asarray(v, float)
        t = np.asarray(t, float)
        return self.a * v**self.b * (1.0 - np.exp(-t / self.tau))

    def mean_rise(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Physics mean for the temperature rise above room: ``c * E(V,t) * t^-p``."""
        t = np.asarray(t, float)
        return self.scale * self.fluence(v, t) * t ** (-self.time_exponent)

    def _features(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Normalized (V, log t) coordinates, so one lengthscale per axis is meaningful."""
        x = (np.asarray(v, float) - V_LO) / (V_HI - V_LO)
        lo, hi = np.log10(T_LO), np.log10(T_HI)
        y = (np.log10(np.asarray(t, float)) - lo) / (hi - lo)
        return np.column_stack([x, y])

    def fit(self, v: np.ndarray, t: np.ndarray, tmax_c: np.ndarray) -> "ThermalGP":
        """Fit the mean scale and then a GP on the residual.

        :param v: node voltages.
        :param t: node flash times (ms).
        :param tmax_c: measured peak temperatures (deg C).
        """
        rise = np.asarray(tmax_c, float) - T_ROOM_C
        flu = self.fluence(v, t)
        tt = np.asarray(t, float)

        def mean_model(x, c, p):
            return c * x[0] * x[1] ** (-p)

        (self.scale, self.time_exponent), _ = curve_fit(
            mean_model, np.vstack([flu, tt]), rise, p0=[1.0, 0.5], maxfev=60000
        )
        resid = rise - self.mean_rise(v, t)
        kernel = ConstantKernel(np.var(resid) + 1.0, AMPLITUDE_BOUNDS) * Matern(
            list(LENGTHSCALE0), LENGTHSCALE_BOUNDS, nu=2.5
        ) + WhiteKernel(1.0, NUGGET_BOUNDS)
        self._gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False, n_restarts_optimizer=GP_RESTARTS
        ).fit(self._features(v, t), resid)
        return self

    def predict(self, v: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Peak temperature (deg C) and its standard deviation at ``(v, t)``.

        The standard deviation is the GP's posterior uncertainty on the residual. It collapses at
        the measured nodes and grows away from them -- most sharply across the un-noded time gap,
        which is the behaviour the spline could not express.

        :param v: flash voltages.
        :param t: flash times (ms).
        """
        v = np.atleast_1d(np.asarray(v, float))
        t = np.atleast_1d(np.asarray(t, float))
        mu, sd = self._gp.predict(self._features(v, t), return_std=True)
        return T_ROOM_C + self.mean_rise(v, t) + mu, sd

    def tmax(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Peak temperature only, matching the ``ThermalModel`` protocol's signature."""
        return self.predict(v, t)[0]


def build(path=BOLOMETER_XLSX) -> ThermalGP:
    """Fit the fluence law from the bolometer and the GP from the measured temperature table.

    :param path: bolometer workbook.
    """
    v, t, e = load_fluence(path)
    a, b, tau, rel = fit_fluence(v, t, e)
    vv, tt = np.meshgrid(FLASH_V, FLASH_T)
    model = ThermalGP(a=a, b=b, tau=tau, fluence_rel_err=rel)
    return model.fit(vv.ravel(), tt.ravel(), FLASH_TMAX.ravel())
