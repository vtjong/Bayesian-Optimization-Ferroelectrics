"""The boundary surrogate: an exact GP on the LATENT logit field, fitted by MAP rather than MLE.

Three decisions here are load-bearing, and each was made because the naive version demonstrably
fails at the sample sizes this campaign runs at.

REGRESS THE LOGIT, NOT THE FRACTION. The crystalline fraction lives in (0, 1) while a Gaussian
field lives on all of R, so a GP fitted directly to X predicts outside [0, 1] and violates its own
noise assumption near the rails. Regressing ``f = logit(X)`` fixes three things at once: the
boundary becomes the ZERO LEVEL SET of a smooth field, which is a far easier object to locate than
a contour of a saturating one; the transition STEEPNESS survives, and steepness is the part of the
signal carrying mechanism information, which a classifier would discard; and JMAK is close to
linear in logit space, so the latent field is smooth and slowly varying exactly where the data is,
which is what a stationary kernel models well. The cost is bounded and stated: readings at or
beyond the rails are clipped before the transform, manufacturing small plateaus at +/- 6.9.

FIT BY MAP, NOT MLE. At n of order 15 the marginal likelihood is not unimodal. It grows a second,
degenerate basin -- one lengthscale collapsed to its floor, the other pushed to its ceiling, the
noise inflated to absorb the residual -- asserting "the response is constant along one axis plus
white noise along the other". That basin can win by a hair, and the boundary it implies is
nonsense. Measured on this campaign's own candidate designs at n = 16, a plain-MLE fit lands in it
between 23% and 72% of the time, and it hits REPLICATED OR CLUSTERED designs hardest -- 72% for the
dwell ladder, whose whole structure is repeated conditions. Any design comparison fitted by plain
MLE therefore partly measures this artefact, and penalises exactly the designs the campaign cares
about. Weakly informative log-normal priors remove it: they encode instrument metrology and window
geometry, never the mechanism, and act only as tie-breakers between near-degenerate basins.

NOISE IS PER-POINT, NOT SCALAR. The logit is variance-inflating: by the delta method
``Var(logit X) = Var(X) / [X(1-X)]^2``, so a readout with roughly constant absolute error in X has
a LATENT noise that varies by more than an order of magnitude across the range -- about 0.12 at
X = 0.5, 0.33 at X = 0.9, and 1.5 at X = 0.98 for a 3% absolute error. Fitting a single scalar
splits that difference, which is exactly backwards: it over-trusts near-rail points, which carry
almost no information about where the boundary is, and under-weights mid-transition points, which
are the only ones that locate it. The shape of the per-point variance is therefore imposed
analytically and only its scale is fitted.

The plug-in uses the SMOOTHED fraction, never the observed one. Reading the noise off the observed
value correlates the weight with the residual -- a point that happened to scatter high is then told
it is precise -- so a first homoscedastic pass supplies the smoothed mean that the second pass
weights by.

CHOLESKY, NEVER A RAW INVERSE. Gram matrices go near-singular whenever two conditions nearly
coincide -- which replicates do by construction -- so the factorization is done once and reused for
both the mean and the variance.

MATERN 5/2, NOT SQUARED-EXPONENTIAL. SE sample paths are analytic; forcing that smoothness onto a
steep crystallization ramp produces ringing beside the boundary, and the fit compensates by
shrinking the lengthscale globally, wrecking the calm regions. Matern 5/2 paths are twice
differentiable and no more.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from .synthetic import T_HI, V_HI, V_LO

LOGIT_CLIP = 1.0e-3  # bounds the latent to about +/- 6.9
_JITTER = 1.0e-8  # insures the Cholesky against coincident conditions
_SQRT5 = np.sqrt(5.0)

# Priors on (sigma_f, ell_V, ell_logt, sigma_n), in log space. Broad -- about a factor of two per
# sigma -- so they only break ties. A lengthscale of 0.02 would make the map unlearnable at any
# feasible budget and one of 3 asserts no structure at all; both are what the degenerate basin
# reaches for, and both sit many sigma out.
PRIOR_MU = np.log([3.0, 0.25, 0.25, 0.20])
PRIOR_SD = np.array([1.0, 0.75, 0.75, 1.0])
BOUNDS_LO = np.log([0.3, 0.02, 0.02, 0.01])
BOUNDS_HI = np.log([30.0, 3.0, 3.0, 5.0])
N_RESTARTS = 6

# Caps the per-point variance inflation. At the logit clip the raw factor would be ~1e6, which is
# numerically pointless -- a rail point carries no information either way -- so it is bounded.
MAX_NOISE_INFLATION = 40.0


def normalize_inputs(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map ``(V, t)`` to the unit box, with time in log10.

    The kernel measures distance and raw units are unusable: voltage spans 210 while flash time
    spans two decades, so a raw metric would be ~99% voltage. Log time additionally makes the
    boundary geometry closer to isotropic.

    :param v: flash voltages.
    :param t: flash times (ms).
    """
    x = (np.asarray(v, float) - V_LO) / (V_HI - V_LO)
    lo, hi = np.log10(0.1), np.log10(T_HI)
    return np.column_stack([x, (np.log10(np.asarray(t, float)) - lo) / (hi - lo)])


def to_latent(y: np.ndarray) -> np.ndarray:
    """Map readings to the latent logit field, self-normalizing first.

    The readout's floor and span are not known -- large-signal permittivity has an uncalibrated
    zero and gain -- so the observed range places the readings in (0, 1) before the transform. A
    design must never rely on the instrument reporting a fraction directly.

    :param y: observed readings, in whatever units the instrument reports.
    """
    y = np.asarray(y, float)
    span = float(np.ptp(y))
    if span <= 0.0:
        return np.zeros_like(y)
    x = np.clip((y - float(np.min(y))) / span, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(x / (1.0 - x))


def matern52(a: np.ndarray, b: np.ndarray, sigma_f: float, ells) -> np.ndarray:
    """Matern 5/2 kernel with per-axis lengthscales (ARD).

    :param a: left inputs, shape (n, 2).
    :param b: right inputs, shape (m, 2).
    :param sigma_f: signal standard deviation.
    :param ells: correlation lengths, one per axis.
    """
    d = (np.atleast_2d(a)[:, None, :] - np.atleast_2d(b)[None, :, :]) / np.asarray(ells, float)
    r = np.sqrt(np.sum(d * d, axis=-1))
    return sigma_f**2 * (1.0 + _SQRT5 * r + 5.0 * r * r / 3.0) * np.exp(-_SQRT5 * r)


@dataclass
class BoundarySurrogate:
    """Exact GP on the latent logit field, with a constant mean and MAP hyperparameters.

    :param n_restarts: restarts for the hyperparameter search.
    :param seed: RNG seed for the dispersed restarts.
    :param priors: maximize LML + log prior; set False to recover pure MLE for comparison.
    """

    n_restarts: int = N_RESTARTS
    seed: int = 0
    priors: bool = True
    _x: np.ndarray = field(default=None, repr=False)
    _y: np.ndarray = field(default=None, repr=False)
    _mean: float = field(default=0.0, repr=False)
    _chol: np.ndarray = field(default=None, repr=False)
    _alpha: np.ndarray = field(default=None, repr=False)
    _hypers: dict = field(default=None, repr=False)
    _smooth: np.ndarray = field(default=None, repr=False)

    def _noise_shape(self) -> np.ndarray:
        """Relative per-point latent noise, from the delta method on the logit transform.

        Unity at mid-transition and growing toward the rails, so the fitted scale is interpretable
        as the latent noise of a point sitting on the boundary.
        """
        if self._smooth is None:
            return np.ones(self._y.size)
        x = np.clip(self._smooth, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
        return np.minimum(0.25 / (x * (1.0 - x)), MAX_NOISE_INFLATION)

    def _factorize(self, sigma_f: float, ells, sigma_n: float) -> None:
        """Fix hyperparameters and rebuild the Cholesky factorization."""
        self._hypers = {"sigma_f": float(sigma_f), "ells": np.asarray(ells, float),
                        "sigma_n": float(sigma_n)}
        self._mean = float(np.mean(self._y))
        centred = self._y - self._mean
        gram = matern52(self._x, self._x, sigma_f, ells)
        gram += np.diag((sigma_n * self._noise_shape()) ** 2 + _JITTER)
        self._chol = np.linalg.cholesky(gram)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, centred))

    def _log_marginal(self, log_theta: np.ndarray) -> float:
        """Log marginal likelihood at log-hyperparameters ``(sigma_f, ell_1, ell_2, sigma_n)``."""
        sigma_f, l1, l2, sigma_n = np.exp(log_theta)
        n = self._y.size
        centred = self._y - np.mean(self._y)
        gram = matern52(self._x, self._x, sigma_f, (l1, l2))
        gram += np.diag((sigma_n * self._noise_shape()) ** 2 + _JITTER)
        try:
            chol = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            return -np.inf  # an invalid corner of hyperparameter space
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centred))
        return float(
            -0.5 * centred @ alpha
            - np.sum(np.log(np.diag(chol)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def _log_prior(self, log_theta: np.ndarray) -> float:
        """Log density of the weakly informative log-normal priors, up to a constant."""
        z = (log_theta - PRIOR_MU) / PRIOR_SD
        return float(-0.5 * np.sum(z * z))

    def fit(self, v: np.ndarray, t: np.ndarray, y: np.ndarray) -> "BoundarySurrogate":
        """Fit to observed readings at ``(v, t)`` by multi-restart L-BFGS-B in log space.

        :param v: as-fired flash voltages.
        :param t: as-fired flash times (ms).
        :param y: observed readings.
        """
        self._x = normalize_inputs(v, t)
        self._y = to_latent(y)
        self._smooth = None  # first pass is homoscedastic; see the module note

        rng = np.random.default_rng(self.seed)
        starts = [np.log([np.std(self._y) + 0.5, 0.3, 0.3, 0.3])]
        starts += [rng.uniform(BOUNDS_LO, BOUNDS_HI) for _ in range(self.n_restarts - 1)]

        def objective(log_theta):
            value = self._log_marginal(log_theta)
            return -(value + (self._log_prior(log_theta) if self.priors else 0.0))

        best_value, best_theta = np.inf, starts[0]
        for start in starts:
            res = minimize(
                objective, start, method="L-BFGS-B", bounds=list(zip(BOUNDS_LO, BOUNDS_HI))
            )
            if res.fun < best_value:
                best_value, best_theta = res.fun, res.x
        sigma_f, l1, l2, sigma_n = np.exp(best_theta)
        self._factorize(sigma_f, (l1, l2), sigma_n)

        # Second pass: weight by the SMOOTHED fraction, then refit the scale with the shape fixed.
        self._smooth = self.fraction(v, t)
        best_value, best_theta2 = np.inf, best_theta
        for start in (best_theta, *starts[:2]):
            res = minimize(
                objective, start, method="L-BFGS-B", bounds=list(zip(BOUNDS_LO, BOUNDS_HI))
            )
            if res.fun < best_value:
                best_value, best_theta2 = res.fun, res.x
        sigma_f, l1, l2, sigma_n = np.exp(best_theta2)
        self._factorize(sigma_f, (l1, l2), sigma_n)
        return self

    def latent(self, v: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and sd of the latent field; the boundary is where the mean is zero.

        :param v: flash voltages.
        :param t: flash times (ms).
        """
        q = normalize_inputs(v, t)
        h = self._hypers
        cross = matern52(self._x, q, h["sigma_f"], h["ells"])
        mu = self._mean + cross.T @ self._alpha
        solved = np.linalg.solve(self._chol, cross)
        var = h["sigma_f"] ** 2 - np.sum(solved * solved, axis=0)
        return mu, np.sqrt(np.maximum(var, 0.0))

    def fraction(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Posterior mean crystalline fraction, back through the logistic.

        :param v: flash voltages.
        :param t: flash times (ms).
        """
        return 1.0 / (1.0 + np.exp(-np.clip(self.latent(v, t)[0], -500.0, 500.0)))

    def crystalline_side(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Which side of the boundary each condition falls on.

        The latent zero-crossing IS the boundary, so this needs no threshold on a saturating
        quantity and is invariant to the readout's unknown floor and span.

        :param v: flash voltages.
        :param t: flash times (ms).
        """
        return self.latent(v, t)[0] > 0.0

    @property
    def hyperparameters(self) -> dict:
        """Fitted amplitude, lengthscales and noise, for diagnosing degenerate fits."""
        h = self._hypers
        return {
            "sigma_f": h["sigma_f"],
            "ell_v": float(h["ells"][0]),
            "ell_logt": float(h["ells"][1]),
            "sigma_n": h["sigma_n"],
        }
