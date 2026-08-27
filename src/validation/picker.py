"""Active experiment-picker for crystallization-boundary mapping (Layer 2).

Maps the amorphous->crystalline boundary in (V, t) by sequentially choosing where to measure,
using an entropy-family acquisition on a GP with HETEROSCEDASTIC observation noise (noise is
worst near the boundary -- partially crystallized, leaky films / degraded pyrometry).

Three acquisitions (all entropy-family level-set utilities):
  * predictive entropy -- class entropy from the PREDICTIVE variance (latent + noise); peaks at
    the boundary but also chases high-noise regions;
  * BALD -- boundary-weighted information gain about the LATENT function,
        a(x) = 0.5*log(1 + s^2/sigma_n^2) * H( Phi((mu-theta)/s) ),
    which down-weights high-noise regions (s = latent std, sigma_n = observation-noise std);
  * latent level-set entropy (LSE) -- boundary class entropy on the LATENT variance only,
        a(x) = H( Phi((mu-theta)/s) ) -- boundary-focused and noise-aware without chasing or
    fleeing the noise band.

A light spreading penalty keeps shots distributed along the whole boundary, and a
convergence-delta / patience rule (after a labmate's phase-mapping pipeline) can stop the loop.

EMPIRICAL FINDING (see run_picker_demo; calibrated boundary, 16 seeds +/- SEM): for the
boundary-MAPPING objective the three are COMPARABLE -- within ~1 SEM across the realistic
(permittivity) noise regime. LSE ties predictive entropy at low noise and edges ahead at very
high noise; BALD is competitive at moderate noise and lags only at the noise extremes. No
acquisition robustly dominates -- seeding and boundary coverage matter more. We keep LSE as the
DEFAULT: a principled level-set utility that is best-or-tied at the noise extremes and never worst
by a meaningful margin. BALD and predictive entropy are retained as baselines -- the comparison
itself is the justification. (An earlier draft's "LSE dominates / BALD hurts" claim did not survive
calibration + more seeds and was retracted.)

The simulation's ground truth is a ``BoundaryModel`` from ``kinetics``, carried in a frozen
``BoundaryConfig`` and passed explicitly (dependency injection) -- there is no mutable module
state, so the simulation is reentrant and both the truth and the noise level are arguments rather
than hidden globals.

Which ground truth is used MATTERS. The default is the physics-default cooling law, whose boundary
carries ~50 C of kinetic tilt. Any sizing conclusion (seed size, batch size, shots to tolerance)
is conditional on that choice, because a tilted boundary is a harder target than a plain Tmax
level set: it cannot be written t = f(V) and its position depends on dwell as well as peak
temperature. Re-run the studies against ``build_ensemble()["isoT"]`` to see how much of a given
conclusion was an artifact of assuming zero tilt.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from physics.constants import NOISE_BOUNDARY, NOISE_FLOOR
from physics.kinetics import DEFAULT_MODEL, BoundaryModel, build_ensemble
from physics.thermal_model import FLASH_T, T_HI, V_HI, V_LO

# Boundary search is restricted to flash times the measured table actually supports.
T_SEARCH_LO = float(FLASH_T[1])

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# GP hyperparameters. Matern 5/2 rather than a squared exponential: SE sample paths are analytic
# (infinitely differentiable), and forcing that smoothness prior onto a steep crystallization ramp
# produces ringing flanking the boundary, which the fit then compensates for by shrinking the
# lengthscale globally. Matern 5/2 paths are twice differentiable -- smooth enough for a physical
# response surface, rough enough to take a steep ramp without ringing.
KERNEL_NU = 2.5
KERNEL_LENGTHSCALE0 = (0.15, 0.15)  # initial ARD lengthscales in the normalized box
KERNEL_LENGTHSCALE_BOUNDS = (0.03, 1.0)  # physical-intuition bound; stabilizes the fit at small n
KERNEL_AMPLITUDE_BOUNDS = (1e-2, 1e2)
GP_RESTARTS = 8  # multi-restart guards against the degenerate small-n likelihood basin
JITTER = 1e-4


def default_truth() -> BoundaryModel:
    """The ground truth used when a caller does not name one: the physics-default cooling law."""
    return build_ensemble()[DEFAULT_MODEL]


@dataclass(frozen=True)
class BoundaryConfig:
    """Ground-truth boundary model + heteroscedastic-noise parameters for the synthetic study.

    Grouping these in one immutable object (instead of mutable module globals that callers
    reassign) makes the simulation reentrant and both the truth and the noise level explicit.

    :param truth: boundary model supplying the true crystalline fraction over (V, t).
    :param theta: crystallization threshold; the boundary is the level set ``f = theta``.
    :param noise_floor: baseline observation-noise std ``s0``.
    :param noise_boundary: extra noise scale at the boundary ``s1``; ``sigma_n = s0 + s1 f (1-f)``.
    """

    truth: BoundaryModel = field(default_factory=default_truth)
    theta: float = 0.5
    noise_floor: float = NOISE_FLOOR
    noise_boundary: float = NOISE_BOUNDARY


DEFAULT = BoundaryConfig()


# --- ground truth (the picker never sees it; used to generate readings and to score) -----
def true_f(V: np.ndarray, t: np.ndarray, cfg: BoundaryConfig = DEFAULT) -> np.ndarray:
    """True crystalline fraction from the configured boundary model. Re-entrant in (V, t)."""
    return cfg.truth.fraction(V, t)


def noise_sigma(f: np.ndarray, cfg: BoundaryConfig = DEFAULT) -> np.ndarray:
    """Heteroscedastic observation-noise std: largest at the boundary (``f = 0.5``)."""
    return cfg.noise_floor + cfg.noise_boundary * f * (1.0 - f)


def measure(
    V: np.ndarray, t: np.ndarray, rng: Generator, cfg: BoundaryConfig = DEFAULT
) -> np.ndarray:
    """Noisy continuous readout at (V, t) (e.g. a permittivity-derived crystalline fraction).

    :param V: flash voltage(s).
    :param t: flash time(s).
    :param rng: random generator for the observation noise.
    :param cfg: ground-truth boundary and noise parameters.
    """
    f = true_f(V, t, cfg)
    return np.clip(f + rng.normal(0.0, noise_sigma(f, cfg)), 0.0, 1.0)


def _scaled_alpha(var: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convert a noise variance in READING units to the units sklearn's ``alpha`` expects.

    With ``normalize_y=True`` sklearn standardizes the target before adding ``alpha`` to the Gram
    diagonal, so ``alpha`` is a variance in STANDARDIZED units. Passing a raw-units variance makes
    the assumed noise wrong by a factor of ``var(y)`` -- for this readout that understated it by
    about 2.5x in sd, i.e. the GP was substantially overconfident exactly where the boundary is.

    :param var: noise variance in reading units.
    :param y: training targets, whose spread sets the standardization.
    """
    scale = float(np.var(np.asarray(y, float)))
    return np.asarray(var, float) / (scale if scale > 1e-12 else 1.0)


# --- normalized coordinates + candidate grid --------------------------------------------
def _norm(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unit-box coordinates, with time on a LOG axis.

    Matches the coordinates the seed design is built in. The boundary is closer to isotropic in
    (V, log t) than in (V, t), and using two different geometries for the design and the learner
    would mean the sizing studies do not describe the campaign that actually runs.
    """
    x = (np.asarray(V, float) - V_LO) / (V_HI - V_LO)
    lo, hi = np.log10(T_SEARCH_LO), np.log10(T_HI)
    return np.column_stack([x, (np.log10(np.asarray(t, float)) - lo) / (hi - lo)])


def candidate_grid(nv: int = 45, nt: int = 45) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate conditions, restricted to flash times the measured table supports.

    The table has no node below ``T_SEARCH_LO``, so peak temperatures quoted there are an artifact
    of the spline; proposing shots in that gap would be proposing conditions the campaign forbids.
    """
    vs = np.linspace(V_LO, V_HI, nv)
    ts = np.geomspace(T_SEARCH_LO, T_HI, nt)
    VV, TT = np.meshgrid(vs, ts)
    return VV.ravel(), TT.ravel()


def lhs_seed(n: int, rng: Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Latin-hypercube seed over the SUPPORTED design box, stratified in (V, log t).

    Better than i.i.d. uniform: one sample per row/column stratum, so the GP sees the whole box
    before active learning starts. Returns (V, t) arrays of length n.
    """
    u = qmc.LatinHypercube(d=2, seed=rng).random(n)
    lo, hi = np.log10(T_SEARCH_LO), np.log10(T_HI)
    return V_LO + u[:, 0] * (V_HI - V_LO), 10.0 ** (lo + u[:, 1] * (hi - lo))


# --- GP fit / predict (heteroscedastic: per-point noise variance) -----------------------
def fit_gp(V: np.ndarray, t: np.ndarray, y: np.ndarray, cfg: BoundaryConfig = DEFAULT):
    """Fit a GP to the continuous readout with per-point (heteroscedastic) noise variance.

    :param V: flash voltage(s) of the measured points.
    :param t: flash time(s) of the measured points.
    :param y: measured crystalline fractions in [0, 1].
    :param cfg: boundary/noise parameters (sets the per-point noise variance).
    """
    Xn = _norm(np.asarray(V), np.asarray(t))
    y = np.asarray(y, float)

    def _kernel():
        return ConstantKernel(1.0, KERNEL_AMPLITUDE_BOUNDS) * Matern(
            list(KERNEL_LENGTHSCALE0), KERNEL_LENGTHSCALE_BOUNDS, nu=KERNEL_NU
        )

    # Pass 1: a homoscedastic fit whose only job is to produce a smoothed mean. The per-point
    # noise must NOT be read off the observation itself -- sigma_n(f) peaks at f = 1/2, so a
    # reading that strays far from the truth is assigned a SMALLER variance and the GP then trusts
    # the outlier more (corr(|y - f|, sigma_n(y)) = -0.94 at mid-transition). Taking sigma from a
    # smoothed mean instead breaks that feedback; this is the standard plug-in step for
    # heteroscedastic GP regression.
    flat = float(np.mean(noise_sigma(np.clip(y, 0, 1), cfg))) ** 2 + JITTER
    warm = GaussianProcessRegressor(
        kernel=_kernel(), alpha=_scaled_alpha(flat, y), normalize_y=True, n_restarts_optimizer=1
    ).fit(Xn, y)
    mu = np.clip(warm.predict(Xn), 0.0, 1.0)

    # Pass 2: per-point noise from the smoothed mean.
    var = noise_sigma(mu, cfg) ** 2 + JITTER
    gp = GaussianProcessRegressor(
        kernel=_kernel(),
        alpha=_scaled_alpha(var, y),
        normalize_y=True,
        n_restarts_optimizer=GP_RESTARTS,
    )
    gp.fit(Xn, y)
    return gp


def predict(gp, V: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu, s) where s is the LATENT (epistemic) std, excluding observation noise."""
    mu, s = gp.predict(_norm(np.asarray(V), np.asarray(t)), return_std=True)
    return mu, np.maximum(s, 1e-6)


# --- acquisitions (entropy-family level-set utilities) ----------------------------------
def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def acq_entropy(mu: np.ndarray, s: np.ndarray, sig_n: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Predictive class entropy (latent + noise variance) -> peaks at boundary AND high noise."""
    return _binary_entropy(norm.cdf((mu - theta) / np.sqrt(s**2 + sig_n**2)))


def noise_weighted_boundary_entropy(
    mu: np.ndarray, s: np.ndarray, sig_n: np.ndarray, theta: float = 0.5
) -> np.ndarray:
    """Boundary entropy weighted by the latent-vs-noise information ratio.

    NAMING: this is NOT BALD (Houlsby et al. 2011). BALD is the single quantity
    I(y; f) = H[E p] - E[H p]; this is a PRODUCT of the regression information gain
    0.5*log(1 + s^2/sigma_n^2) with a separate boundary entropy, which carries no
    information-theoretic interpretation. It was previously misnamed ``bald``. Kept as an
    empirical baseline only.
    """
    info = 0.5 * np.log1p(s**2 / sig_n**2)  # reducible-vs-noise information gain
    boundary = _binary_entropy(norm.cdf((mu - theta) / s))  # focus on the boundary (latent)
    return info * boundary


def latent_class_entropy(
    mu: np.ndarray, s: np.ndarray, sig_n: np.ndarray, theta: float = 0.5
) -> np.ndarray:
    """Binary class entropy from the LATENT (reducible) uncertainty only. The default.

    High where mu ~ theta and the latent class is still uncertain. It scores by reducible
    uncertainty rather than total predictive spread, so unlike predictive entropy it does not
    chase the irreducible-noise band.

    NAMING: this is NOT the LSE algorithm of Gotovos et al. (IJCAI 2013), which classifies points
    by whether their GP confidence interval clears h +/- eps and then samples the most ambiguous
    unclassified point. This is a plain latent class entropy; it was previously misnamed ``lse``.

    LIMITATION, recorded because this docstring used to claim the opposite: the utility does NOT
    self-suppress at an already-measured point. At mu = theta the argument is 0 for ANY s, so
    a(x) = ln 2 -- its global maximum -- even as the latent sd collapses to zero. Re-visiting is
    prevented by the separate spreading penalty, not by this term. Use ``straddle`` if you want an
    acquisition with the self-suppression property.
    """
    return _binary_entropy(norm.cdf((mu - theta) / s))


def straddle(
    mu: np.ndarray, s: np.ndarray, sig_n: np.ndarray, theta: float = 0.5, beta: float = 1.96
) -> np.ndarray:
    """Straddle utility ``beta*s - |mu - theta|`` (Bryan et al., NIPS 2005).

    Unlike the entropy family this decays to zero as the latent sd collapses, so it genuinely
    stops re-measuring a boundary point once that point is resolved.
    """
    return beta * s - np.abs(mu - theta)


def targeted_mse(
    mu: np.ndarray, s: np.ndarray, sig_n: np.ndarray, theta: float = 0.5
) -> np.ndarray:
    """Targeted mean-squared error: kriging variance weighted toward the threshold. THE DEFAULT.

    ``a(x) = s^2 * W(x)`` with ``W`` a Gaussian in ``mu - theta`` of width ``sqrt(s^2 + sig_n^2)``,
    so the utility is large only where the surface is BOTH uncertain and plausibly on the boundary.

    Two properties the entropy family lacks. It self-suppresses: ``s -> 0`` at a measured point
    drives the whole utility to zero, so an already-resolved boundary point cannot be chosen again
    and the spreading penalty stops being load-bearing. And the width uses the TOTAL spread
    ``s^2 + sig_n^2``, so a point whose ambiguity is pure measurement noise is down-weighted
    relative to one that more data would actually resolve.

    Chosen over straddle because the level-set benchmarking literature finds targeted-variance
    methods the strongest option specifically in low dimension, which is where this campaign sits;
    straddle remains available and is the better-known choice.
    """
    total = s**2 + sig_n**2
    weight = np.exp(-0.5 * (mu - theta) ** 2 / total) / np.sqrt(2.0 * np.pi * total)
    return s**2 * weight


# Strategy registry: acquisition name -> a(mu, s, sig_n, theta) scoring callable.
ACQUISITIONS: Dict[str, Callable[..., np.ndarray]] = {
    "predictive_entropy": acq_entropy,
    "noise_weighted": noise_weighted_boundary_entropy,
    "latent_entropy": latent_class_entropy,
    "straddle": straddle,
    "targeted_mse": targeted_mse,
}
# NOT latent_entropy. H(Phi((mu-theta)/s)) equals its maximum ln 2 on the whole predicted contour
# for ANY s -- verified identical to machine precision from s = 0.01 to s = 50 -- so the utility is
# a flat plateau along the boundary and the argmax is settled by optimizer tie-breaking rather than
# by information. That is the criterion which failed to converge on three of four benchmarks in the
# founding level-set paper (Bryan et al., NIPS 2005) and was indistinguishable from random search on
# the fourth. The spreading penalty was the only thing preventing it from re-measuring one point.
DEFAULT_ACQ = "targeted_mse"


def _spread(Vc: np.ndarray, tc: np.ndarray, Vs, ts, r: float = 0.08) -> np.ndarray:
    """Down-weight candidates near already-sampled points (uniform fidelity along the boundary)."""
    Cn, Sn = _norm(Vc, tc), _norm(np.asarray(Vs), np.asarray(ts))
    d2 = ((Cn[:, None, :] - Sn[None, :, :]) ** 2).sum(-1)
    return 1.0 - np.exp(-d2 / (2 * r**2)).max(axis=1)


# --- the active loop --------------------------------------------------------------------
def run_active(
    acq: str = DEFAULT_ACQ,
    n_seed: int = 10,
    n_iter: int = 25,
    seed: int = 0,
    cfg: BoundaryConfig = DEFAULT,
) -> Dict:
    """Seed with LHS, then pick each next shot by the acquisition. Returns the run history.

    :param acq: acquisition name (key of ``ACQUISITIONS``).
    :param n_seed: number of space-filling LHS seed points.
    :param n_iter: number of sequential active-learning picks.
    :param seed: RNG seed for reproducibility.
    :param cfg: ground-truth boundary and noise parameters.
    """
    rng = np.random.default_rng(seed)
    Vseed, tseed = lhs_seed(n_seed, rng)  # space-filling LHS seed over (V, t)
    Vs, ts = list(Vseed), list(tseed)
    ys = list(measure(np.array(Vs), np.array(ts), rng, cfg))
    Vc, tc = candidate_grid()
    f_true_grid = true_f(Vc, tc, cfg)
    true_lbl = (f_true_grid > cfg.theta).astype(int)
    hi_noise = noise_sigma(f_true_grid, cfg) > 0.6 * noise_sigma(np.array([0.5]), cfg)[0]
    acq_fn = ACQUISITIONS[acq]

    err_hist, hinoise_hist = [], []
    for _ in range(n_iter):
        gp = fit_gp(Vs, ts, ys, cfg)
        mu, s = predict(gp, Vc, tc)
        sig_n = noise_sigma(np.clip(mu, 0, 1), cfg)
        a = acq_fn(mu, s, sig_n, cfg.theta) * _spread(Vc, tc, Vs, ts)
        prob = norm.cdf((mu - cfg.theta) / np.sqrt(s**2 + sig_n**2))
        err_hist.append(float(np.mean((prob > 0.5).astype(int) != true_lbl)))
        j = int(np.argmax(a))
        Vn, tn = Vc[j], tc[j]
        Vs.append(Vn)
        ts.append(tn)
        ys.append(float(measure(np.array([Vn]), np.array([tn]), rng, cfg)[0]))
        hinoise_hist.append(hi_noise[j])

    return {
        "acq": acq,
        "V": np.array(Vs),
        "t": np.array(ts),
        "err": np.array(err_hist),
        "frac_hi_noise": float(np.mean(hinoise_hist)),
    }


# --- batch (q-per-round) proposal: greedy "qEntropy" ------------------------------------
def _conditioned_gp(kernel, V, t, y, cfg: BoundaryConfig = DEFAULT):
    """GP conditioned on (V,t,y) at FIXED (already-fitted) hyperparameters -- no re-optimize."""
    Xn = _norm(np.asarray(V), np.asarray(t))
    y = np.asarray(y, float)
    var = noise_sigma(np.clip(y, 0, 1), cfg) ** 2 + JITTER
    g = GaussianProcessRegressor(
        kernel=kernel, alpha=_scaled_alpha(var, y), optimizer=None, normalize_y=True
    )
    g.fit(Xn, y)
    return g


def propose_batch(gp, Vs, ts, ys, q, acq, Vc, tc, cfg: BoundaryConfig = DEFAULT):
    """Greedy batch of q points -- the entropy-family analogue of qUCB/qEI's sequential greedy.

    After each pick, FANTASIZE its label at the posterior mean (Kriging believer) and re-condition
    at fixed hyperparameters; the latent variance near that point collapses, so the next pick moves
    away. The GP's own correlation spreads the batch along the boundary -- no measuring in between.

    :param gp: a fitted GP (its kernel hyperparameters are reused for the conditioning steps).
    :param Vs: voltages measured so far.
    :param ts: times measured so far.
    :param ys: crystalline fractions measured so far.
    :param q: batch size (points to propose this round).
    :param acq: acquisition name (key of ``ACQUISITIONS``).
    :param Vc: candidate-grid voltages.
    :param tc: candidate-grid times.
    :param cfg: boundary/noise parameters.
    :returns: ``(Vb, tb)`` -- the q proposed (voltage, time) points.
    """
    acq_fn = ACQUISITIONS[acq]
    kernel = gp.kernel_
    Va, ta, ya = list(Vs), list(ts), list(ys)
    Vb, tb = [], []
    for _ in range(q):
        g = _conditioned_gp(kernel, Va, ta, ya, cfg)
        mu, s = predict(g, Vc, tc)
        a = acq_fn(mu, s, noise_sigma(np.clip(mu, 0, 1), cfg), cfg.theta) * _spread(Vc, tc, Va, ta)
        j = int(np.argmax(a))
        Vb.append(Vc[j])
        tb.append(tc[j])
        Va.append(Vc[j])
        ta.append(tc[j])
        ya.append(float(mu[j]))  # Kriging-believer fantasy
    return np.array(Vb), np.array(tb)


def run_active_batch(
    q: int = 4,
    n_seed: int = 10,
    n_rounds: int = 5,
    acq: str = DEFAULT_ACQ,
    seed: int = 0,
    cfg: BoundaryConfig = DEFAULT,
) -> Dict:
    """Batch active learning: LHS seed, then each round propose q points (greedy qEntropy), measure
    all q together, refit. Models beamtime, where 1-shot-per-round is infeasible.

    :param q: batch size (conditions measured together per round).
    :param n_seed: number of space-filling LHS seed points.
    :param n_rounds: number of batch rounds.
    :param acq: acquisition name (key of ``ACQUISITIONS``).
    :param seed: RNG seed for reproducibility.
    :param cfg: ground-truth boundary and noise parameters.
    :returns: history dict; ``err`` is the boundary-map error at the start of each round plus a
        final value after the last batch.
    """
    rng = np.random.default_rng(seed)
    Vseed, tseed = lhs_seed(n_seed, rng)
    Vs, ts = list(Vseed), list(tseed)
    ys = list(measure(np.array(Vs), np.array(ts), rng, cfg))
    Vc, tc = candidate_grid()
    true_lbl = (true_f(Vc, tc, cfg) > cfg.theta).astype(int)

    def boundary_err(g) -> float:
        mu, s = predict(g, Vc, tc)
        prob = norm.cdf((mu - cfg.theta) / np.sqrt(s**2 + noise_sigma(np.clip(mu, 0, 1), cfg) ** 2))
        return float(np.mean((prob > 0.5).astype(int) != true_lbl))

    err_hist, batches = [], []
    for _ in range(n_rounds):
        gp = fit_gp(Vs, ts, ys, cfg)  # refit + re-optimize on REAL data
        err_hist.append(boundary_err(gp))
        Vb, tb = propose_batch(gp, Vs, ts, ys, q, acq, Vc, tc, cfg)
        batches.append((Vb, tb))
        yb = measure(Vb, tb, rng, cfg)  # all q measured together
        Vs += list(Vb)
        ts += list(tb)
        ys += list(yb)
    err_hist.append(boundary_err(fit_gp(Vs, ts, ys, cfg)))

    return {
        "acq": acq,
        "q": q,
        "V": np.array(Vs),
        "t": np.array(ts),
        "err": np.array(err_hist),
        "batches": batches,
        "n_measured": len(ys),
    }
