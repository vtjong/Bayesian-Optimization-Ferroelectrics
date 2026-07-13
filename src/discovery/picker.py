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

The simulation's ground-truth boundary and noise level are carried in a frozen ``BoundaryConfig``
that is passed explicitly (dependency injection) -- there is no mutable module state, so the
simulation is reentrant and the noise level is an argument rather than a hidden global.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from .synthetic import T_HI, T_LO, V_HI, V_LO, tmax

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class BoundaryConfig:
    """Ground-truth boundary + heteroscedastic-noise parameters for the synthetic study.

    Grouping these in one immutable object (instead of mutable module globals that callers
    reassign) makes the simulation reentrant and the noise level an explicit argument.

    :param theta: crystallization threshold; the boundary is the level set ``f = theta``.
    :param t_star: onset temperature (deg C); the true fraction is a sigmoid of ``Tmax - t_star``.
    :param sharp: sigmoid sharpness (1/deg C); ``0.12`` gives a ~37 C 10-90% transition width.
    :param noise_floor: baseline observation-noise std ``s0``.
    :param noise_boundary: extra noise scale at the boundary ``s1``; ``sigma_n = s0 + s1 f (1-f)``.
    """

    theta: float = 0.5
    t_star: float = 380.0
    sharp: float = 0.12
    noise_floor: float = 0.02
    noise_boundary: float = 0.30


DEFAULT = BoundaryConfig()


# --- ground truth (fixed function of (V,t)); the picker never sees it -------------------
def true_f(V: np.ndarray, t: np.ndarray, cfg: BoundaryConfig = DEFAULT) -> np.ndarray:
    """True crystalline fraction: a sigmoid of ``(Tmax - t_star)``. Boundary is re-entrant."""
    return 1.0 / (1.0 + np.exp(-cfg.sharp * (tmax(V, t) - cfg.t_star)))


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


# --- normalized coordinates + candidate grid --------------------------------------------
def _norm(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack([(V - V_LO) / (V_HI - V_LO), (t - T_LO) / (T_HI - T_LO)])


def candidate_grid(nv: int = 45, nt: int = 45) -> Tuple[np.ndarray, np.ndarray]:
    vs = np.linspace(V_LO, V_HI, nv)
    ts = np.linspace(T_LO, T_HI, nt)
    VV, TT = np.meshgrid(vs, ts)
    return VV.ravel(), TT.ravel()


def lhs_seed(n: int, rng: Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Latin-hypercube seed over the (V, t) design box (space-filling; low discrepancy).

    Better than i.i.d. uniform: one sample per row/column stratum, so the GP sees the whole box
    before active learning starts. Returns (V, t) arrays of length n.
    """
    u = qmc.LatinHypercube(d=2, seed=rng).random(n)
    return V_LO + u[:, 0] * (V_HI - V_LO), T_LO + u[:, 1] * (T_HI - T_LO)


# --- GP fit / predict (heteroscedastic: per-point noise variance) -----------------------
def fit_gp(V: np.ndarray, t: np.ndarray, y: np.ndarray, cfg: BoundaryConfig = DEFAULT):
    """Fit a GP to the continuous readout with per-point (heteroscedastic) noise variance.

    :param V: flash voltage(s) of the measured points.
    :param t: flash time(s) of the measured points.
    :param y: measured crystalline fractions in [0, 1].
    :param cfg: boundary/noise parameters (sets the per-point noise variance).
    """
    Xn = _norm(np.asarray(V), np.asarray(t))
    var = noise_sigma(np.clip(y, 0, 1), cfg) ** 2 + 1e-4  # noise var estimated from the reading
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF([0.15, 0.15], (0.03, 1.0))
    gp = GaussianProcessRegressor(kernel=k, alpha=var, normalize_y=True, n_restarts_optimizer=1)
    gp.fit(Xn, np.asarray(y))
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


def acq_bald(mu: np.ndarray, s: np.ndarray, sig_n: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Boundary-weighted information gain about the latent function (down-weights noise)."""
    info = 0.5 * np.log1p(s**2 / sig_n**2)  # reducible-vs-noise information gain
    boundary = _binary_entropy(norm.cdf((mu - theta) / s))  # focus on the boundary (latent)
    return info * boundary


def acq_lse(mu: np.ndarray, s: np.ndarray, sig_n: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Level-set entropy on the LATENT (reducible) uncertainty: boundary-focused AND noise-aware.

    High where mu ~ theta and the LATENT class is still uncertain; unlike predictive entropy it
    stops re-chasing a boundary point once its reducible uncertainty is resolved, and unlike BALD
    it does not flee the boundary. A principled, robust default for noisy boundary mapping --
    comparable to entropy/BALD in practice (see module docstring).
    """
    return _binary_entropy(norm.cdf((mu - theta) / s))


# Strategy registry: acquisition name -> a(mu, s, sig_n, theta) scoring callable.
ACQUISITIONS: Dict[str, Callable[..., np.ndarray]] = {
    "entropy": acq_entropy,
    "bald": acq_bald,
    "lse": acq_lse,
}


def _spread(Vc: np.ndarray, tc: np.ndarray, Vs, ts, r: float = 0.08) -> np.ndarray:
    """Down-weight candidates near already-sampled points (uniform fidelity along the boundary)."""
    Cn, Sn = _norm(Vc, tc), _norm(np.asarray(Vs), np.asarray(ts))
    d2 = ((Cn[:, None, :] - Sn[None, :, :]) ** 2).sum(-1)
    return 1.0 - np.exp(-d2 / (2 * r**2)).max(axis=1)


# --- the active loop --------------------------------------------------------------------
def run_active(
    acq: str = "lse",
    n_seed: int = 10,
    n_iter: int = 25,
    seed: int = 0,
    cfg: BoundaryConfig = DEFAULT,
) -> Dict:
    """Seed with LHS, then pick each next shot by the acquisition. Returns the run history.

    :param acq: acquisition name (key of ``ACQUISITIONS``): ``"lse"``, ``"entropy"``, or ``"bald"``.
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
    var = noise_sigma(np.clip(np.asarray(y), 0, 1), cfg) ** 2 + 1e-4
    g = GaussianProcessRegressor(kernel=kernel, alpha=var, optimizer=None, normalize_y=True)
    g.fit(Xn, np.asarray(y))
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
    acq: str = "lse",
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
