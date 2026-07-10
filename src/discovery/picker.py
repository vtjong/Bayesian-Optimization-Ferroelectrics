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
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import warnings

from .synthetic import T_HI, T_LO, V_HI, V_LO, tmax

warnings.filterwarnings("ignore", category=ConvergenceWarning)

THETA = 0.5                      # crystallization threshold (boundary = level set f = THETA)
T_STAR = 390.0                   # boundary Tmax = T_STAR: MEASURED flash-lamp crystallization
                                 # onset (flash T50=388 C, RTA 357 C); see src/run_calibration.py
_SHARP = 0.12                    # transition sharpness: ~37 C 10-90% width, matching the
                                 # measured onset (flash ~11 C, RTA ~40 C); run_calibration.py
_S0, _S1 = 0.02, 0.30            # heteroscedastic noise: sigma_n = S0 + S1 * f*(1-f)  (worst at f=0.5)


# --- ground truth (fixed function of (V,t)); the picker never sees it -------------------
def true_f(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """True crystalline fraction: a sigmoid of (Tmax - T_STAR). Boundary is re-entrant."""
    return 1.0 / (1.0 + np.exp(-_SHARP * (tmax(V, t) - T_STAR)))


def noise_sigma(f: np.ndarray) -> np.ndarray:
    """Heteroscedastic observation-noise std: largest at the boundary (f = 0.5)."""
    return _S0 + _S1 * f * (1.0 - f)


def measure(V: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy continuous readout at (V,t) (e.g. a permittivity-derived crystalline fraction)."""
    f = true_f(V, t)
    return np.clip(f + rng.normal(0.0, noise_sigma(f)), 0.0, 1.0)


# --- normalized coordinates for the GP --------------------------------------------------
def _norm(V, t):
    return np.column_stack([(V - V_LO) / (V_HI - V_LO), (t - T_LO) / (T_HI - T_LO)])


def candidate_grid(nv=45, nt=45):
    vs = np.linspace(V_LO, V_HI, nv)
    ts = np.linspace(T_LO, T_HI, nt)
    VV, TT = np.meshgrid(vs, ts)
    return VV.ravel(), TT.ravel()


def lhs_seed(n, rng):
    """Latin-hypercube seed over the (V, t) design box (space-filling; low discrepancy).

    Better than i.i.d. uniform: one sample per row/column stratum, so the GP sees the whole box
    before active learning starts. Returns (V, t) arrays of length n.
    """
    u = qmc.LatinHypercube(d=2, seed=rng).random(n)
    return V_LO + u[:, 0] * (V_HI - V_LO), T_LO + u[:, 1] * (T_HI - T_LO)


# --- GP fit / predict (heteroscedastic: per-point noise variance) -----------------------
def fit_gp(V, t, y):
    """Fit a GP to the continuous readout with per-point (heteroscedastic) noise variance."""
    Xn = _norm(np.asarray(V), np.asarray(t))
    var = noise_sigma(np.clip(y, 0, 1)) ** 2 + 1e-4        # noise var estimated from the reading
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF([0.15, 0.15], (0.03, 1.0))
    gp = GaussianProcessRegressor(kernel=k, alpha=var, normalize_y=True,
                                  n_restarts_optimizer=1)
    gp.fit(Xn, np.asarray(y))
    return gp


def predict(gp, V, t):
    """Return (mu, s) where s is the LATENT (epistemic) std, excluding observation noise."""
    mu, s = gp.predict(_norm(np.asarray(V), np.asarray(t)), return_std=True)
    return mu, np.maximum(s, 1e-6)


# --- acquisitions -----------------------------------------------------------------------
def _binary_entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def acq_entropy(mu, s, sig_n):
    """Predictive class entropy (latent + noise variance) -> peaks at boundary AND high noise."""
    p = norm.cdf((mu - THETA) / np.sqrt(s ** 2 + sig_n ** 2))
    return _binary_entropy(p)


def acq_bald(mu, s, sig_n):
    """Boundary-weighted information gain about the latent function (down-weights noise)."""
    info = 0.5 * np.log1p(s ** 2 / sig_n ** 2)             # reducible-vs-noise information gain
    boundary = _binary_entropy(norm.cdf((mu - THETA) / s))  # focus on the boundary (latent)
    return info * boundary


def acq_lse(mu, s, sig_n):
    """Level-set entropy on the LATENT (reducible) uncertainty: boundary-focused AND
    noise-aware. High where mu ~ theta and the LATENT class is still uncertain; unlike
    predictive entropy it stops re-chasing a boundary point once its reducible uncertainty
    is resolved, and unlike BALD it does not flee the boundary. A principled, robust default for
    noisy boundary mapping -- comparable to entropy/BALD in practice (see module docstring)."""
    return _binary_entropy(norm.cdf((mu - THETA) / s))


def _spread(Vc, tc, Vs, ts, r=0.08):
    """Down-weight candidates near already-sampled points (uniform fidelity along the boundary)."""
    Cn, Sn = _norm(Vc, tc), _norm(np.asarray(Vs), np.asarray(ts))
    d2 = ((Cn[:, None, :] - Sn[None, :, :]) ** 2).sum(-1)
    return 1.0 - np.exp(-d2 / (2 * r ** 2)).max(axis=1)


# --- the active loop --------------------------------------------------------------------
def run_active(acq: str = "lse", n_seed=10, n_iter=25, seed=0) -> Dict:
    """Seed with LHS, then pick each next shot by the acquisition. Returns history."""
    rng = np.random.default_rng(seed)
    Vseed, tseed = lhs_seed(n_seed, rng)     # space-filling LHS seed over (V, t)
    Vs, ts = list(Vseed), list(tseed)
    ys = list(measure(np.array(Vs), np.array(ts), rng))
    Vc, tc = candidate_grid()
    f_true_grid = true_f(Vc, tc)
    true_lbl = (f_true_grid > THETA).astype(int)
    hi_noise = noise_sigma(f_true_grid) > 0.6 * noise_sigma(np.array([0.5]))[0]

    err_hist, hinoise_hist = [], []
    prob_prev = None
    conv_count = 0
    for it in range(n_iter):
        gp = fit_gp(Vs, ts, ys)
        mu, s = predict(gp, Vc, tc)
        sig_n = noise_sigma(np.clip(mu, 0, 1))
        a = {"entropy": acq_entropy, "bald": acq_bald, "lse": acq_lse}[acq](mu, s, sig_n)
        a = a * _spread(Vc, tc, Vs, ts)                    # spreading
        # boundary-map error (misclassification area) + convergence probe
        prob = norm.cdf((mu - THETA) / np.sqrt(s ** 2 + sig_n ** 2))
        err_hist.append(float(np.mean((prob > 0.5).astype(int) != true_lbl)))
        if prob_prev is not None:
            conv_count = conv_count + 1 if np.mean(np.abs(prob - prob_prev)) < 0.02 else 0
        prob_prev = prob
        # pick next shot
        j = int(np.argmax(a))
        Vn, tn = Vc[j], tc[j]
        Vs.append(Vn); ts.append(tn); ys.append(float(measure(np.array([Vn]), np.array([tn]), rng)[0]))
        hinoise_hist.append(hi_noise[j])

    return {
        "acq": acq, "V": np.array(Vs), "t": np.array(ts),
        "err": np.array(err_hist),
        "frac_hi_noise": float(np.mean(hinoise_hist)),
        "converged_iter": None,
    }


# --- batch (q-per-round) proposal: greedy "qEntropy" ------------------------------------
def _conditioned_gp(kernel, V, t, y):
    """GP conditioned on (V,t,y) at FIXED (already-fitted) hyperparameters -- no re-optimize."""
    Xn = _norm(np.asarray(V), np.asarray(t))
    var = noise_sigma(np.clip(np.asarray(y), 0, 1)) ** 2 + 1e-4
    g = GaussianProcessRegressor(kernel=kernel, alpha=var, optimizer=None, normalize_y=True)
    g.fit(Xn, np.asarray(y))
    return g


def propose_batch(gp, Vs, ts, ys, q, acq, Vc, tc):
    """Greedy batch of q points -- the entropy-family analogue of qUCB/qEI's sequential greedy.

    After each pick, FANTASIZE its label at the posterior mean (Kriging believer) and re-condition
    at fixed hyperparameters; the latent variance near that point collapses, so the next pick moves
    away. The GP's own correlation spreads the batch along the boundary -- no measuring in between.
    Returns q (V, t) points to run together this round.
    """
    acq_fn = {"entropy": acq_entropy, "bald": acq_bald, "lse": acq_lse}[acq]
    kernel = gp.kernel_
    Va, ta, ya = list(Vs), list(ts), list(ys)
    Vb, tb = [], []
    for _ in range(q):
        g = _conditioned_gp(kernel, Va, ta, ya)
        mu, s = predict(g, Vc, tc)
        a = acq_fn(mu, s, noise_sigma(np.clip(mu, 0, 1))) * _spread(Vc, tc, Va, ta)
        j = int(np.argmax(a))
        Vb.append(Vc[j]); tb.append(tc[j])
        Va.append(Vc[j]); ta.append(tc[j]); ya.append(float(mu[j]))   # Kriging-believer fantasy
    return np.array(Vb), np.array(tb)


def run_active_batch(q=4, n_seed=10, n_rounds=5, acq="lse", seed=0) -> Dict:
    """Batch active learning: LHS seed, then each round propose q points (greedy qEntropy), MEASURE
    ALL q together, refit. Models beamtime, where 1-shot-per-round is infeasible. `err` is the
    boundary-map error at the start of each round plus a final value after the last batch."""
    rng = np.random.default_rng(seed)
    Vseed, tseed = lhs_seed(n_seed, rng)
    Vs, ts = list(Vseed), list(tseed)
    ys = list(measure(np.array(Vs), np.array(ts), rng))
    Vc, tc = candidate_grid()
    true_lbl = (true_f(Vc, tc) > THETA).astype(int)

    def boundary_err(g):
        mu, s = predict(g, Vc, tc)
        prob = norm.cdf((mu - THETA) / np.sqrt(s ** 2 + noise_sigma(np.clip(mu, 0, 1)) ** 2))
        return float(np.mean((prob > 0.5).astype(int) != true_lbl))

    err_hist, batches = [], []
    for _ in range(n_rounds):
        gp = fit_gp(Vs, ts, ys)                       # refit + re-optimize on REAL data
        err_hist.append(boundary_err(gp))
        Vb, tb = propose_batch(gp, Vs, ts, ys, q, acq, Vc, tc)
        batches.append((Vb, tb))
        yb = measure(Vb, tb, rng)                     # all q measured together
        Vs += list(Vb); ts += list(tb); ys += list(yb)
    err_hist.append(boundary_err(fit_gp(Vs, ts, ys)))

    return {
        "acq": acq, "q": q, "V": np.array(Vs), "t": np.array(ts),
        "err": np.array(err_hist), "batches": batches, "n_measured": len(ys),
    }
