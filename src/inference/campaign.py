"""In-silico closed-loop experimental-campaign simulator (tests the boss's round plan).

Synthetic ground truth over (V, t): crystallinity (an onset boundary) and orthorhombic
fraction (a process WINDOW — a band with a lower onset and an upper over-budget edge). A
GP surrogate is refit each round and a level-set "straddle" acquisition proposes the next
batch, mapping the polar-phase window. We compare active learning vs random vs
space-filling baselines and sweep batch/round schedules — testing the campaign BEFORE any
XRD budget is spent. (PROTOTYPE: single-output window target; multi-output + the
mechanism-discrimination layer reuse inference.power_study.)
"""

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

torch.set_default_dtype(torch.double)


def true_world(V, t):
    """Ground truth: (crystallinity onset, orthorhombic-fraction window) over (V, t)."""
    K = (V ** 2) * (0.4 + 0.6 * t)  # effective thermal budget (single-pulse warp of V,t)
    cryst = 1.0 / (1.0 + np.exp(-(K - 0.33) / 0.05))
    ortho = cryst * np.exp(-((K - 0.55) / 0.16) ** 2)  # Goldilocks window in K
    return cryst, ortho


def _grid(n):
    g = np.linspace(0, 1, n)
    vv, tt = np.meshgrid(g, g)
    return np.column_stack([vv.ravel(), tt.ravel()])


def window_f1(pred, true, thr=0.5):
    """F1 of the predicted polar-phase window vs the true window (overlap accuracy)."""
    p, tgt = pred > thr, true > thr
    tp, fp, fn = np.sum(p & tgt), np.sum(p & ~tgt), np.sum(~p & tgt)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 1.0


def _fit(X, y):
    gp = SingleTaskGP(torch.tensor(X), torch.tensor(y).unsqueeze(-1),
                      outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    return gp


def _predict(gp, X):
    with torch.no_grad():
        post = gp.posterior(torch.tensor(X))
        mu = post.mean.squeeze(-1).numpy()
        sd = post.variance.squeeze(-1).clamp_min(1e-9).sqrt().numpy()
    return mu, sd


def _straddle_batch(gp, cand, q, thr=0.5, kappa=1.96, min_dist=0.09):
    """Greedy level-set batch: high uncertainty near the window contour, spread out."""
    mu, sd = _predict(gp, cand)
    score = kappa * sd - np.abs(mu - thr)
    avail = np.ones(len(cand), bool)
    chosen = []
    for _ in range(q):
        i = int(np.argmax(np.where(avail, score, -np.inf)))
        chosen.append(i)
        avail &= np.linalg.norm(cand - cand[i], axis=1) > min_dist
        if not avail.any():
            break
    return cand[chosen]


def run_campaign(strategy="active", q=4, n_rounds=8, n0=8, seed=0,
                 noise=0.04, cand_n=28, eval_n=60):
    """Run one campaign; return history of (n_experiments, window_F1) over rounds."""
    rng = np.random.default_rng(seed)
    eval_X = _grid(eval_n)
    true_ortho = true_world(eval_X[:, 0], eval_X[:, 1])[1]
    cand = _grid(cand_n)
    sobol = qmc.Sobol(d=2, seed=seed)

    X = qmc.LatinHypercube(d=2, seed=seed).random(n0)  # seed round
    y = np.clip(true_world(X[:, 0], X[:, 1])[1] + rng.normal(0, noise, len(X)), 0, 1)

    history = []
    for _ in range(n_rounds):
        gp = _fit(X, y)
        mu, _ = _predict(gp, eval_X)
        history.append((len(X), window_f1(mu, true_ortho)))
        if strategy == "active":
            new_x = _straddle_batch(gp, cand, q)
        elif strategy == "random":
            new_x = rng.uniform(0, 1, (q, 2))
        else:  # space-filling
            new_x = sobol.random(q)
        new_y = np.clip(true_world(new_x[:, 0], new_x[:, 1])[1]
                        + rng.normal(0, noise, len(new_x)), 0, 1)
        X, y = np.vstack([X, new_x]), np.concatenate([y, new_y])

    gp = _fit(X, y)
    mu, _ = _predict(gp, eval_X)
    history.append((len(X), window_f1(mu, true_ortho)))
    return history


def average_runs(strategy, q, n_rounds, reps=4, **kw):
    """Average window-F1 over seeds at each experiment count."""
    hs = [run_campaign(strategy, q, n_rounds, seed=s, **kw) for s in range(reps)]
    n_exp = [h[0] for h in hs[0]]
    f1 = np.mean([[step[1] for step in h] for h in hs], axis=0)
    return np.array(n_exp), f1


def experiments_to_target(n_exp, f1, target=0.8):
    """First experiment count reaching the target F1, or None."""
    hits = [n for n, v in zip(n_exp, f1) if v >= target]
    return int(hits[0]) if hits else None
