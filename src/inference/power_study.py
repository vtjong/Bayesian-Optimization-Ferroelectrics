"""Design-stage Bayesian power / preposterior study — the go/no-go gate (REVISION 1 #1).

For each candidate model taken as the TRUTH, simulate data at a given sample size n and
noise sigma (optionally with thermal-calibration error injected into the design
coordinate), compute the log10 Bayes factor of the true vs the competing model, and
report the probability of correctly selecting the generator above a decision threshold.

This answers the panel's #1 feasibility question BEFORE any XRD budget is spent:
*can ~20-40 noisy points discriminate these mechanism shapes at all?*
"""

from typing import Dict, List

import numpy as np

from .evidence import log_evidence, make_param_grid
from .forward_models import MODELS


def _design(n: int, rng: np.random.Generator, cal_err: float) -> np.ndarray:
    """n design points in u-space; cal_err mimics thermal-model error in the coordinate."""
    u = np.linspace(0.05, 0.95, n)
    if cal_err > 0:
        u = np.clip(u + rng.normal(0.0, cal_err, n), 0.01, 0.99)
    return u


def run_power_study(
    n_list=(20, 30, 40),
    sigma_list=(0.045, 0.1),
    reps: int = 200,
    cal_err: float = 0.03,
    threshold_log10bf: float = 1.0,
    grid_n: int = 50,
    seed: int = 0,
) -> List[Dict]:
    """Return rows of {true_model, n, sigma, p_correct, median_log10bf}.

    p_correct = fraction of replicates where log10 BF(true vs other) > threshold.
    """
    rng = np.random.default_rng(seed)
    models = list(MODELS.values())
    grids = {m.name: make_param_grid(m, grid_n) for m in models}
    rows: List[Dict] = []

    for truth in models:
        other = next(m for m in models if m.name != truth.name)
        lo, hi = truth.prior_lo, truth.prior_hi
        for n in n_list:
            for sigma in sigma_list:
                bfs = np.empty(reps)
                for r in range(reps):
                    theta_true = rng.uniform(lo, hi)
                    u = _design(n, rng, cal_err)
                    alpha = np.clip(truth.predict(u, theta_true), 1e-4, 1 - 1e-4)
                    y = np.clip(alpha + rng.normal(0.0, sigma, n), 0.0, 1.0)
                    z_true = log_evidence(truth, u, y, sigma, grids[truth.name])
                    z_other = log_evidence(other, u, y, sigma, grids[other.name])
                    bfs[r] = (z_true - z_other) / np.log(10)
                rows.append({
                    "true_model": truth.name,
                    "n": n,
                    "sigma": sigma,
                    "p_correct": float(np.mean(bfs > threshold_log10bf)),
                    "median_log10bf": float(np.median(bfs)),
                })
    return rows


def min_n_for_power(rows: List[Dict], target: float = 0.8) -> Dict:
    """Smallest n reaching target p_correct, per (true_model, sigma); None if never."""
    out: Dict = {}
    keys = {(r["true_model"], r["sigma"]) for r in rows}
    for tm, sig in sorted(keys):
        ns = sorted(r["n"] for r in rows
                    if r["true_model"] == tm and r["sigma"] == sig and r["p_correct"] >= target)
        out[f"{tm}@sigma={sig}"] = (ns[0] if ns else None)
    return out
