"""Functional regression of crystalline fraction on the full temperature trace.

Each shot is represented by its time-temperature OCCUPANCY histogram g(T): the time the trace
spends in each temperature bin. The controlling quantity is then a weighted integral

    phi-integral = sum_b phi(T_b) g(T_b),

a LINEAR functional of g with weighting phi(T). A smoothness-regularized linear regression of
logit(crystalline fraction) on g recovers phi(T) directly. This GENERALIZES the discrete charts
(each chart is a fixed phi: Tmax = weight on the top bin; dwell = a step; TBac(Ea) = an Arrhenius
shape) and can recover an UNLISTED weighting -- e.g. a temperature WINDOW [T1,T2] -- that the
discrete menu cannot represent. Single pulse: the trace is set by (V,t); we use its full shape.
"""

from typing import Tuple

import numpy as np

from .synthetic import READOUT_SIGMA, _rank, _trace, sample_design

# Temperature bins (deg C) spanning the reachable range.
TEMP_BINS = np.linspace(25.0, 800.0, 41)
TEMP_CENTERS = 0.5 * (TEMP_BINS[:-1] + TEMP_BINS[1:])


def occupancy(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Time-temperature occupancy histogram per shot (ms in each temperature bin)."""
    n, nb = len(V), len(TEMP_CENTERS)
    G = np.zeros((n, nb))
    for i, (vi, ti) in enumerate(zip(V, t)):
        s, T = _trace(float(vi), float(ti))
        dt = np.gradient(s)
        idx = np.digitize(T, TEMP_BINS) - 1
        ok = (idx >= 0) & (idx < nb)
        np.add.at(G[i], idx[ok], dt[ok])
    return G


def phi_window(V: np.ndarray, t: np.ndarray, lo: float = 500.0, hi: float = 560.0) -> np.ndarray:
    """An UNLISTED controlling quantity: time the trace spends in the window [lo, hi] (deg C).

    Not representable as Tmax, dwell-above-a-single-threshold, or TBac(Ea): it is band-limited
    (upper AND lower bounded), and is non-monotonic in Tmax (a very hot shot races through the
    window, a moderate shot lingers in it).
    """
    G = occupancy(V, t)
    mask = (TEMP_CENTERS >= lo) & (TEMP_CENTERS <= hi)
    return G[:, mask].sum(axis=1)


def make_window_dataset(
    n: int,
    readout: str,
    rng: np.random.Generator,
    lo: float = 500.0,
    hi: float = 560.0,
    k: float = 40.0,
):
    """(V, t, y) where crystallization is controlled by time-in-window [lo, hi]."""
    V, t = sample_design(n, rng)
    r = _rank(phi_window(V, t, lo, hi))
    p = 1.0 / (1.0 + np.exp(-k * (r - 0.5)))
    sigma = READOUT_SIGMA[readout]
    if sigma is None:
        y = (rng.uniform(size=n) < p).astype(float)
    else:
        y = np.clip(p + rng.normal(0.0, sigma, n), 0.0, 1.0)
    return V, t, y


def fit_weighting(
    G: np.ndarray, y: np.ndarray, lam: float = 50.0, eps: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Smoothness-regularized regression of logit(y) on occupancy -> learned weighting.

    Solves (X'X + lam D'D) c = X'y_logit with D a 2nd-difference operator on the weighting
    (adjacent temperature bins should weight similarly). Returns the STANDARDIZED per-bin
    weights (stable; the learned importance of each temperature bin), the per-bin occupancy
    std (use to mask rarely-visited bins for plotting), and the fit R2.
    """
    yl = np.log(np.clip(y, eps, 1 - eps) / (1 - np.clip(y, eps, 1 - eps)))
    sd = G.std(0) + 1e-9
    Gs = (G - G.mean(0)) / sd
    X = np.column_stack([np.ones(len(y)), Gs])
    nb = Gs.shape[1]
    D = np.zeros((nb - 2, nb))
    for i in range(nb - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    P = np.zeros((nb + 1, nb + 1))
    P[1:, 1:] = lam * (D.T @ D)
    c = np.linalg.solve(X.T @ X + P, X.T @ yl)
    pred = X @ c
    r2 = 1.0 - np.sum((yl - pred) ** 2) / np.sum((yl - yl.mean()) ** 2)
    return c[1:], sd, float(r2)  # standardized per-bin weights, occupancy std, fit R2
