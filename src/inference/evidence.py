"""Grid-quadrature marginal likelihood for low-dimensional kinetic models (PROTOTYPE).

Computes the evidence p(D|M) by integrating the (Gaussian) likelihood over a uniform
prior grid. For 2-parameter models this is accurate and fast and deliberately avoids the
Laplace approximation the review panel flagged as unreliable at small n. The production
pipeline uses nested sampling (dynesty/UltraNest) per plan REVISION 1 #3; grid quadrature
is the right tool here because a *design-stage power study* needs a trustworthy, unbiased
evidence at n~20-40 across thousands of synthetic replicates.
"""

import numpy as np
from scipy.special import logsumexp

from .forward_models import KineticModel


def make_param_grid(model: KineticModel, n_per_dim: int = 50) -> np.ndarray:
    """Uniform grid over the model's prior box, shape (n_per_dim**2, 2)."""
    g0 = np.linspace(model.prior_lo[0], model.prior_hi[0], n_per_dim)
    g1 = np.linspace(model.prior_lo[1], model.prior_hi[1], n_per_dim)
    gg0, gg1 = np.meshgrid(g0, g1)
    return np.column_stack([gg0.ravel(), gg1.ravel()])


def log_evidence(
    model: KineticModel,
    u: np.ndarray,
    y: np.ndarray,
    sigma: float,
    theta_grid: np.ndarray,
) -> float:
    """log p(D|M) under a Gaussian likelihood with known sigma and a uniform prior grid.

    With a uniform prior over the grid, Z = mean_g lik(theta_g), so
    log Z = logsumexp(loglik) - log(n_grid).
    """
    pred = model.predict(u, theta_grid)  # (n_grid, D)
    resid = (y[None, :] - pred) / sigma
    loglik = -0.5 * np.sum(resid ** 2, axis=1) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))
    return float(logsumexp(loglik) - np.log(theta_grid.shape[0]))


def log10_bayes_factor(
    model_a: KineticModel,
    model_b: KineticModel,
    u: np.ndarray,
    y: np.ndarray,
    sigma: float,
    grids: dict,
) -> float:
    """log10 BF in favor of model_a over model_b for data (u, y)."""
    za = log_evidence(model_a, u, y, sigma, grids[model_a.name])
    zb = log_evidence(model_b, u, y, sigma, grids[model_b.name])
    return (za - zb) / np.log(10)
