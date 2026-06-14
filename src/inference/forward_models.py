"""Candidate kinetic forward models for the design-stage power study (PROTOTYPE).

Each model maps a normalized design coordinate ``u in [0, 1]`` (a stand-in for thermal
budget / annealing condition) to crystallinity ``alpha in [0, 1]`` with 2 parameters and
a uniform prior box. The two models have genuinely different *shapes* (curvature), which
is what Bayesian model comparison must exploit.

PROTOTYPE SCOPE (see plan REVISION 1 #2/#8): 1-D design + Gaussian likelihood. The
production version uses the real JMAK/Avrami (nucleation- vs growth-limited) and
single/dual-Ea Arrhenius forms evaluated over a transient T(t), with a bounded
(Beta / logit-normal) heteroscedastic likelihood and nested-sampling evidence.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class KineticModel:
    """A candidate kinetic model with a vectorized predictor and a uniform prior box."""

    name: str
    predict: Callable  # predict(u: (D,), theta: (...,2)) -> alpha: (...,D)
    prior_lo: np.ndarray  # shape (2,)
    prior_hi: np.ndarray  # shape (2,)
    param_names: Tuple[str, str]


def _jmak(u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """JMAK/Avrami: alpha = 1 - exp(-(k*u)^n). theta = (k, n)."""
    k = theta[..., 0:1]
    n = theta[..., 1:2]
    return 1.0 - np.exp(-np.power(np.clip(k * u, 1e-9, None), n))


def _gompertz(u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gompertz (asymmetric sigmoid): alpha = exp(-b*exp(-c*u)). theta = (b, c)."""
    b = theta[..., 0:1]
    c = theta[..., 1:2]
    return np.exp(-b * np.exp(-c * u))


JMAK = KineticModel("JMAK", _jmak, np.array([0.5, 1.0]), np.array([3.0, 4.0]), ("k", "n"))
GOMPERTZ = KineticModel("Gompertz", _gompertz, np.array([1.0, 2.0]), np.array([6.0, 8.0]), ("b", "c"))

MODELS: Dict[str, KineticModel] = {"jmak": JMAK, "gompertz": GOMPERTZ}
