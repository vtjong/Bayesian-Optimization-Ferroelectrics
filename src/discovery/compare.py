"""Chart comparison by log marginal likelihood (boss's framework §4) — GPyTorch/BoTorch.

Fit ONE GP per chart on the SAME outcomes, holding the model specification fixed (ARD over
the 2 chart dims) and letting each chart fit its own hyperparameters. Score each by the log
marginal likelihood (LML); the highest-LML chart is the coordinate system in which the
boundary is simplest = the controlling quantity.

  * continuous readout -> exact GP regression on the logit of the crystalline fraction
    (BoTorch SingleTaskGP, fit by fit_gpytorch_mll; LML from ExactMarginalLogLikelihood).
  * binary readout     -> variational GP classification (GPyTorch ApproximateGP +
    BernoulliLikelihood); the ELBO is the LML surrogate (matches the boss's SVGP classifier).

Uses the team's GPyTorch/BoTorch stack (mirrors src/fit.py), not sklearn.
LMLs become a posterior over charts with a temperature, w ~ exp(LML / tau), tau = n/10.
"""

import warnings
from typing import Dict

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy)

from .charts import ea_of_chart, tbac_family_names

warnings.filterwarnings("ignore")  # BoTorch input-scaling / convergence chatter
torch.set_default_dtype(torch.double)

_EPS = 0.02  # logit clip for continuous fractions


def _logit(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, _EPS, 1 - _EPS)
    return np.log(y / (1 - y))


def _lml_continuous(X: np.ndarray, y: np.ndarray) -> float:
    """Exact GP regression on logit(y); full log marginal likelihood."""
    tx = torch.as_tensor(X, dtype=torch.double)
    ty = torch.as_tensor(_logit(y), dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(tx, ty, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.train()
    out = model(*model.train_inputs)
    # ExactMarginalLogLikelihood normalizes by n; multiply back to a full LML
    return float(mll(out, model.train_targets).item() * len(y))


class _VarGPC(ApproximateGP):
    """Minimal variational GP classifier (RBF-ARD over 2 dims), boss's SVGP analogue."""

    def __init__(self, inducing: torch.Tensor):
        vdist = CholeskyVariationalDistribution(inducing.size(0))
        vstrat = VariationalStrategy(self, inducing, vdist,
                                     learn_inducing_locations=True)
        super().__init__(vstrat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def _lml_binary(X: np.ndarray, y: np.ndarray, epochs: int = 100) -> float:
    """Variational GP classification; ELBO as the LML surrogate (full, un-normalized)."""
    tx = torch.as_tensor(X, dtype=torch.double)
    ty = torch.as_tensor(y, dtype=torch.double)
    m = min(len(y), 64)
    idx = torch.randperm(len(y))[:m]
    model = _VarGPC(tx[idx].clone()).double()
    lik = BernoulliLikelihood().double()
    model.train()
    lik.train()
    opt = torch.optim.Adam(list(model.parameters()) + list(lik.parameters()), lr=0.1)
    mll = VariationalELBO(lik, model, num_data=len(y))
    for _ in range(epochs):
        opt.zero_grad()
        loss = -mll(model(tx), ty)
        loss.backward()
        opt.step()
    model.eval()
    lik.eval()
    with torch.no_grad():
        elbo = mll(model(tx), ty).item() * len(y)
    return float(elbo)


def chart_lml(X: np.ndarray, y: np.ndarray, readout: str) -> float:
    """Log marginal likelihood (or ELBO surrogate) of one chart's GP fit."""
    if readout == "binary":
        return _lml_binary(X, y)
    return _lml_continuous(X, y)


def compare(charts: Dict[str, np.ndarray], y: np.ndarray, readout: str) -> Dict:
    """Score every chart; return LMLs, tempered weights, winner, and recovered Ea."""
    lml = {name: chart_lml(X, y, readout) for name, X in charts.items()}
    n = len(y)
    tau = max(n / 10.0, 1.0)
    vals = np.array(list(lml.values()))
    w = np.exp((vals - vals.max()) / tau)
    w /= w.sum()
    weights = dict(zip(lml.keys(), w))

    winner = max(weights, key=weights.get)
    fam = tbac_family_names()
    best_fam = max(fam, key=lambda nm: lml[nm])
    return {
        "lml": lml,
        "weights": weights,
        "winner": winner,
        "top_weight": float(weights[winner]),
        "recovered_ea": ea_of_chart(best_fam),
        "tbac_family_won": winner in fam,
    }
