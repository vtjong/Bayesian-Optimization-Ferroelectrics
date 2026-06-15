"""Marginal-likelihood GP fitting via BoTorch (multi-restart L-BFGS).

Provides :func:`fit_gp_mll`, the principled primary fitter for the project's
exact GP models. It optimizes the exact marginal log-likelihood with BoTorch's
``fit_gpytorch_mll`` (multi-restart L-BFGS), which is more robust and far
cheaper than the hand-rolled 3000-epoch Adam loop in :mod:`trainer` (kept as a
fallback). It also exposes an opt-in learnable input warp (BoTorch
``Warp``), mirroring the reference pattern in ``src/run_warp_demo.py``.

This module is additive: existing callers that use ``trainer.train_gp_model``
are unaffected.
"""

from typing import List, Optional, Tuple

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from models.factory import create_gp_model, create_kernel, make_noise_prior


def fit_gp_mll(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
) -> Tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.Likelihood]:
    """Fit a GP by maximizing the exact marginal log-likelihood.

    Uses BoTorch's ``fit_gpytorch_mll`` (multi-restart L-BFGS) as the primary,
    principled fitter. This replaces the need to hand-tune a long Adam loop and
    is robust to the initialization of kernel hyperparameters.

    :param model: GP model to fit (in-place)
    :param likelihood: The model's likelihood
    :return: (fitted_model, fitted_likelihood) in eval mode
    """
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    likelihood.eval()
    return model, likelihood


def build_and_fit_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel_type: str = "matern",
    lengthscale: Optional[List[float]] = None,
    noise: float = 0.07,
    matern_nu: float = 0.5,
    min_lengthscale: float = 0.03,
    learn_noise: bool = True,
    lengthscale_prior: Optional[str] = "lognormal",
    warp_dims: Optional[List[int]] = None,
    verbose: bool = False,
) -> Tuple[gpytorch.likelihoods.Likelihood, gpytorch.models.GP]:
    """Construct and marginal-likelihood-fit a GP in one call.

    Convenience wrapper that builds a model via
    :func:`models.factory.create_gp_model`, optionally wraps the inputs in a
    learnable Kumaraswamy warp, and fits with :func:`fit_gp_mll`. Intended as
    the principled default path; the legacy Adam loop remains available in
    :mod:`trainer`.

    :param train_x: Training inputs (n_samples, n_features), scaled to [0, 1]
    :param train_y: Training targets (n_samples,)
    :param kernel_type: "matern" or "rbf"
    :param lengthscale: Initial per-dim lengthscale (defaults to ones)
    :param noise: Noise std (prior center if learned, fixed value otherwise)
    :param matern_nu: Matern smoothness (0.5, 1.5, 2.5)
    :param min_lengthscale: Numerical lengthscale floor
    :param learn_noise: Learn noise via GaussianLikelihood + prior when True
    :param lengthscale_prior: ARD lengthscale prior ("lognormal"/"gamma"/None)
    :param warp_dims: Optional list of input dims to warp (opt-in); None = off
    :param verbose: Forward verbosity to the factory
    :return: (fitted_likelihood, fitted_model) in eval mode
    """
    num_dims = train_x.shape[-1]
    if lengthscale is None:
        lengthscale = [1.0] * num_dims

    if warp_dims is not None:
        # Warp path: use BoTorch's SingleTaskGP, which correctly transforms both
        # train and test inputs through the learnable Warp (mirrors
        # run_warp_demo.py). Our custom ExactGPModel does not swap cached train
        # inputs on eval(), so we delegate warped models to SingleTaskGP.
        kernel = create_kernel(
            kernel_type=kernel_type,
            num_dims=num_dims,
            lengthscale=lengthscale,
            min_lengthscale=min_lengthscale,
            matern_nu=matern_nu,
            lengthscale_prior=lengthscale_prior,
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=make_noise_prior(noise_center=noise),
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
        )
        model = SingleTaskGP(
            train_x,
            train_y.unsqueeze(-1),
            likelihood=likelihood,
            covar_module=kernel,
            input_transform=Warp(d=num_dims, indices=warp_dims),
            outcome_transform=Standardize(m=1),
        )
        model, likelihood = fit_gp_mll(model, likelihood)
        return likelihood, model

    likelihood, model, _ = create_gp_model(
        train_x=train_x,
        train_y=train_y,
        kernel_type=kernel_type,
        lengthscale=lengthscale,
        noise=noise,
        num_dims=num_dims,
        min_lengthscale=min_lengthscale,
        matern_nu=matern_nu,
        learn_noise=learn_noise,
        lengthscale_prior=lengthscale_prior,
        verbose=verbose,
    )

    model, likelihood = fit_gp_mll(model, likelihood)
    return likelihood, model
