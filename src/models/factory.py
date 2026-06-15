"""GP model factory functions for kernel and model construction."""

from typing import Optional, Tuple

import gpytorch
import torch

from .gp import ExactGPModel


def make_lengthscale_prior(
    prior_type: str = "lognormal",
) -> Optional[gpytorch.priors.Prior]:
    """Build a weakly-informative ARD lengthscale prior.

    The inputs are MinMax-scaled to roughly [0, 1], so a lengthscale near
    O(1) is sensible. These priors gently regularize the marginal-likelihood
    fit away from degenerate (tiny / huge) lengthscales without hard-coding a
    value; the numerical floor is handled separately by ``min_lengthscale``.

    :param prior_type: "lognormal" (default), "gamma", or "none"
    :return: A GPyTorch prior, or None when ``prior_type == "none"``
    :raises ValueError: If ``prior_type`` is not recognized
    """
    if prior_type == "none":
        return None
    if prior_type == "lognormal":
        # Mode ~0.7, broad: weakly favors lengthscales of order one.
        return gpytorch.priors.LogNormalPrior(loc=0.0, scale=0.75)
    if prior_type == "gamma":
        # Concentration/rate chosen to put mass over [~0.2, ~3].
        return gpytorch.priors.GammaPrior(concentration=3.0, rate=3.0)
    raise ValueError(
        f"Unknown lengthscale prior: {prior_type}. "
        "Choose 'lognormal', 'gamma', or 'none'."
    )


def make_noise_prior(noise_center: float = 0.07) -> gpytorch.priors.Prior:
    """Build a weakly-informative noise-variance prior centered on measurement.

    Used when learning observation noise via a ``GaussianLikelihood``. The
    measured experimental noise std is ~0.045-0.1, i.e. a variance of roughly
    ``noise_center**2``. A GammaPrior with mean near that variance keeps the
    learned noise physically plausible on small (n~40) datasets.

    :param noise_center: Approximate noise std (default 0.07, mid of 0.045-0.1)
    :return: GammaPrior over the noise variance
    """
    var_center = noise_center**2
    # Gamma(concentration=2) mean = concentration / rate -> rate sets the mean.
    concentration = 2.0
    rate = concentration / max(var_center, 1e-6)
    return gpytorch.priors.GammaPrior(concentration=concentration, rate=rate)


def create_kernel(
    kernel_type: str,
    num_dims: int,
    lengthscale: list,
    min_lengthscale: float = 0.03,
    matern_nu: float = 0.5,
    lengthscale_prior: Optional[str] = None,
) -> gpytorch.kernels.Kernel:
    """Create GP kernel with ARD and outputscale.

    Supports RBF (infinitely smooth) and Matérn (finite smoothness) kernels
    with automatic relevance determination for feature importance learning.

    :param kernel_type: Kernel type ("rbf" or "matern")
    :param num_dims: Number of input dimensions for ARD
    :param lengthscale: Initial lengthscale per dimension
    :param min_lengthscale: Minimum lengthscale constraint
    :param matern_nu: Matérn smoothness (0.5, 1.5, or 2.5)
    :param lengthscale_prior: Optional weakly-informative ARD lengthscale prior
        ("lognormal", "gamma", or None for back-compat / no prior)
    :return: Kernel with ScaleKernel wrapper
    :raises ValueError: If kernel_type not in ["rbf", "matern"]
    """
    ls_prior = (
        make_lengthscale_prior(lengthscale_prior)
        if lengthscale_prior is not None
        else None
    )
    constraint = gpytorch.constraints.GreaterThan(min_lengthscale)

    if kernel_type == "rbf":
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=num_dims,
                lengthscale_constraint=constraint,
                lengthscale_prior=ls_prior,
            )
        )

    elif kernel_type == "matern":
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                ard_num_dims=num_dims,
                lengthscale=torch.tensor(lengthscale),
                lengthscale_constraint=constraint,
                lengthscale_prior=ls_prior,
                nu=matern_nu,
            )
        )

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Choose 'rbf' or 'matern'.")


def create_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel_type: str,
    lengthscale: list,
    noise: float = 0.1,
    num_dims: int = 2,
    min_lengthscale: float = 0.03,
    matern_nu: float = 0.5,
    learn_noise: bool = False,
    lengthscale_prior: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[gpytorch.likelihoods.Likelihood, ExactGPModel, list]:
    """Create complete GP model.

    By default uses ``FixedNoiseGaussianLikelihood`` assuming known observation
    noise from experimental measurements (faster, stable on small datasets and
    backward-compatible with existing callers). Set ``learn_noise=True`` to use
    a learned ``GaussianLikelihood`` with a weakly-informative noise prior
    centered on the measured noise level instead.

    :param train_x: Training inputs (n_samples, n_features)
    :param train_y: Training targets (n_samples,)
    :param kernel_type: Kernel type ("rbf" or "matern")
    :param lengthscale: Initial lengthscale per dimension
    :param noise: Observation noise level (std dev). When ``learn_noise`` is
        True this seeds the noise prior center and initial value.
    :param num_dims: Number of input dimensions
    :param min_lengthscale: Minimum lengthscale constraint
    :param matern_nu: Matérn smoothness parameter
    :param learn_noise: If True, learn noise via GaussianLikelihood + prior;
        if False (default), use fixed homoscedastic noise (back-compat)
    :param lengthscale_prior: Optional ARD lengthscale prior ("lognormal",
        "gamma", or None). None preserves the original prior-free behavior.
    :param verbose: Print the kernel configuration when True
    :return: (likelihood, model, lengthscale)
    """
    # Create kernel
    kernel = create_kernel(
        kernel_type=kernel_type,
        num_dims=num_dims,
        lengthscale=lengthscale,
        min_lengthscale=min_lengthscale,
        matern_nu=matern_nu,
        lengthscale_prior=lengthscale_prior,
    )

    if verbose:
        print(f"Kernel configuration: {kernel}")

    if learn_noise:
        # Learned homoscedastic noise with a prior centered on measurement.
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=make_noise_prior(noise_center=noise),
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
        )
        likelihood.noise = max(noise**2, 1e-4)
    else:
        # Create fixed noise likelihood (constant across all observations)
        num_samples = len(train_x)
        noise_tensor = noise * torch.ones(num_samples)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise_tensor)

    # Create GP model
    model = ExactGPModel(train_x, train_y, likelihood, kernel)

    return likelihood, model, lengthscale


# Backward compatibility aliases
kernel_func = create_kernel
make_model = create_gp_model
