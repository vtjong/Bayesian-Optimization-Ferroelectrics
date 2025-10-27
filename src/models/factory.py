"""GP model factory functions for kernel and model construction."""

from typing import Tuple

import gpytorch
import torch

from .gp import ExactGPModel


def create_kernel(
    kernel_type: str,
    num_dims: int,
    lengthscale: list,
    min_lengthscale: float = 0.03,
    matern_nu: float = 0.5,
) -> gpytorch.kernels.Kernel:
    """Create GP kernel with ARD and outputscale.

    Supports RBF (infinitely smooth) and Matérn (finite smoothness) kernels
    with automatic relevance determination for feature importance learning.

    :param kernel_type: Kernel type ("rbf" or "matern")
    :param num_dims: Number of input dimensions for ARD
    :param lengthscale: Initial lengthscale per dimension
    :param min_lengthscale: Minimum lengthscale constraint
    :param matern_nu: Matérn smoothness (0.5, 1.5, or 2.5)
    :return: Kernel with ScaleKernel wrapper
    :raises ValueError: If kernel_type not in ["rbf", "matern"]
    """
    if kernel_type == "rbf":
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_dims))

    elif kernel_type == "matern":
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                ard_num_dims=num_dims,
                lengthscale=torch.tensor(lengthscale),
                lengthscale_constraint=gpytorch.constraints.GreaterThan(min_lengthscale),
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
) -> Tuple[gpytorch.likelihoods.Likelihood, ExactGPModel, list]:
    """Create complete GP model with fixed noise likelihood.

    Uses FixedNoiseGaussianLikelihood assuming known observation noise
    from experimental measurements. Faster and more stable than learning
    noise for small datasets.

    :param train_x: Training inputs (n_samples, n_features)
    :param train_y: Training targets (n_samples,)
    :param kernel_type: Kernel type ("rbf" or "matern")
    :param lengthscale: Initial lengthscale per dimension
    :param noise: Observation noise level (std dev)
    :param num_dims: Number of input dimensions
    :param min_lengthscale: Minimum lengthscale constraint
    :param matern_nu: Matérn smoothness parameter
    :return: (likelihood, model, lengthscale)
    """
    # Create kernel
    kernel = create_kernel(
        kernel_type=kernel_type,
        num_dims=num_dims,
        lengthscale=lengthscale,
        min_lengthscale=min_lengthscale,
        matern_nu=matern_nu,
    )

    print(f"Kernel configuration: {kernel}")

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
