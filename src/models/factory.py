"""Gaussian Process model utilities for kernel and model construction.

This module provides factory functions for creating GP models with
different kernel configurations (RBF, Matern) and fixed noise likelihoods.
"""

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
    """Create a GP kernel with specified hyperparameters.

    ML Reasoning:
    -------------
    Kernel Choice Impacts GP Behavior:
    - RBF (Radial Basis Function): Infinitely differentiable, smooth
      predictions. Good for well-behaved functions but can oversmooth.
    - Matern: Finite differentiability controlled by nu parameter:
      * nu=0.5: Once differentiable (similar to exponential kernel)
      * nu=1.5: Twice differentiable (good for most physical systems)
      * nu=2.5: Smoother, five times differentiable

    ARD (Automatic Relevance Determination):
    - Separate lengthscale per dimension allows learning feature importance
    - Small lengthscale = sensitive to that dimension (important feature)
    - Large lengthscale = insensitive to that dimension (less important)

    Lengthscale Constraints:
    - Minimum lengthscale prevents overfitting to noise
    - Too small: model fits every wiggle (high variance)
    - Too large: model underfit (high bias)

    :param kernel_type: Kernel type ("rbf" or "matern")
    :type kernel_type: str
    :param num_dims: Number of input dimensions for ARD
    :type num_dims: int
    :param lengthscale: Initial lengthscale values (one per dimension)
    :type lengthscale: list
    :param min_lengthscale: Minimum allowed lengthscale constraint
    :type min_lengthscale: float
    :param matern_nu: Smoothness parameter for Matern kernel
        (0.5, 1.5, or 2.5)
    :type matern_nu: float
    :return: Configured kernel with outputscale and ARD lengthscales
    :rtype: gpytorch.kernels.Kernel
    :raises ValueError: If kernel_type is not "rbf" or "matern"
    """
    if kernel_type == "rbf":
        # RBF kernel: exp(-0.5 * ||x-x'||² / lengthscale²)
        # Infinitely differentiable - produces very smooth functions
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_dims))

    elif kernel_type == "matern":
        # Matern kernel with ARD lengthscales
        # nu controls smoothness: 0.5 (rough) < 1.5 < 2.5 (smooth)
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                ard_num_dims=num_dims,
                lengthscale=torch.tensor(lengthscale),
                lengthscale_constraint=gpytorch.constraints.GreaterThan(min_lengthscale),
                nu=matern_nu,
            )
        )

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. " f"Choose 'rbf' or 'matern'.")


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
    """Create a complete GP model with likelihood and kernel.

    ML Reasoning:
    -------------
    Fixed Noise Likelihood:
    - We use FixedNoiseGaussianLikelihood instead of learning noise
    - Assumption: Observation noise is known/estimated from experiments
    - Advantages: Faster optimization, more stable for small datasets
    - Disadvantage: If noise estimate is wrong, calibration suffers

    Model Components:
    - Likelihood: Models observation noise (fixed at specified level)
    - Kernel: Models correlation structure in latent function
    - Mean: Constant mean (learned during training)

    Why Print Kernel:
    - Sanity check hyperparameters before training
    - Verify lengthscale constraints are applied
    - Debug if model behaves unexpectedly

    :param train_x: Training inputs of shape (n_samples, n_features)
    :type train_x: torch.Tensor
    :param train_y: Training targets of shape (n_samples,)
    :type train_y: torch.Tensor
    :param kernel_type: Type of kernel ("rbf" or "matern")
    :type kernel_type: str
    :param lengthscale: Initial lengthscale values
    :type lengthscale: list
    :param noise: Observation noise level (standard deviation)
    :type noise: float
    :param num_dims: Number of input dimensions
    :type num_dims: int
    :param min_lengthscale: Minimum lengthscale constraint
    :type min_lengthscale: float
    :param matern_nu: Matern kernel smoothness parameter
    :type matern_nu: float
    :return: Tuple of (likelihood, model, lengthscale) where likelihood
        is FixedNoiseGaussianLikelihood, model is ExactGPModel, and
        lengthscale is the initial lengthscale list
    :rtype: Tuple[gpytorch.likelihoods.Likelihood, ExactGPModel, list]
    """
    # Create kernel with specified hyperparameters
    kernel = create_kernel(
        kernel_type=kernel_type,
        num_dims=num_dims,
        lengthscale=lengthscale,
        min_lengthscale=min_lengthscale,
        matern_nu=matern_nu,
    )

    # Print kernel configuration for debugging
    print(kernel)

    # Create fixed noise likelihood
    # Noise is constant across all observations
    num_samples = len(train_x)
    noise_tensor = noise * torch.ones(num_samples)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise_tensor)

    # Create GP model
    model = ExactGPModel(train_x, train_y, likelihood, kernel)

    return likelihood, model, lengthscale


# Backward compatibility aliases
kernel_func = create_kernel
make_model = create_gp_model
