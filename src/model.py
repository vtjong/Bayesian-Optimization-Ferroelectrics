"""Gaussian Process model for Bayesian Optimization.

This module defines a GP model compatible with both GPyTorch and BoTorch,
enabling exact GP inference for small-to-medium datasets and integration
with BoTorch's acquisition functions for Bayesian Optimization.
"""

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    """Exact Gaussian Process model with customizable kernel.

    This model combines:
    - GPyTorch's ExactGP for efficient exact inference
    - BoTorch's GPyTorchModel interface for acquisition functions
    - Constant mean prior (can be learned during training)
    - User-specified kernel (e.g., Matern, RBF, etc.)

    The model is suitable for:
    - Small to medium datasets (< 10,000 points)
    - Exact posterior inference (no approximations)
    - Single-output regression tasks
    - Bayesian Optimization workflows

    Example:
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood, kernel)

        # Training
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Prediction
        model.eval()
        with torch.no_grad():
            posterior = model(test_x)

    :ivar mean_module: Constant mean function
    :ivar covar_module: Kernel/covariance function
    """

    _num_outputs = 1  # Single-output GP

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel,
    ) -> None:
        """Initialize the Exact GP model.

        :param train_x: Training inputs of shape (n_samples, n_features)
        :param train_y: Training targets of shape (n_samples,)
        :param likelihood: Likelihood function (e.g., GaussianLikelihood)
        :param kernel: Covariance kernel (e.g., MaternKernel, RBFKernel)
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        # Prior mean (constant, but trainable)
        self.mean_module = gpytorch.means.ConstantMean()

        # Covariance function (kernel)
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the GP prior distribution at input points.

        This defines the forward pass through the GP model, computing
        the predictive mean and covariance at the given input locations.

        :param x: Input locations of shape (n_points, n_features) or
            (batch_size, n_points, n_features)
        :return: Multivariate normal distribution representing the
            GP prior at the input locations
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def num_outputs(self) -> int:
        """Get the number of outputs.

        :return: Number of outputs (always 1 for this model)
        """
        return self._num_outputs


# Backward compatibility alias
GPModel = ExactGPModel
