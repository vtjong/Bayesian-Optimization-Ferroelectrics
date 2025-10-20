"""Thompson Sampling acquisition for Bayesian Optimization.

This module implements Thompson Sampling as an acquisition strategy for
Bayesian Optimization. Unlike traditional acquisition functions (EI, UCB),
Thompson Sampling draws samples from the GP posterior and optimizes those
samples directly, providing a principled exploration-exploitation tradeoff.

Adapted from: https://botorch.org/tutorials/thompson_sampling
"""

import time
from contextlib import ExitStack
from typing import Literal, Tuple

import gpytorch.settings as gpts
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.generation import MaxPosteriorSampling
from torch.quasirandom import SobolEngine


class ThompsonSampler:
    """Thompson Sampling for GP-based Bayesian Optimization.

    This class implements Thompson Sampling with multiple sampling
    strategies for posterior draws (Cholesky, CIQ, Lanczos). It uses
    BoTorch's MaxPosteriorSampling to find points that maximize sampled
    functions from the GP posterior.

    The sampler supports:
    - Exact sampling (Cholesky decomposition)
    - Contour Integral Quadrature (CIQ) for scalability
    - Lanczos iterations for approximate sampling
    - Sobol sequences for quasi-random candidate generation

    Example:
        sampler = ThompsonSampler(model, likelihood, scaler, seed=42)
        X_suggested, Y_predicted = sampler.run_optimization(
            n_cands=900,
            n_init=1,
            max_evals=4,
            batch_size=1,
            sampler="ciq"
        )

    :ivar model: Trained GP model
    :ivar likelihood: GP likelihood function
    :ivar scaler: Data scaler for inverse transform
    :ivar seed: Random seed for reproducibility
    :ivar dim: Dimensionality of the parameter space
    """

    def __init__(
        self,
        model,
        likelihood,
        scaler,
        seed: int,
        dim: int = 2,
    ) -> None:
        """Initialize Thompson Sampler.

        :param model: Fitted Gaussian Process model
        :param likelihood: GP likelihood (e.g., GaussianLikelihood)
        :param scaler: Scaler for transforming back to original space
        :param seed: Random seed for Sobol sequence generation
        :param dim: Number of input dimensions
        """
        self.model = model
        self.likelihood = likelihood
        self.scaler = scaler
        self.seed = seed
        self.dim = dim

    def inverse_transform_candidates(self, candidates: torch.Tensor) -> np.ndarray:
        """Transform scaled candidates back to original parameter space.

        :param candidates: Scaled candidate points
        :return: Candidates in original space, rounded to 2 decimals
        """
        original_space = self.scaler.inverse_transform(candidates)
        return np.round(original_space, 2)

    def predict_mean(self, candidates: torch.Tensor, batch_size: int) -> np.ndarray:
        """Predict mean values for candidate points.

        :param candidates: Input points to predict
        :param batch_size: Number of candidates (1 for single point)
        :return: Predicted mean values, rounded to 2 decimals
        """
        with torch.no_grad():
            posterior = self.likelihood(self.model(candidates))
            y_pred_mean = posterior.mean.detach().numpy()

        if batch_size == 1:
            return np.round(y_pred_mean.item(), 2)
        return np.round(y_pred_mean, 2)

    def generate_sobol_candidates(self, n_points: int) -> torch.Tensor:
        """Generate quasi-random candidates using Sobol sequences.

        Sobol sequences provide better space-filling properties than
        pure random sampling, leading to more efficient exploration.

        :param n_points: Number of candidate points to generate
        :return: Tensor of shape (n_points, dim) in [0, 1]^dim
        """
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        return sobol.draw(n=n_points)

    def generate_batch(
        self,
        n_candidates: int,
        batch_size: int,
        sampler: Literal["cholesky", "ciq", "lanczos", "rff"] = "ciq",
    ) -> torch.Tensor:
        """Generate batch of points using Thompson Sampling.

        This method:
        1. Generates candidate points via Sobol sequences
        2. Configures GPyTorch settings for the chosen sampler
        3. Draws posterior samples and finds their maxima

        Sampler options:
        - cholesky: Exact sampling (slow for large n)
        - ciq: Contour Integral Quadrature (scalable)
        - lanczos: Lanczos iterations (approximate)
        - rff: Random Fourier Features (fast approximate)

        :param n_candidates: Number of candidate points to evaluate
        :param batch_size: Number of points to select
        :param sampler: Sampling strategy to use
        :return: Selected candidate points of shape (batch_size, dim)
        """
        # Generate candidate points
        X_candidates = self.generate_sobol_candidates(n_candidates)

        # Configure GPyTorch settings based on sampler type
        with ExitStack() as es:
            if sampler == "cholesky":
                # Exact sampling via Cholesky decomposition
                es.enter_context(gpts.max_cholesky_size(float("inf")))

            elif sampler == "ciq":
                # Contour Integral Quadrature sampling
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(True))
                es.enter_context(gpts.minres_tolerance(2e-3))
                es.enter_context(gpts.num_contour_quadrature(15))

            elif sampler == "lanczos":
                # Lanczos iterations for approximate sampling
                es.enter_context(
                    gpts.fast_computations(
                        covar_root_decomposition=True,
                        log_prob=True,
                        solves=True,
                    )
                )
                es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(False))

            elif sampler == "rff":
                # Random Fourier Features
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

            # Draw samples and find maxima
            with torch.no_grad():
                thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
                X_next = thompson_sampling(X_candidates, num_samples=batch_size)

        return X_next

    def run_optimization(
        self,
        n_cands: int,
        n_init: int,
        max_evals: int,
        batch_size: int,
        sampler: Literal["cholesky", "ciq", "lanczos", "rff"] = "ciq",
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Run Thompson Sampling optimization loop.

        Iteratively generates batches of candidate points, evaluates them
        via the GP model, and tracks the best values found.

        :param n_cands: Number of candidates to generate per iteration
        :param n_init: Number of initial random points
        :param max_evals: Maximum number of evaluations to perform
        :param batch_size: Points to select per iteration
        :param sampler: Sampling strategy (cholesky, ciq, lanczos, rff)
        :return: Tuple of (suggested_points, predicted_values) where
            suggested_points are in original parameter space and
            predicted_values are GP predictions
        """
        # Initialize with random Sobol points
        X = self.generate_sobol_candidates(n_init)
        Y = torch.tensor([self.predict_mean(x.unsqueeze(0), 1) for x in X])
        print(f"{len(X)}) Best value: {Y.max().item():.2e}")

        # Iteratively generate batches
        while len(X) < max_evals:
            start = time.monotonic()
            X_next = self.generate_batch(n_cands, batch_size, sampler)
            end = time.monotonic()
            print(f"Generated batch in {end - start:.3f} seconds")

            # Evaluate new candidates
            Y_next = torch.tensor([self.predict_mean(x.unsqueeze(0), 1) for x in X_next])

            # Append to history
            X = torch.cat((X, X_next), dim=0)
            Y = torch.cat((Y, Y_next), dim=0)

            print(f"{len(X)}) Best value: {Y.max().item():.2e}")

        # Transform back to original space
        X_original = self.inverse_transform_candidates(X)
        return X_original, Y

    def visualize_convergence(
        self,
        Y_values: torch.Tensor,
        optimum: float,
        n_candidates: int,
        max_evals: int,
        sampler_name: str = "CIQ",
    ) -> None:
        """Visualize Thompson Sampling convergence.

        Plots the cumulative maximum (best value found so far) over
        the course of optimization iterations.

        :param Y_values: Predicted values at each iteration
        :param optimum: Known global optimum (for comparison)
        :param n_candidates: Number of candidates used per iteration
        :param max_evals: Total number of evaluations
        :param sampler_name: Name of sampler for legend
        """
        fig = plt.figure(figsize=(10, 8))
        matplotlib.rcParams.update({"font.size": 20})
        fig.add_subplot(1, 1, 1)

        # Plot cumulative maximum
        cumulative_max = Y_values.cummax(dim=0)[0]
        iterations = 1 + np.arange(len(cumulative_max))

        plt.plot(
            iterations[0::2],
            cumulative_max[0::2],
            c="g",
            marker="*",
            linestyle="-",
            markersize=12,
            label=f"{sampler_name}-{n_candidates}",
        )

        # Plot global optimum reference line
        plt.plot(
            [0, max_evals],
            [optimum, optimum],
            "k--",
            lw=3,
            label="Global optimal value",
        )

        plt.xlabel("Number of evaluations", fontsize=18)
        plt.ylabel("Best value found", fontsize=18)
        plt.title("Thompson Sampling Convergence", fontsize=24)
        plt.xlim([0, max_evals])
        plt.ylim([0, 5])
        plt.grid(True)
        plt.tight_layout()
        plt.legend(loc="lower right", ncol=1, fontsize=18)
        plt.show()


# Backward compatibility alias
ThompsonSampling = ThompsonSampler
