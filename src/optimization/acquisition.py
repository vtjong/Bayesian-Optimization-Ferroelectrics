"""Bayesian Optimization acquisition function utilities."""

from typing import Dict, List, Tuple

import gpytorch
import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler


def optimize_acquisition_function(
    acq_function,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = 20,
    raw_samples: int = 900,
) -> torch.Tensor:
    """Optimize acquisition function to find next candidate points.

    :param acq_function: BoTorch acquisition function
    :param bounds: Parameter bounds [[lower], [upper]]
    :param q: Number of candidates to generate jointly
    :param num_restarts: Number of optimization restarts
    :param raw_samples: Number of raw samples for initialization
    :return: Optimized candidate points
    """
    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={},
    )
    return candidates


def suggest_next_experiments_analytic(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    beta: float = 5.0,
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Suggest next experiments using analytic acquisition functions.

    Uses Expected Improvement (EI), Probability of Improvement (PI),
    and Upper Confidence Bound (UCB) to suggest single candidate point.

    :param model: Trained GP model
    :param likelihood: Trained likelihood
    :param train_y: Training targets (for computing best observation)
    :param bounds: Parameter bounds [[lower], [upper]]
    :param beta: UCB exploration parameter
    :return: Dict mapping acquisition function name to (candidate, predicted_value)
    """
    model.eval()
    likelihood.eval()

    y_best = train_y.max()

    # Create acquisition functions
    acq_functions = {
        "EI": ExpectedImprovement(model, y_best),
        "PI": ProbabilityOfImprovement(model, y_best),
        "UCB": UpperConfidenceBound(model, beta),
    }

    suggestions = {}
    for name, acq_func in acq_functions.items():
        # Optimize acquisition function
        candidate = optimize_acquisition_function(acq_func, bounds, q=1)

        # Get predicted value
        with torch.no_grad():
            pred_mean = likelihood(model(candidate)).mean.item()

        suggestions[name] = (candidate.numpy(), pred_mean)

    return suggestions


def suggest_next_experiments_mc(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    q: int = 4,
    beta: float = 5.0,
    seed: int = 1,
    acq_functions: List[str] = ["qEI", "qPI", "qUCB"],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Suggest next experiments using Monte Carlo acquisition functions.

    Uses qEI, qPI, qUCB to suggest multiple candidate points jointly.
    MC sampling enables batch optimization of acquisition functions.

    :param model: Trained GP model
    :param likelihood: Trained likelihood
    :param train_y: Training targets
    :param bounds: Parameter bounds
    :param q: Number of candidates to suggest jointly
    :param beta: UCB exploration parameter
    :param seed: Random seed for MC sampling
    :param acq_functions: List of acquisition function names to use
    :return: Dict mapping function name to (candidates, predicted_values)
    """
    model.eval()
    likelihood.eval()

    y_best = train_y.max()

    # Setup MC sampler
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]), seed=seed)

    # Create acquisition functions
    acq_func_map = {
        "qEI": qExpectedImprovement(model, y_best, sampler=sampler),
        "qPI": qProbabilityOfImprovement(model, y_best, sampler=sampler),
        "qUCB": qUpperConfidenceBound(model, beta, sampler=sampler),
    }

    suggestions = {}
    for name in acq_functions:
        if name not in acq_func_map:
            continue

        acq_func = acq_func_map[name]

        # Optimize to find q candidates jointly
        candidates = optimize_acquisition_function(acq_func, bounds, q=q)

        # Get predicted values for each candidate
        with torch.no_grad():
            pred_means = likelihood(model(candidates)).mean.numpy()

        suggestions[name] = (candidates.numpy(), pred_means)

    return suggestions


def format_suggestions(
    suggestions: Dict,
    scaler,
    feature_names: List[str] = None,
) -> None:
    """Pretty-print acquisition function suggestions.

    :param suggestions: Dict from suggest_next_experiments_*
    :param scaler: Fitted scaler to inverse transform candidates
    :param feature_names: Names of input features
    """
    if feature_names is None:
        feature_names = ["Feature 1", "Feature 2"]

    print("\n" + "=" * 70)
    print("SUGGESTED NEXT EXPERIMENTS")
    print("=" * 70)

    for acq_name, (candidates, predictions) in suggestions.items():
        print(f"\n{acq_name}:")

        # Handle single vs batch candidates
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
            predictions = [predictions]

        # Inverse transform to original scale
        candidates_original = scaler.inverse_transform(torch.from_numpy(candidates).float()).numpy()

        for i, (cand, pred) in enumerate(zip(candidates_original, predictions)):
            print(f"  Candidate {i + 1}:")
            for fname, val in zip(feature_names, cand):
                print(f"    {fname}: {val:.3f}")
            print(f"    Predicted FOM: {pred:.3f}")
