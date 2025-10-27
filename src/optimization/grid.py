"""Grid generation utilities for Gaussian Process parameter space.

This module creates candidate search grids for Bayesian Optimization.
The grid is NOT a held-out test set, but rather a discretized search
space used for acquisition function optimization and visualization.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class CandidateGrid:
    """Immutable container for candidate search grid.

    :ivar candidate_grid: Grid points for each parameter dimension
    :ivar candidate_points: Cartesian product of all candidate points
    """

    candidate_grid: torch.Tensor
    candidate_points: torch.Tensor

    # Backward compatibility aliases
    @property
    def test_grid(self) -> torch.Tensor:
        """Legacy name for candidate_grid."""
        return self.candidate_grid

    @property
    def test_arr(self) -> torch.Tensor:
        """Legacy name for candidate_points."""
        return self.candidate_points


def create_candidate_grid(train_x: torch.Tensor, num_points: int = 30) -> CandidateGrid:
    """Create candidate search grid for Bayesian Optimization.

    This is the main entry point for grid generation. It creates a dense
    discretization of the parameter space for acquisition function
    evaluation and GP visualization.

    :param train_x: Observed training inputs of shape
        (n_samples, n_features)
    :param num_points: Number of grid points per dimension
    :return: CandidateGrid containing grid and all point combinations
    """
    num_params = train_x.size(dim=1)

    # Extract bounds from observed data
    grid_bounds = [
        (train_x[:, i].min().item(), train_x[:, i].max().item()) for i in range(num_params)
    ]

    # Build 1D grids for each parameter dimension
    candidate_grid = _build_parameter_grid(num_points, num_params, grid_bounds)

    # Generate all combinations for acquisition function evaluation
    grid_dims = tuple(candidate_grid[:, i] for i in range(num_params))
    candidate_points = torch.cartesian_prod(*grid_dims)

    return CandidateGrid(candidate_grid, candidate_points)


def _compute_grid_spacing(param_min: float, param_max: float, grid_size: int) -> float:
    """Compute spacing between grid points with padding.

    Adds buffer zone beyond observed data bounds to allow exploration
    of slightly extrapolated parameter regions.

    :param param_min: Minimum observed value for the parameter
    :param param_max: Maximum observed value for the parameter
    :param grid_size: Total number of grid points
    :return: Grid spacing value for padding beyond min/max bounds
    """
    return (param_max - param_min) / (grid_size - 2)


def _build_parameter_grid(
    grid_size: int,
    num_params: int,
    grid_bounds: List[Tuple[float, float]],
) -> torch.Tensor:
    """Build grid spanning parameter space with padding.

    Creates a uniform grid for each parameter dimension, extending
    slightly beyond observed data bounds to enable exploration.

    :param grid_size: Number of points per parameter dimension
    :param num_params: Number of parameters (dimensions)
    :param grid_bounds: List of (min, max) tuples for each parameter
    :return: Tensor of shape (grid_size, num_params) with grid points
    """
    grid = torch.zeros(grid_size, num_params)

    for param_idx in range(num_params):
        param_min, param_max = grid_bounds[param_idx]
        spacing = _compute_grid_spacing(param_min, param_max, grid_size)

        # Create linearly spaced points with padding on both ends
        grid[:, param_idx] = torch.linspace(
            param_min - spacing,
            param_max + spacing,
            grid_size,
        )

    return grid


# Backward compatibility wrapper
class Grid:
    """Legacy class wrapper for backward compatibility.

    Deprecated: Use create_candidate_grid() function instead.
    """

    def __init__(self, train_x: torch.Tensor, num_points: int = 30) -> None:
        """Initialize Grid with training data bounds.

        :param train_x: Observed training inputs
        :param num_points: Number of grid points per parameter dimension
        """
        result = create_candidate_grid(train_x, num_points)
        self.candidate_grid = result.candidate_grid
        self.candidate_points = result.candidate_points
        self.test_grid = result.test_grid
        self.test_arr = result.test_arr
