"""Adapter: build a :class:`PhaseMapResult` from a trained GP.

This is the ONLY place in the visualization package that touches the GP / torch stack,
keeping :mod:`visualization.phase_map` dependent on numpy arrays alone (Dependency
Inversion). Reuses the existing grid + prediction utilities
(:func:`optimization.grid.create_candidate_grid`, :func:`evaluator.predict_on_grid`)
and the fitted scaler to express the map in physical units.
"""

from typing import Optional

import numpy as np
import torch

from optimization.grid import create_candidate_grid

from .phase_map import PhaseMapResult


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _physical_axis(scaler, scaled_1d: np.ndarray, index: int, n_dims: int) -> np.ndarray:
    """Map a 1-D scaled grid for one feature back to physical units.

    MinMax scaling is independent per feature, so the other columns are dummies.
    """
    arr = np.zeros((len(scaled_1d), n_dims), dtype=float)
    arr[:, index] = np.asarray(scaled_1d, dtype=float)
    phys = _to_numpy(scaler.inverse_transform(arr))
    return phys[:, index]


def build_phase_map_result(
    model,
    likelihood,
    train_x: torch.Tensor,
    scaler,
    *,
    num_points: int = 60,
    threshold: Optional[float] = None,
    x_index: int = 0,
    y_index: int = 1,
    x_label: str = "Pulse Time (ms)",
    y_label: str = "Voltage (V)",
    value_label: str = "FOM",
    train_y: Optional[torch.Tensor] = None,
) -> PhaseMapResult:
    """Predict the GP over a 2-D grid and package it as a :class:`PhaseMapResult`.

    :param model: Trained GP model (inputs in scaled [0,1] space)
    :param likelihood: Trained likelihood
    :param train_x: Scaled training inputs (n_samples, n_dims) — sets the grid bounds
    :param scaler: Fitted ``TorchMinMaxScaler`` used to label axes in physical units
    :param num_points: Grid resolution per axis
    :param threshold: Optional level-set value drawn as the boundary contour
    :param x_index, y_index: Which input dims map to the x/y axes
    :param x_label, y_label, value_label: Axis / colorbar labels
    :param train_y: Optional targets, attached for point coloring/reference
    :return: A populated :class:`PhaseMapResult` (physical-unit axes, (ny,nx) grids)
    """
    grid_1d = create_candidate_grid(train_x, num_points=num_points).candidate_grid
    n_dims = grid_1d.shape[1]
    gx = _to_numpy(grid_1d[:, x_index])
    gy = _to_numpy(grid_1d[:, y_index])
    xx, yy = np.meshgrid(gx, gy)  # (num_points, num_points)

    # Full scaled input grid; any non-(x,y) dims held at their training-column mean.
    col_means = _to_numpy(train_x).mean(axis=0)
    flat = np.tile(col_means, (xx.size, 1)).astype(float)
    flat[:, x_index] = xx.ravel()
    flat[:, y_index] = yy.ravel()

    # Predict via posterior() so this works for BOTH ExactGPModel and a warped
    # SingleTaskGP (which un-standardizes in posterior(), not in forward()).
    model.eval()
    with torch.no_grad():
        post = model.posterior(torch.from_numpy(flat).float())
    mean = _to_numpy(post.mean.squeeze(-1)).reshape(xx.shape)
    std = _to_numpy(post.variance.squeeze(-1).clamp_min(1e-12).sqrt()).reshape(xx.shape)

    obs = _to_numpy(scaler.inverse_transform(train_x))
    obs_value = _to_numpy(train_y) if train_y is not None else None

    return PhaseMapResult(
        x_coords=_physical_axis(scaler, gx, x_index, n_dims),
        y_coords=_physical_axis(scaler, gy, y_index, n_dims),
        mean=mean,
        std=std,
        x_label=x_label,
        y_label=y_label,
        value_label=value_label,
        threshold=threshold,
        obs_x=obs[:, x_index],
        obs_y=obs[:, y_index],
        obs_value=obs_value,
    )
