"""GP model evaluation utilities."""

from typing import Tuple

import gpytorch
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate GP model on test data.

    Returns predictions, uncertainties, and performance metrics.

    :param model: Trained GP model
    :param likelihood: Trained likelihood
    :param test_x: Test inputs
    :param test_y: Test targets
    :return: (predictions, std_devs, metrics_dict)
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        f_pred = model(test_x)
        y_pred = likelihood(f_pred)

        y_pred_mean = y_pred.mean.numpy()
        y_pred_std = torch.sqrt(f_pred.variance).numpy()

    # Compute metrics
    test_y_np = test_y.numpy()
    rmse = np.sqrt(mean_squared_error(test_y_np, y_pred_mean))
    mae = mean_absolute_error(test_y_np, y_pred_mean)
    r2 = r2_score(test_y_np, y_pred_mean)
    spearman = spearmanr(test_y_np, y_pred_mean)[0]

    metrics = {"RMSE": rmse, "MAE": mae, "R² score": r2, "Spearman": spearman}

    return y_pred_mean, y_pred_std, metrics


def predict_on_grid(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    grid_x: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions on a grid for visualization.

    :param model: Trained GP model
    :param likelihood: Trained likelihood
    :param grid_x: Grid points (n_grid_points, n_features)
    :return: (mean_predictions, std_devs)
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        f_pred = model(grid_x)
        y_pred = likelihood(f_pred)

        y_mean = y_pred.mean.numpy()
        y_std = torch.sqrt(f_pred.variance).numpy()

    return y_mean, y_std
