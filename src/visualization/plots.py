"""Visualization and evaluation utilities for Gaussian Process models.

This module provides plotting functions for:
- 3D surface visualizations of GP predictions
- Acquisition function visualizations
- Training/test performance metrics
- Prediction vs. ground truth comparisons
- Training loss curves
"""

from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import DEFAULT_STYLE

# Plotting constants (styling from the shared PlotStyle; see visualization.base).
DEFAULT_COLOR = DEFAULT_STYLE.accent_color
DEFAULT_COLORSCALE = "Burg"  # plotly colorscale (interactive 3D)
DEFAULT_FONT_SIZE = DEFAULT_STYLE.title_fontsize
DEFAULT_DPI = 200  # interactive preview DPI (PNGs use save_dpi)
CONFIDENCE_INTERVAL_ALPHA = DEFAULT_STYLE.grid_alpha
CONFIDENCE_INTERVAL_COLOR = "grey"


def configure_plot_style() -> None:
    """Configure matplotlib with aesthetically pleasing settings.

    Sets the style, line width, DPI, and font size for all subsequent
    matplotlib plots. This should be called once at the beginning of
    a plotting session.
    """
    plt.style.use("bmh")
    mpl.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = DEFAULT_DPI
    plt.rcParams["font.size"] = DEFAULT_FONT_SIZE


def visualize_gp_surface(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_grid: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """Visualize GP predictions as a 3D surface with training data.

    Creates an interactive 3D plot showing the GP's predicted surface
    over the parameter space, overlaid with the observed training data
    points.

    :param train_x: Training inputs of shape (n_samples, 2)
    :param train_y: Training targets of shape (n_samples,)
    :param test_grid: Grid of test points of shape (n_grid, 2)
    :param predictions: GP predictions at grid points
    """
    # Create GP surface
    fig = go.Figure(
        data=[
            go.Surface(
                z=predictions.numpy(),
                x=test_grid[:, 0],
                y=test_grid[:, 1],
                opacity=0.8,
                colorscale=DEFAULT_COLORSCALE,
                colorbar=dict(thickness=15, len=0.5),
                name="GP regression",
            )
        ]
    )

    # Add training data points
    fig.add_trace(
        go.Scatter3d(
            x=train_x[:, 0],
            y=train_x[:, 1],
            z=train_y.numpy(),
            mode="markers",
            marker={"color": DEFAULT_COLOR},
            name="training data",
        )
    )

    # Configure layout
    fig.update_layout(
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="Pulse Width (msec)",
            yaxis_title="Energy density new cone (J/cm^2)",
            zaxis_title="2 Qsw/(U+|D|) 1e6",
        ),
        template="ggplot2",
    )

    # Set camera angle
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2.75, y=1.75, z=1),
    )
    fig.update_layout(scene_camera=camera)
    fig.show()


def visualize_acquisition_function(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_grid: torch.Tensor,
    predictions: torch.Tensor,
    upper_confidence: np.ndarray,
    lower_confidence: np.ndarray,
    acquisition_maxima: Dict[str, Tuple[int, int]],
) -> None:
    """Visualize GP with uncertainty bounds and acquisition maxima.

    Creates an interactive 3D plot showing:
    - GP mean predictions
    - Confidence interval surfaces
    - Training data
    - Points that maximize various acquisition functions

    :param train_x: Training inputs of shape (n_samples, 2)
    :param train_y: Training targets of shape (n_samples,)
    :param test_grid: Grid of test points of shape (n_grid, 2)
    :param predictions: GP mean predictions at grid points
    :param upper_confidence: Upper confidence bound surface
    :param lower_confidence: Lower confidence bound surface
    :param acquisition_maxima: Dict mapping acquisition function names
        to (row, col) indices of their maxima on the grid
    """
    # Create GP mean surface
    fig = go.Figure(
        data=[
            go.Surface(
                z=predictions.numpy(),
                x=test_grid[:, 0],
                y=test_grid[:, 1],
                opacity=0.8,
                colorscale=DEFAULT_COLORSCALE,
                colorbar=dict(thickness=15, len=0.5),
                name="GP regression",
            )
        ]
    )

    # Add confidence interval surfaces
    fig.add_trace(
        go.Surface(
            z=upper_confidence,
            x=test_grid[:, 0],
            y=test_grid[:, 1],
            opacity=0.2,
            colorscale=DEFAULT_COLORSCALE,
            showscale=False,
            name="Upper confidence bound",
        )
    )

    fig.add_trace(
        go.Surface(
            z=lower_confidence,
            x=test_grid[:, 0],
            y=test_grid[:, 1],
            colorscale=DEFAULT_COLORSCALE,
            opacity=0.2,
            showscale=False,
            name="Lower confidence bound",
        )
    )

    # Add training data
    fig.add_trace(
        go.Scatter3d(
            x=train_x[:, 0],
            y=train_x[:, 1],
            z=train_y.numpy(),
            mode="markers",
            name="training data",
            marker={"color": DEFAULT_COLOR},
        )
    )

    # Add acquisition function maxima
    for acq_name, (row_idx, col_idx) in acquisition_maxima.items():
        fig.add_trace(
            go.Scatter3d(
                x=[test_grid[col_idx, 0]],
                y=[test_grid[row_idx, 1]],
                z=[predictions[row_idx, col_idx]],
                mode="markers",
                name=f"max({acq_name})",
            )
        )

    # Configure layout
    fig.update_layout(
        width=1200,
        height=750,
        margin=dict(r=20, l=10, b=15, t=10),
        legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="right", x=0.85),
        scene=dict(
            xaxis_title="Pulse Width (msec)",
            yaxis_title="Energy density new cone (J/cm^2)",
            zaxis_title="2 Qsw/(U+|D|) 1e6",
        ),
        template="ggplot2",
    )

    # Set camera angle
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=0.3, z=0.75),
    )
    fig.update_layout(scene_camera=camera)
    fig.show()


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """Compute regression performance metrics.

    Calculates RMSE, MAE, Spearman correlation, and R² score.

    :param y_true: Ground truth values
    :param y_pred: Predicted values
    :return: List of [RMSE, MAE, Spearman, R²], rounded to 3 decimals
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)

    return [np.round(val, 3) for val in [rmse, mae, spearman, r2]]


def _plot_predictions_vs_actual(
    ax: plt.Axes,
    y_true: torch.Tensor,
    y_pred: np.ndarray,
    y_uncertainty: np.ndarray,
    mae: float,
    dataset_type: str,
    font_size: int,
) -> None:
    """Plot predictions vs. actual values with confidence interval.

    :param ax: Matplotlib axes to plot on
    :param y_true: Ground truth values
    :param y_pred: Predicted values
    :param y_uncertainty: Prediction uncertainty (std dev)
    :param mae: Mean absolute error for the title
    :param dataset_type: Either "train" or "test"
    :param font_size: Font size for labels and title
    """
    # Scatter plot of predictions vs. actuals
    ax.scatter(y_true, y_pred, color=DEFAULT_COLOR)

    # Plot diagonal reference line
    low_lim = int(min(y_true).item())
    upp_lim = int(np.ceil(max(y_true).item()))
    ax.plot(
        np.linspace(low_lim, upp_lim),
        np.linspace(low_lim, upp_lim),
        "k--",
    )
    ax.set_xlim(low_lim, upp_lim)
    ax.set_ylim(low_lim, upp_lim)

    # Plot 95% confidence interval
    # Sort to enable fill_between to work correctly
    sorted_indices = np.argsort(y_true)
    sorted_y_true = np.sort(y_true)
    lower_bound = (y_pred - y_uncertainty)[sorted_indices]
    upper_bound = (y_pred + y_uncertainty)[sorted_indices]

    ax.fill_between(
        sorted_y_true,
        lower_bound,
        upper_bound,
        color=CONFIDENCE_INTERVAL_COLOR,
        alpha=CONFIDENCE_INTERVAL_ALPHA,
        label="95% confidence interval",
    )

    # Labels and title
    ax.set_xlabel("Ground Truth 2 Qsw/(U+|D|) 1e6", fontsize=font_size)
    ax.set_ylabel("Prediction 2 Qsw/(U+|D|) 1e6", fontsize=font_size)

    title_type = "Training" if dataset_type == "train" else "Test"
    ax.set_title(
        f"GP {title_type} Results (MAE={mae:.2f} [%])",
        fontsize=font_size,
    )
    ax.legend()


def _plot_training_loss(
    ax: plt.Axes,
    loss_history: List[float],
    log_interval: int,
    font_size: int,
) -> None:
    """Plot training loss over epochs.

    :param ax: Matplotlib axes to plot on
    :param loss_history: List of loss values recorded during training
    :param log_interval: Number of epochs between loss recordings
    :param font_size: Font size for labels and title
    """
    epochs = np.arange(len(loss_history)) * log_interval
    ax.plot(epochs, loss_history, "o-", color=DEFAULT_COLOR)
    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_ylabel("Marginal Log Likelihood Loss", fontsize=font_size)
    ax.set_title(
        f"Training Loss (Loss={loss_history[-1]:.2f})",
        fontsize=font_size,
    )


def plot_gp_results(
    y_true: torch.Tensor,
    y_pred: np.ndarray,
    loss_history: Optional[List[float]],
    y_uncertainty: np.ndarray,
    dataset_type: str = "train",
    log_interval: int = 500,
) -> None:
    """Plot comprehensive GP model evaluation results.

    Creates a figure with:
    1. Predictions vs. actual values with confidence intervals
    2. Training loss curve (if provided)
    3. Printed metrics table (RMSE, MAE, Spearman, R²)

    :param y_true: Ground truth values of shape (n_samples,)
    :param y_pred: Predicted mean values
    :param loss_history: Training loss values (None if not training)
    :param y_uncertainty: Prediction uncertainty (std dev)
    :param dataset_type: Either "train" or "test"
    :param log_interval: Epochs between loss recordings (default: 500)
    """
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 4.5))
    font_size = DEFAULT_FONT_SIZE

    # Compute and print metrics
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "Spearman", "R² score"],
            "Value": metrics,
        }
    )
    print(metrics_df)

    # Plot predictions vs. actual
    _plot_predictions_vs_actual(
        axes[0],
        y_true,
        y_pred,
        y_uncertainty,
        mae=metrics[1],
        dataset_type=dataset_type,
        font_size=font_size,
    )

    # Plot training loss if available
    if loss_history:
        _plot_training_loss(axes[1], loss_history, log_interval, font_size)
    else:
        axes[1].axis("off")

    # Turn off third subplot (reserved for future use)
    axes[2].axis("off")

    # Configure tick parameters
    for ax in axes:
        ax.tick_params(
            direction="in",
            length=5,
            width=1,
            labelsize=font_size * 0.8,
        )

    plt.subplots_adjust(wspace=0.4)
    plt.show()


# Backward compatibility aliases
prettyplot = configure_plot_style
vis_pred = visualize_gp_surface
vis_acq = visualize_acquisition_function
get_err = compute_regression_metrics
plot_err = _plot_predictions_vs_actual
plot_training_loss = _plot_training_loss
plot_gp_res = plot_gp_results
