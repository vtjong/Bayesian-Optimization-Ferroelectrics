"""Visualization utilities for GP models and Bayesian Optimization.

Modules:
--------
- plots: 3D plotting, acquisition visualization, and evaluation metrics
"""

from .plots import (
    compute_regression_metrics,
    configure_plot_style,
    plot_gp_results,
    visualize_acquisition_function,
    visualize_gp_surface,
)

# Backward compatibility
prettyplot = configure_plot_style
vis_pred = visualize_gp_surface
vis_acq = visualize_acquisition_function
get_err = compute_regression_metrics
plot_gp_res = plot_gp_results

__all__ = [
    "configure_plot_style",
    "visualize_gp_surface",
    "visualize_acquisition_function",
    "compute_regression_metrics",
    "plot_gp_results",
    # Backward compatibility
    "prettyplot",
    "vis_pred",
    "vis_acq",
    "get_err",
    "plot_gp_res",
]
