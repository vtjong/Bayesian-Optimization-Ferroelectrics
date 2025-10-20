"""Backward compatibility shim for plotting imports.

This module provides backward compatibility for old import paths.
New code should use: from visualization import ...
"""

from visualization import (
    compute_regression_metrics,
    configure_plot_style,
    get_err,
    plot_gp_res,
    plot_gp_results,
    prettyplot,
    vis_acq,
    vis_pred,
    visualize_acquisition_function,
    visualize_gp_surface,
)

__all__ = [
    "configure_plot_style",
    "prettyplot",
    "visualize_gp_surface",
    "vis_pred",
    "visualize_acquisition_function",
    "vis_acq",
    "compute_regression_metrics",
    "get_err",
    "plot_gp_results",
    "plot_gp_res",
]
