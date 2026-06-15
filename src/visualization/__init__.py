"""Visualization utilities for GP models and Bayesian Optimization.

Modules:
--------
- base: shared plotting foundation (BasePlotter, PlotStyle, save_figure)
- plots: 3D plotting, acquisition visualization, and evaluation metrics
- phase_map: (V, t) crystallinity / phase-region maps
- structures: HfO2 crystal-structure rendering (optional pymatgen/ase deps)
"""

from .base import DEFAULT_STYLE, BasePlotter, PlotStyle, save_figure
from .plots import (
    compute_regression_metrics,
    configure_plot_style,
    plot_gp_results,
    visualize_acquisition_function,
    visualize_gp_surface,
)
from .structures import CrystalStructureVisualizer

# Backward compatibility
prettyplot = configure_plot_style
vis_pred = visualize_gp_surface
vis_acq = visualize_acquisition_function
get_err = compute_regression_metrics
plot_gp_res = plot_gp_results

__all__ = [
    # Shared foundation
    "BasePlotter",
    "PlotStyle",
    "DEFAULT_STYLE",
    "save_figure",
    # GP/result plots
    "configure_plot_style",
    "visualize_gp_surface",
    "visualize_acquisition_function",
    "compute_regression_metrics",
    "plot_gp_results",
    # Crystal structures
    "CrystalStructureVisualizer",
    # Backward compatibility
    "prettyplot",
    "vis_pred",
    "vis_acq",
    "get_err",
    "plot_gp_res",
]
