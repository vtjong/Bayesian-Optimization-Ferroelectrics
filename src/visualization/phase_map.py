"""(Voltage, time) -> crystallinity / phase-region maps with the boundary contour.

Defines :class:`PhaseMapResult` (a frozen value object holding a grid of predictions in
physical units) and :class:`PhaseMapPlotter` (a :class:`BasePlotter` that draws the
continuous crystallinity heatmap with its level-set boundary, the uncertainty map, and
the categorical phase-region map). The plotter depends only on numpy arrays — building
a result from a trained GP lives in :mod:`visualization.grid_predictor` (the adapter),
so this module has no gpytorch/torch coupling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .base import DEFAULT_STYLE, BasePlotter, save_figure


@dataclass(frozen=True)
class PhaseMapResult:
    """Grid of model predictions over a 2-D (x, y) control space, in physical units.

    :param x_coords: 1-D physical values along the x axis (length nx)
    :param y_coords: 1-D physical values along the y axis (length ny)
    :param mean: Predicted continuous target on the grid, shape (ny, nx)
    :param std: Predictive standard deviation on the grid, shape (ny, nx)
    :param x_label: x-axis label (e.g. "Pulse Time (ms)")
    :param y_label: y-axis label (e.g. "Energy density (J/cm^2)")
    :param value_label: Name of the predicted quantity (e.g. "FOM" / "Crystallinity")
    :param threshold: Optional level-set value to draw as the boundary contour
    :param obs_x: Optional observed x coordinates (physical units)
    :param obs_y: Optional observed y coordinates (physical units)
    :param obs_value: Optional observed target values (for point coloring)
    :param phase_labels: Optional categorical phase index per grid cell, shape (ny, nx)
    :param phase_names: Optional display names for the phase categories
    """

    x_coords: np.ndarray
    y_coords: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    x_label: str = "x"
    y_label: str = "y"
    value_label: str = "value"
    threshold: Optional[float] = None
    obs_x: Optional[np.ndarray] = None
    obs_y: Optional[np.ndarray] = None
    obs_value: Optional[np.ndarray] = None
    phase_labels: Optional[np.ndarray] = None
    phase_names: Optional[Tuple[str, ...]] = None


class PhaseMapPlotter(BasePlotter):
    """Render :class:`PhaseMapResult` objects as crystallinity / phase-region maps."""

    def __init__(self, style=DEFAULT_STYLE):
        self.style = style

    @property
    def name(self) -> str:
        return "phase_map"

    def plot(self, result: PhaseMapResult, save_path: Optional[str] = None) -> plt.Figure:
        """Default plot: the continuous crystallinity map with its boundary contour."""
        return self.plot_crystallinity_map(result, save_path=save_path)

    def _overlay_observations(self, ax: plt.Axes, result: PhaseMapResult) -> None:
        """Scatter the observed experiments on top of a map, if provided."""
        if result.obs_x is None or result.obs_y is None:
            return
        ax.scatter(
            result.obs_x,
            result.obs_y,
            c="white",
            edgecolors="black",
            s=40,
            linewidths=0.8,
            zorder=5,
            label="experiments",
        )

    def plot_crystallinity_map(
        self, result: PhaseMapResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Filled contour of the predicted target, with the level-set boundary contour."""
        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(
            result.x_coords, result.y_coords, result.mean, levels=20,
            cmap=self.style.sequential_cmap,
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(result.value_label, fontsize=self.style.label_fontsize)

        if result.threshold is not None:
            cs = ax.contour(
                result.x_coords, result.y_coords, result.mean,
                levels=[result.threshold], colors="red", linewidths=2.0,
            )
            ax.clabel(cs, fmt=f"{result.value_label}={result.threshold:g}", fontsize=9)

        self._overlay_observations(ax, result)
        ax.set_xlabel(result.x_label, fontsize=self.style.label_fontsize)
        ax.set_ylabel(result.y_label, fontsize=self.style.label_fontsize)
        ax.set_title(
            f"Predicted {result.value_label} map",
            fontsize=self.style.title_fontsize, fontweight=self.style.title_weight,
        )
        if result.obs_x is not None:
            ax.legend(loc="best")
        plt.tight_layout()
        save_figure(fig, save_path)
        return fig

    def plot_uncertainty_map(
        self, result: PhaseMapResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Filled contour of the predictive standard deviation (epistemic uncertainty)."""
        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(
            result.x_coords, result.y_coords, result.std, levels=20, cmap="magma",
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(f"std({result.value_label})", fontsize=self.style.label_fontsize)
        self._overlay_observations(ax, result)
        ax.set_xlabel(result.x_label, fontsize=self.style.label_fontsize)
        ax.set_ylabel(result.y_label, fontsize=self.style.label_fontsize)
        ax.set_title(
            f"Predictive uncertainty in {result.value_label}",
            fontsize=self.style.title_fontsize, fontweight=self.style.title_weight,
        )
        if result.obs_x is not None:
            ax.legend(loc="best")
        plt.tight_layout()
        save_figure(fig, save_path)
        return fig

    def plot_phase_regions(
        self, result: PhaseMapResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Discrete map of the argmax phase per cell, with a legend.

        Requires ``phase_labels`` (and ideally ``phase_names``) on the result; raises a
        clear error otherwise. Lights up once XRD phase labels feed the model.
        """
        if result.phase_labels is None:
            raise ValueError(
                "plot_phase_regions requires result.phase_labels (categorical phase "
                "index per grid cell). Populate it from a phase classifier/XRD labels."
            )
        labels = np.asarray(result.phase_labels)
        n_classes = int(labels.max()) + 1
        names = result.phase_names or tuple(f"phase {i}" for i in range(n_classes))

        cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])
        norm = BoundaryNorm(np.arange(-0.5, n_classes, 1), cmap.N)

        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(
            result.x_coords, result.y_coords, labels, cmap=cmap, norm=norm,
            shading="auto",
        )
        cbar = fig.colorbar(mesh, ax=ax, ticks=range(n_classes))
        cbar.ax.set_yticklabels(names)
        self._overlay_observations(ax, result)
        ax.set_xlabel(result.x_label, fontsize=self.style.label_fontsize)
        ax.set_ylabel(result.y_label, fontsize=self.style.label_fontsize)
        ax.set_title(
            "Predicted phase regions",
            fontsize=self.style.title_fontsize, fontweight=self.style.title_weight,
        )
        plt.tight_layout()
        save_figure(fig, save_path)
        return fig
