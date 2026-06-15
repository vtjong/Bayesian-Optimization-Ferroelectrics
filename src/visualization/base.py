"""Shared plotting foundation: styling, figure saving, and a base plotter contract.

Centralizes figure styling (:class:`PlotStyle`) and the save-to-disk operation
(:func:`save_figure`) so individual plotters/visualizers no longer duplicate the
``plt.savefig(..., dpi=300, bbox_inches="tight")`` pattern or scatter magic-number
font sizes. Provides :class:`BasePlotter`, an ABC mirroring
``analysis.core.BaseAnalyzer`` for consistent plotter design across the codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotStyle:
    """Immutable styling constants shared across all figures (single source of truth).

    Holds the font sizes, alphas, colors, and save resolution that were previously
    hard-coded in each plot method. Frozen so it can be safely shared as a default.
    """

    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 11
    title_weight: str = "bold"
    save_dpi: int = 300
    grid_alpha: float = 0.3
    bar_alpha: float = 0.8
    accent_color: str = "#72356c"
    diverging_cmap: str = "coolwarm"
    sequential_cmap: str = "viridis"


# Default shared style instance — import and reuse rather than re-specifying constants.
DEFAULT_STYLE = PlotStyle()


def save_figure(
    fig: plt.Figure,
    save_path: Optional[Union[str, Path]],
    dpi: int = DEFAULT_STYLE.save_dpi,
) -> None:
    """Save a figure to disk if a path is given; a no-op otherwise.

    Centralizes the repeated ``if save_path: plt.savefig(...)`` pattern and creates
    parent directories as needed. Operates on the specific ``fig`` (not pyplot global
    state) so it is safe when several figures are open.

    :param fig: Figure to save
    :param save_path: Destination path, or None to skip saving
    :param dpi: Resolution in dots per inch (defaults to ``PlotStyle.save_dpi``)
    """
    if save_path is None:
        return
    path = Path(save_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


class BasePlotter(ABC):
    """Abstract base for plotters that render a result into a matplotlib Figure.

    Mirrors ``analysis.core.BaseAnalyzer``: concrete plotters expose a ``name`` and a
    ``plot`` method returning the Figure. Subclasses should persist via
    :func:`save_figure` so save/styling logic stays centralized (DRY).
    """

    #: Styling used by the plotter; override per-instance to customize.
    style: PlotStyle = DEFAULT_STYLE

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the plotter (used in logging/titles)."""

    @abstractmethod
    def plot(self, *args, **kwargs) -> plt.Figure:
        """Produce and return a matplotlib Figure."""
