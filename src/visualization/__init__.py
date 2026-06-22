"""Visualization utilities.

Modules:
--------
- base: shared plotting foundation (BasePlotter, PlotStyle, save_figure)
- concepts: figures for the interactive learning app (gitignored)
- structures: HfO2 crystal-structure rendering (optional pymatgen/ase deps)
"""

from .base import DEFAULT_STYLE, BasePlotter, PlotStyle, save_figure
from .structures import CrystalStructureVisualizer

__all__ = [
    "BasePlotter",
    "PlotStyle",
    "DEFAULT_STYLE",
    "save_figure",
    "CrystalStructureVisualizer",
]
