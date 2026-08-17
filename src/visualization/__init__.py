"""Visualization utilities.

Modules:
--------
- base: shared plotting foundation (BasePlotter, PlotStyle, save_figure)
- concepts: figures for the interactive learning app (personal tooling, not committed)
- structures: HfO2 crystal-structure rendering (personal tooling, not committed; needs the
  optional pymatgen/ase extras in requirements-viz.txt)

``structures`` is imported lazily and its absence is not an error: the campaign code needs only
``base``, so a checkout without the personal tooling must still import cleanly.
"""

from .base import DEFAULT_STYLE, BasePlotter, PlotStyle, save_figure

__all__ = [
    "BasePlotter",
    "PlotStyle",
    "DEFAULT_STYLE",
    "save_figure",
]

try:  # optional: only present alongside the (uncommitted) interactive tooling
    from .structures import CrystalStructureVisualizer

    __all__.append("CrystalStructureVisualizer")
except ImportError:  # pragma: no cover - depends on local checkout contents
    pass
