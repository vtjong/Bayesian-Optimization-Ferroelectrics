"""Structure renderers: draw a crystal structure to a static matplotlib figure.

`StructureRenderer` is the abstraction; `AseMplRenderer` converts a pymatgen
``Structure`` to ASE atoms and renders them with ``ase.visualize.plot.plot_atoms``.
Heavy deps (ase / pymatgen) are import-guarded so importing this module never requires
them — only instantiating the renderer does. Figure saving/styling is delegated to the
shared ``visualization.base`` foundation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from ..base import DEFAULT_STYLE, PlotStyle, save_figure

try:
    from ase.visualize.plot import plot_atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    ASE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the optional dep
    plot_atoms = None  # type: ignore
    AseAtomsAdaptor = None  # type: ignore
    ASE_AVAILABLE = False


_MISSING_ASE_MSG = (
    "ase + pymatgen are required to render crystal structures. "
    "Install with: pip install ase pymatgen  (or: pip install -r requirements-viz.txt)"
)


class StructureRenderer(ABC):
    """Abstraction for rendering a pymatgen ``Structure`` to a figure."""

    @abstractmethod
    def render_on_ax(self, structure, ax: plt.Axes, title: Optional[str] = None) -> None:
        """Draw the structure onto an existing matplotlib axis."""

    @abstractmethod
    def render(
        self, structure, title: Optional[str] = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Render the structure to its own figure (optionally saved)."""


class AseMplRenderer(StructureRenderer):
    """Render a structure as a static figure via ASE + matplotlib.

    :param rotation: ASE rotation string controlling the viewing angle
    :param supercell: Number of unit-cell repeats along (a, b, c)
    :param atom_radii: Relative atom radius for the ball-and-stick drawing
    :param style: Shared :class:`PlotStyle` for titles/save resolution
    """

    def __init__(
        self,
        rotation: str = "10x,-10y,0z",
        supercell: Tuple[int, int, int] = (1, 1, 1),
        atom_radii: float = 0.5,
        style: PlotStyle = DEFAULT_STYLE,
    ):
        if not ASE_AVAILABLE:
            raise ImportError(_MISSING_ASE_MSG)
        self.rotation = rotation
        self.supercell = supercell
        self.atom_radii = atom_radii
        self.style = style

    def render_on_ax(self, structure, ax: plt.Axes, title: Optional[str] = None) -> None:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        if self.supercell != (1, 1, 1):
            atoms = atoms.repeat(self.supercell)
        plot_atoms(atoms, ax, rotation=self.rotation, radii=self.atom_radii)
        ax.set_axis_off()
        if title:
            ax.set_title(
                title,
                fontsize=self.style.title_fontsize,
                fontweight=self.style.title_weight,
            )

    def render(
        self, structure, title: Optional[str] = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5, 5))
        self.render_on_ax(structure, ax, title=title)
        plt.tight_layout()
        save_figure(fig, save_path)
        return fig
