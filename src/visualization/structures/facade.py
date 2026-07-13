"""Facade tying a structure provider + renderer into one simple entry point.

`CrystalStructureVisualizer` is the public API for the structures subpackage. By
default it fetches live from the Materials Project (cached to CIF) and renders static
figures via ASE+matplotlib, but any `StructureProvider` / `StructureRenderer` can be
injected (Dependency Inversion) for testing or alternative backends.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt

from ..base import DEFAULT_STYLE, save_figure
from .phase_registry import available_phase_keys, get_phase
from .provider import (
    CachedStructureProvider,
    MaterialsProjectProvider,
    StructureProvider,
)
from .renderer import AseMplRenderer, StructureRenderer


class CrystalStructureVisualizer:
    """Fetch HfO2 polymorph structures and render them to static figures.

    :param provider: Structure source (default: Materials Project + CIF cache)
    :param renderer: Figure renderer (default: ASE + matplotlib static figures)
    """

    def __init__(
        self,
        provider: Optional[StructureProvider] = None,
        renderer: Optional[StructureRenderer] = None,
    ):
        # Defaults are constructed lazily so importing this module never needs the
        # optional heavy deps; they are only required once a default is built here.
        self._provider = provider or CachedStructureProvider(MaterialsProjectProvider())
        self._renderer = renderer or AseMplRenderer()

    def _title_for(self, phase_key: str) -> str:
        phase = get_phase(phase_key)
        suffix = "  ★ ferroelectric" if phase.ferroelectric else ""
        return f"{phase.name} — {phase.space_group}{suffix}"

    def render_phase(self, phase_key: str, save_path: Optional[str] = None) -> plt.Figure:
        """Render a single phase to its own figure."""
        structure = self._provider.get_structure(phase_key)
        return self._renderer.render(
            structure, title=self._title_for(phase_key), save_path=save_path
        )

    def render_all_phases(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Render every registered phase; optionally save each as ``hfo2_<key>.png``."""
        figures: Dict[str, plt.Figure] = {}
        for key in available_phase_keys():
            save_path = str(Path(save_dir) / f"hfo2_{key}.png") if save_dir else None
            figures[key] = self.render_phase(key, save_path=save_path)
        return figures

    def render_comparison_grid(self, save_path: Optional[str] = None) -> plt.Figure:
        """Render all four phases side by side in a 2x2 comparison figure."""
        keys = available_phase_keys()
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, key in zip(axes.flat, keys):
            structure = self._provider.get_structure(key)
            self._renderer.render_on_ax(structure, ax, title=self._title_for(key))
        # Hide any unused panels if fewer than 4 phases are registered.
        for ax in axes.flat[len(keys) :]:
            ax.set_axis_off()
        fig.suptitle(
            "HfO2 polymorphs — only the polar orthorhombic phase is ferroelectric",
            fontsize=DEFAULT_STYLE.title_fontsize,
            fontweight=DEFAULT_STYLE.title_weight,
        )
        plt.tight_layout()
        save_figure(fig, save_path)
        return fig
