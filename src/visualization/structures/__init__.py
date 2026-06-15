"""Crystal-structure visualization for HfO2 polymorphs.

Public API:
- ``CrystalStructureVisualizer``: fetch + render the HfO2 phases (facade)
- Providers/renderers/registry for advanced/customized use

Heavy deps (pymatgen / mp-api / ase) are import-guarded in the submodules, so importing
this package never requires them — only constructing a Materials Project provider or an
ASE renderer does.
"""

from .facade import CrystalStructureVisualizer
from .phase_registry import HFO2_PHASES, HfO2Phase, available_phase_keys, get_phase
from .provider import (
    CachedStructureProvider,
    MaterialsProjectProvider,
    StructureProvider,
)
from .renderer import AseMplRenderer, StructureRenderer

__all__ = [
    "CrystalStructureVisualizer",
    "StructureProvider",
    "MaterialsProjectProvider",
    "CachedStructureProvider",
    "StructureRenderer",
    "AseMplRenderer",
    "HfO2Phase",
    "HFO2_PHASES",
    "available_phase_keys",
    "get_phase",
]
