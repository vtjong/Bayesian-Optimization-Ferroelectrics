"""Structure providers: fetch HfO2 polymorph structures (Strategy + Repository).

`StructureProvider` is the abstraction; `MaterialsProjectProvider` fetches live from
the Materials Project; `CachedStructureProvider` is a Decorator that caches fetched
structures as CIF on disk so repeat runs work offline. Heavy materials-science deps
(pymatgen / mp-api) are import-guarded (mirroring the SALib `SOBOL_AVAILABLE` pattern
in `analysis.analyzers`), so importing this module never requires them — only
instantiating a provider does.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .phase_registry import available_phase_keys, get_phase

try:
    from pymatgen.core import Structure

    PYMATGEN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the optional dep
    Structure = None  # type: ignore
    PYMATGEN_AVAILABLE = False


_MISSING_PYMATGEN_MSG = (
    "pymatgen + mp-api are required for crystal-structure visualization. "
    "Install with: pip install pymatgen mp-api  (or: pip install -r requirements-viz.txt)"
)


class StructureProvider(ABC):
    """Abstraction for obtaining a crystal structure for a phase key."""

    @abstractmethod
    def get_structure(self, phase_key: str):
        """Return a pymatgen ``Structure`` for the given phase key."""

    @abstractmethod
    def available_phases(self) -> List[str]:
        """Return the phase keys this provider can supply."""


class MaterialsProjectProvider(StructureProvider):
    """Fetch HfO2 polymorphs live from the Materials Project.

    Resolves each phase by its Materials Project id when one is registered, otherwise
    queries by formula + space-group number and returns the most stable match. The API
    key is read from ``api_key`` or the ``MP_API_KEY`` environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        if not PYMATGEN_AVAILABLE:
            raise ImportError(_MISSING_PYMATGEN_MSG)
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Materials Project API key required. Set the MP_API_KEY environment "
                "variable or pass api_key=... (free key: https://materialsproject.org)."
            )

    def available_phases(self) -> List[str]:
        return available_phase_keys()

    def get_structure(self, phase_key: str):
        phase = get_phase(phase_key)
        from mp_api.client import MPRester

        with MPRester(self.api_key) as mpr:
            if phase.mp_id:
                try:
                    return mpr.get_structure_by_material_id(phase.mp_id)
                except Exception:
                    pass  # fall back to a space-group query below
            return self._query_by_spacegroup(mpr, phase.spacegroup_number, phase_key)

    @staticmethod
    def _query_by_spacegroup(mpr, spacegroup_number: int, phase_key: str):
        """Return the most stable HfO2 structure with the given space-group number."""
        docs = mpr.materials.summary.search(
            formula="HfO2",
            spacegroup_number=spacegroup_number,
            fields=["material_id", "structure", "energy_above_hull"],
        )
        if not docs:
            raise ValueError(
                f"No Materials Project HfO2 entry for space group "
                f"#{spacegroup_number} ('{phase_key}')."
            )
        best = min(
            docs,
            key=lambda d: d.energy_above_hull if d.energy_above_hull is not None else 1e9,
        )
        return best.structure


class CachedStructureProvider(StructureProvider):
    """Decorator that caches an inner provider's structures as CIF files on disk.

    First call fetches via the wrapped provider and writes ``<cache_dir>/<key>.cif``;
    subsequent calls read the CIF directly (offline, deterministic). Caching is kept
    orthogonal to fetching so any provider can be wrapped.
    """

    def __init__(self, inner: StructureProvider, cache_dir: str = "data/structures"):
        if not PYMATGEN_AVAILABLE:
            raise ImportError(_MISSING_PYMATGEN_MSG)
        self._inner = inner
        self._cache_dir = Path(cache_dir)

    def available_phases(self) -> List[str]:
        return self._inner.available_phases()

    def get_structure(self, phase_key: str):
        cache_file = self._cache_dir / f"{phase_key}.cif"
        if cache_file.exists():
            return Structure.from_file(str(cache_file))
        structure = self._inner.get_structure(phase_key)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        structure.to(filename=str(cache_file))
        return structure
