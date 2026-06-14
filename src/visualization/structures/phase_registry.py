"""Registry of the HfO2 polymorphs relevant to ferroelectric HZO.

Single source of truth mapping a human-friendly phase key to its crystallographic
metadata (Materials Project id, space group, polarity) and a one-line physical
description. Used by the provider to resolve structures and by the renderer/facade
for titles and the ferroelectric annotation.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HfO2Phase:
    """Crystallographic metadata for one HfO2 polymorph.

    :param key: Short lookup key (e.g. "orthorhombic")
    :param name: Display name (e.g. "Orthorhombic (o)")
    :param mp_id: Materials Project id hint (may be empty -> resolve by space group)
    :param space_group: Hermann-Mauguin symbol + number (e.g. "Pca2_1 (#29)")
    :param spacegroup_number: International space-group number (for MP query fallback)
    :param polar: Whether the structure is polar (has a net dipole)
    :param ferroelectric: Whether this is the ferroelectric phase of interest
    :param description: One-line physical description
    """

    key: str
    name: str
    mp_id: str
    space_group: str
    spacegroup_number: int
    polar: bool
    ferroelectric: bool
    description: str


# The four polymorphs that matter for HZO crystallization. The polar orthorhombic
# Pca2_1 (#29) phase is the ferroelectric target; note the *non*-polar orthorhombic
# Pbca (#61) is a different phase and is intentionally not used here.
HFO2_PHASES: Dict[str, HfO2Phase] = {
    "monoclinic": HfO2Phase(
        key="monoclinic",
        name="Monoclinic (m)",
        mp_id="mp-352",
        space_group="P2_1/c (#14)",
        spacegroup_number=14,
        polar=False,
        ferroelectric=False,
        description=(
            "Thermodynamically stable, non-polar baddeleyite phase — the 'default' "
            "HfO2 forms; non-ferroelectric."
        ),
    ),
    "tetragonal": HfO2Phase(
        key="tetragonal",
        name="Tetragonal (t)",
        mp_id="mp-1018721",
        space_group="P4_2/nmc (#137)",
        spacegroup_number=137,
        polar=False,
        ferroelectric=False,
        description=(
            "High-symmetry non-polar phase; gives antiferroelectric-like loops, "
            "favored by higher Zr content."
        ),
    ),
    "orthorhombic": HfO2Phase(
        key="orthorhombic",
        name="Orthorhombic (o)",
        mp_id="",  # resolve by space group #29 (avoids hardcoding an uncertain mp-id)
        space_group="Pca2_1 (#29)",
        spacegroup_number=29,
        polar=True,
        ferroelectric=True,
        description=(
            "Polar metastable phase — the SOURCE of ferroelectricity in HZO and the "
            "target of the anneal."
        ),
    ),
    "cubic": HfO2Phase(
        key="cubic",
        name="Cubic (c)",
        mp_id="mp-550893",
        space_group="Fm-3m (#225)",
        spacegroup_number=225,
        polar=False,
        ferroelectric=False,
        description="High-symmetry non-polar fluorite phase.",
    ),
}


def available_phase_keys() -> List[str]:
    """Return the registered phase keys (stable order)."""
    return list(HFO2_PHASES.keys())


def get_phase(phase_key: str) -> HfO2Phase:
    """Look up a phase by key.

    :param phase_key: One of :func:`available_phase_keys`
    :return: The :class:`HfO2Phase` metadata
    :raises KeyError: If the key is unknown (message lists valid keys)
    """
    try:
        return HFO2_PHASES[phase_key]
    except KeyError:
        raise KeyError(
            f"Unknown phase '{phase_key}'. Available: {available_phase_keys()}"
        )
