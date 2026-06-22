"""Thermal model for flash-lamp annealing: turn (V, t) into the temperature the film feels.

- model.py:       simulate_profile(V, t) -> T(t)  (lumped transient heat/quench model)
- descriptors.py: extract_descriptors(T(t)) -> {Tmax, K, heating/cooling rate, dwell, ...}

These thermal descriptors are the physics features the kinetic mechanism models consume.
PROTOTYPE — the production upgrade is a 1-D transient through-thickness stack validated
against measured traces.
"""

from .descriptors import extract_descriptors
from .model import (
    ThermalParams,
    ThermalParams1D,
    simulate_profile,
    simulate_profile_1d,
)

__all__ = [
    "simulate_profile",
    "ThermalParams",
    "simulate_profile_1d",
    "ThermalParams1D",
    "extract_descriptors",
]
