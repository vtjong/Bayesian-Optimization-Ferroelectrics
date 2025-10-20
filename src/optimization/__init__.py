"""Bayesian Optimization utilities.

Modules:
--------
- grid: Candidate grid generation for parameter space exploration
- acquisition: Thompson Sampling and acquisition function optimization
"""

from .acquisition import ThompsonSampler
from .grid import CandidateGrid, Grid, create_candidate_grid

# Backward compatibility
ThompsonSampling = ThompsonSampler

__all__ = [
    "CandidateGrid",
    "Grid",
    "create_candidate_grid",
    "ThompsonSampler",
    "ThompsonSampling",
]
