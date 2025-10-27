"""Bayesian Optimization utilities.

Modules:
--------
- grid: Candidate grid generation for parameter space exploration
- acquisition: Acquisition function utilities (EI, PI, UCB, qEI, qPI, qUCB)
- thompson_sampler: Thompson Sampling implementation
"""

from .grid import CandidateGrid, Grid, create_candidate_grid
from .thompson_sampler import ThompsonSampler

# Backward compatibility
ThompsonSampling = ThompsonSampler

__all__ = [
    "CandidateGrid",
    "Grid",
    "create_candidate_grid",
    "ThompsonSampler",
    "ThompsonSampling",
]
