"""Backward compatibility shim for grid imports.

This module provides backward compatibility for old import paths.
New code should use: from optimization import Grid
"""

from optimization import CandidateGrid, Grid, create_candidate_grid

__all__ = ["Grid", "CandidateGrid", "create_candidate_grid"]
