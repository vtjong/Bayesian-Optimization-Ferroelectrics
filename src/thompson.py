"""Backward compatibility shim for Thompson Sampling imports.

This module provides backward compatibility for old import paths.
New code should use: from optimization import ThompsonSampler
"""

from optimization import ThompsonSampler, ThompsonSampling

__all__ = ["ThompsonSampler", "ThompsonSampling"]
