"""Backward compatibility shim for model imports.

This module provides backward compatibility for old import paths.
New code should use: from models import ExactGPModel
"""

from models import ExactGPModel, GPModel

__all__ = ["ExactGPModel", "GPModel"]
