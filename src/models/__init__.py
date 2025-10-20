"""GP model package for Bayesian Optimization.

Modules:
--------
- gp: Core Gaussian Process model definitions
- factory: Kernel and model creation utilities
"""

from .factory import create_gp_model, create_kernel
from .gp import ExactGPModel

# Backward compatibility
GPModel = ExactGPModel
kernel_func = create_kernel
make_model = create_gp_model

__all__ = [
    "ExactGPModel",
    "GPModel",
    "create_kernel",
    "create_gp_model",
    "kernel_func",
    "make_model",
]
