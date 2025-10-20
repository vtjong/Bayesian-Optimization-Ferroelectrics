"""Preprocessing package for ferroelectric experiments.

Modules:
--------
- loaders: Data loading and exploratory visualization
- transforms: Scaling and preprocessing utilities
"""

from .loaders import display_data, read_dat
from .transforms import TorchMinMaxScaler, datasetmaker

# Backward compatibility
MinMaxScalerTorch = TorchMinMaxScaler

__all__ = [
    "read_dat",
    "display_data",
    "datasetmaker",
    "TorchMinMaxScaler",
    "MinMaxScalerTorch",
]
