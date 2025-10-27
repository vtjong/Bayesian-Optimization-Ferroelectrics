"""Preprocessing package for ferroelectric experiments.

Modules:
--------
- loaders: Data loading and exploratory visualization
- transforms: Scaling and preprocessing utilities
"""

from .loaders import load_experimental_data, plot_input_output_scatter_matrix
from .transforms import TorchMinMaxScaler, prepare_gp_training_tensors

# Backward compatibility
read_dat = load_experimental_data
display_data = plot_input_output_scatter_matrix
datasetmaker = prepare_gp_training_tensors
MinMaxScalerTorch = TorchMinMaxScaler

__all__ = [
    "load_experimental_data",
    "plot_input_output_scatter_matrix",
    "prepare_gp_training_tensors",
    "TorchMinMaxScaler",
    # Backward compatibility
    "read_dat",
    "display_data",
    "datasetmaker",
    "MinMaxScalerTorch",
]
