"""Helper functions for analysis module.

Utility functions to keep main logic concise and reusable.
"""

from typing import List

import numpy as np
import torch


def convert_tensor_to_numpy(
    tensor_or_array: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Convert tensor to numpy array if needed.

    :param tensor_or_array: Input tensor or numpy array
    :return: Numpy array
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().numpy()
    return tensor_or_array


def generate_default_feature_names(n_features: int) -> List[str]:
    """Generate default feature names.

    :param n_features: Number of features
    :return: List of feature names
    """
    return [f"Parameter_{i+1}" for i in range(n_features)]
