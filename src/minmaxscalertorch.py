"""Backward compatibility shim for scaler imports.

This module provides backward compatibility for old import paths.
New code should use: from preprocessing import TorchMinMaxScaler
"""

from preprocessing import MinMaxScalerTorch, TorchMinMaxScaler

__all__ = ["TorchMinMaxScaler", "MinMaxScalerTorch"]
