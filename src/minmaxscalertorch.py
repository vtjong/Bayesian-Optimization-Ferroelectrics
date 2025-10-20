"""PyTorch-compatible MinMax scaling utilities.

This module provides a wrapper around scikit-learn's MinMaxScaler that
works seamlessly with PyTorch tensors, automatically handling conversions
between numpy arrays and torch tensors.
"""

from typing import Optional, Union

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


class TorchMinMaxScaler:
    """Adapter for MinMaxScaler that works with PyTorch tensors.

    This is a composition-based wrapper (not inheritance) that:
    - Accepts torch.Tensor or numpy arrays as input
    - Returns torch.Tensor outputs
    - Preserves scikit-learn's scaling logic
    - Maintains fitted parameters (min_, scale_, etc.)

    Example:
        scaler = TorchMinMaxScaler()
        X_scaled = scaler.fit_transform(train_x)
        X_new_scaled = scaler.transform(test_x)
        X_original = scaler.inverse_transform(X_scaled)

    :ivar min_: Per-feature minimum values seen during fit
    :ivar scale_: Per-feature scaling factors
    :ivar data_min_: Per-feature minimum in training data
    :ivar data_max_: Per-feature maximum in training data
    """

    def __init__(
        self,
        feature_range: tuple = (0, 1),
        copy: bool = True,
        clip: bool = False,
    ) -> None:
        """Initialize the scaler with sklearn MinMaxScaler parameters.

        :param feature_range: Desired range of transformed data
        :param copy: Set to False to perform inplace scaling
        :param clip: Set to True to clip transformed values to feature_range
        """
        self._scaler = MinMaxScaler(feature_range=feature_range, copy=copy, clip=clip)

    def fit(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y: Optional[torch.Tensor] = None,
    ) -> "TorchMinMaxScaler":
        """Compute min and max for later scaling.

        :param X: Training data of shape (n_samples, n_features)
        :param y: Ignored, present for scikit-learn compatibility
        :return: Self for method chaining
        """
        X_numpy = self._to_numpy(X)
        self._scaler.fit(X_numpy, y)
        return self

    def transform(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Scale features to the specified range.

        :param X: Data to transform of shape (n_samples, n_features)
        :return: Transformed data as torch.Tensor
        :raises ValueError: If scaler has not been fitted yet
        """
        self._check_is_fitted()
        X_numpy = self._to_numpy(X)
        X_scaled = self._scaler.transform(X_numpy)
        return torch.from_numpy(X_scaled).float()

    def fit_transform(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fit to data, then transform it.

        :param X: Training data of shape (n_samples, n_features)
        :param y: Ignored, present for scikit-learn compatibility
        :return: Transformed data as torch.Tensor
        """
        X_numpy = self._to_numpy(X)
        X_scaled = self._scaler.fit_transform(X_numpy, y)
        return torch.from_numpy(X_scaled).float()

    def inverse_transform(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Undo the scaling transformation.

        :param X: Scaled data of shape (n_samples, n_features)
        :return: Data in original scale as torch.Tensor
        :raises ValueError: If scaler has not been fitted yet
        """
        self._check_is_fitted()
        X_numpy = self._to_numpy(X)
        X_original = self._scaler.inverse_transform(X_numpy)
        return torch.from_numpy(X_original).float()

    @property
    def min_(self) -> Optional[torch.Tensor]:
        """Get per-feature adjustment for minimum.

        :return: Tensor of min values or None if not fitted
        """
        if hasattr(self._scaler, "min_"):
            return torch.from_numpy(self._scaler.min_).float()
        return None

    @property
    def scale_(self) -> Optional[torch.Tensor]:
        """Get per-feature relative scaling.

        :return: Tensor of scale values or None if not fitted
        """
        if hasattr(self._scaler, "scale_"):
            return torch.from_numpy(self._scaler.scale_).float()
        return None

    @property
    def data_min_(self) -> Optional[torch.Tensor]:
        """Get per-feature minimum seen during fit.

        :return: Tensor of data min values or None if not fitted
        """
        if hasattr(self._scaler, "data_min_"):
            return torch.from_numpy(self._scaler.data_min_).float()
        return None

    @property
    def data_max_(self) -> Optional[torch.Tensor]:
        """Get per-feature maximum seen during fit.

        :return: Tensor of data max values or None if not fitted
        """
        if hasattr(self._scaler, "data_max_"):
            return torch.from_numpy(self._scaler.data_max_).float()
        return None

    def _to_numpy(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array.

        :param X: Input data as tensor or array
        :return: Data as numpy array
        """
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        return X

    def _check_is_fitted(self) -> None:
        """Check if the scaler has been fitted.

        :raises ValueError: If scaler has not been fitted yet
        """
        if not hasattr(self._scaler, "scale_"):
            raise ValueError(
                "This TorchMinMaxScaler instance is not fitted yet. "
                "Call 'fit' or 'fit_transform' before using 'transform' "
                "or 'inverse_transform'."
            )


# Backward compatibility alias
MinMaxScalerTorch = TorchMinMaxScaler
