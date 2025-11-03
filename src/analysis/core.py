from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import torch


@dataclass(frozen=True)
class CorrelationResult:
    """Result container for correlation analysis outputs."""

    pearson: np.ndarray
    spearman: np.ndarray
    partial: np.ndarray
    param_param_matrix: np.ndarray
    feature_names: Tuple[str, ...]


@dataclass(frozen=True)
class FeatureImportanceResult:
    """Result container for feature importance analysis outputs.

    :param importance_scores: Main importance scores (n_features,)
    :param confidence_intervals: Optional uncertainty measures (n_features,)
    :param feature_names: Names of features
    :param method_name: Name of analysis method
    :param metadata: Method-specific data (e.g., gradients, raw values)
    """

    importance_scores: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    feature_names: Tuple[str, ...]
    method_name: str = "unknown"
    metadata: Dict[str, any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers.

    All concrete analyzers must implement the analyze() method.
    """

    @abstractmethod
    def analyze(self, *args, **kwargs):
        """Execute analysis and return result."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return name of analyzer for logging/identification."""
        pass


class DataAnalyzer(BaseAnalyzer):
    """Base class for analyzers that only need data (no model)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Initialize with data.

        :param X: Input features (n_samples, n_features)
        :param y: Target values (n_samples,)
        :param feature_names: Names of features
        """
        self.X = X
        self.y = y
        self.feature_names = tuple(feature_names)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input data."""
        if self.X.shape[0] != len(self.y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got {self.X.shape[0]} and {len(self.y)}"
            )
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X columns must match feature_names length. "
                f"Got {self.X.shape[1]} and {len(self.feature_names)}"
            )


class ModelAnalyzer(BaseAnalyzer):
    """Base class for analyzers that require a trained model."""

    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelihood: gpytorch.likelihoods.Likelihood,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ):
        """Initialize with model and data.

        :param model: Trained GP model
        :param likelihood: GP likelihood
        :param X: Input features (n_samples, n_features)
        :param y: Target values (n_samples,)
        :param feature_names: Names of features
        """
        self.model = model
        self.likelihood = likelihood
        self.X = X
        self.y = y
        self.feature_names = tuple(feature_names)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate inputs."""
        if self.X.shape[0] != len(self.y):
            raise ValueError("X and y must have same number of samples")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("X columns must match feature_names length")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Helper to make predictions with model.

        :param X: Input features
        :return: Predictions
        """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            f_pred = self.model(X_tensor)
            y_pred = self.likelihood(f_pred)
            return y_pred.mean.numpy()
