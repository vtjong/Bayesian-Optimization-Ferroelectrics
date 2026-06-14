"""Unified interface for parameter analysis.

Orchestrates analysis computation, visualization, and report generation.
"""

from typing import List, Optional

import gpytorch
import torch

from .analyzers import (
    ARDImportanceAnalyzer,
    CorrelationAnalyzer,
    SobolSensitivityAnalyzer,
)
from .core import CorrelationResult, FeatureImportanceResult
from .helpers import convert_tensor_to_numpy, generate_default_feature_names
from .report_builder import AnalysisReportBuilder
from .visualizer import AnalysisVisualizer


class ParameterAnalyzer:
    """Unified interface for comprehensive parameter analysis.

    Orchestrates analysis computation, visualization, and report generation.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        feature_names: Optional[List[str]] = None,
    ):
        """Initialize analyzer.

        :param train_x: Training inputs (n_samples, n_features)
        :param train_y: Training targets (n_samples,)
        :param feature_names: Names of input features/parameters
        """
        X_np = convert_tensor_to_numpy(train_x)
        y_np = convert_tensor_to_numpy(train_y)

        if feature_names is None:
            feature_names = generate_default_feature_names(X_np.shape[1])

        self.feature_names = feature_names
        self.X_np = X_np
        self.y_np = y_np

        self._correlation_analyzer = CorrelationAnalyzer(X_np, y_np, feature_names)
        self._visualizer = AnalysisVisualizer()
        self._correlation_result: Optional[CorrelationResult] = None
        self._importance_results: List[FeatureImportanceResult] = []

    @property
    def importance_results(self) -> List[FeatureImportanceResult]:
        """Feature-importance results computed so far (e.g. ARD, Sobol)."""
        return list(self._importance_results)

    @property
    def correlation_result(self) -> Optional[CorrelationResult]:
        """The most recent correlation result, or None if not yet computed."""
        return self._correlation_result

    def compute_correlations(self, methods: Optional[List[str]] = None) -> CorrelationResult:
        """Compute correlation coefficients.

        :param methods: List of correlation methods ['pearson', 'spearman']
        :return: CorrelationResult with correlation matrices
        """
        if methods is None:
            methods = ["pearson", "spearman"]

        result = self._correlation_analyzer.analyze(methods=methods)
        self._correlation_result = result
        return result

    def extract_ard_importance(
        self, model: gpytorch.models.ExactGP, inverse_lengthscale: bool = True
    ) -> FeatureImportanceResult:
        """Extract ARD importance from GP lengthscales.

        :param model: Trained GP model
        :param inverse_lengthscale: Use inverse lengthscale as importance
        :return: FeatureImportanceResult
        """
        analyzer = ARDImportanceAnalyzer(
            model=model,
            feature_names=list(self.feature_names),
        )

        result = analyzer.analyze(inverse_lengthscale=inverse_lengthscale)
        self._importance_results.append(result)
        return result

    def compute_sobol_indices(
        self,
        model: gpytorch.models.ExactGP,
        likelihood: gpytorch.likelihoods.Likelihood,
        n_samples: int = 10000,
        n_bootstrap: int = 100,
    ) -> FeatureImportanceResult:
        """Compute Sobol global sensitivity indices.

        :param model: Trained GP model
        :param likelihood: GP likelihood
        :param n_samples: Number of samples for Sobol analysis
        :param n_bootstrap: Bootstrap samples for uncertainty
        :return: FeatureImportanceResult with Sobol indices
        """
        analyzer = SobolSensitivityAnalyzer(
            model=model,
            likelihood=likelihood,
            X=self.X_np,
            y=self.y_np,
            feature_names=list(self.feature_names),
        )

        result = analyzer.analyze(n_samples=n_samples, n_bootstrap=n_bootstrap)
        self._importance_results.append(result)
        return result

    def generate_summary_report(self):
        """Generate summary report with all analysis results.

        Aggregates correlation and importance results into a single DataFrame.
        """
        builder = AnalysisReportBuilder(tuple(self.feature_names))

        if self._correlation_result:
            builder.add_correlation(self._correlation_result)

        for result in self._importance_results:
            builder.add_importance(result)

        return builder.build()

    def plot_correlation_matrix(self, save_path: Optional[str] = None):
        """Plot correlation matrix between parameters."""
        if self._correlation_result is None:
            raise ValueError("Run compute_correlations() first")
        return self._visualizer.plot_correlation_matrix(
            self._correlation_result, save_path=save_path
        )

    def plot_feature_importance_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of different feature importance metrics."""
        if not self._importance_results:
            raise ValueError("No importance results available")
        return self._visualizer.plot_feature_importance_comparison(
            self._importance_results,
            tuple(self.feature_names),
            save_path=save_path,
        )

    def plot_correlation_barplot(self, save_path: Optional[str] = None):
        """Plot bar plot of parameter-FOM correlations."""
        if self._correlation_result is None:
            raise ValueError("Run compute_correlations() first")
        return self._visualizer.plot_correlation_barplot(
            self._correlation_result, save_path=save_path
        )
