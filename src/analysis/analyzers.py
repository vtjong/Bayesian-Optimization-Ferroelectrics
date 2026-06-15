"""Concrete analyzer implementations.

Each analyzer implements BaseAnalyzer and provides a specific analysis method.
"""

from typing import Dict, List, Optional

import gpytorch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from .core import (
    BaseAnalyzer,
    CorrelationResult,
    DataAnalyzer,
    FeatureImportanceResult,
    ModelAnalyzer,
)

try:
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sample

    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False


class ARDImportanceAnalyzer(BaseAnalyzer):
    """Analyzer for ARD lengthscale-based feature importance.

    Only needs model, not likelihood.
    """

    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        feature_names: List[str],
    ):
        """Initialize with model only.

        :param model: Trained GP model
        :param feature_names: Feature names
        """
        self.model = model
        self.feature_names = tuple(feature_names)

    @property
    def name(self) -> str:
        return "ARD_Importance"

    def analyze(self, inverse_lengthscale: bool = True) -> FeatureImportanceResult:
        """Extract feature importance from ARD lengthscales.

        :param inverse_lengthscale: Use 1/lengthscale as importance
        :return: Feature importance result
        """
        self.model.eval()

        # Extract lengthscales
        base_kernel = (
            self.model.covar_module.base_kernel
            if hasattr(self.model.covar_module, "base_kernel")
            else self.model.covar_module
        )

        lengthscales = base_kernel.lengthscale.detach().cpu().numpy()
        if lengthscales.ndim > 1:
            lengthscales = lengthscales.flatten()

        # Compute importance
        if inverse_lengthscale:
            importance = 1.0 / lengthscales
            importance = importance / importance.sum()
        else:
            importance = -lengthscales
            importance = (importance - importance.min()) / (
                importance.max() - importance.min() + 1e-10
            )

        return FeatureImportanceResult(
            importance_scores=importance,
            feature_names=self.feature_names,
            method_name=self.name,
            metadata={"lengthscales": lengthscales},
        )


class CorrelationAnalyzer(DataAnalyzer):
    """Analyzer for computing correlations (Pearson, Spearman, Partial)."""

    @property
    def name(self) -> str:
        return "Correlation"

    def analyze(self, methods: Optional[List[str]] = None) -> CorrelationResult:
        """Compute correlation coefficients.

        :param methods: Correlation methods to use ['pearson', 'spearman']
        :return: Correlation result
        """
        if methods is None:
            methods = ["pearson", "spearman"]

        n_features = self.X.shape[1]
        pearson_vals = np.zeros(n_features)
        spearman_vals = np.zeros(n_features)

        for i in range(n_features):
            if "pearson" in methods:
                pearson_vals[i], _ = pearsonr(self.X[:, i], self.y)
            if "spearman" in methods:
                spearman_vals[i], _ = spearmanr(self.X[:, i], self.y)

        # Parameter-parameter correlations
        param_param_matrix = np.corrcoef(self.X.T)

        # Partial correlations
        partial_corr = self._compute_partial_correlations()

        return CorrelationResult(
            pearson=pearson_vals,
            spearman=spearman_vals,
            partial=partial_corr,
            param_param_matrix=param_param_matrix,
            feature_names=self.feature_names,
        )

    def _compute_partial_correlations(self) -> np.ndarray:
        """Compute partial correlations controlling for other parameters."""
        n_features = self.X.shape[1]
        partial_corr = np.zeros(n_features)

        for i in range(n_features):
            X_other = np.delete(self.X, i, axis=1)

            # Residuals from regressing parameter i on others
            reg_i = LinearRegression().fit(X_other, self.X[:, i])
            residuals_i = self.X[:, i] - reg_i.predict(X_other)

            # Residuals from regressing y on other parameters
            reg_y = LinearRegression().fit(X_other, self.y)
            residuals_y = self.y - reg_y.predict(X_other)

            # Partial correlation
            partial_corr[i], _ = pearsonr(residuals_i, residuals_y)

        return partial_corr


class SobolSensitivityAnalyzer(ModelAnalyzer):
    """Analyzer for Sobol global sensitivity indices."""

    @property
    def name(self) -> str:
        return "Sobol_Sensitivity"

    def analyze(self, n_samples: int = 10000, n_bootstrap: int = 100) -> FeatureImportanceResult:
        """Compute Sobol sensitivity indices.

        :param n_samples: Number of samples for Sobol analysis
        :param n_bootstrap: Bootstrap samples (unused)
        :return: Sobol indices result
        """
        if not SOBOL_AVAILABLE:
            raise ImportError("SALib required. Install with: pip install SALib")

        problem = self._create_sobol_problem()
        sobol_samples = self._generate_sobol_samples(problem, n_samples)
        Y_pred = self._predict(sobol_samples)
        Si = sobol.analyze(problem, Y_pred, calc_second_order=False)

        return self._build_sobol_result(Si)

    def _create_sobol_problem(self) -> Dict:
        """Create Sobol problem structure.

        :return: Problem dict for SALib
        """
        return {
            "num_vars": self.X.shape[1],
            "names": list(self.feature_names),
            "bounds": [[0.0, 1.0]] * self.X.shape[1],
        }

    def _generate_sobol_samples(self, problem: Dict, n_samples: int) -> np.ndarray:
        """Generate Sobol samples.

        :param problem: Problem structure
        :param n_samples: Number of samples
        :return: Sobol samples array
        """
        return sobol_sample.sample(problem, n_samples, calc_second_order=False)

    def _build_sobol_result(self, Si: Dict) -> FeatureImportanceResult:
        """Build FeatureImportanceResult from Sobol indices.

        :param Si: Sobol indices dict from SALib
        :return: Feature importance result
        """
        return FeatureImportanceResult(
            importance_scores=Si["S1"],
            confidence_intervals=Si["S1_conf"],
            feature_names=self.feature_names,
            method_name=self.name,
            metadata={
                "total_order": Si["ST"],
                "total_order_conf": Si["ST_conf"],
            },
        )
