"""Visualization utilities for analysis results.

Handles all plotting and figure generation separately from computation.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .core import CorrelationResult, FeatureImportanceResult


class AnalysisVisualizer:
    """Handles visualization for analysis results.

    Provides static methods for plotting correlation matrices, importance
    comparisons, and correlation barplots.
    """

    @staticmethod
    def plot_correlation_matrix(
        result: CorrelationResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot correlation matrix between parameters.

        :param result: CorrelationResult from CorrelationAnalyzer
        :param save_path: Optional path to save figure
        :return: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            result.param_param_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            xticklabels=result.feature_names,
            yticklabels=result.feature_names,
            ax=ax,
        )
        ax.set_title(
            "Parameter-Parameter Correlation Matrix",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_correlation_barplot(
        result: CorrelationResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot bar plot of parameter-FOM correlations.

        :param result: CorrelationResult
        :param save_path: Optional path to save figure
        :return: Matplotlib figure
        """
        x = np.arange(len(result.feature_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x - width / 2, result.pearson, width, label="Pearson", alpha=0.8)
        ax.bar(x + width / 2, result.spearman, width, label="Spearman", alpha=0.8)

        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Parameters", fontsize=12)
        ax.set_ylabel("Correlation with FOM", fontsize=12)
        ax.set_title("Parameter-FOM Correlations", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(result.feature_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_feature_importance_comparison(
        results: list[FeatureImportanceResult],
        feature_names: tuple[str, ...],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot comparison of different feature importance metrics.

        :param results: List of FeatureImportanceResult objects
        :param feature_names: Feature names
        :param save_path: Optional path to save figure
        :return: Matplotlib figure
        """
        if not results:
            raise ValueError("No importance results provided")

        normalized_results, method_names = AnalysisVisualizer._normalize_importance_scores(results)
        fig = AnalysisVisualizer._create_importance_comparison_plot(
            normalized_results, method_names, feature_names, save_path
        )
        return fig

    @staticmethod
    def _normalize_importance_scores(
        results: list[FeatureImportanceResult],
    ) -> tuple[list[np.ndarray], list[str]]:
        """Normalize importance scores to [0, 1] for comparison.

        :param results: List of FeatureImportanceResult
        :return: Tuple of (normalized_arrays, method_names)
        """
        normalized_results = []
        method_names = []

        for result in results:
            imp = result.importance_scores
            imp_min, imp_max = imp.min(), imp.max()
            if imp_max > imp_min:
                imp_normalized = (imp - imp_min) / (imp_max - imp_min)
            else:
                imp_normalized = np.ones_like(imp)
            normalized_results.append(imp_normalized)
            method_names.append(result.method_name)

        return normalized_results, method_names

    @staticmethod
    def _create_importance_comparison_plot(
        normalized_results: list[np.ndarray],
        method_names: list[str],
        feature_names: tuple[str, ...],
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create the actual comparison plot.

        :param normalized_results: Normalized importance arrays
        :param method_names: Names of methods
        :param feature_names: Feature names
        :param save_path: Optional save path
        :return: Matplotlib figure
        """
        x = np.arange(len(feature_names))
        width = 0.8 / len(method_names)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (method, imp_data) in enumerate(zip(method_names, normalized_results)):
            offset = (i - len(method_names) / 2) * width + width / 2
            ax.bar(x + offset, imp_data, width, label=method, alpha=0.8)

        ax.set_xlabel("Parameters", fontsize=12)
        ax.set_ylabel("Importance Score (Normalized)", fontsize=12)
        ax.set_title("Feature Importance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
