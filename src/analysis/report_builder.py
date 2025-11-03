"""Report builder for constructing analysis reports.

Provides incremental construction of analysis reports from correlation
and importance results.
"""

from typing import Dict, List, Optional

import pandas as pd

from .core import CorrelationResult, FeatureImportanceResult


class AnalysisReportBuilder:
    """Constructs comprehensive analysis reports incrementally.

    Supports method chaining for adding correlation and importance results
    before building the final report DataFrame.
    """

    def __init__(self, feature_names: tuple[str, ...]):
        """Initialize builder.

        :param feature_names: Names of features/parameters
        """
        self.feature_names = feature_names
        self._correlation_result: Optional[CorrelationResult] = None
        self._importance_results: List[FeatureImportanceResult] = []
        self._metadata: Dict[str, any] = {}

    def add_correlation(self, result: CorrelationResult) -> "AnalysisReportBuilder":
        """Add correlation analysis result.

        :param result: CorrelationResult
        :return: Self for method chaining
        """
        self._correlation_result = result
        return self

    def add_importance(self, result: FeatureImportanceResult) -> "AnalysisReportBuilder":
        """Add feature importance result.

        :param result: FeatureImportanceResult
        :return: Self for method chaining
        """
        self._importance_results.append(result)
        return self

    def add_metadata(self, key: str, value: any) -> "AnalysisReportBuilder":
        """Add metadata to report.

        :param key: Metadata key
        :param value: Metadata value
        :return: Self for method chaining
        """
        self._metadata[key] = value
        return self

    def build(self) -> pd.DataFrame:
        """Build and return comprehensive report DataFrame.

        :return: DataFrame with all analysis results
        """
        summary_data = []

        for i, name in enumerate(self.feature_names):
            row = self._build_row_for_feature(i, name)
            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def export_csv(self, filepath: str) -> None:
        """Export report to CSV.

        :param filepath: Path to save CSV
        """
        df = self.build()
        df.to_csv(filepath, index=False)

    def _build_row_for_feature(self, index: int, name: str) -> Dict[str, any]:
        """Build a single row for a feature.

        :param index: Feature index
        :param name: Feature name
        :return: Dict representing one row
        """
        row = {"Parameter": name}

        # Add correlation results
        if self._correlation_result:
            row["Pearson_Corr"] = self._correlation_result.pearson[index]
            row["Spearman_Corr"] = self._correlation_result.spearman[index]
            row["Partial_Corr"] = self._correlation_result.partial[index]

        # Add importance results
        for result in self._importance_results:
            col_name = result.method_name
            row[col_name] = result.importance_scores[index]

            if result.confidence_intervals is not None:
                row[f"{col_name}_Std"] = result.confidence_intervals[index]

        return row
