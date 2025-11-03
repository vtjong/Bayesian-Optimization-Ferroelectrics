"""Parameter correlation and feature importance analysis.

Provides unified interface for parameter analysis with backward compatibility.
Implementation is modularized across separate modules:
- core: Base classes and result types
- analyzers: Analysis computation implementations
- facade: Unified interface layer
- visualizer: Visualization utilities
- report_builder: Report construction
"""

# Import from refactored modules - maintains backward compatibility
from .facade import ParameterAnalyzer

# Export for backward compatibility
__all__ = ["ParameterAnalyzer"]
