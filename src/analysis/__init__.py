"""
Quant Analytics Tool - Analysis Module

AFML-compliant financial data analysis components.
Provides comprehensive statistical and technical analysis capabilities
for quantitative finance research and development.

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

from .returns import (
    ReturnAnalyzer,
    calculate_simple_returns,
    calculate_log_returns,
    calculate_cumulative_returns,
)

from .volatility import (
    VolatilityAnalyzer,
    calculate_simple_volatility,
    calculate_ewma_volatility,
    calculate_garman_klass_volatility,
)

from .statistics import (
    StatisticsAnalyzer,
    calculate_basic_statistics,
    calculate_risk_metrics,
    analyze_distribution,
)

from .correlation import (
    CorrelationAnalyzer,
    calculate_correlation_matrix,
    calculate_rolling_correlation,
)

__all__ = [
    # Return Analysis
    "ReturnAnalyzer",
    "calculate_simple_returns",
    "calculate_log_returns",
    "calculate_cumulative_returns",
    # Volatility Analysis
    "VolatilityAnalyzer",
    "calculate_simple_volatility",
    "calculate_ewma_volatility",
    "calculate_garman_klass_volatility",
    # Statistical Analysis
    "StatisticsAnalyzer",
    "calculate_basic_statistics",
    "calculate_risk_metrics",
    "analyze_distribution",
    # Correlation Analysis
    "CorrelationAnalyzer",
    "calculate_correlation_matrix",
    "calculate_rolling_correlation",
]

__version__ = "1.0.0"
__author__ = "Quant Analytics Tool Team"
__description__ = "AFML-compliant financial data analysis module"
