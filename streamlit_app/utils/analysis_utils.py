"""
Analysis Utilities

This module provides utility functions for statistical analysis workflows,
integrating src/analysis modules with UI components for better separation of concerns.

Design Principles:
- Separation of Concerns: UI logic vs Business logic
- Testability: Pure Python functions without Streamlit dependencies
- Reusability: Functions can be used across different pages
- Maintainability: Easy to modify business logic without touching UI
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
from pathlib import Path
import sys

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.analysis.statistics import (
        StatisticsAnalyzer,
        BasicStatistics,
        RiskMetrics,
        DistributionAnalysis,
    )
    from src.analysis.returns import ReturnAnalyzer, ReturnStatistics
    from src.analysis.volatility import VolatilityAnalyzer, VolatilityStatistics
    from src.analysis.correlation import CorrelationAnalyzer, CorrelationStatistics
    from src.config import settings
except ImportError as e:
    # Handle import errors gracefully for testing
    print(f"Import warning in analysis_utils: {e}")


class AnalysisManager:
    """Manager class for statistical analysis operations"""

    def __init__(self):
        self.statistics_analyzer = StatisticsAnalyzer()
        self.return_analyzer = ReturnAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()

    def initialize_session_state(self, session_state: Dict) -> None:
        """Initialize session state for analysis"""

        if "analysis_cache" not in session_state:
            session_state["analysis_cache"] = {}

        if "analysis_results" not in session_state:
            session_state["analysis_results"] = {}

    def analyze_basic_statistics(
        self,
        data: Union[pd.Series, pd.DataFrame],
        ticker: str,
        session_state: Dict,
        data_type: str = "price",
    ) -> Tuple[bool, str]:
        """
        Perform basic statistical analysis

        Args:
            data: Price or return data
            ticker: Symbol identifier
            session_state: Streamlit session state
            data_type: Type of data ('price', 'returns', 'features')

        Returns:
            Tuple of (success, message)
        """

        try:
            results = {}
            analysis_key = f"{ticker}_{data_type}_statistics"

            if isinstance(data, pd.DataFrame):
                # Analyze each column
                for column in data.columns:
                    if data[column].dtype in ["float64", "int64"]:
                        series_data = data[column].dropna()
                        if len(series_data) > 10:  # Minimum data requirement
                            basic_stats = (
                                self.statistics_analyzer.calculate_basic_statistics(
                                    series_data
                                )
                            )
                            results[column] = basic_stats
            else:
                # Single series analysis
                if data.dtype in ["float64", "int64"]:
                    clean_data = data.dropna()
                    if len(clean_data) > 10:
                        basic_stats = (
                            self.statistics_analyzer.calculate_basic_statistics(
                                clean_data
                            )
                        )
                        results[ticker] = basic_stats

            if not results:
                return False, "No valid data for statistical analysis"

            # Store results
            session_state["analysis_cache"][analysis_key] = {
                "results": results,
                "type": "basic_statistics",
                "data_type": data_type,
                "calculated_at": datetime.now(),
                "ticker": ticker,
            }

            return True, f"Basic statistics calculated for {len(results)} series"

        except Exception as e:
            return False, f"Statistical analysis failed: {str(e)}"

    def analyze_distribution(
        self,
        data: Union[pd.Series, pd.DataFrame],
        ticker: str,
        session_state: Dict,
        data_type: str = "returns",
    ) -> Tuple[bool, str]:
        """
        Perform distribution analysis including normality tests

        Args:
            data: Data for distribution analysis
            ticker: Symbol identifier
            session_state: Streamlit session state
            data_type: Type of data being analyzed

        Returns:
            Tuple of (success, message)
        """

        try:
            results = {}
            analysis_key = f"{ticker}_{data_type}_distribution"

            if isinstance(data, pd.DataFrame):
                # Analyze each column
                for column in data.columns:
                    if data[column].dtype in ["float64", "int64"]:
                        series_data = data[column].dropna()
                        if len(series_data) > 50:  # Minimum for normality test
                            dist_analysis = (
                                self.statistics_analyzer.analyze_distribution(
                                    series_data
                                )
                            )
                            results[column] = dist_analysis
            else:
                # Single series analysis
                if data.dtype in ["float64", "int64"]:
                    clean_data = data.dropna()
                    if len(clean_data) > 50:
                        dist_analysis = self.statistics_analyzer.analyze_distribution(
                            clean_data
                        )
                        results[ticker] = dist_analysis

            if not results:
                return (
                    False,
                    "Insufficient data for distribution analysis (need >50 observations)",
                )

            # Store results
            session_state["analysis_cache"][analysis_key] = {
                "results": results,
                "type": "distribution_analysis",
                "data_type": data_type,
                "calculated_at": datetime.now(),
                "ticker": ticker,
            }

            return True, f"Distribution analysis completed for {len(results)} series"

        except Exception as e:
            return False, f"Distribution analysis failed: {str(e)}"

    def analyze_returns(
        self,
        price_data: pd.Series,
        ticker: str,
        session_state: Dict,
        return_type: str = "simple",
    ) -> Tuple[bool, str]:
        """
        Perform comprehensive return analysis

        Args:
            price_data: Price series
            ticker: Symbol identifier
            session_state: Streamlit session state
            return_type: Type of returns ('simple' or 'log')

        Returns:
            Tuple of (success, message)
        """

        try:
            clean_prices = price_data.dropna()
            if len(clean_prices) < 20:
                return False, "Insufficient price data for return analysis"

            # Calculate return statistics
            return_stats = self.return_analyzer.analyze_returns(
                clean_prices, return_type
            )

            # Calculate returns series
            if return_type == "simple":
                returns = self.return_analyzer.calculate_simple_returns(clean_prices)
            else:
                returns = self.return_analyzer.calculate_log_returns(clean_prices)

            # Store results
            analysis_key = f"{ticker}_returns_analysis"
            session_state["analysis_cache"][analysis_key] = {
                "return_statistics": return_stats,
                "returns_series": returns,
                "return_type": return_type,
                "type": "returns_analysis",
                "calculated_at": datetime.now(),
                "ticker": ticker,
            }

            return True, f"Return analysis completed for {ticker}"

        except Exception as e:
            return False, f"Return analysis failed: {str(e)}"

    def analyze_volatility(
        self,
        price_data: pd.Series,
        ticker: str,
        session_state: Dict,
        ohlc_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """
        Perform volatility analysis

        Args:
            price_data: Price series
            ticker: Symbol identifier
            session_state: Streamlit session state
            ohlc_data: Optional OHLC data for advanced volatility measures

        Returns:
            Tuple of (success, message)
        """

        try:
            clean_prices = price_data.dropna()
            if len(clean_prices) < 30:
                return False, "Insufficient price data for volatility analysis"

            # Calculate returns for volatility analysis
            returns = self.return_analyzer.calculate_simple_returns(clean_prices)

            # Perform volatility analysis
            vol_stats = self.volatility_analyzer.analyze_volatility(returns, ohlc_data)

            # Calculate various volatility measures
            simple_vol = self.volatility_analyzer.calculate_simple_volatility(returns)
            ewma_vol = self.volatility_analyzer.calculate_ewma_volatility(returns)

            # Store results
            analysis_key = f"{ticker}_volatility_analysis"
            session_state["analysis_cache"][analysis_key] = {
                "volatility_statistics": vol_stats,
                "simple_volatility": simple_vol,
                "ewma_volatility": ewma_vol,
                "type": "volatility_analysis",
                "calculated_at": datetime.now(),
                "ticker": ticker,
            }

            return True, f"Volatility analysis completed for {ticker}"

        except Exception as e:
            return False, f"Volatility analysis failed: {str(e)}"

    def analyze_correlation(
        self,
        data: pd.DataFrame,
        ticker: str,
        session_state: Dict,
        method: str = "pearson",
    ) -> Tuple[bool, str]:
        """
        Perform correlation analysis

        Args:
            data: Multi-column DataFrame
            ticker: Symbol identifier
            session_state: Streamlit session state
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Tuple of (success, message)
        """

        try:
            if len(data.columns) < 2:
                return False, "Need at least 2 columns for correlation analysis"

            # Calculate correlation matrix
            corr_matrix = self.correlation_analyzer.calculate_correlation_matrix(
                data, method
            )

            # Analyze correlation structure
            corr_stats = self.correlation_analyzer.analyze_correlation_structure(data)

            # Store results
            analysis_key = f"{ticker}_correlation_analysis"
            session_state["analysis_cache"][analysis_key] = {
                "correlation_matrix": corr_matrix,
                "correlation_statistics": corr_stats,
                "method": method,
                "type": "correlation_analysis",
                "calculated_at": datetime.now(),
                "ticker": ticker,
            }

            return True, f"Correlation analysis completed for {ticker}"

        except Exception as e:
            return False, f"Correlation analysis failed: {str(e)}"

    def analyze_feature_quality(
        self,
        feature_data: pd.DataFrame,
        ticker: str,
        session_state: Dict,
        target_column: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Perform comprehensive feature quality analysis

        Args:
            feature_data: Feature DataFrame
            ticker: Symbol identifier
            session_state: Streamlit session state
            target_column: Optional target column for supervised analysis

        Returns:
            Tuple of (success, message)
        """

        try:
            results = {}

            # Basic statistics for each feature
            basic_stats_success, basic_message = self.analyze_basic_statistics(
                feature_data, ticker, session_state, "features"
            )

            # Distribution analysis for each feature
            dist_success, dist_message = self.analyze_distribution(
                feature_data, ticker, session_state, "features"
            )

            # Correlation analysis among features
            corr_success, corr_message = self.analyze_correlation(
                feature_data, ticker, session_state
            )

            # Compile results
            analysis_key = f"{ticker}_feature_quality"
            session_state["analysis_cache"][analysis_key] = {
                "basic_statistics": basic_stats_success,
                "distribution_analysis": dist_success,
                "correlation_analysis": corr_success,
                "type": "feature_quality",
                "calculated_at": datetime.now(),
                "ticker": ticker,
                "feature_count": len(feature_data.columns),
                "sample_count": len(feature_data),
            }

            successful_analyses = sum([basic_stats_success, dist_success, corr_success])
            return (
                True,
                f"Feature quality analysis completed ({successful_analyses}/3 analyses successful)",
            )

        except Exception as e:
            return False, f"Feature quality analysis failed: {str(e)}"

    def get_analysis_summary(self, session_state: Dict) -> Dict[str, Any]:
        """Get summary of available analysis results"""

        analysis_cache = session_state.get("analysis_cache", {})

        analysis_types = {}
        for key, result in analysis_cache.items():
            analysis_type = result.get("type", "unknown")
            if analysis_type not in analysis_types:
                analysis_types[analysis_type] = 0
            analysis_types[analysis_type] += 1

        return {
            "total_analyses": len(analysis_cache),
            "analysis_types": analysis_types,
            "available_analyses": list(analysis_cache.keys()),
        }

    def get_analysis_results(
        self, analysis_key: str, session_state: Dict
    ) -> Optional[Dict[str, Any]]:
        """Get analysis results by key"""

        return session_state.get("analysis_cache", {}).get(analysis_key)

    def format_statistics_for_display(
        self, basic_stats: BasicStatistics
    ) -> Dict[str, str]:
        """Format BasicStatistics for UI display"""

        return {
            "Count": f"{basic_stats.count:,}",
            "Mean": f"{basic_stats.mean:.6f}",
            "Std Dev": f"{basic_stats.std:.6f}",
            "Min": f"{basic_stats.min:.6f}",
            "25%": f"{basic_stats.percentile_25:.6f}",
            "Median": f"{basic_stats.median:.6f}",
            "75%": f"{basic_stats.percentile_75:.6f}",
            "Max": f"{basic_stats.max:.6f}",
            "Skewness": f"{basic_stats.skewness:.4f}",
            "Kurtosis": f"{basic_stats.kurtosis:.4f}",
        }

    def format_distribution_for_display(
        self, dist_analysis: DistributionAnalysis
    ) -> Dict[str, str]:
        """Format DistributionAnalysis for UI display"""

        # Determine normality based on p-value
        is_normal = "Yes" if dist_analysis.normality_p_value > 0.05 else "No"
        has_arch = "Yes" if dist_analysis.has_arch_effects else "No"
        has_autocorr = "Yes" if dist_analysis.has_autocorrelation else "No"

        return {
            "Normality Test Statistic": f"{dist_analysis.normality_test_statistic:.4f}",
            "Normality P-Value": f"{dist_analysis.normality_p_value:.6f}",
            "Is Normal (α=0.05)": is_normal,
            "ARCH Effects Present": has_arch,
            "ARCH Test P-Value": f"{dist_analysis.arch_test_p_value:.6f}",
            "Autocorrelation Present": has_autocorr,
            "Autocorr Test P-Value": f"{dist_analysis.ljung_box_p_value:.6f}",
        }
