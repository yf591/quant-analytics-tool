"""
Test suite for statistics analysis functionality.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

from src.analysis.statistics import (
    StatisticsAnalyzer,
    BasicStatistics,
    RiskMetrics,
)


class TestStatisticsAnalyzer:
    """Test cases for StatisticsAnalyzer class."""

    def setup_method(self):
        """Set up test data."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate data with known statistical properties
        data = np.random.normal(0.001, 0.02, 252)
        self.data = pd.Series(data, index=dates, name="data")

        # Initialize analyzer
        self.analyzer = StatisticsAnalyzer()

    def test_initialization(self):
        """Test StatisticsAnalyzer initialization."""
        analyzer = StatisticsAnalyzer()
        assert analyzer is not None

    def test_basic_statistics_calculation(self):
        """Test basic statistics calculation."""
        stats = self.analyzer.calculate_basic_statistics(self.data)

        # Check that statistics are properly calculated
        assert isinstance(stats, BasicStatistics)
        assert stats.count == len(self.data)
        assert np.isfinite(stats.mean)
        assert stats.std > 0
        assert (
            stats.min
            <= stats.percentile_25
            <= stats.median
            <= stats.percentile_75
            <= stats.max
        )
        assert np.isfinite(stats.skewness)
        assert np.isfinite(stats.kurtosis)

    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        # Create price series from returns
        price_series = (1 + self.data).cumprod() * 100
        risk_metrics = self.analyzer.calculate_risk_metrics(self.data, price_series)

        # Check that risk metrics are properly calculated
        assert isinstance(risk_metrics, RiskMetrics)
        # Note: VaR is calculated as positive value representing loss
        assert risk_metrics.var_95 >= 0  # VaR should be positive (loss amount)
        assert risk_metrics.var_99 >= 0  # VaR should be positive
        assert risk_metrics.var_99 >= risk_metrics.var_95  # 99% VaR should be higher
        assert (
            risk_metrics.cvar_95 >= risk_metrics.var_95
        )  # CVaR should be higher than VaR
        assert (
            risk_metrics.cvar_99 >= risk_metrics.var_99
        )  # CVaR should be higher than VaR
        assert np.isfinite(risk_metrics.downside_deviation)
        assert np.isfinite(risk_metrics.sortino_ratio)
        assert np.isfinite(risk_metrics.calmar_ratio)

    def test_normality_tests(self):
        """Test normality testing functions."""
        # Test with normal data
        normal_data = pd.Series(np.random.normal(0, 1, 1000))
        normality_results = self.analyzer.test_normality(normal_data)

        # Check that results are returned as tuple (statistic, p_value)
        assert isinstance(normality_results, tuple)
        assert len(normality_results) == 2
        statistic, p_value = normality_results
        assert np.isfinite(statistic)
        assert 0 <= p_value <= 1

    def test_distribution_analysis(self):
        """Test distribution analysis functionality."""
        # Test distribution analysis
        dist_analysis = self.analyzer.analyze_distribution(self.data)

        # Check the actual attributes of DistributionAnalysis
        assert hasattr(dist_analysis, "normality_test_statistic")
        assert hasattr(dist_analysis, "normality_p_value")
        assert hasattr(dist_analysis, "is_normal")
        assert hasattr(dist_analysis, "autocorrelation_lag1")

        # Check values are reasonable
        assert np.isfinite(dist_analysis.normality_test_statistic)
        assert 0 <= dist_analysis.normality_p_value <= 1
        assert isinstance(dist_analysis.is_normal, (bool, np.bool_))

    def test_var_calculation(self):
        """Test VaR calculation."""
        var_95 = self.analyzer.calculate_var(self.data, confidence_level=0.05)
        var_99 = self.analyzer.calculate_var(self.data, confidence_level=0.01)

        # Check that VaR values are positive (representing loss amounts)
        assert var_95 >= 0
        assert var_99 >= 0
        assert (
            var_99 >= var_95
        )  # 99% VaR should be higher (more extreme)    def test_basic_statistics_to_dict(self):
        """Test BasicStatistics to_dict method."""
        stats = self.analyzer.calculate_basic_statistics(self.data)
        stats_dict = stats.to_dict()

        # Check that all expected keys are present
        expected_keys = [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "median",
            "75%",
            "max",
            "skewness",
            "kurtosis",
        ]

        for key in expected_keys:
            assert key in stats_dict
            assert np.isfinite(stats_dict[key]) or key == "count"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty series
        empty_series = pd.Series([], dtype=float)

        try:
            result = self.analyzer.calculate_basic_statistics(empty_series)
            # If no error raised, check that result indicates empty data
            assert result.count == 0
        except (ValueError, ZeroDivisionError, IndexError):
            # These errors are acceptable for empty data
            pass

        # Test with series containing only NaN
        nan_series = pd.Series([np.nan, np.nan, np.nan])

        try:
            result = self.analyzer.calculate_basic_statistics(nan_series)
            # If successful, result should reflect no valid data
            assert result.count == 0 or np.isnan(result.mean)
        except (ValueError, ZeroDivisionError, IndexError):
            # These errors are acceptable for NaN-only data
            pass


class TestBasicStatistics:
    """Test cases for BasicStatistics dataclass."""

    def test_basic_statistics_creation(self):
        """Test BasicStatistics creation and access."""
        stats = BasicStatistics(
            count=100,
            mean=0.001,
            std=0.02,
            min=-0.05,
            percentile_25=-0.01,
            median=0.0,
            percentile_75=0.01,
            max=0.06,
            skewness=0.1,
            kurtosis=3.2,
        )

        assert stats.count == 100
        assert stats.mean == 0.001
        assert stats.std == 0.02
        assert stats.min == -0.05
        assert stats.percentile_25 == -0.01
        assert stats.median == 0.0
        assert stats.percentile_75 == 0.01
        assert stats.max == 0.06
        assert stats.skewness == 0.1
        assert stats.kurtosis == 3.2

    def test_basic_statistics_to_dict(self):
        """Test BasicStatistics to_dict conversion."""
        stats = BasicStatistics(
            count=100,
            mean=0.001,
            std=0.02,
            min=-0.05,
            percentile_25=-0.01,
            median=0.0,
            percentile_75=0.01,
            max=0.06,
            skewness=0.1,
            kurtosis=3.2,
        )

        stats_dict = stats.to_dict()

        assert stats_dict["count"] == 100
        assert stats_dict["mean"] == 0.001
        assert stats_dict["std"] == 0.02
        assert stats_dict["min"] == -0.05
        assert stats_dict["25%"] == -0.01
        assert stats_dict["median"] == 0.0
        assert stats_dict["75%"] == 0.01
        assert stats_dict["max"] == 0.06
        assert stats_dict["skewness"] == 0.1
        assert stats_dict["kurtosis"] == 3.2


def run_tests():
    """Run all tests."""
    test_stats = TestStatisticsAnalyzer()
    test_stats.setup_method()

    print("Testing StatisticsAnalyzer initialization...")
    test_stats.test_initialization()

    print("Testing basic statistics calculation...")
    test_stats.test_basic_statistics_calculation()

    print("Testing risk metrics calculation...")
    test_stats.test_risk_metrics_calculation()

    print("Testing normality tests...")
    test_stats.test_normality_tests()

    print("Testing distribution fitting...")
    test_stats.test_distribution_fitting()

    print("Testing comprehensive analysis...")
    test_stats.test_comprehensive_analysis()

    print("Testing BasicStatistics to_dict...")
    test_stats.test_basic_statistics_to_dict()

    print("Testing error handling...")
    test_stats.test_error_handling()

    # Test BasicStatistics dataclass
    test_basic_stats = TestBasicStatistics()

    print("Testing BasicStatistics creation...")
    test_basic_stats.test_basic_statistics_creation()

    print("Testing BasicStatistics dict conversion...")
    test_basic_stats.test_basic_statistics_to_dict()

    print("All statistics analysis tests passed!")


if __name__ == "__main__":
    run_tests()
