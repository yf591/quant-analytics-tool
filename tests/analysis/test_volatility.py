"""
Test suite for volatility analysis functionality.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

from src.analysis.volatility import (
    VolatilityAnalyzer,
    VolatilityStatistics,
)


class TestVolatilityAnalyzer:
    """Test cases for VolatilityAnalyzer class."""

    def setup_method(self):
        """Set up test data."""
        # Create sample return data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate returns with varying volatility
        returns = np.random.normal(0.001, 0.02, 252)
        self.returns = pd.Series(returns, index=dates, name="returns")

        # Initialize analyzer
        self.analyzer = VolatilityAnalyzer(window=30)

    def test_initialization(self):
        """Test VolatilityAnalyzer initialization."""
        analyzer = VolatilityAnalyzer(window=20)
        assert analyzer.window == 20

        # Test default initialization
        analyzer_default = VolatilityAnalyzer()
        assert analyzer_default.window == 30

    def test_simple_volatility_calculation(self):
        """Test simple volatility calculation."""
        volatility = self.analyzer.calculate_simple_volatility(self.returns)

        # Check that volatility Series is not empty
        assert len(volatility) > 0

        # Check that all values are positive and finite
        assert (volatility > 0).all()
        assert volatility.isna().sum() == 0

        # Check that it's a pandas Series
        assert isinstance(volatility, pd.Series)

    def test_rolling_volatility_calculation(self):
        """Test rolling volatility calculation."""
        # Use simple_volatility since rolling is handled internally
        rolling_vol = self.analyzer.calculate_simple_volatility(self.returns, window=20)

        # Check that all values are positive
        assert (rolling_vol > 0).all()

        # Check that no values are NaN
        assert rolling_vol.isna().sum() == 0

    def test_ewm_volatility_calculation(self):
        """Test exponentially weighted moving average volatility."""
        ewm_vol = self.analyzer.calculate_ewma_volatility(
            self.returns, lambda_param=0.1
        )

        # Check that all values are positive
        assert (ewm_vol > 0).all()

        # Check that no values are NaN
        assert ewm_vol.isna().sum() == 0

        # Check that ewm_vol has reasonable length
        assert len(ewm_vol) > 0

    def test_realized_volatility_calculation(self):
        """Test realized volatility calculation."""
        realized_vol = self.analyzer.calculate_realized_volatility(self.returns)

        # Check that volatility Series is not empty
        assert len(realized_vol) > 0

        # Check that all values are positive and finite
        assert (realized_vol > 0).all()
        assert realized_vol.isna().sum() == 0

    def test_volatility_analysis(self):
        """Test comprehensive volatility analysis."""
        stats = self.analyzer.analyze_volatility(self.returns)

        # Check that all statistics are present and valid
        assert isinstance(stats, VolatilityStatistics)
        assert stats.current_volatility > 0
        assert stats.average_volatility > 0
        assert stats.volatility_std >= 0
        assert np.isfinite(stats.volatility_skewness)
        assert np.isfinite(stats.volatility_kurtosis)

    def test_volatility_statistics_to_dict(self):
        """Test VolatilityStatistics to_dict method."""
        stats = self.analyzer.analyze_volatility(self.returns)
        stats_dict = stats.to_dict()

        # Check that all expected keys are present
        expected_keys = [
            "current_volatility",
            "average_volatility",
            "volatility_std",
            "volatility_skewness",
            "volatility_kurtosis",
            "garch_persistence",
        ]

        for key in expected_keys:
            assert key in stats_dict

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty series
        empty_series = pd.Series([], dtype=float)

        try:
            result = self.analyzer.calculate_simple_volatility(empty_series)
            # If no error raised, check result is empty
            assert len(result) == 0
        except (ValueError, ZeroDivisionError, IndexError):
            # These errors are acceptable
            pass

        # Test with insufficient data for rolling calculation
        short_series = pd.Series([0.1, 0.2], name="short")
        analyzer_long_window = VolatilityAnalyzer(window=100)

        try:
            rolling_vol = analyzer_long_window.calculate_simple_volatility(short_series)
            # If successful, result should be empty or small
            assert len(rolling_vol) <= 2
        except (ValueError, IndexError):
            # These errors are acceptable for insufficient data
            pass


class TestVolatilityStatistics:
    """Test cases for VolatilityStatistics dataclass."""

    def test_volatility_statistics_creation(self):
        """Test VolatilityStatistics creation and access."""
        stats = VolatilityStatistics(
            current_volatility=0.25,
            average_volatility=0.22,
            volatility_std=0.05,
            volatility_skewness=0.1,
            volatility_kurtosis=2.8,
        )

        assert stats.current_volatility == 0.25
        assert stats.average_volatility == 0.22
        assert stats.volatility_std == 0.05
        assert stats.volatility_skewness == 0.1
        assert stats.volatility_kurtosis == 2.8
        assert stats.garch_persistence is None

    def test_volatility_statistics_to_dict(self):
        """Test VolatilityStatistics to_dict conversion."""
        stats = VolatilityStatistics(
            current_volatility=0.25,
            average_volatility=0.22,
            volatility_std=0.05,
            volatility_skewness=0.1,
            volatility_kurtosis=2.8,
            garch_persistence=0.95,
        )

        stats_dict = stats.to_dict()

        assert stats_dict["current_volatility"] == 0.25
        assert stats_dict["average_volatility"] == 0.22
        assert stats_dict["volatility_std"] == 0.05
        assert stats_dict["volatility_skewness"] == 0.1
        assert stats_dict["volatility_kurtosis"] == 2.8
        assert stats_dict["garch_persistence"] == 0.95


def run_tests():
    """Run all tests."""
    test_vol = TestVolatilityAnalyzer()
    test_vol.setup_method()

    print("Testing VolatilityAnalyzer initialization...")
    test_vol.test_initialization()

    print("Testing simple volatility calculation...")
    test_vol.test_simple_volatility_calculation()

    print("Testing rolling volatility calculation...")
    test_vol.test_rolling_volatility_calculation()

    print("Testing EWM volatility calculation...")
    test_vol.test_ewm_volatility_calculation()

    print("Testing realized volatility calculation...")
    test_vol.test_realized_volatility_calculation()

    print("Testing comprehensive volatility analysis...")
    test_vol.test_volatility_analysis()

    print("Testing VolatilityStatistics to_dict...")
    test_vol.test_volatility_statistics_to_dict()

    print("Testing error handling...")
    test_vol.test_error_handling()

    # Test VolatilityStatistics dataclass
    test_stats = TestVolatilityStatistics()

    print("Testing VolatilityStatistics creation...")
    test_stats.test_volatility_statistics_creation()

    print("Testing VolatilityStatistics dict conversion...")
    test_stats.test_volatility_statistics_to_dict()

    print("All volatility analysis tests passed!")


if __name__ == "__main__":
    run_tests()
