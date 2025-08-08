"""
Test suite for return analysis functionality.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from analysis.returns import (
    ReturnAnalyzer,
    calculate_simple_returns,
    calculate_log_returns,
)


class TestReturnAnalyzer:
    """Test cases for ReturnAnalyzer class."""

    def setup_method(self):
        """Set up test data."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate price series with some trends
        price_changes = np.random.normal(0.001, 0.02, 252)
        prices = [100]  # Starting price

        for change in price_changes:
            prices.append(prices[-1] * (1 + change))

        self.prices = pd.Series(prices[1:], index=dates, name="price")
        self.analyzer = ReturnAnalyzer()

    def test_initialization(self):
        """Test ReturnAnalyzer initialization."""
        analyzer = ReturnAnalyzer(risk_free_rate=0.02)
        assert analyzer.risk_free_rate == 0.02

        # Test default initialization
        analyzer_default = ReturnAnalyzer()
        assert analyzer_default.risk_free_rate == 0.0

    def test_simple_returns_calculation(self):
        """Test simple returns calculation."""
        returns = self.analyzer.calculate_simple_returns(self.prices)

        # Check that returns have the correct length
        assert len(returns) == len(self.prices) - 1

        # Check that returns are properly calculated
        expected_first_return = (
            self.prices.iloc[1] - self.prices.iloc[0]
        ) / self.prices.iloc[0]
        assert abs(returns.iloc[0] - expected_first_return) < 1e-10

        # Check that no returns are NaN (except possibly the first which should be dropped)
        assert not returns.isna().any()

    def test_log_returns_calculation(self):
        """Test log returns calculation."""
        log_returns = self.analyzer.calculate_log_returns(self.prices)

        # Check that returns have the correct length
        assert len(log_returns) == len(self.prices) - 1

        # Check that log returns are properly calculated
        expected_first_log_return = np.log(self.prices.iloc[1] / self.prices.iloc[0])
        assert abs(log_returns.iloc[0] - expected_first_log_return) < 1e-10

        # Check that no returns are NaN
        assert not log_returns.isna().any()

    def test_cumulative_returns_simple(self):
        """Test cumulative returns calculation with simple method."""
        returns = self.analyzer.calculate_simple_returns(self.prices)
        cum_returns = self.analyzer.calculate_cumulative_returns(
            returns, method="simple"
        )

        # Check that cumulative returns have the correct length
        assert len(cum_returns) == len(returns)

        # Check that the final cumulative return matches total return
        total_return = (self.prices.iloc[-1] / self.prices.iloc[0]) - 1
        final_cum_return = cum_returns.iloc[-1]

        # Should be approximately equal (small numerical differences expected)
        assert abs(final_cum_return - total_return) < 1e-6

    def test_cumulative_returns_log(self):
        """Test cumulative returns calculation with log method."""
        log_returns = self.analyzer.calculate_log_returns(self.prices)
        cum_returns = self.analyzer.calculate_cumulative_returns(
            log_returns, method="log"
        )

        # Check that cumulative returns have the correct length
        assert len(cum_returns) == len(log_returns)

        # Check that the final cumulative return is reasonable
        assert not np.isnan(cum_returns.iloc[-1])
        assert np.isfinite(cum_returns.iloc[-1])

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        drawdown = self.analyzer.calculate_drawdown(self.prices)

        # Check that drawdown has the correct length
        assert len(drawdown) == len(self.prices)

        # Check that all drawdown values are <= 0
        assert (drawdown <= 0).all()

        # Check that the first value is 0 (no drawdown at start)
        assert drawdown.iloc[0] == 0

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        max_dd = self.analyzer.calculate_max_drawdown(self.prices)

        # Check that max drawdown is negative or zero
        assert max_dd <= 0

        # Check that it's a finite number
        assert np.isfinite(max_dd)

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = self.analyzer.calculate_simple_returns(self.prices)
        sharpe = self.analyzer.calculate_sharpe_ratio(returns, annualize=True)

        # Check that Sharpe ratio is a finite number
        assert np.isfinite(sharpe)

        # Test with different risk-free rate
        analyzer_rf = ReturnAnalyzer(risk_free_rate=0.02)
        sharpe_rf = analyzer_rf.calculate_sharpe_ratio(returns, annualize=True)

        # Sharpe with risk-free rate should be different
        assert sharpe != sharpe_rf

    def test_analyze_returns_simple(self):
        """Test comprehensive return analysis with simple returns."""
        stats = self.analyzer.analyze_returns(self.prices, return_type="simple")

        # Check that all statistics are present and finite
        assert np.isfinite(stats.mean)
        assert np.isfinite(stats.std)
        assert stats.std > 0  # Standard deviation should be positive
        assert np.isfinite(stats.skewness)
        assert np.isfinite(stats.kurtosis)
        assert np.isfinite(stats.sharpe_ratio)
        assert stats.max_drawdown <= 0  # Max drawdown should be negative
        assert np.isfinite(stats.total_return)
        assert np.isfinite(stats.annualized_return)
        assert np.isfinite(stats.annualized_volatility)
        assert stats.annualized_volatility > 0  # Volatility should be positive

    def test_analyze_returns_log(self):
        """Test comprehensive return analysis with log returns."""
        stats = self.analyzer.analyze_returns(self.prices, return_type="log")

        # Check that all statistics are present and finite
        assert np.isfinite(stats.mean)
        assert np.isfinite(stats.std)
        assert stats.std > 0
        assert np.isfinite(stats.sharpe_ratio)
        assert np.isfinite(stats.total_return)
        assert np.isfinite(stats.annualized_return)
        assert np.isfinite(stats.annualized_volatility)

    def test_return_statistics_to_dict(self):
        """Test ReturnStatistics to_dict method."""
        stats = self.analyzer.analyze_returns(self.prices)
        stats_dict = stats.to_dict()

        # Check that all expected keys are present
        expected_keys = [
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "sharpe_ratio",
            "max_drawdown",
            "total_return",
            "annualized_return",
            "annualized_volatility",
        ]

        for key in expected_keys:
            assert key in stats_dict
            assert np.isfinite(stats_dict[key])

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        try:
            # Test with invalid return type
            self.analyzer.analyze_returns(self.prices, return_type="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            # Test with invalid cumulative returns method
            returns = self.analyzer.calculate_simple_returns(self.prices)
            self.analyzer.calculate_cumulative_returns(returns, method="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestStandaloneFunctions:
    """Test cases for standalone functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        price_changes = np.random.normal(0.001, 0.02, 100)
        prices = [100]

        for change in price_changes:
            prices.append(prices[-1] * (1 + change))

        self.prices = pd.Series(prices[1:], index=dates, name="price")

    def test_calculate_simple_returns_function(self):
        """Test standalone calculate_simple_returns function."""
        returns = calculate_simple_returns(self.prices)

        # Check basic properties
        assert len(returns) == len(self.prices) - 1
        assert not returns.isna().any()
        assert isinstance(returns, pd.Series)

    def test_calculate_log_returns_function(self):
        """Test standalone calculate_log_returns function."""
        log_returns = calculate_log_returns(self.prices)

        # Check basic properties
        assert len(log_returns) == len(self.prices) - 1
        assert not log_returns.isna().any()
        assert isinstance(log_returns, pd.Series)

    def test_standalone_vs_class_methods(self):
        """Test that standalone functions give same results as class methods."""
        analyzer = ReturnAnalyzer()

        # Simple returns
        returns_class = analyzer.calculate_simple_returns(self.prices)
        returns_function = calculate_simple_returns(self.prices)
        pd.testing.assert_series_equal(returns_class, returns_function)

        # Log returns
        log_returns_class = analyzer.calculate_log_returns(self.prices)
        log_returns_function = calculate_log_returns(self.prices)
        pd.testing.assert_series_equal(log_returns_class, log_returns_function)


def run_tests():
    """Run all tests."""
    test_return = TestReturnAnalyzer()
    test_return.setup_method()

    print("Testing ReturnAnalyzer initialization...")
    test_return.test_initialization()

    print("Testing simple returns calculation...")
    test_return.test_simple_returns_calculation()

    print("Testing log returns calculation...")
    test_return.test_log_returns_calculation()

    print("Testing cumulative returns (simple)...")
    test_return.test_cumulative_returns_simple()

    print("Testing cumulative returns (log)...")
    test_return.test_cumulative_returns_log()

    print("Testing drawdown calculation...")
    test_return.test_drawdown_calculation()

    print("Testing max drawdown calculation...")
    test_return.test_max_drawdown_calculation()

    print("Testing Sharpe ratio calculation...")
    test_return.test_sharpe_ratio_calculation()

    print("Testing comprehensive analysis (simple)...")
    test_return.test_analyze_returns_simple()

    print("Testing comprehensive analysis (log)...")
    test_return.test_analyze_returns_log()

    print("Testing ReturnStatistics to_dict...")
    test_return.test_return_statistics_to_dict()

    print("Testing error handling...")
    test_return.test_error_handling()

    # Test standalone functions
    test_standalone = TestStandaloneFunctions()
    test_standalone.setup_method()

    print("Testing standalone simple returns function...")
    test_standalone.test_calculate_simple_returns_function()

    print("Testing standalone log returns function...")
    test_standalone.test_calculate_log_returns_function()

    print("Testing standalone vs class methods...")
    test_standalone.test_standalone_vs_class_methods()

    print("All return analysis tests passed!")


if __name__ == "__main__":
    run_tests()
