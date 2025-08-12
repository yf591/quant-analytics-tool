"""
Test suite for correlation analysis functionality.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

from src.analysis.correlation import (
    CorrelationAnalyzer,
    CorrelationStatistics,
)


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer class."""

    def setup_method(self):
        """Set up test data."""
        # Create sample multi-asset data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate correlated data
        n_assets = 4
        returns = np.random.multivariate_normal(
            mean=[0.001] * n_assets,
            cov=np.array(
                [
                    [0.0004, 0.0002, 0.0001, 0.0001],
                    [0.0002, 0.0004, 0.0001, 0.0001],
                    [0.0001, 0.0001, 0.0004, 0.0002],
                    [0.0001, 0.0001, 0.0002, 0.0004],
                ]
            ),
            size=252,
        )

        self.data = pd.DataFrame(
            returns, index=dates, columns=["Asset_A", "Asset_B", "Asset_C", "Asset_D"]
        )

        # Initialize analyzer
        self.analyzer = CorrelationAnalyzer(method="pearson")

    def test_initialization(self):
        """Test CorrelationAnalyzer initialization."""
        analyzer = CorrelationAnalyzer(method="spearman")
        assert analyzer.method == "spearman"

        # Test default initialization
        analyzer_default = CorrelationAnalyzer()
        assert analyzer_default.method == "pearson"

    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation."""
        corr_matrix = self.analyzer.calculate_correlation_matrix(self.data)

        # Check matrix properties
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (4, 4)

        # Check diagonal elements are 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)

        # Check symmetry
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.values.T)

        # Check correlation values are between -1 and 1
        assert (corr_matrix.abs() <= 1.0).all().all()

    def test_rolling_correlation_calculation(self):
        """Test rolling correlation calculation."""
        rolling_corr = self.analyzer.calculate_rolling_correlation(
            self.data["Asset_A"], self.data["Asset_B"], window=30
        )

        # Check that correlations are valid
        valid_corr = rolling_corr.dropna()
        assert len(valid_corr) > 0
        assert (valid_corr.abs() <= 1.0).all()

    def test_dynamic_correlation_calculation(self):
        """Test dynamic correlation calculation."""
        dynamic_corr = self.analyzer.calculate_dynamic_correlation(self.data)

        # Check that we get a time series of correlation matrices
        assert isinstance(dynamic_corr, pd.DataFrame)
        assert len(dynamic_corr) > 0

    def test_correlation_stability_analysis(self):
        """Test correlation stability analysis."""
        stability = self.analyzer.calculate_correlation_stability(self.data)

        # Check that stability is a finite number
        assert np.isfinite(stability)
        assert isinstance(stability, (float, np.floating))
        assert (
            stability >= 0
        )  # Stability should be non-negative    def test_comprehensive_correlation_analysis(self):
        """Test comprehensive correlation analysis."""
        stats = self.analyzer.analyze_correlation_structure(self.data)

        # Check that all statistics are present and valid
        assert isinstance(stats, CorrelationStatistics)
        assert isinstance(stats.correlation_matrix, pd.DataFrame)
        assert np.isfinite(stats.average_correlation)
        assert np.isfinite(stats.max_correlation)
        assert np.isfinite(stats.min_correlation)
        assert stats.correlation_stability >= 0
        assert len(stats.eigenvalues) == stats.correlation_matrix.shape[0]
        assert stats.condition_number > 0

    def test_correlation_statistics_to_dict(self):
        """Test CorrelationStatistics to_dict method."""
        stats = self.analyzer.analyze_correlation_structure(self.data)
        stats_dict = stats.to_dict()

        # Check that all expected keys are present
        expected_keys = [
            "correlation_matrix",
            "average_correlation",
            "max_correlation",
            "min_correlation",
            "correlation_stability",
            "eigenvalues",
            "condition_number",
        ]

        for key in expected_keys:
            assert key in stats_dict

    def test_different_correlation_methods(self):
        """Test different correlation calculation methods."""
        # Test Pearson correlation
        pearson_analyzer = CorrelationAnalyzer(method="pearson")
        pearson_corr = pearson_analyzer.calculate_correlation_matrix(self.data)

        # Test Spearman correlation
        spearman_analyzer = CorrelationAnalyzer(method="spearman")
        spearman_corr = spearman_analyzer.calculate_correlation_matrix(self.data)

        # Test Kendall correlation
        kendall_analyzer = CorrelationAnalyzer(method="kendall")
        kendall_corr = kendall_analyzer.calculate_correlation_matrix(self.data)

        # All should be valid correlation matrices
        for corr in [pearson_corr, spearman_corr, kendall_corr]:
            assert isinstance(corr, pd.DataFrame)
            assert corr.shape == (4, 4)
            np.testing.assert_array_almost_equal(np.diag(corr), 1.0)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with insufficient data
        small_data = self.data.head(2)

        try:
            result = self.analyzer.analyze_correlation_structure(small_data)
            # If no error raised, result should be reasonable or empty
            assert result is not None
        except (ValueError, np.linalg.LinAlgError, IndexError):
            # These errors are acceptable for insufficient data
            pass

        # Test with invalid method
        try:
            invalid_analyzer = CorrelationAnalyzer(method="invalid_method")
            corr_matrix = invalid_analyzer.calculate_correlation_matrix(self.data)
            # This might work or fail depending on implementation
        except (ValueError, AttributeError):
            # These errors are acceptable for invalid method
            pass


class TestCorrelationStatistics:
    """Test cases for CorrelationStatistics dataclass."""

    def test_correlation_statistics_creation(self):
        """Test CorrelationStatistics creation and access."""
        # Create a simple correlation matrix for testing
        corr_matrix = pd.DataFrame(
            {"A": [1.0, 0.5, 0.3], "B": [0.5, 1.0, 0.7], "C": [0.3, 0.7, 1.0]},
            index=["A", "B", "C"],
        )

        eigenvals = np.array([2.2, 0.6, 0.2])

        stats = CorrelationStatistics(
            correlation_matrix=corr_matrix,
            average_correlation=0.5,
            max_correlation=0.7,
            min_correlation=0.3,
            correlation_stability=0.8,
            eigenvalues=eigenvals,
            condition_number=11.0,
        )

        assert isinstance(stats.correlation_matrix, pd.DataFrame)
        assert stats.average_correlation == 0.5
        assert stats.max_correlation == 0.7
        assert stats.min_correlation == 0.3
        assert stats.correlation_stability == 0.8
        np.testing.assert_array_equal(stats.eigenvalues, eigenvals)
        assert stats.condition_number == 11.0

    def test_correlation_statistics_to_dict(self):
        """Test CorrelationStatistics to_dict conversion."""
        corr_matrix = pd.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]}, index=["A", "B"])

        eigenvals = np.array([1.5, 0.5])

        stats = CorrelationStatistics(
            correlation_matrix=corr_matrix,
            average_correlation=0.5,
            max_correlation=0.5,
            min_correlation=0.5,
            correlation_stability=0.9,
            eigenvalues=eigenvals,
            condition_number=3.0,
        )

        stats_dict = stats.to_dict()

        assert "correlation_matrix" in stats_dict
        assert stats_dict["average_correlation"] == 0.5
        assert stats_dict["max_correlation"] == 0.5
        assert stats_dict["min_correlation"] == 0.5
        assert stats_dict["correlation_stability"] == 0.9
        assert stats_dict["eigenvalues"] == eigenvals.tolist()
        assert stats_dict["condition_number"] == 3.0


def run_tests():
    """Run all tests."""
    test_corr = TestCorrelationAnalyzer()
    test_corr.setup_method()

    print("Testing CorrelationAnalyzer initialization...")
    test_corr.test_initialization()

    print("Testing correlation matrix calculation...")
    test_corr.test_correlation_matrix_calculation()

    print("Testing rolling correlation calculation...")
    test_corr.test_rolling_correlation_calculation()

    print("Testing dynamic correlation calculation...")
    test_corr.test_dynamic_correlation_calculation()

    print("Testing correlation stability analysis...")
    test_corr.test_correlation_stability_analysis()

    print("Testing comprehensive correlation analysis...")
    test_corr.test_comprehensive_correlation_analysis()

    print("Testing CorrelationStatistics to_dict...")
    test_corr.test_correlation_statistics_to_dict()

    print("Testing different correlation methods...")
    test_corr.test_different_correlation_methods()

    print("Testing error handling...")
    test_corr.test_error_handling()

    # Test CorrelationStatistics dataclass
    test_stats = TestCorrelationStatistics()

    print("Testing CorrelationStatistics creation...")
    test_stats.test_correlation_statistics_creation()

    print("Testing CorrelationStatistics dict conversion...")
    test_stats.test_correlation_statistics_to_dict()

    print("All correlation analysis tests passed!")


if __name__ == "__main__":
    run_tests()
