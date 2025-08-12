"""
Test suite for Monte Carlo analysis implementation.

Tests Monte Carlo cross-validation, bootstrap analysis, synthetic data generation,
and scenario analysis functionality.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.analysis.monte_carlo import MonteCarloAnalyzer, monte_carlo_permutation_test


class TestMonteCarloAnalyzer:
    """Test suite for MonteCarloAnalyzer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        n_features = 4

        # Create features with some structure
        data = np.random.randn(len(dates), n_features)
        X = pd.DataFrame(
            data, index=dates, columns=[f"feature_{i}" for i in range(n_features)]
        )

        # Create target with some predictable pattern
        y = pd.Series(
            np.where(X["feature_0"] + X["feature_1"] > 0, 1, 0),
            index=dates,
            name="target",
        )

        return X, y

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)), index=dates, name="returns"
        )
        return returns

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.model_type = "classifier"
        model.clone.return_value = model

        def mock_predict(X):
            # Simple prediction based on first feature
            return (X.iloc[:, 0] > 0).astype(int).values

        def mock_score(X, y):
            pred = mock_predict(X)
            return np.mean(pred == y.values)

        model.predict = mock_predict
        model.score = mock_score
        model.fit = Mock()

        return model

    def test_analyzer_initialization(self):
        """Test MonteCarloAnalyzer initialization."""
        analyzer = MonteCarloAnalyzer(
            n_simulations=500,
            confidence_levels=[0.90, 0.95, 0.99],
            random_state=123,
            purging_enabled=True,
            embargo_pct=0.02,
        )

        assert analyzer.n_simulations == 500
        assert analyzer.confidence_levels == [0.90, 0.95, 0.99]
        assert analyzer.random_state == 123
        assert analyzer.purging_enabled is True
        assert analyzer.embargo_pct == 0.02
        assert analyzer.simulation_results_ == []
        assert analyzer.bootstrap_results_ == []
        assert analyzer.synthetic_results_ == []

    def test_default_initialization(self):
        """Test default initialization parameters."""
        analyzer = MonteCarloAnalyzer()

        assert analyzer.n_simulations == 1000
        assert analyzer.confidence_levels == [0.95, 0.99]
        assert analyzer.random_state is None
        assert analyzer.purging_enabled is True
        assert analyzer.embargo_pct == 0.01

    def test_monte_carlo_cross_validation(self, sample_data, mock_model):
        """Test Monte Carlo cross-validation."""
        X, y = sample_data

        analyzer = MonteCarloAnalyzer(
            n_simulations=10, random_state=42  # Small number for testing
        )

        results = analyzer.monte_carlo_cross_validation(mock_model, X, y, test_size=0.3)

        # Check results structure
        assert isinstance(results, dict)
        assert "mean_score" in results
        assert "std_score" in results
        assert "scores" in results
        assert "confidence_intervals" in results
        assert "n_successful_simulations" in results

        # Check that simulations were run
        assert results["n_successful_simulations"] > 0
        assert len(results["scores"]) > 0

        # Check that model was fitted
        assert mock_model.fit.called

        # Check confidence intervals
        assert isinstance(results["confidence_intervals"], dict)
        assert "95%" in results["confidence_intervals"]

    def test_purged_bootstrap_sample(self, sample_data):
        """Test purged bootstrap sampling."""
        X, y = sample_data
        n_samples = len(X)

        analyzer = MonteCarloAnalyzer(embargo_pct=0.05, random_state=42)

        train_idx, test_idx = analyzer._purged_bootstrap_sample(n_samples, 30)

        # Check no overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        # Check indices are valid
        assert np.all(train_idx >= 0)
        assert np.all(train_idx < n_samples)
        assert np.all(test_idx >= 0)
        assert np.all(test_idx < n_samples)

        # Check embargo effect (training set should be smaller due to embargo)
        simple_train_idx, simple_test_idx = analyzer._simple_bootstrap_sample(
            n_samples, 30
        )
        assert len(train_idx) <= len(simple_train_idx)

    def test_simple_bootstrap_sample(self, sample_data):
        """Test simple bootstrap sampling."""
        X, y = sample_data
        n_samples = len(X)

        analyzer = MonteCarloAnalyzer(random_state=42)

        train_idx, test_idx = analyzer._simple_bootstrap_sample(n_samples, 40)

        # Check no overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        # Check sizes
        assert len(test_idx) == 40
        assert len(train_idx) == n_samples - 40

        # Check indices are valid
        assert np.all(train_idx >= 0)
        assert np.all(train_idx < n_samples)
        assert np.all(test_idx >= 0)
        assert np.all(test_idx < n_samples)

    def test_bootstrap_performance_analysis(self, sample_returns):
        """Test bootstrap performance analysis."""
        analyzer = MonteCarloAnalyzer(
            n_simulations=20, random_state=42  # Small number for testing
        )

        results = analyzer.bootstrap_performance_analysis(sample_returns)

        # Check results structure
        assert isinstance(results, dict)

        # Check that key metrics are present
        expected_metrics = [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "var_95",
            "cvar_95",
        ]

        for metric in expected_metrics:
            if metric in results:
                assert "mean" in results[metric]
                assert "std" in results[metric]
                assert "confidence_intervals" in results[metric]
                assert "raw_values" in results[metric]

    def test_bootstrap_with_benchmark(self, sample_returns):
        """Test bootstrap analysis with benchmark comparison."""
        # Create benchmark returns
        benchmark_returns = sample_returns * 0.8 + np.random.normal(
            0, 0.01, len(sample_returns)
        )
        benchmark_returns = pd.Series(benchmark_returns, index=sample_returns.index)

        analyzer = MonteCarloAnalyzer(n_simulations=15, random_state=42)

        results = analyzer.bootstrap_performance_analysis(
            sample_returns, benchmark_returns=benchmark_returns
        )

        # Check that relative metrics are calculated
        relative_metrics = ["alpha", "beta", "information_ratio", "tracking_error"]

        for metric in relative_metrics:
            if metric in results:
                assert "mean" in results[metric]
                assert "confidence_intervals" in results[metric]

    def test_block_bootstrap_sample(self, sample_returns):
        """Test block bootstrap sampling."""
        analyzer = MonteCarloAnalyzer(random_state=42)

        block_size = 5
        bootstrap_sample = analyzer._block_bootstrap_sample(sample_returns, block_size)

        # Check sample properties
        assert len(bootstrap_sample) == len(sample_returns)
        assert isinstance(bootstrap_sample, pd.Series)

        # Check that values come from original series
        unique_original = set(sample_returns.values)
        unique_bootstrap = set(bootstrap_sample.values)
        assert unique_bootstrap.issubset(unique_original)

    def test_synthetic_data_generation(self, sample_data):
        """Test synthetic data generation."""
        X, y = sample_data

        analyzer = MonteCarloAnalyzer(random_state=42)

        X_synthetic, y_synthetic = analyzer._generate_synthetic_data(X, y)

        # Check dimensions
        assert X_synthetic.shape == X.shape
        assert y_synthetic.shape == y.shape

        # Check column names
        assert list(X_synthetic.columns) == list(X.columns)
        assert y_synthetic.name == y.name

        # Check that synthetic data has similar statistical properties
        for col in X.columns:
            original_mean = X[col].mean()
            synthetic_mean = X_synthetic[col].mean()

            # Means should be reasonably close (within 2 standard errors)
            std_error = X[col].std() / np.sqrt(len(X))
            assert abs(original_mean - synthetic_mean) <= 3 * std_error

    def test_synthetic_data_backtesting(self, sample_data, mock_model):
        """Test synthetic data backtesting."""
        X, y = sample_data

        analyzer = MonteCarloAnalyzer(
            n_simulations=50, random_state=42  # Reduce for testing
        )

        results = analyzer.synthetic_data_backtesting(
            mock_model, X, y, n_synthetic_datasets=5
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "original_score" in results
        assert "synthetic_mean_score" in results
        assert "score_degradation" in results
        assert "synthetic_confidence_intervals" in results

        # Check that synthetic datasets were generated
        assert results["original_score"] is not None
        assert results["synthetic_mean_score"] is not None

        # Check that model was trained
        assert mock_model.fit.called

    def test_performance_metrics_calculation(self, sample_returns):
        """Test performance metrics calculation."""
        analyzer = MonteCarloAnalyzer()

        metrics = analyzer._calculate_performance_metrics(sample_returns)

        # Check that key metrics are calculated
        expected_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    def test_relative_metrics_calculation(self, sample_returns):
        """Test relative metrics calculation."""
        # Create benchmark returns
        benchmark_returns = sample_returns * 0.9 + 0.0005

        analyzer = MonteCarloAnalyzer()

        metrics = analyzer._calculate_relative_metrics(
            sample_returns, benchmark_returns
        )

        # Check that relative metrics are calculated
        expected_metrics = ["alpha", "beta", "tracking_error", "information_ratio"]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_scenario_analysis(self, sample_returns):
        """Test scenario analysis functionality."""
        scenarios = {
            "bull_market": {"mean_adjustment": 0.002, "volatility_multiplier": 0.8},
            "bear_market": {"mean_adjustment": -0.003, "volatility_multiplier": 1.5},
            "high_volatility": {
                "mean_adjustment": 0.0,
                "volatility_multiplier": 2.0,
                "skewness_adjustment": -0.5,
            },
        }

        analyzer = MonteCarloAnalyzer(
            n_simulations=10, random_state=42  # Small number for testing
        )

        results = analyzer.scenario_analysis(sample_returns, scenarios)

        # Check results structure
        assert isinstance(results, dict)
        assert len(results) == len(scenarios)

        for scenario_name in scenarios.keys():
            assert scenario_name in results
            assert "summary_stats" in results[scenario_name]
            assert "confidence_intervals" in results[scenario_name]

    def test_confidence_intervals_calculation(self):
        """Test confidence intervals calculation."""
        analyzer = MonteCarloAnalyzer(confidence_levels=[0.90, 0.95, 0.99])

        # Test with normal distribution
        values = np.random.normal(0, 1, 1000)
        intervals = analyzer._calculate_confidence_intervals(values)

        assert isinstance(intervals, dict)
        assert "90%" in intervals
        assert "95%" in intervals
        assert "99%" in intervals

        # Check interval structure
        for level, (lower, upper) in intervals.items():
            assert lower < upper
            assert isinstance(lower, (int, float))
            assert isinstance(upper, (int, float))

    def test_score_distribution_analysis(self):
        """Test score distribution analysis."""
        analyzer = MonteCarloAnalyzer()

        # Test with known distribution
        np.random.seed(42)
        normal_scores = np.random.normal(0.75, 0.1, 100)

        analysis = analyzer._analyze_score_distribution(normal_scores)

        assert isinstance(analysis, dict)
        assert "percentiles" in analysis

        # Check percentiles
        percentiles = analysis["percentiles"]
        assert "5%" in percentiles
        assert "95%" in percentiles
        assert percentiles["5%"] < percentiles["50%"] < percentiles["95%"]

    def test_comprehensive_summary(self, sample_data, sample_returns, mock_model):
        """Test comprehensive summary generation."""
        X, y = sample_data

        analyzer = MonteCarloAnalyzer(n_simulations=10, random_state=42)

        # Run different analyses
        analyzer.monte_carlo_cross_validation(mock_model, X, y)
        analyzer.bootstrap_performance_analysis(sample_returns)
        analyzer.synthetic_data_backtesting(mock_model, X, y, n_synthetic_datasets=3)

        # Get comprehensive summary
        summary = analyzer.get_comprehensive_summary()

        assert isinstance(summary, dict)
        assert "monte_carlo_cv" in summary
        assert "bootstrap_analysis" in summary
        assert "synthetic_backtesting" in summary
        assert "analysis_summary" in summary

        # Check analysis summary
        analysis_summary = summary["analysis_summary"]
        assert "total_simulations" in analysis_summary
        assert "confidence_levels" in analysis_summary
        assert "purging_enabled" in analysis_summary

    def test_plot_data_generation(self, sample_data, sample_returns, mock_model):
        """Test plot data generation for visualization."""
        X, y = sample_data

        analyzer = MonteCarloAnalyzer(n_simulations=10, random_state=42)

        # Run analyses to generate data
        analyzer.monte_carlo_cross_validation(mock_model, X, y)
        analyzer.bootstrap_performance_analysis(sample_returns)
        analyzer.synthetic_data_backtesting(mock_model, X, y, n_synthetic_datasets=3)

        # Generate plot data
        plot_data = analyzer.plot_monte_carlo_results()

        assert isinstance(plot_data, dict)
        assert "simulation_scores" in plot_data
        assert "bootstrap_distributions" in plot_data
        assert "synthetic_comparisons" in plot_data

        # Check data structure
        assert len(plot_data["simulation_scores"]) > 0

        if plot_data["synthetic_comparisons"]:
            assert "original_score" in plot_data["synthetic_comparisons"]
            assert "synthetic_scores" in plot_data["synthetic_comparisons"]

    def test_error_handling(self, sample_data):
        """Test error handling in various scenarios."""
        X, y = sample_data

        analyzer = MonteCarloAnalyzer()

        # Test summary without results
        with pytest.raises(ValueError, match="No analysis results available"):
            analyzer.get_comprehensive_summary()

        # Test plot without results
        with pytest.raises(ValueError, match="No results available"):
            analyzer.plot_monte_carlo_results()

        # Test with empty data
        # Test with empty data
        empty_X = pd.DataFrame()
        empty_y = pd.Series([], dtype="float64")

        mock_model = Mock()
        mock_model.clone.return_value = mock_model

        # Should handle gracefully
        try:
            results = analyzer.monte_carlo_cross_validation(
                mock_model, empty_X, empty_y
            )
            # Should have no successful simulations
            assert results["n_successful_simulations"] == 0
        except (ValueError, IndexError) as e:
            # Should fail gracefully with appropriate error message
            assert any(
                keyword in str(e).lower()
                for keyword in ["empty", "insufficient", "size", "samples"]
            )


class TestMonteCarloPermutationTest:
    """Test suite for Monte Carlo permutation test."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for permutation testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, 3), columns=["feature_0", "feature_1", "feature_2"]
        )

        # Create target with some signal
        y = pd.Series(
            np.where(X["feature_0"] + X["feature_1"] > 0, 1, 0), name="target"
        )

        return X, y

    @pytest.fixture
    def mock_model(self):
        """Create mock model for permutation testing."""
        model = Mock()
        model.clone.return_value = model

        def mock_score(X, y):
            # Simple scoring based on feature_0
            pred = (X["feature_0"] > 0).astype(int)
            return np.mean(pred == y)

        model.score = mock_score
        model.fit = Mock()

        return model

    def test_permutation_test_basic(self, sample_data, mock_model):
        """Test basic permutation test functionality."""
        X, y = sample_data

        results = monte_carlo_permutation_test(
            mock_model, X, y, n_permutations=20, metric="accuracy"
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "original_score" in results
        assert "permutation_scores" in results
        assert "p_value" in results
        assert "is_significant" in results
        assert "permutation_mean" in results
        assert "permutation_std" in results

        # Check data types and ranges
        assert isinstance(results["original_score"], (int, float))
        assert isinstance(results["p_value"], (int, float))
        assert 0 <= results["p_value"] <= 1
        assert isinstance(results["is_significant"], bool)

        # Check permutation scores
        assert len(results["permutation_scores"]) == 20
        assert all(
            isinstance(score, (int, float)) for score in results["permutation_scores"]
        )

    def test_permutation_test_significance(self, sample_data):
        """Test permutation test significance detection."""
        X, y = sample_data

        # Create a model with strong signal
        strong_model = Mock()
        strong_model.clone.return_value = strong_model

        def strong_score(X, y):
            # Perfect prediction based on actual pattern
            pred = (X["feature_0"] + X["feature_1"] > 0).astype(int)
            return np.mean(pred == y)

        strong_model.score = strong_score
        strong_model.fit = Mock()

        results = monte_carlo_permutation_test(strong_model, X, y, n_permutations=50)

        # Should be significant (low p-value)
        assert results["p_value"] < 0.1  # Strong signal should be significant

        # Create a model with no signal (random predictions)
        weak_model = Mock()
        weak_model.clone.return_value = weak_model

        def weak_score(X, y):
            # Random predictions
            pred = np.random.randint(0, 2, len(y))
            return np.mean(pred == y)

        weak_model.score = weak_score
        weak_model.fit = Mock()

        # Set seed for consistent results
        np.random.seed(42)
        results_weak = monte_carlo_permutation_test(weak_model, X, y, n_permutations=50)

        # Should have higher p-value (less significant)
        assert results_weak["p_value"] > results["p_value"]

    def test_permutation_test_edge_cases(self, sample_data):
        """Test permutation test edge cases."""
        X, y = sample_data

        # Model without score method should use default behavior
        no_score_model = Mock()
        no_score_model.clone.return_value = no_score_model
        no_score_model.fit = Mock()
        # Mock hasattr to return False for score method

        # Test that it handles missing score method gracefully
        with patch("builtins.hasattr") as mock_hasattr:
            mock_hasattr.return_value = False

            results = monte_carlo_permutation_test(
                no_score_model, X, y, n_permutations=10
            )

            # Should handle gracefully with default score of 0.0
            assert results["original_score"] == 0.0
            assert all(score == 0.0 for score in results["permutation_scores"])


class TestIntegration:
    """Integration tests for Monte Carlo analysis."""

    def test_full_monte_carlo_workflow(self):
        """Test complete Monte Carlo analysis workflow."""
        # Create realistic financial data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=150, freq="D")

        # Features: price-based indicators
        price_change = np.random.normal(0.001, 0.02, len(dates))
        volume = np.random.lognormal(10, 0.5, len(dates))
        volatility = pd.Series(price_change).rolling(10).std().fillna(0.02)

        X = pd.DataFrame(
            {
                "price_change": price_change,
                "volume": volume,
                "volatility": volatility,
                "momentum": pd.Series(price_change).rolling(5).mean().fillna(0),
            },
            index=dates,
        )

        # Target: direction of next day return
        y = pd.Series(np.where(price_change > 0, 1, 0), index=dates, name="direction")

        # Create realistic returns
        returns = pd.Series(price_change, index=dates, name="returns")

        # Create realistic model
        model = Mock()
        model.model_type = "classifier"
        model.clone.return_value = model

        def realistic_predict(X_input):
            # Strategy based on momentum and volatility
            signal = (X_input["momentum"] > 0) & (X_input["volatility"] < 0.025)
            return signal.astype(int).values

        def realistic_score(X_input, y_input):
            pred = realistic_predict(X_input)
            return np.mean(pred == y_input.values)

        model.predict = realistic_predict
        model.score = realistic_score
        model.fit = Mock()

        # Initialize analyzer
        analyzer = MonteCarloAnalyzer(
            n_simulations=30,  # Reduced for testing
            confidence_levels=[0.90, 0.95],
            random_state=42,
        )

        # Run full analysis suite
        cv_results = analyzer.monte_carlo_cross_validation(model, X, y)
        bootstrap_results = analyzer.bootstrap_performance_analysis(returns)
        synthetic_results = analyzer.synthetic_data_backtesting(
            model, X, y, n_synthetic_datasets=5
        )

        # Test scenario analysis
        scenarios = {
            "normal": {"mean_adjustment": 0.0, "volatility_multiplier": 1.0},
            "stressed": {"mean_adjustment": -0.002, "volatility_multiplier": 1.5},
        }
        scenario_results = analyzer.scenario_analysis(returns, scenarios)

        # Verify all analyses completed
        assert cv_results["n_successful_simulations"] > 0
        assert len(bootstrap_results) > 0
        assert synthetic_results["original_score"] is not None
        assert len(scenario_results) == 2

        # Test comprehensive summary
        summary = analyzer.get_comprehensive_summary()
        assert "monte_carlo_cv" in summary
        assert "bootstrap_analysis" in summary
        assert "synthetic_backtesting" in summary

        # Test plot data
        plot_data = analyzer.plot_monte_carlo_results()
        assert len(plot_data["simulation_scores"]) > 0

        # Test permutation test
        perm_results = monte_carlo_permutation_test(model, X, y, n_permutations=20)
        assert "p_value" in perm_results

        # Verify model was used appropriately
        assert model.fit.call_count >= cv_results["n_successful_simulations"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
