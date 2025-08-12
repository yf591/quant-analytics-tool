"""
Test suite for walk-forward analysis implementation.

Tests the walk-forward analysis functionality including purged cross-validation,
time series splits, and performance metrics calculation.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analysis.walk_forward import (
    PurgedGroupTimeSeriesSplit,
    WalkForwardAnalyzer,
    create_time_series_splits,
)


class TestPurgedGroupTimeSeriesSplit:
    """Test suite for PurgedGroupTimeSeriesSplit."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        n_features = 5

        # Create synthetic data
        np.random.seed(42)
        data = np.random.randn(len(dates), n_features)

        df = pd.DataFrame(
            data, index=dates, columns=[f"feature_{i}" for i in range(n_features)]
        )
        target = pd.Series(
            np.random.randint(0, 3, len(dates)), index=dates, name="target"
        )

        return df, target

    def test_split_initialization(self):
        """Test splitter initialization."""
        splitter = PurgedGroupTimeSeriesSplit(
            n_splits=5, embargo_pct=0.02, test_size=0.15, gap_size=10
        )

        assert splitter.n_splits == 5
        assert splitter.embargo_pct == 0.02
        assert splitter.test_size == 0.15
        assert splitter.gap_size == 10

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        splitter = PurgedGroupTimeSeriesSplit(n_splits=3)
        assert splitter.get_n_splits() == 3

    def test_split_generation(self, sample_data):
        """Test train/test split generation."""
        X, y = sample_data

        splitter = PurgedGroupTimeSeriesSplit(n_splits=3, test_size=0.2)
        splits = list(splitter.split(X, y))

        assert len(splits) <= 3  # May be fewer if not enough data

        for train_idx, test_idx in splits:
            # Check indices are valid
            assert len(train_idx) > 0
            assert len(test_idx) > 0

            # Check no overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Check indices are within bounds
            assert np.max(train_idx) < len(X)
            assert np.max(test_idx) < len(X)
            assert np.min(train_idx) >= 0
            assert np.min(test_idx) >= 0

    def test_temporal_order(self, sample_data):
        """Test that splits maintain temporal order."""
        X, y = sample_data

        splitter = PurgedGroupTimeSeriesSplit(n_splits=2, test_size=0.3)
        splits = list(splitter.split(X, y))

        for i in range(len(splits) - 1):
            _, test_idx_current = splits[i]
            _, test_idx_next = splits[i + 1]

            # Later splits should have later test periods
            assert np.min(test_idx_next) > np.min(test_idx_current)

    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data."""
        # Empty data
        empty_df = pd.DataFrame()
        splitter = PurgedGroupTimeSeriesSplit(n_splits=3)

        splits = list(splitter.split(empty_df))
        assert len(splits) == 0

        # Very small data
        small_df = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )
        splits = list(splitter.split(small_df))
        assert len(splits) <= 1  # Should be very few or no splits


class TestWalkForwardAnalyzer:
    """Test suite for WalkForwardAnalyzer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        n_features = 3

        np.random.seed(42)
        # Create features with some time dependency
        data = []
        for i in range(len(dates)):
            if i == 0:
                row = np.random.randn(n_features)
            else:
                # Add some autocorrelation
                row = 0.7 * data[-1] + 0.3 * np.random.randn(n_features)
            data.append(row)

        X = pd.DataFrame(
            data, index=dates, columns=[f"feature_{i}" for i in range(n_features)]
        )

        # Create target with some predictable pattern
        y = pd.Series(
            np.where(X["feature_0"].rolling(5).mean() > 0, 1, 0),
            index=dates,
            name="target",
        )

        return X, y

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.model_type = "classifier"
        model.clone.return_value = model

        # Mock predictions
        def mock_predict(X):
            return np.random.randint(0, 2, len(X))

        def mock_predict_proba(X):
            probs = np.random.rand(len(X), 2)
            return probs / probs.sum(axis=1, keepdims=True)

        model.predict = mock_predict
        model.predict_proba = mock_predict_proba
        model.fit = Mock()

        return model

    def test_analyzer_initialization(self):
        """Test WalkForwardAnalyzer initialization."""
        analyzer = WalkForwardAnalyzer(
            window_size=100, step_size=10, min_train_size=50, embargo_pct=0.02
        )

        assert analyzer.window_size == 100
        assert analyzer.step_size == 10
        assert analyzer.min_train_size == 50
        assert analyzer.embargo_pct == 0.02
        assert analyzer.results_ == []

    def test_walk_forward_split_generation(self, sample_data):
        """Test walk-forward split generation."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer(window_size=100, step_size=20, min_train_size=50)

        splits = analyzer.walk_forward_split(X)

        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Check minimum training size
            assert len(train_idx) >= analyzer.min_train_size

            # Check indices are datetime indices
            assert isinstance(train_idx, pd.Index)
            assert isinstance(test_idx, pd.Index)

            # Check temporal order
            assert train_idx[-1] <= test_idx[0]

    def test_purge_training_set(self, sample_data):
        """Test training set purging functionality."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer()

        # Create overlapping train and test periods
        train_times = pd.Series(
            X.index[100:200], index=X.index[50:150]  # End times  # Start times
        )

        test_times = pd.Series(
            X.index[180:220], index=X.index[160:200]  # End times  # Start times
        )

        purged_idx = analyzer.purge_training_set(train_times, test_times)

        # Should have removed overlapping observations
        assert len(purged_idx) < len(train_times)

    def test_embargo_period_creation(self, sample_data):
        """Test embargo period creation."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer(embargo_pct=0.05)

        test_end = X.index[100]
        embargo_idx = analyzer.create_embargo_period(test_end, X.index)

        # Check embargo period starts after test end (or on the same day if it's the next available period)
        if len(embargo_idx) > 0:
            assert embargo_idx[0] >= test_end

            # Check embargo size is approximately correct
            expected_size = int(len(X) * analyzer.embargo_pct)
            assert abs(len(embargo_idx) - expected_size) <= 1

    def test_walk_forward_analysis_run(self, sample_data, mock_model):
        """Test full walk-forward analysis run."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer(window_size=100, step_size=50, min_train_size=80)

        # Run analysis
        results = analyzer.run_walk_forward_analysis(mock_model, X, y)

        # Check results structure
        assert isinstance(results, dict)
        assert "total_splits" in results
        assert "successful_splits" in results
        assert "overall_metrics" in results
        assert "period_results" in results

        # Check that model was fitted
        assert mock_model.fit.called

        # Check results storage
        assert len(analyzer.results_) > 0

        # Check period results structure
        for result in results["period_results"]:
            assert "split_id" in result
            assert "predictions" in result
            assert "actuals" in result
            assert "accuracy" in result  # For classifier

    def test_performance_summary(self, sample_data, mock_model):
        """Test performance summary generation."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer(window_size=80, step_size=40, min_train_size=60)

        # Run analysis
        analyzer.run_walk_forward_analysis(mock_model, X, y)

        # Get summary
        summary = analyzer.get_performance_summary()

        assert isinstance(summary, dict)
        assert "stability" in summary

        # Check metric statistics
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            if metric_name in summary:
                metric_stats = summary[metric_name]
                assert "mean" in metric_stats
                assert "std" in metric_stats
                assert "min" in metric_stats
                assert "max" in metric_stats

    def test_stability_metrics(self, sample_data, mock_model):
        """Test stability metrics calculation."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer(window_size=80, step_size=40, min_train_size=60)

        # Run analysis
        analyzer.run_walk_forward_analysis(mock_model, X, y)

        # Get summary with stability metrics
        summary = analyzer.get_performance_summary()
        stability = summary["stability"]

        assert "coefficient_of_variation" in stability
        assert "percentage_above_mean" in stability
        assert "max_performance_drawdown" in stability
        assert "performance_trend" in stability

        # Check value ranges
        assert 0 <= stability["percentage_above_mean"] <= 1
        assert stability["max_performance_drawdown"] >= 0

    def test_plot_data_generation(self, sample_data, mock_model):
        """Test plot data generation."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer(window_size=80, step_size=40, min_train_size=60)

        # Run analysis
        analyzer.run_walk_forward_analysis(mock_model, X, y)

        # Generate plot data
        plot_data = analyzer.plot_walk_forward_results()

        assert isinstance(plot_data, dict)
        assert "dates" in plot_data
        assert "performance" in plot_data
        assert "metric_name" in plot_data
        assert "splits_info" in plot_data

        # Check data consistency
        assert len(plot_data["dates"]) == len(plot_data["performance"])
        assert len(plot_data["splits_info"]) == len(plot_data["dates"])

    def test_regression_model_support(self, sample_data):
        """Test support for regression models."""
        X, y = sample_data

        # Create continuous target
        y_continuous = pd.Series(
            np.random.randn(len(y)), index=y.index, name="continuous_target"
        )

        # Mock regression model
        reg_model = Mock()
        reg_model.model_type = "regressor"
        reg_model.clone.return_value = reg_model
        reg_model.predict = lambda X: np.random.randn(len(X))
        reg_model.fit = Mock()

        analyzer = WalkForwardAnalyzer(window_size=80, step_size=40, min_train_size=60)

        # Run analysis
        results = analyzer.run_walk_forward_analysis(reg_model, X, y_continuous)

        # Check regression metrics are calculated
        overall_metrics = results["overall_metrics"]
        assert "mse" in overall_metrics
        assert "mae" in overall_metrics
        assert "rmse" in overall_metrics

        # Check period results contain regression metrics
        for result in results["period_results"]:
            assert "mse" in result
            assert "mae" in result
            assert "rmse" in result

    def test_error_handling(self, sample_data):
        """Test error handling in walk-forward analysis."""
        X, y = sample_data

        analyzer = WalkForwardAnalyzer()

        # Test with no results
        with pytest.raises(ValueError, match="No results available"):
            analyzer.get_performance_summary()

        with pytest.raises(ValueError, match="No results available"):
            analyzer.plot_walk_forward_results()

        # Test with insufficient data
        small_X = X.iloc[:10]
        small_y = y.iloc[:10]

        mock_model = Mock()
        mock_model.model_type = "classifier"
        mock_model.clone.return_value = mock_model

        with pytest.raises(ValueError, match="No valid splits generated"):
            analyzer.run_walk_forward_analysis(mock_model, small_X, small_y)


class TestTimeSeriesSplits:
    """Test suite for time series split utilities."""

    def test_create_time_series_splits(self):
        """Test convenience function for creating time series splits."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = pd.DataFrame(
            np.random.randn(len(dates), 3), index=dates, columns=["A", "B", "C"]
        )

        splits = create_time_series_splits(
            data, n_splits=3, test_size=0.25, embargo_pct=0.02
        )

        assert isinstance(splits, list)
        assert len(splits) <= 3

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestIntegration:
    """Integration tests for walk-forward analysis."""

    def test_full_workflow_integration(self):
        """Test complete walk-forward analysis workflow."""
        # Create realistic financial data
        dates = pd.date_range("2020-01-01", periods=300, freq="D")

        # Features: returns, volatility, momentum
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        volatility = np.abs(np.random.normal(0.02, 0.01, len(dates)))
        momentum = pd.Series(returns).rolling(10).mean().fillna(0)

        X = pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "momentum": momentum,
                "volume": np.random.lognormal(10, 1, len(dates)),
            },
            index=dates,
        )

        # Binary target: up/down movement
        y = pd.Series(np.where(returns > 0, 1, 0), index=dates, name="direction")

        # Create realistic model mock
        model = Mock()
        model.model_type = "classifier"
        model.clone.return_value = model

        # Realistic prediction function
        def predict_func(X_test):
            # Simple strategy: predict up if momentum positive
            return (X_test["momentum"] > 0).astype(int).values

        model.predict = predict_func
        model.predict_proba = lambda X: np.column_stack(
            [1 - predict_func(X), predict_func(X)]
        )
        model.fit = Mock()

        # Run analysis
        analyzer = WalkForwardAnalyzer(window_size=60, step_size=20, min_train_size=40)

        results = analyzer.run_walk_forward_analysis(model, X, y)

        # Validate results
        assert results["successful_splits"] > 0
        assert "accuracy" in results["overall_metrics"]

        # Test performance summary
        summary = analyzer.get_performance_summary()
        assert "stability" in summary

        # Test plot data
        plot_data = analyzer.plot_walk_forward_results()
        assert len(plot_data["dates"]) > 0

        # Verify model was trained multiple times
        assert model.fit.call_count >= results["successful_splits"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
