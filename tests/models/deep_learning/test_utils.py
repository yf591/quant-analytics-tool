"""
Tests for Deep Learning Utilities

This module contains comprehensive tests for the utility classes:
- DeepLearningEvaluator
- FinancialMetrics
- ModelComparison
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

from src.models.deep_learning.utils import (
    DeepLearningEvaluator,
    FinancialMetrics,
    ModelComparison,
)
from src.models.deep_learning import QuantLSTMClassifier, QuantLSTMRegressor


class TestDeepLearningEvaluator:
    """Test DeepLearningEvaluator class."""

    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data and model."""
        np.random.seed(42)
        n_samples = 150
        n_features = 8

        X = np.random.randn(n_samples, n_features)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

        # Create and fit a simple model
        model = QuantLSTMClassifier(
            sequence_length=15, lstm_units=[8], epochs=2, verbose=0
        )
        model.fit(X, y)

        # Split data for testing - need enough samples for sequence creation
        X_test = X[-30:]  # Use last 30 samples for testing
        y_test = y[-30:]

        # Adjust test data to match expected prediction length
        expected_pred_length = len(X_test) - model.sequence_length + 1
        y_test = y_test[-expected_pred_length:]

        return model, X_test, y_test

    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data and model."""
        np.random.seed(42)
        n_samples = 150
        n_features = 6

        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

        # Create and fit a simple model
        model = QuantLSTMRegressor(
            sequence_length=10, lstm_units=[8], epochs=2, verbose=0
        )
        model.fit(X, y)

        # Split data for testing - need enough samples for sequence creation
        X_test = X[-30:]
        y_test = y[-30:]

        # Adjust test data to match expected prediction length
        expected_pred_length = len(X_test) - model.sequence_length + 1
        y_test = y_test[-expected_pred_length:]

        return model, X_test, y_test

    def test_evaluator_initialization(self, sample_classification_data):
        """Test evaluator initialization."""
        model, X_test, y_test = sample_classification_data

        evaluator = DeepLearningEvaluator(model, "Test LSTM")

        assert evaluator.model == model
        assert evaluator.model_name == "Test LSTM"

    def test_evaluate_classifier(self, sample_classification_data):
        """Test classifier evaluation."""
        model, X_test, y_test = sample_classification_data

        evaluator = DeepLearningEvaluator(model, "Test LSTM Classifier")

        # Test evaluation without plots
        results = evaluator.evaluate_classifier(X_test, y_test, plot_results=False)

        # Check that all expected metrics are present
        expected_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "confusion_matrix",
            "classification_report",
            "predictions",
            "probabilities",
        ]

        for metric in expected_metrics:
            assert metric in results

        # Check metric types and ranges
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1_score"] <= 1
        assert 0 <= results["roc_auc"] <= 1

        # Check prediction shapes
        expected_samples = len(X_test) - model.sequence_length + 1
        assert len(results["predictions"]) == expected_samples
        assert results["probabilities"].shape == (expected_samples, 2)

        # Check confusion matrix shape
        assert results["confusion_matrix"].shape == (2, 2)

    def test_evaluate_regressor(self, sample_regression_data):
        """Test regressor evaluation."""
        model, X_test, y_test = sample_regression_data

        evaluator = DeepLearningEvaluator(model, "Test LSTM Regressor")

        # Test evaluation without plots
        results = evaluator.evaluate_regressor(X_test, y_test, plot_results=False)

        # Check that all expected metrics are present
        expected_metrics = [
            "mse",
            "rmse",
            "mae",
            "r2_score",
            "directional_accuracy",
            "hit_ratio",
            "predictions",
            "actual",
        ]

        for metric in expected_metrics:
            assert metric in results

        # Check metric properties
        assert results["mse"] >= 0
        assert results["rmse"] >= 0
        assert results["mae"] >= 0
        assert 0 <= results["directional_accuracy"] <= 1
        assert 0 <= results["hit_ratio"] <= 1

        # Check that RMSE = sqrt(MSE)
        assert abs(results["rmse"] - np.sqrt(results["mse"])) < 1e-10

        # Check prediction shapes
        expected_samples = len(X_test) - model.sequence_length + 1
        assert len(results["predictions"]) == expected_samples

    def test_directional_accuracy_calculation(self):
        """Test directional accuracy calculation."""
        evaluator = DeepLearningEvaluator(Mock(), "Test")

        # Test perfect directional accuracy
        y_true = np.array([1, 2, 1, 3, 2])
        y_pred = np.array([1.1, 2.1, 0.9, 3.1, 1.9])

        dir_acc = evaluator._calculate_directional_accuracy(y_true, y_pred)
        assert dir_acc == 1.0  # Perfect directional accuracy

        # Test mixed directional accuracy
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])  # Same directions

        dir_acc = evaluator._calculate_directional_accuracy(y_true, y_pred)
        assert dir_acc == 1.0  # All directions correct

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation."""
        evaluator = DeepLearningEvaluator(Mock(), "Test")

        # Test perfect hit ratio
        y_true = np.array([1, -1, 2, -2, 0.5])
        y_pred = np.array([0.8, -0.8, 1.5, -1.5, 0.3])

        hit_ratio = evaluator._calculate_hit_ratio(y_true, y_pred)
        assert hit_ratio == 1.0

        # Test zero hit ratio
        y_true = np.array([1, -1, 2, -2])
        y_pred = np.array([-1, 1, -2, 2])  # Opposite signs

        hit_ratio = evaluator._calculate_hit_ratio(y_true, y_pred)
        assert hit_ratio == 0.0

    @patch("matplotlib.pyplot.show")
    def test_plot_methods_no_errors(self, mock_show, sample_classification_data):
        """Test that plotting methods don't raise errors."""
        model, X_test, y_test = sample_classification_data

        evaluator = DeepLearningEvaluator(model, "Test LSTM")

        # Test classification plots
        results = evaluator.evaluate_classifier(X_test, y_test, plot_results=True)
        assert mock_show.called

        # Reset mock
        mock_show.reset_mock()

    @pytest.mark.skip(reason="Requires pydot installation for plot_model functionality")
    def test_model_architecture_plot(self, sample_classification_data):
        """Test model architecture plotting."""
        model, X_test, y_test = sample_classification_data

        evaluator = DeepLearningEvaluator(model, "Test LSTM")

        # Test with mock to avoid pydot dependency - simplified approach
        with patch("tensorflow.keras.utils.plot_model") as mock_plot:
            # Mock pydot check to return True
            mock_plot.side_effect = None  # No exception
            try:
                evaluator.plot_model_architecture()
                # If no exception, plot_model was called successfully
                mock_plot.assert_called_once()
            except ImportError:
                # Expected when pydot is not installed
                mock_plot.assert_called_once()


class TestFinancialMetrics:
    """Test FinancialMetrics class."""

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Test with positive returns
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
        sharpe = FinancialMetrics.sharpe_ratio(returns)

        # Sharpe ratio should be finite and reasonable
        assert np.isfinite(sharpe)

        # Test with zero standard deviation
        zero_std_returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe_zero = FinancialMetrics.sharpe_ratio(zero_std_returns)
        assert sharpe_zero == 0.0

        # Test with risk-free rate
        sharpe_rf = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.001)
        assert np.isfinite(sharpe_rf)

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Test with declining returns
        returns = np.array([0.1, -0.05, -0.1, -0.05, 0.2])
        max_dd = FinancialMetrics.max_drawdown(returns)

        # Max drawdown should be negative or zero
        assert max_dd <= 0

        # Test with only positive returns
        positive_returns = np.array([0.01, 0.02, 0.01, 0.03])
        max_dd_positive = FinancialMetrics.max_drawdown(positive_returns)
        assert max_dd_positive <= 0

    def test_information_ratio(self):
        """Test information ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
        benchmark = np.array([0.008, 0.015, -0.005, 0.025, 0.002])

        info_ratio = FinancialMetrics.information_ratio(returns, benchmark)
        assert np.isfinite(info_ratio)

        # Test with identical returns (zero tracking error)
        info_ratio_zero = FinancialMetrics.information_ratio(returns, returns)
        assert info_ratio_zero == 0.0

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
        calmar = FinancialMetrics.calmar_ratio(returns)

        # Calmar ratio should be finite
        assert np.isfinite(calmar)

        # Test with zero max drawdown
        positive_returns = np.array([0.01, 0.02, 0.03, 0.04])
        with patch.object(
            FinancialMetrics, "max_drawdown", return_value=0.0
        ) as mock_dd:
            calmar_zero = FinancialMetrics.calmar_ratio(positive_returns)
            assert calmar_zero == 0.0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
        sortino = FinancialMetrics.sortino_ratio(returns)

        assert np.isfinite(sortino)

        # Test with no downside returns
        positive_returns = np.array([0.01, 0.02, 0.03, 0.04])
        sortino_positive = FinancialMetrics.sortino_ratio(positive_returns)
        assert sortino_positive == 0.0

        # Test with target return
        sortino_target = FinancialMetrics.sortino_ratio(returns, target_return=0.01)
        assert np.isfinite(sortino_target)

    def test_omega_ratio(self):
        """Test Omega ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
        omega = FinancialMetrics.omega_ratio(returns)

        assert np.isfinite(omega)
        assert omega > 0

        # Test with threshold
        omega_threshold = FinancialMetrics.omega_ratio(returns, threshold=0.005)
        assert np.isfinite(omega_threshold)

        # Test with only positive returns
        positive_returns = np.array([0.01, 0.02, 0.03, 0.04])
        omega_positive = FinancialMetrics.omega_ratio(positive_returns)
        assert omega_positive == float("inf")

        # Test with only negative returns
        negative_returns = np.array([-0.01, -0.02, -0.03])
        omega_negative = FinancialMetrics.omega_ratio(negative_returns)
        assert omega_negative == 0.0  # No gains, some losses = 0


class TestModelComparison:
    """Test ModelComparison class."""

    @pytest.fixture
    def comparison_data(self):
        """Generate data for model comparison."""
        np.random.seed(42)
        n_samples = 100
        n_features = 6

        X = np.random.randn(n_samples, n_features)
        y_class = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        y_reg = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

        # Split for testing with enough samples for sequence creation
        X_test = X[-30:]  # Use enough samples for sequences
        y_class_test = y_class[-30:]
        y_reg_test = y_reg[-30:]

        return X, y_class, y_reg, X_test, y_class_test, y_reg_test

    def test_model_comparison_initialization(self):
        """Test ModelComparison initialization."""
        comparison = ModelComparison()

        assert comparison.models == {}
        assert comparison.results == {}

    def test_add_classification_model(self, comparison_data):
        """Test adding classification models for comparison."""
        X, y_class, y_reg, X_test, y_class_test, y_reg_test = comparison_data

        comparison = ModelComparison()

        # Create and fit models
        model1 = QuantLSTMClassifier(
            sequence_length=10, lstm_units=[6], epochs=2, verbose=0
        )
        model1.fit(X, y_class)

        model2 = QuantLSTMClassifier(
            sequence_length=8, lstm_units=[4], epochs=2, verbose=0
        )
        model2.fit(X, y_class)

        # Adjust test targets to match prediction length
        expected_length1 = len(X_test) - model1.sequence_length + 1
        expected_length2 = len(X_test) - model2.sequence_length + 1

        y_test_adj1 = y_class_test[-expected_length1:]
        y_test_adj2 = y_class_test[-expected_length2:]

        # Add models to comparison
        comparison.add_model("LSTM-6", model1, X_test, y_test_adj1)
        comparison.add_model("LSTM-4", model2, X_test, y_test_adj2)

        # Check that models and results were added
        assert "LSTM-6" in comparison.models
        assert "LSTM-4" in comparison.models
        assert "LSTM-6" in comparison.results
        assert "LSTM-4" in comparison.results

        # Check that results contain classification metrics
        assert "accuracy" in comparison.results["LSTM-6"]
        assert "f1_score" in comparison.results["LSTM-6"]

    def test_add_regression_model(self, comparison_data):
        """Test adding regression models for comparison."""
        X, y_class, y_reg, X_test, y_class_test, y_reg_test = comparison_data

        comparison = ModelComparison()

        # Create and fit models
        model1 = QuantLSTMRegressor(
            sequence_length=10, lstm_units=[6], epochs=2, verbose=0
        )
        model1.fit(X, y_reg)

        # Adjust test targets to match prediction length
        expected_length = len(X_test) - model1.sequence_length + 1
        y_test_adj = y_reg_test[-expected_length:]

        # Add model to comparison
        comparison.add_model("LSTM-Reg", model1, X_test, y_test_adj)

        # Check that results contain regression metrics
        assert "r2_score" in comparison.results["LSTM-Reg"]
        assert "rmse" in comparison.results["LSTM-Reg"]

    @patch("matplotlib.pyplot.show")
    def test_compare_metrics_classification(self, mock_show, comparison_data):
        """Test metrics comparison for classification models."""
        X, y_class, y_reg, X_test, y_class_test, y_reg_test = comparison_data

        comparison = ModelComparison()

        # Add classification models
        model1 = QuantLSTMClassifier(
            sequence_length=8, lstm_units=[4], epochs=2, verbose=0
        )
        model1.fit(X, y_class)

        # Adjust test targets to match prediction length
        expected_length = len(X_test) - model1.sequence_length + 1
        y_test_adj = y_class_test[-expected_length:]

        comparison.add_model("Model1", model1, X_test, y_test_adj)

        # Test comparison
        df_comparison = comparison.compare_metrics()

        # Check that comparison DataFrame was returned
        assert df_comparison is not None
        assert "Model1" in df_comparison.columns
        assert len(df_comparison.index) == 5  # 5 classification metrics

        # Check that plot was called
        assert mock_show.called

    @patch("matplotlib.pyplot.show")
    def test_compare_metrics_regression(self, mock_show, comparison_data):
        """Test metrics comparison for regression models."""
        X, y_class, y_reg, X_test, y_class_test, y_reg_test = comparison_data

        comparison = ModelComparison()

        # Add regression models
        model1 = QuantLSTMRegressor(
            sequence_length=8, lstm_units=[4], epochs=2, verbose=0
        )
        model1.fit(X, y_reg)

        # Adjust test targets to match prediction length
        expected_length = len(X_test) - model1.sequence_length + 1
        y_test_adj = y_reg_test[-expected_length:]

        comparison.add_model("Reg-Model1", model1, X_test, y_test_adj)

        # Test comparison
        df_comparison = comparison.compare_metrics()

        # Check that comparison DataFrame was returned
        assert df_comparison is not None
        assert "Reg-Model1" in df_comparison.columns
        assert len(df_comparison.index) == 5  # 5 regression metrics

    def test_print_summary(self, comparison_data, capsys):
        """Test print summary functionality."""
        X, y_class, y_reg, X_test, y_class_test, y_reg_test = comparison_data

        comparison = ModelComparison()

        # Test empty comparison
        comparison.print_summary()
        captured = capsys.readouterr()
        assert "No models to compare" in captured.out

        # Add a model and test summary
        model1 = QuantLSTMClassifier(
            sequence_length=8, lstm_units=[4], epochs=2, verbose=0
        )
        model1.fit(X, y_class)

        # Adjust test targets to match prediction length
        expected_length = len(X_test) - model1.sequence_length + 1
        y_test_adj = y_class_test[-expected_length:]

        comparison.add_model("Test-Model", model1, X_test, y_test_adj)

        comparison.print_summary()
        captured = capsys.readouterr()
        assert "MODEL COMPARISON SUMMARY" in captured.out
        assert "Test-Model" in captured.out
        assert "Accuracy:" in captured.out

    def test_empty_comparison_methods(self):
        """Test methods on empty comparison."""
        comparison = ModelComparison()

        # Test compare_metrics with no models
        result = comparison.compare_metrics()
        assert result is None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
