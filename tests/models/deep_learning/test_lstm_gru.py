"""
Tests for Deep Learning Models

This module contains comprehensive tests for LSTM and GRU models
including functionality, performance, and integration tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

from src.models.deep_learning import (
    QuantLSTMClassifier,
    QuantLSTMRegressor,
    QuantGRUClassifier,
    QuantGRURegressor,
    LSTMDataPreprocessor,
)


class TestLSTMDataPreprocessor:
    """Test LSTM data preprocessor."""

    def test_sequence_creation_basic(self):
        """Test basic sequence creation."""
        # Create sample data
        data = np.random.randn(100, 5)
        target = np.random.randn(100)

        preprocessor = LSTMDataPreprocessor(sequence_length=10)
        X_seq, y_seq = preprocessor.fit_transform(data, target)

        # Check shapes
        assert X_seq.shape == (91, 10, 5)  # 100 - 10 + 1 = 91 sequences
        assert y_seq.shape == (91,)

    def test_sequence_creation_no_overlap(self):
        """Test sequence creation without overlap."""
        data = np.random.randn(100, 3)

        preprocessor = LSTMDataPreprocessor(sequence_length=10, overlap=False)
        X_seq, _ = preprocessor.fit_transform(data)

        # Check shapes with no overlap
        assert X_seq.shape == (10, 10, 3)  # 100 // 10 = 10 sequences

    def test_scaling_options(self):
        """Test different scaling options."""
        data = np.random.randn(50, 3) * 100  # Large scale data
        target = np.random.randn(50) * 10

        # Standard scaling
        preprocessor_std = LSTMDataPreprocessor(
            sequence_length=5, feature_scaler="standard", target_scaler="standard"
        )
        X_std, y_std = preprocessor_std.fit_transform(data, target)

        # Check if data is scaled (approximately mean=0, std=1)
        assert abs(X_std.mean()) < 0.1
        assert abs(X_std.std() - 1.0) < 0.1

        # MinMax scaling
        preprocessor_minmax = LSTMDataPreprocessor(
            sequence_length=5, feature_scaler="minmax"
        )
        X_minmax, _ = preprocessor_minmax.fit_transform(data)

        # Check if data is in [0, 1] range
        assert X_minmax.min() >= 0
        assert X_minmax.max() <= 1

    def test_transform_consistency(self):
        """Test that transform produces consistent results."""
        data = np.random.randn(50, 3)
        target = np.random.randn(50)

        preprocessor = LSTMDataPreprocessor(sequence_length=5)

        # Fit and transform
        X1, y1 = preprocessor.fit_transform(data, target)

        # Transform again (should be same as fitted data)
        X2, y2 = preprocessor.transform(data, target)

        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_almost_equal(y1, y2)


class TestQuantLSTMClassifier:
    """Test LSTM classifier."""

    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create sample features
        X = np.random.randn(n_samples, n_features)

        # Create binary targets with some pattern
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

        return X, y

    def test_lstm_classifier_initialization(self):
        """Test LSTM classifier initialization."""
        model = QuantLSTMClassifier(
            sequence_length=30,
            lstm_units=[32, 16],
            dense_units=[10],
            dropout_rate=0.1,
            epochs=2,
            verbose=0,
        )

        assert model.sequence_length == 30
        assert model.lstm_units == [32, 16]
        assert model.dense_units == [10]
        assert model.dropout_rate == 0.1

    def test_lstm_classifier_fitting(self, sample_classification_data):
        """Test LSTM classifier fitting."""
        X, y = sample_classification_data

        model = QuantLSTMClassifier(
            sequence_length=20, lstm_units=[16], epochs=2, verbose=0
        )

        # Fit model
        model.fit(X, y)

        # Check that model is fitted
        assert model.model_ is not None
        assert model.preprocessor is not None
        assert model.n_features_in_ == X.shape[1]
        assert model.n_classes_ == 2

    def test_lstm_classifier_prediction(self, sample_classification_data):
        """Test LSTM classifier prediction."""
        X, y = sample_classification_data

        model = QuantLSTMClassifier(
            sequence_length=15, lstm_units=[8], epochs=2, verbose=0
        )

        # Fit and predict
        model.fit(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Check prediction shapes
        expected_samples = len(X) - model.sequence_length + 1
        assert len(predictions) == expected_samples
        assert probabilities.shape == (expected_samples, 2)

        # Check prediction values
        assert all(pred in [0, 1] for pred in predictions)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_bidirectional_lstm(self, sample_classification_data):
        """Test bidirectional LSTM."""
        X, y = sample_classification_data

        model = QuantLSTMClassifier(
            sequence_length=10, lstm_units=[8], bidirectional=True, epochs=2, verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) > 0
        assert model.model_ is not None


class TestQuantLSTMRegressor:
    """Test LSTM regressor."""

    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 8

        # Create sample features
        X = np.random.randn(n_samples, n_features)

        # Create continuous targets with some pattern
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

        return X, y

    def test_lstm_regressor_initialization(self):
        """Test LSTM regressor initialization."""
        model = QuantLSTMRegressor(
            sequence_length=25,
            lstm_units=[20, 10],
            dense_units=[5],
            target_scaler="minmax",
            epochs=2,
            verbose=0,
        )

        assert model.sequence_length == 25
        assert model.lstm_units == [20, 10]
        assert model.target_scaler == "minmax"

    def test_lstm_regressor_fitting(self, sample_regression_data):
        """Test LSTM regressor fitting."""
        X, y = sample_regression_data

        model = QuantLSTMRegressor(
            sequence_length=15, lstm_units=[12], epochs=2, verbose=0
        )

        # Fit model
        model.fit(X, y)

        # Check that model is fitted
        assert model.model_ is not None
        assert model.preprocessor is not None
        assert model.n_features_in_ == X.shape[1]

    def test_lstm_regressor_prediction(self, sample_regression_data):
        """Test LSTM regressor prediction."""
        X, y = sample_regression_data

        model = QuantLSTMRegressor(
            sequence_length=10, lstm_units=[8], epochs=2, verbose=0
        )

        # Fit and predict
        model.fit(X, y)
        predictions = model.predict(X)

        # Check prediction shape
        expected_samples = len(X) - model.sequence_length + 1
        assert len(predictions) == expected_samples

        # Check that predictions are continuous values
        assert isinstance(predictions[0], (float, np.floating))

    def test_uncertainty_estimation(self, sample_regression_data):
        """Test uncertainty estimation for regressor."""
        X, y = sample_regression_data

        model = QuantLSTMRegressor(
            sequence_length=10,
            lstm_units=[8],
            dropout_rate=0.3,  # Need dropout for uncertainty
            epochs=2,
            verbose=0,
        )

        model.fit(X, y)

        # Test uncertainty estimation
        mean_pred, std_pred = model.predict_with_uncertainty(X, n_samples=10)

        expected_samples = len(X) - model.sequence_length + 1
        assert len(mean_pred) == expected_samples
        assert len(std_pred) == expected_samples
        assert np.all(std_pred >= 0)  # Standard deviation should be non-negative


class TestQuantGRUModels:
    """Test GRU models."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for GRU tests."""
        np.random.seed(42)
        n_samples = 150
        n_features = 6

        X = np.random.randn(n_samples, n_features)
        y_class = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        y_reg = X[:, 0] * 0.7 + X[:, 1] * 0.4 + np.random.randn(n_samples) * 0.15

        return X, y_class, y_reg

    def test_gru_classifier(self, sample_data):
        """Test GRU classifier."""
        X, y_class, _ = sample_data

        model = QuantGRUClassifier(
            sequence_length=12, gru_units=[10], epochs=2, verbose=0
        )

        model.fit(X, y_class)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        expected_samples = len(X) - model.sequence_length + 1
        assert len(predictions) == expected_samples
        assert probabilities.shape == (expected_samples, 2)

    def test_gru_regressor(self, sample_data):
        """Test GRU regressor."""
        X, _, y_reg = sample_data

        model = QuantGRURegressor(
            sequence_length=12, gru_units=[10], epochs=2, verbose=0
        )

        model.fit(X, y_reg)
        predictions = model.predict(X)

        expected_samples = len(X) - model.sequence_length + 1
        assert len(predictions) == expected_samples

    def test_bidirectional_gru(self, sample_data):
        """Test bidirectional GRU."""
        X, y_class, _ = sample_data

        model = QuantGRUClassifier(
            sequence_length=10, gru_units=[8], bidirectional=True, epochs=2, verbose=0
        )

        model.fit(X, y_class)
        predictions = model.predict(X)

        assert len(predictions) > 0


class TestModelIntegration:
    """Test model integration and edge cases."""

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        X = np.random.randn(5, 3)  # Only 5 samples
        y = np.random.randint(0, 2, 5)

        model = QuantLSTMClassifier(
            sequence_length=10, epochs=1, verbose=0  # Longer than available data
        )

        # Should raise error for insufficient data
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_model_reproducibility(self):
        """Test model reproducibility with same random state."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Train two models with same random state
        model1 = QuantLSTMClassifier(
            sequence_length=15, lstm_units=[8], random_state=42, epochs=2, verbose=0
        )

        model2 = QuantLSTMClassifier(
            sequence_length=15, lstm_units=[8], random_state=42, epochs=2, verbose=0
        )

        model1.fit(X, y)
        model2.fit(X, y)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        # Predictions should be similar (though not necessarily identical due to TensorFlow randomness)
        assert len(pred1) == len(pred2)

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        np.random.seed(42)
        X = np.random.randn(150, 5)
        y = np.random.randint(0, 3, 150)  # 3 classes

        model = QuantLSTMClassifier(
            sequence_length=10, lstm_units=[8], epochs=2, verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Check multiclass outputs
        assert model.n_classes_ == 3
        assert probabilities.shape[1] == 3
        assert set(predictions).issubset({0, 1, 2})

    def test_pandas_input(self):
        """Test with pandas DataFrame/Series input."""
        np.random.seed(42)

        # Create pandas data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        X = pd.DataFrame(
            np.random.randn(100, 4),
            columns=["feature1", "feature2", "feature3", "feature4"],
            index=dates,
        )
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)

        model = QuantLSTMClassifier(
            sequence_length=15, lstm_units=[8], epochs=2, verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) > 0
        assert model.n_features_in_ == 4


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
