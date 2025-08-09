"""
Tests for Transformer architecture components.

This module tests the Transformer-based models for financial time series analysis.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Importing modules under test
try:
    from src.models.advanced.transformer import (
        TransformerClassifier,
        TransformerRegressor,
        FinancialTransformer,
        PositionalEncoding,
        TransformerBlock,
        create_transformer_config,
    )

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


@pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE, reason="Transformer module not available"
)
class TestPositionalEncoding:
    """Test PositionalEncoding component."""

    def test_positional_encoding_creation(self):
        """Test PositionalEncoding layer creation."""
        seq_len = 60
        d_model = 64

        pos_encoding = PositionalEncoding(seq_len, d_model)
        assert pos_encoding is not None

    def test_positional_encoding_output_shape(self):
        """Test PositionalEncoding output shape."""
        seq_len = 60
        d_model = 64
        batch_size = 32

        pos_encoding = PositionalEncoding(seq_len, d_model)

        # Create dummy input
        inputs = tf.random.normal((batch_size, seq_len, d_model))
        output = pos_encoding(inputs)

        assert output.shape == (batch_size, seq_len, d_model)


@pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE, reason="Transformer module not available"
)
class TestTransformerBlock:
    """Test TransformerBlock component."""

    def test_transformer_block_creation(self):
        """Test TransformerBlock creation."""
        d_model = 64
        num_heads = 8
        dff = 256

        transformer_block = TransformerBlock(d_model, num_heads, dff)
        assert transformer_block is not None

    def test_transformer_block_output_shape(self):
        """Test TransformerBlock output shape."""
        d_model = 64
        num_heads = 8
        dff = 256
        seq_len = 60
        batch_size = 32

        transformer_block = TransformerBlock(d_model, num_heads, dff)

        # Create dummy input
        inputs = tf.random.normal((batch_size, seq_len, d_model))
        output = transformer_block(inputs)

        assert output.shape == (batch_size, seq_len, d_model)


@pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE, reason="Transformer module not available"
)
class TestFinancialTransformer:
    """Test FinancialTransformer model."""

    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial time series data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        seq_length = 60

        # Generate financial-like data
        data = np.random.randn(n_samples, n_features)
        prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100

        # Create sequences
        X = []
        y = []
        for i in range(seq_length, n_samples):
            X.append(data[i - seq_length : i])
            y.append(
                1 if prices[i] > prices[i - 1] else 0
            )  # Simple directional prediction

        return np.array(X), np.array(y)

    def test_transformer_creation(self, sample_financial_data):
        """Test FinancialTransformer creation."""
        X, y = sample_financial_data

        config = create_transformer_config(
            input_dim=X.shape[-1], sequence_length=X.shape[1], num_classes=2
        )

        transformer = FinancialTransformer(config)
        assert transformer is not None

    def test_transformer_compilation(self, sample_financial_data):
        """Test FinancialTransformer compilation."""
        X, y = sample_financial_data

        config = create_transformer_config(
            input_dim=X.shape[-1], sequence_length=X.shape[1], num_classes=2
        )

        transformer = FinancialTransformer(config)
        transformer.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        assert transformer.optimizer is not None

    def test_transformer_prediction_shape(self, sample_financial_data):
        """Test FinancialTransformer prediction output shape."""
        X, y = sample_financial_data

        config = create_transformer_config(
            input_dim=X.shape[-1], sequence_length=X.shape[1], num_classes=2
        )

        transformer = FinancialTransformer(config)
        transformer.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Test prediction
        test_sample = X[:10]  # First 10 samples
        predictions = transformer.predict(test_sample)

        assert predictions.shape[0] == 10
        assert predictions.shape[1] == 2  # Binary classification


@pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE, reason="Transformer module not available"
)
class TestTransformerClassifier:
    """Test TransformerClassifier wrapper."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for classification."""
        np.random.seed(42)
        n_samples = 500
        n_features = 8
        seq_length = 30

        X = np.random.randn(n_samples, seq_length, n_features)
        y = np.random.choice([0, 1, 2], size=n_samples)  # 3-class classification

        return X, y

    def test_transformer_classifier_creation(self, sample_data):
        """Test TransformerClassifier creation."""
        X, y = sample_data

        classifier = TransformerClassifier(
            input_dim=X.shape[-1], sequence_length=X.shape[1], num_classes=3
        )

        assert classifier is not None

    def test_transformer_classifier_fit(self, sample_data):
        """Test TransformerClassifier training."""
        X, y = sample_data

        classifier = TransformerClassifier(
            input_dim=X.shape[-1],
            sequence_length=X.shape[1],
            num_classes=3,
            epochs=1,  # Quick test
            verbose=0,
        )

        # Fit should not raise an exception
        classifier.fit(X, y)

        assert hasattr(classifier, "model")
        assert classifier.model is not None

    def test_transformer_classifier_predict(self, sample_data):
        """Test TransformerClassifier prediction."""
        X, y = sample_data

        classifier = TransformerClassifier(
            input_dim=X.shape[-1],
            sequence_length=X.shape[1],
            num_classes=3,
            epochs=1,
            verbose=0,
        )

        classifier.fit(X, y)
        predictions = classifier.predict(X[:10])

        assert len(predictions) == 10
        assert all(pred in [0, 1, 2] for pred in predictions)


@pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE, reason="Transformer module not available"
)
class TestTransformerRegressor:
    """Test TransformerRegressor wrapper."""

    @pytest.fixture
    def sample_regression_data(self):
        """Create sample data for regression."""
        np.random.seed(42)
        n_samples = 500
        n_features = 8
        seq_length = 30

        X = np.random.randn(n_samples, seq_length, n_features)
        y = np.random.randn(n_samples)  # Continuous target

        return X, y

    def test_transformer_regressor_creation(self, sample_regression_data):
        """Test TransformerRegressor creation."""
        X, y = sample_regression_data

        regressor = TransformerRegressor(
            input_dim=X.shape[-1], sequence_length=X.shape[1]
        )

        assert regressor is not None

    def test_transformer_regressor_fit(self, sample_regression_data):
        """Test TransformerRegressor training."""
        X, y = sample_regression_data

        regressor = TransformerRegressor(
            input_dim=X.shape[-1], sequence_length=X.shape[1], epochs=1, verbose=0
        )

        regressor.fit(X, y)

        assert hasattr(regressor, "model")
        assert regressor.model is not None

    def test_transformer_regressor_predict(self, sample_regression_data):
        """Test TransformerRegressor prediction."""
        X, y = sample_regression_data

        regressor = TransformerRegressor(
            input_dim=X.shape[-1], sequence_length=X.shape[1], epochs=1, verbose=0
        )

        regressor.fit(X, y)
        predictions = regressor.predict(X[:10])

        assert len(predictions) == 10
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)


@pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE, reason="Transformer module not available"
)
class TestTransformerConfig:
    """Test transformer configuration utilities."""

    def test_create_transformer_config(self):
        """Test transformer configuration creation."""
        config = create_transformer_config(
            input_dim=10, sequence_length=60, num_classes=3
        )

        assert config is not None
        assert "input_dim" in config
        assert "sequence_length" in config
        assert "num_classes" in config
        assert config["input_dim"] == 10
        assert config["sequence_length"] == 60
        assert config["num_classes"] == 3

    def test_create_transformer_config_defaults(self):
        """Test transformer configuration with defaults."""
        config = create_transformer_config(input_dim=5, sequence_length=30)

        assert config is not None
        assert "d_model" in config
        assert "num_heads" in config
        assert "num_layers" in config


# Integration tests
@pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE, reason="Transformer module not available"
)
class TestTransformerIntegration:
    """Integration tests for Transformer components."""

    def test_transformer_with_real_financial_data_structure(self):
        """Test transformer with realistic financial data structure."""
        # Simulate realistic financial features
        np.random.seed(42)
        n_samples = 300
        seq_length = 60

        # Common financial features
        features = ["open", "high", "low", "close", "volume", "sma_20", "rsi", "macd"]
        n_features = len(features)

        # Generate data with financial characteristics
        X = np.random.randn(n_samples, seq_length, n_features)

        # Ensure volume is positive
        X[:, :, 4] = np.abs(X[:, :, 4])

        # RSI should be between 0-100
        X[:, :, 6] = (X[:, :, 6] + 1) * 50  # Transform to 0-100 range

        # Create directional labels (up/down/sideways)
        y = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])

        classifier = TransformerClassifier(
            input_dim=n_features,
            sequence_length=seq_length,
            num_classes=3,
            epochs=1,
            verbose=0,
        )

        # Should handle financial data without errors
        classifier.fit(X, y)
        predictions = classifier.predict(X[:5])

        assert len(predictions) == 5
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_transformer_memory_efficiency(self):
        """Test transformer memory usage with larger datasets."""
        # Test with larger dataset to check memory efficiency
        n_samples = 1000
        seq_length = 100
        n_features = 20

        X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
        y = np.random.choice([0, 1], size=n_samples)

        classifier = TransformerClassifier(
            input_dim=n_features,
            sequence_length=seq_length,
            num_classes=2,
            epochs=1,
            batch_size=32,  # Smaller batch size for memory efficiency
            verbose=0,
        )

        # Should handle larger data without memory issues
        classifier.fit(X, y)
        predictions = classifier.predict(X[:10])

        assert len(predictions) == 10


if __name__ == "__main__":
    pytest.main([__file__])
