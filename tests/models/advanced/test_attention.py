"""
Tests for Attention mechanisms and visualization components.

This module tests the attention-based models and visualization tools
for financial time series analysis.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Importing modules under test
try:
    from src.models.advanced.attention import (
        AttentionLayer,
        MultiHeadAttention,
        TemporalAttention,
        AttentionVisualizer,
        create_attention_model,
    )

    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False


@pytest.mark.skipif(not ATTENTION_AVAILABLE, reason="Attention module not available")
class TestAttentionLayer:
    """Test basic AttentionLayer component."""

    def test_attention_layer_creation(self):
        """Test AttentionLayer creation."""
        units = 64

        attention_layer = AttentionLayer(units)
        assert attention_layer is not None

    def test_attention_layer_output_shape(self):
        """Test AttentionLayer output shape."""
        units = 64
        seq_len = 60
        batch_size = 32
        input_dim = 10

        attention_layer = AttentionLayer(units)

        # Create dummy input
        inputs = tf.random.normal((batch_size, seq_len, input_dim))
        attended_output, attention_weights = attention_layer(inputs)

        # Output should be (batch_size, input_dim) after attention pooling
        assert attended_output.shape == (batch_size, input_dim)
        # Attention weights should be (batch_size, seq_len)
        assert attention_weights.shape == (batch_size, seq_len)


@pytest.mark.skipif(not ATTENTION_AVAILABLE, reason="Attention module not available")
class TestMultiHeadAttention:
    """Test MultiHeadAttention component."""

    def test_multihead_attention_creation(self):
        """Test MultiHeadAttention creation."""
        d_model = 64
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads)
        assert mha is not None

    def test_multihead_attention_output_shape(self):
        """Test MultiHeadAttention output shape."""
        d_model = 64
        num_heads = 8
        seq_len = 60
        batch_size = 32

        mha = MultiHeadAttention(d_model, num_heads)

        # Create dummy input
        inputs = tf.random.normal((batch_size, seq_len, d_model))
        output, attention_weights = mha(inputs, inputs, inputs)  # self-attention

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_multihead_attention_with_mask(self):
        """Test MultiHeadAttention with attention mask."""
        d_model = 64
        num_heads = 8
        seq_len = 60
        batch_size = 32

        mha = MultiHeadAttention(d_model, num_heads)

        # Create dummy input and mask
        inputs = tf.random.normal((batch_size, seq_len, d_model))
        # Create attention mask with proper shape for multi-head attention
        # Shape should be [batch_size, 1, 1, seq_len] for broadcasting
        base_mask = tf.cast(tf.random.uniform((batch_size, seq_len)) > 0.5, tf.float32)
        mask = tf.expand_dims(tf.expand_dims(base_mask, 1), 1) * -1e9

        output, attention_weights = mha(inputs, inputs, inputs, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


@pytest.mark.skipif(not ATTENTION_AVAILABLE, reason="Attention module not available")
class TestTemporalAttention:
    """Test TemporalAttention component."""

    def test_temporal_attention_creation(self):
        """Test TemporalAttention creation."""
        units = 64
        time_steps = 60

        temporal_attention = TemporalAttention(units, time_steps)
        assert temporal_attention is not None

    def test_temporal_attention_output_shape(self):
        """Test TemporalAttention output shape with financial data."""
        units = 64
        seq_len = 60
        batch_size = 32
        n_features = 8

        temporal_attention = TemporalAttention(units, seq_len)

        # Create financial-like sequential data
        inputs = tf.random.normal((batch_size, seq_len, n_features))
        output = temporal_attention(inputs)

        # Should return attention weights and attended features
        assert output is not None

    def test_temporal_attention_with_timestamps(self):
        """Test TemporalAttention with timestamp information."""
        units = 64
        seq_len = 60
        batch_size = 32
        n_features = 8

        temporal_attention = TemporalAttention(units, seq_len)

        # Create inputs with temporal information
        inputs = tf.random.normal((batch_size, seq_len, n_features))

        # Add timestamp encoding (e.g., day of week, hour)
        temporal_features = tf.random.uniform((batch_size, seq_len, 2))
        enhanced_inputs = tf.concat([inputs, temporal_features], axis=-1)

        output = temporal_attention(enhanced_inputs)

        assert output is not None


@pytest.mark.skipif(not ATTENTION_AVAILABLE, reason="Attention module not available")
class TestAttentionVisualizer:
    """Test AttentionVisualizer component."""

    @pytest.fixture
    def sample_attention_data(self):
        """Create sample attention weights and data."""
        seq_len = 60
        batch_size = 1  # Single sample for visualization
        n_features = 8
        num_heads = 4

        # Sample input data
        data = np.random.randn(batch_size, seq_len, n_features)

        # Sample attention weights (head, batch, seq_len, seq_len)
        raw_weights = np.random.randn(num_heads, batch_size, seq_len, seq_len)
        # Apply softmax manually since np.random.softmax doesn't exist
        attention_weights = np.exp(raw_weights) / np.sum(
            np.exp(raw_weights), axis=-1, keepdims=True
        )

        # Sample timestamps
        timestamps = pd.date_range("2023-01-01", periods=seq_len, freq="1h")

        return data, attention_weights, timestamps

    def test_attention_visualizer_creation(self):
        """Test AttentionVisualizer creation."""
        visualizer = AttentionVisualizer()
        assert visualizer is not None

    @patch("matplotlib.pyplot.show")
    def test_attention_heatmap_generation(self, mock_show, sample_attention_data):
        """Test attention heatmap generation."""
        data, attention_weights, timestamps = sample_attention_data

        visualizer = AttentionVisualizer()

        # Should generate heatmap without errors
        visualizer.plot_attention_heatmap(
            attention_weights[0, 0], timestamps  # Single head, single sample
        )
        # If we get here without exception, the test passes
        assert True

    @patch("matplotlib.pyplot.show")
    def test_temporal_attention_plot(self, mock_show, sample_attention_data):
        """Test temporal attention plotting."""
        data, attention_weights, timestamps = sample_attention_data

        visualizer = AttentionVisualizer()

        # Test temporal attention visualization
        visualizer.plot_temporal_attention(
            data[0],  # Single sample
            attention_weights[:, 0],  # All heads, single sample
            timestamps,
            feature_names=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "sma",
                "rsi",
                "macd",
            ],
        )
        # If we get here without exception, the test passes
        assert True

    def test_attention_statistics(self, sample_attention_data):
        """Test attention statistics calculation."""
        data, attention_weights, timestamps = sample_attention_data

        visualizer = AttentionVisualizer()

        # Calculate attention statistics
        stats = visualizer.calculate_attention_statistics(attention_weights)

        assert "mean_attention" in stats
        assert "attention_entropy" in stats
        assert "max_attention_position" in stats


@pytest.mark.skipif(not ATTENTION_AVAILABLE, reason="Attention module not available")
class TestAttentionModelCreation:
    """Test attention model creation utilities."""

    def test_create_attention_model_lstm(self):
        """Test creating LSTM model with attention."""
        input_shape = (60, 8)  # seq_len, n_features

        model = create_attention_model(
            input_shape=input_shape,
            attention_type="temporal",
            output_dim=3,
        )

        assert model is not None
        assert len(model.layers) > 1

    def test_create_attention_model_gru(self):
        """Test creating GRU model with attention."""
        input_shape = (60, 8)

        model = create_attention_model(
            input_shape=input_shape,
            attention_type="multi_head",
            output_dim=2,
        )

        assert model is not None

    def test_create_attention_model_prediction(self):
        """Test attention model prediction capability."""
        np.random.seed(42)
        input_shape = (30, 5)
        n_samples = 100

        # Create model
        model = create_attention_model(
            input_shape=input_shape,
            attention_type="temporal",
            output_dim=2,
        )

        # Compile model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Generate sample data
        X = np.random.randn(n_samples, *input_shape)
        y = np.random.choice([0, 1], size=n_samples)

        # Quick training
        model.fit(X, y, epochs=1, verbose=0)

        # Test prediction
        predictions = model.predict(X[:5], verbose=0)

        assert predictions.shape[0] == 5
        assert predictions.shape[1] == 2  # num_classes


# Integration tests
@pytest.mark.skipif(not ATTENTION_AVAILABLE, reason="Attention module not available")
class TestAttentionIntegration:
    """Integration tests for Attention components."""

    @pytest.fixture
    def financial_sequence_data(self):
        """Create realistic financial sequence data."""
        np.random.seed(42)
        n_samples = 200
        seq_len = 60

        # Financial features
        features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_20",
            "ema_12",
            "rsi",
            "macd",
            "bb_upper",
        ]
        n_features = len(features)

        # Generate correlated financial data
        base_prices = np.cumsum(np.random.randn(n_samples + seq_len) * 0.01) + 100

        X = []
        y = []

        for i in range(seq_len, n_samples + seq_len):
            # Create OHLC data
            prices = base_prices[i - seq_len : i]

            # Open, High, Low, Close
            ohlc = np.column_stack(
                [
                    prices,  # Open (simplified)
                    prices + np.random.uniform(0, 0.5, seq_len),  # High
                    prices - np.random.uniform(0, 0.5, seq_len),  # Low
                    prices + np.random.uniform(-0.2, 0.2, seq_len),  # Close
                ]
            )

            # Volume
            volume = np.random.exponential(1000, seq_len)

            # Technical indicators (simplified)
            sma_20 = np.convolve(prices, np.ones(20) / 20, mode="same")
            ema_12 = prices  # Simplified
            rsi = np.random.uniform(0, 100, seq_len)
            macd = np.random.randn(seq_len)
            bb_upper = sma_20 + 2 * np.std(prices)

            # Combine features
            features_data = np.column_stack(
                [
                    ohlc,
                    volume.reshape(-1, 1),
                    sma_20.reshape(-1, 1),
                    ema_12.reshape(-1, 1),
                    rsi.reshape(-1, 1),
                    macd.reshape(-1, 1),
                    bb_upper.reshape(-1, 1),
                ]
            )

            X.append(features_data)

            # Binary classification: price up or down
            y.append(1 if prices[-1] > prices[-2] else 0)

        return np.array(X), np.array(y), features

    def test_attention_with_financial_data(self, financial_sequence_data):
        """Test attention mechanisms with realistic financial data."""
        X, y, feature_names = financial_sequence_data

        # Create attention model
        model = create_attention_model(
            input_shape=X.shape[1:],
            attention_type="temporal",
            output_dim=2,
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Training should work without errors
        history = model.fit(X, y, epochs=1, validation_split=0.2, verbose=0)

        assert "loss" in history.history
        assert "accuracy" in history.history

    def test_attention_visualization_integration(self, financial_sequence_data):
        """Test attention visualization with real financial data."""
        X, y, feature_names = financial_sequence_data

        # Create model with attention
        model = create_attention_model(
            input_shape=X.shape[1:],
            attention_type="multi_head",
            output_dim=2,
        )

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Quick training
        model.fit(X, y, epochs=1, verbose=0)

        # Get attention weights for visualization
        # This would require model modification to return attention weights
        # For now, just test that the visualizer can handle the data format
        visualizer = AttentionVisualizer()

        # Create sample attention weights
        seq_len = X.shape[1]
        raw_weights = np.random.randn(4, 1, seq_len, seq_len)  # 4 heads
        attention_weights = np.exp(raw_weights) / np.sum(
            np.exp(raw_weights), axis=-1, keepdims=True
        )

        timestamps = pd.date_range("2023-01-01", periods=seq_len, freq="1h")

        # Test statistics calculation
        stats = visualizer.calculate_attention_statistics(attention_weights)

        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_attention_memory_efficiency(self):
        """Test attention mechanisms with larger datasets."""
        # Test memory efficiency with larger sequences
        n_samples = 500
        seq_len = 120  # Longer sequences
        n_features = 15

        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y = np.random.choice([0, 1], size=n_samples)

        # Create attention model
        model = create_attention_model(
            input_shape=(seq_len, n_features),
            attention_type="temporal",
            output_dim=2,
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Should handle larger data efficiently
        history = model.fit(
            X,
            y,
            epochs=1,
            batch_size=16,  # Smaller batch size for memory efficiency
            verbose=0,
        )

        assert "loss" in history.history


if __name__ == "__main__":
    pytest.main([__file__])
