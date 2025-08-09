"""
Attention Mechanisms for Financial Time Series

This module implements various attention mechanisms specifically designed
for financial time series analysis, including temporal attention and
attention visualization tools.

Based on:
- Multi-head attention mechanisms
- Temporal attention for financial patterns
- Attention visualization for interpretability
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from dataclasses import dataclass


class AttentionLayer(layers.Layer):
    """Basic attention layer for financial time series."""

    def __init__(self, units: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="attention_bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )

        self.V = self.add_weight(
            name="attention_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x, mask=None):
        # Calculate attention scores
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1))
        if self.use_bias:
            e = e + self.b

        # Calculate attention weights
        attention_scores = tf.tensordot(e, self.V, axes=1)
        attention_scores = tf.squeeze(attention_scores, axis=-1)

        # Apply mask if provided
        if mask is not None:
            attention_scores += mask * -1e9

        # Apply softmax to get normalized weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Apply attention weights
        attended_output = tf.reduce_sum(
            x * tf.expand_dims(attention_weights, axis=-1), axis=1
        )

        return attended_output, attention_weights


class MultiHeadAttention(layers.Layer):
    """Multi-head attention specifically designed for financial data."""

    def __init__(
        self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.dropout_rate = dropout_rate

        # Linear projections for Q, K, V
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        # Output projection
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights and apply to values."""

        # Calculate attention scores
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale by square root of key dimension
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, q, k, v, mask=None, training=None):
        batch_size = tf.shape(q)[0]

        # Linear projections
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Apply scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.dense(concat_attention)

        return output, attention_weights


class TemporalAttention(layers.Layer):
    """Temporal attention mechanism for financial time series patterns."""

    def __init__(self, units: int, time_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.time_steps = time_steps

    def build(self, input_shape):
        # Temporal embedding weights
        self.temporal_embedding = self.add_weight(
            name="temporal_embedding",
            shape=(self.time_steps, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Attention computation weights
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.attention_bias = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        self.attention_output = self.add_weight(
            name="attention_output",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Get temporal embeddings for current sequence length
        temporal_emb = self.temporal_embedding[:seq_len, :]

        # Broadcast temporal embeddings to match batch size
        temporal_emb = tf.tile(tf.expand_dims(temporal_emb, 0), [batch_size, 1, 1])

        # Concatenate input features with temporal embeddings
        combined_input = tf.concat([x, temporal_emb], axis=-1)

        # Compute attention scores
        attention_hidden = tf.nn.tanh(
            tf.tensordot(combined_input, self.attention_weights, axes=1)
            + self.attention_bias
        )

        attention_scores = tf.tensordot(attention_hidden, self.attention_output, axes=1)
        attention_scores = tf.squeeze(attention_scores, axis=-1)

        # Apply mask if provided
        if mask is not None:
            attention_scores += mask * -1e9

        # Apply softmax to get normalized weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Apply attention weights to input
        attended_output = tf.reduce_sum(
            x * tf.expand_dims(attention_weights, axis=-1), axis=1
        )

        return attended_output, attention_weights


class AttentionVisualizer:
    """Utility class for visualizing attention weights."""

    def __init__(self):
        self.attention_weights = None
        self.feature_names = None

    def store_attention_weights(self, weights: np.ndarray, feature_names: List[str]):
        """Store attention weights and feature names for visualization."""
        self.attention_weights = weights
        self.feature_names = feature_names

    def plot_attention_heatmap(
        self, sample_idx: int = 0, figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot attention heatmap for a specific sample.

        Args:
            sample_idx: Index of the sample to visualize
            figsize: Figure size tuple
        """
        if self.attention_weights is None:
            raise ValueError(
                "No attention weights stored. Call store_attention_weights first."
            )

        # Extract attention weights for the specified sample
        if len(self.attention_weights.shape) == 3:  # Multi-head attention
            weights = np.mean(self.attention_weights[sample_idx], axis=0)
        else:
            weights = self.attention_weights[sample_idx]

        # Create the heatmap
        plt.figure(figsize=figsize)

        if (
            self.feature_names is not None
            and len(self.feature_names) == weights.shape[1]
        ):
            # If we have feature names, use them as labels
            sns.heatmap(
                weights,
                xticklabels=self.feature_names,
                yticklabels=[f"Time_{i}" for i in range(weights.shape[0])],
                cmap="Blues",
                annot=False,
                cbar_kws={"label": "Attention Weight"},
            )
        else:
            # Otherwise, use numeric labels
            sns.heatmap(
                weights,
                cmap="Blues",
                annot=False,
                cbar_kws={"label": "Attention Weight"},
            )

        plt.title(f"Attention Weights Heatmap (Sample {sample_idx})")
        plt.xlabel("Features")
        plt.ylabel("Time Steps")
        plt.tight_layout()
        plt.show()

    def plot_temporal_attention(
        self, sample_idx: int = 0, figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot temporal attention weights over time.

        Args:
            sample_idx: Index of the sample to visualize
            figsize: Figure size tuple
        """
        if self.attention_weights is None:
            raise ValueError(
                "No attention weights stored. Call store_attention_weights first."
            )

        # Extract temporal attention (average across features)
        if len(self.attention_weights.shape) == 3:
            temporal_weights = np.mean(self.attention_weights[sample_idx], axis=-1)
        else:
            temporal_weights = np.mean(self.attention_weights[sample_idx], axis=-1)

        plt.figure(figsize=figsize)
        plt.plot(range(len(temporal_weights)), temporal_weights, "b-", linewidth=2)
        plt.fill_between(range(len(temporal_weights)), temporal_weights, alpha=0.3)

        plt.title(f"Temporal Attention Weights (Sample {sample_idx})")
        plt.xlabel("Time Steps")
        plt.ylabel("Attention Weight")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_feature_attention(
        self, time_step: int = -1, figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot feature attention weights for a specific time step.

        Args:
            time_step: Time step to visualize (-1 for last time step)
            figsize: Figure size tuple
        """
        if self.attention_weights is None:
            raise ValueError(
                "No attention weights stored. Call store_attention_weights first."
            )

        # Extract feature attention for the specified time step
        if len(self.attention_weights.shape) == 3:
            feature_weights = np.mean(self.attention_weights[:, :, time_step], axis=0)
        else:
            feature_weights = np.mean(self.attention_weights[:, time_step, :], axis=0)

        plt.figure(figsize=figsize)

        x_labels = (
            self.feature_names
            if self.feature_names
            else [f"Feature_{i}" for i in range(len(feature_weights))]
        )

        plt.bar(range(len(feature_weights)), feature_weights)
        plt.title(f"Feature Attention Weights (Time Step {time_step})")
        plt.xlabel("Features")
        plt.ylabel("Attention Weight")
        plt.xticks(range(len(feature_weights)), x_labels, rotation=45)
        plt.tight_layout()
        plt.show()


def create_attention_model(
    input_shape: Tuple[int, int],
    attention_type: str = "multi_head",
    units: int = 64,
    num_heads: int = 8,
    output_dim: int = 1,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """
    Create a model with specified attention mechanism.

    Args:
        input_shape: Shape of input data (time_steps, features)
        attention_type: Type of attention ("basic", "multi_head", "temporal")
        units: Number of attention units
        num_heads: Number of heads for multi-head attention
        output_dim: Output dimension
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model with attention mechanism
    """
    inputs = keras.Input(shape=input_shape)

    # Apply attention mechanism based on type
    if attention_type == "basic":
        attended_output, attention_weights = AttentionLayer(units)(inputs)

    elif attention_type == "multi_head":
        attended_output, attention_weights = MultiHeadAttention(
            d_model=units, num_heads=num_heads, dropout_rate=dropout_rate
        )(inputs, inputs, inputs)

        # Global average pooling for multi-head attention output
        attended_output = layers.GlobalAveragePooling1D()(attended_output)

    elif attention_type == "temporal":
        attended_output, attention_weights = TemporalAttention(
            units=units, time_steps=input_shape[0]
        )(inputs)

    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

    # Add output layers
    x = layers.Dropout(dropout_rate)(attended_output)
    x = layers.Dense(units // 2, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(output_dim)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
