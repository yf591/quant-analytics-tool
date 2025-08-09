"""
Transformer Architecture for Financial Time Series

This module implements Transformer models specifically designed for financial
time series prediction, following state-of-the-art attention mechanisms.

Based on:
- "Attention Is All You Need" (Vaswani et al., 2017)
- Financial time series adaptations
- AFML principles for financial machine learning
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from dataclasses import dataclass

from ..base import BaseFinancialModel, ModelConfig, ModelResults


# Simple utility functions for data preparation
def create_sequences(
    data: np.ndarray, sequence_length: int, prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series modeling."""
    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length : i + sequence_length + prediction_horizon])
    return np.array(X), np.array(y)


def prepare_financial_data(
    df: pd.DataFrame, target_col: str, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare financial data for transformer training."""
    # Simple implementation
    features = df.drop(columns=[target_col]).values
    target = df[target_col].values
    return create_sequences(features, sequence_length), create_sequences(
        target.reshape(-1, 1), sequence_length
    )


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Transformer models."""

    # Architecture parameters
    d_model: int = 64  # Model dimension
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer blocks
    dff: int = 256  # Feed-forward dimension
    dropout_rate: float = 0.1

    # Sequence parameters
    sequence_length: int = 60
    prediction_horizon: int = 1

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10

    # Financial parameters
    use_positional_encoding: bool = True
    use_temporal_embedding: bool = True
    financial_features: List[str] = None


class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer model."""

    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )

        pos_encoding = np.zeros((self.sequence_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(self.sequence_length, self.d_model),
            initializer="zeros",
            trainable=False,
        )
        self.pos_encoding.assign(pos_encoding)

    def call(self, x):
        return x + self.pos_encoding[: tf.shape(x)[1], :]


class TransformerBlock(layers.Layer):
    """Single transformer block with multi-head attention and feed-forward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        # Feed-forward network
        self.ffn = keras.Sequential(
            [
                layers.Dense(dff, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(d_model),
            ]
        )

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=None, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.mha(x, x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class FinancialTransformer(keras.Model):
    """Core Transformer model for financial time series."""

    def __init__(self, config: TransformerConfig, output_dim: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.output_dim = output_dim

        # Input projection
        self.input_projection = layers.Dense(config.d_model)

        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                config.sequence_length, config.d_model
            )

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dff=config.dff,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.num_layers)
        ]

        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling1D()

        # Output layers
        self.dropout = layers.Dropout(config.dropout_rate)
        self.dense1 = layers.Dense(config.d_model // 2, activation="relu")
        self.output_layer = layers.Dense(output_dim)

    def call(self, x, training=None, mask=None):
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        if self.config.use_positional_encoding:
            x = self.pos_encoding(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training, mask=mask)

        # Global pooling and output
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.output_layer(x)

        return output


class TransformerClassifier(BaseFinancialModel):
    """Transformer model for financial classification tasks."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        self.model = None
        self.scaler = None
        self.is_fitted = False

    def _build_model(self) -> Any:
        """Build the model architecture."""
        if hasattr(self.config, "d_model"):
            return FinancialTransformer(config=self.config, output_dim=2)
        else:
            # Fallback for old-style configs
            return None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None
    ) -> ModelResults:
        """
        Fit the transformer classifier.

        Args:
            X: Feature dataframe
            y: Target series (classification labels)
            validation_data: Optional validation data tuple (X_val, y_val)

        Returns:
            ModelResults object with training metrics
        """
        try:
            # Prepare data
            X_seq, y_processed, self.scaler = prepare_financial_data(
                X, y, sequence_length=self.config.sequence_length, classification=True
            )

            # Determine number of classes
            num_classes = len(np.unique(y_processed))

            # Build model
            self.model = FinancialTransformer(
                config=self.config, output_dim=num_classes
            )

            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate
                ),
                loss=(
                    "sparse_categorical_crossentropy"
                    if num_classes > 2
                    else "binary_crossentropy"
                ),
                metrics=["accuracy"],
            )

            # Prepare callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7
                ),
            ]

            # Train model
            history = self.model.fit(
                X_seq,
                y_processed,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1,
            )

            self.is_fitted = True

            # Calculate final training score
            train_score = max(history.history["accuracy"])

            return ModelResults(
                model=self.model,
                training_score=train_score,
                validation_score=max(history.history.get("val_accuracy", [0])),
                training_history=history.history,
                feature_importance=None,  # Attention weights can serve this purpose
                model_config=self.config,
            )

        except Exception as e:
            raise Exception(f"Error in TransformerClassifier fit: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Prepare sequences
            X_seq = create_sequences(
                self.scaler.transform(X), self.config.sequence_length
            )

            # Make predictions
            predictions = self.model.predict(X_seq)

            # Convert to class predictions
            if predictions.shape[1] > 1:
                return np.argmax(predictions, axis=1)
            else:
                return (predictions > 0.5).astype(int).flatten()

        except Exception as e:
            raise Exception(f"Error in TransformerClassifier predict: {str(e)}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            X_seq = create_sequences(
                self.scaler.transform(X), self.config.sequence_length
            )

            predictions = self.model.predict(X_seq)

            # Apply softmax for multi-class or sigmoid for binary
            if predictions.shape[1] > 1:
                return tf.nn.softmax(predictions).numpy()
            else:
                proba = tf.nn.sigmoid(predictions).numpy()
                return np.column_stack([1 - proba, proba])

        except Exception as e:
            raise Exception(f"Error in TransformerClassifier predict_proba: {str(e)}")


class TransformerRegressor(BaseFinancialModel):
    """Transformer model for financial regression tasks."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        self.model = None
        self.scaler = None
        self.is_fitted = False

    def _build_model(self) -> Any:
        """Build the model architecture."""
        if hasattr(self.config, "d_model"):
            return FinancialTransformer(config=self.config, output_dim=1)
        else:
            # Fallback for old-style configs
            return None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None
    ) -> ModelResults:
        """
        Fit the transformer regressor.

        Args:
            X: Feature dataframe
            y: Target series (continuous values)
            validation_data: Optional validation data tuple (X_val, y_val)

        Returns:
            ModelResults object with training metrics
        """
        try:
            # Prepare data
            X_seq, y_processed, self.scaler = prepare_financial_data(
                X, y, sequence_length=self.config.sequence_length, classification=False
            )

            # Build model
            self.model = FinancialTransformer(config=self.config, output_dim=1)

            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate
                ),
                loss="mse",
                metrics=["mae"],
            )

            # Prepare callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7
                ),
            ]

            # Train model
            history = self.model.fit(
                X_seq,
                y_processed,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1,
            )

            self.is_fitted = True

            # Calculate R² score as training score
            train_predictions = self.model.predict(X_seq)
            train_score = 1 - np.mean(
                (y_processed - train_predictions.flatten()) ** 2
            ) / np.var(y_processed)

            return ModelResults(
                model=self.model,
                training_score=train_score,
                validation_score=min(history.history.get("val_loss", [float("inf")])),
                training_history=history.history,
                feature_importance=None,
                model_config=self.config,
            )

        except Exception as e:
            raise Exception(f"Error in TransformerRegressor fit: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            X_seq = create_sequences(
                self.scaler.transform(X), self.config.sequence_length
            )

            predictions = self.model.predict(X_seq)
            return predictions.flatten()

        except Exception as e:
            raise Exception(f"Error in TransformerRegressor predict: {str(e)}")


def create_transformer_config(
    d_model: int = 64,
    num_heads: int = 8,
    num_layers: int = 4,
    sequence_length: int = 60,
    **kwargs,
) -> TransformerConfig:
    """
    Create a transformer configuration with sensible defaults.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        sequence_length: Input sequence length
        **kwargs: Additional configuration parameters

    Returns:
        TransformerConfig object
    """
    config = TransformerConfig(
        model_type="transformer",
        hyperparameters=kwargs,
        training_config={},
        validation_config={},
        feature_config={},
    )

    # Set transformer-specific parameters
    config.d_model = d_model
    config.num_heads = num_heads
    config.num_layers = num_layers
    config.sequence_length = sequence_length

    return config
