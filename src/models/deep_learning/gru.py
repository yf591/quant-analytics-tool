"""
GRU Models for Financial Time Series Analysis

This module implements GRU (Gated Recurrent Unit) classifiers and regressors
optimized for financial time series prediction with advanced features.

Based on financial machine learning best practices and AFML principles.
Implements both unidirectional and bidirectional GRU architectures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GRU,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    TimeDistributed,
    BatchNormalization,
    Activation,
    Lambda,
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

from ..base import BaseClassifier, BaseRegressor, register_model
from .lstm import LSTMDataPreprocessor  # Reuse LSTM preprocessor


# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")


@register_model("gru_classifier")
class QuantGRUClassifier(BaseClassifier):
    """
    GRU Classifier optimized for financial time series classification.

    Implements GRU neural networks with financial-specific features including
    bidirectional processing and robust training procedures. GRU is often faster
    than LSTM while maintaining similar performance.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        gru_units: List[int] = [50, 50],
        dense_units: List[int] = [25],
        dropout_rate: float = 0.2,
        recurrent_dropout: float = 0.2,
        bidirectional: bool = False,
        activation: str = "tanh",
        dense_activation: str = "relu",
        output_activation: str = "softmax",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        l1_reg: float = 0.0,
        l2_reg: float = 0.01,
        batch_norm: bool = True,
        feature_scaler: str = "standard",
        random_state: int = 42,
        verbose: int = 1,
        **kwargs,
    ):
        """
        Initialize GRU Classifier.

        Args:
            sequence_length: Number of time steps in each sequence
            gru_units: List of GRU layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate for regularization
            recurrent_dropout: Recurrent dropout rate
            bidirectional: Whether to use bidirectional GRU
            activation: GRU activation function
            dense_activation: Dense layer activation function
            output_activation: Output layer activation function
            optimizer: Optimizer type
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            validation_split: Validation data split ratio
            early_stopping_patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            batch_norm: Whether to use batch normalization
            feature_scaler: Feature scaling method
            random_state: Random state for reproducibility
            verbose: Verbosity level
        """
        super().__init__(**kwargs)

        # Model architecture parameters
        self.sequence_length = sequence_length
        self.gru_units = gru_units if isinstance(gru_units, list) else [gru_units]
        self.dense_units = (
            dense_units if isinstance(dense_units, list) else [dense_units]
        )
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        self.activation = activation
        self.dense_activation = dense_activation
        self.output_activation = output_activation

        # Training parameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience

        # Regularization
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_norm = batch_norm

        # Data preprocessing
        self.feature_scaler = feature_scaler
        self.random_state = random_state
        self.verbose = verbose

        # Initialize components
        self.model_ = None
        self.preprocessor = None
        self.n_features_in_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.history_ = None
        self.training_time = None

        # Override training_history from base class to be dictionary
        self.training_history = {}

        # Set random seeds for reproducibility
        self._set_random_seeds()

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def _build_model(self, input_shape: Tuple[int, int], n_classes: int) -> Model:
        """
        Build GRU model architecture.

        Args:
            input_shape: Shape of input data (timesteps, features)
            n_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        model = Sequential()

        # Input layer
        model.add(Input(shape=input_shape))

        # GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1

            if self.bidirectional:
                gru_layer = Bidirectional(
                    GRU(
                        units,
                        return_sequences=return_sequences,
                        activation=self.activation,
                        recurrent_dropout=self.recurrent_dropout,
                        kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    )
                )
            else:
                gru_layer = GRU(
                    units,
                    return_sequences=return_sequences,
                    activation=self.activation,
                    recurrent_dropout=self.recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                )

            model.add(gru_layer)

            # Batch normalization
            if self.batch_norm:
                model.add(BatchNormalization())

            # Dropout
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        # Dense layers
        for units in self.dense_units:
            model.add(
                Dense(
                    units,
                    activation=self.dense_activation,
                    kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                )
            )

            if self.batch_norm:
                model.add(BatchNormalization())

            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        # Output layer
        if n_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
        else:
            model.add(Dense(n_classes, activation=self.output_activation))

        return model

    def _get_optimizer(self):
        """Get optimizer instance."""
        if self.optimizer.lower() == "adam":
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "rmsprop":
            return RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _get_callbacks(self):
        """Get training callbacks."""
        callbacks_list = []

        # Early stopping
        if self.early_stopping_patience > 0:
            callbacks_list.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                )
            )

        # Learning rate reduction
        if self.reduce_lr_patience > 0:
            callbacks_list.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=self.verbose,
                )
            )

        return callbacks_list

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "QuantGRUClassifier":
        """
        Fit GRU classifier.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights (optional)
            **kwargs: Additional fit parameters

        Returns:
            Fitted classifier
        """
        start_time = datetime.now()

        # Validate inputs
        X, y = self._validate_input(X, y)

        # Store classes and dimensions
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        # Initialize preprocessor (reuse LSTM preprocessor)
        self.preprocessor = LSTMDataPreprocessor(
            sequence_length=self.sequence_length, feature_scaler=self.feature_scaler
        )

        # Preprocess data
        X_sequences, y_sequences = self.preprocessor.fit_transform(X, y)

        if len(X_sequences) == 0:
            raise ValueError(
                f"Not enough data to create sequences. Need at least {self.sequence_length} samples."
            )

        # Convert labels to categorical if multiclass
        if self.n_classes_ > 2:
            y_categorical = tf.keras.utils.to_categorical(
                y_sequences, num_classes=self.n_classes_
            )
        else:
            y_categorical = y_sequences

        # Build model
        input_shape = (self.sequence_length, self.n_features_in_)
        self.model_ = self._build_model(input_shape, self.n_classes_)

        # Compile model
        optimizer = self._get_optimizer()

        if self.n_classes_ == 2:
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            loss = "categorical_crossentropy"
            metrics = ["accuracy"]

        self.model_.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Get callbacks
        callbacks = self._get_callbacks()

        # Train model
        self.history_ = self.model_.fit(
            X_sequences,
            y_categorical,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=self.verbose,
            **kwargs,
        )

        self.training_time = datetime.now() - start_time

        # Update training history
        self.training_history.update(
            {
                "training_time": self.training_time,
                "final_loss": self.history_.history["loss"][-1],
                "final_val_loss": self.history_.history.get("val_loss", [None])[-1],
                "final_accuracy": self.history_.history["accuracy"][-1],
                "final_val_accuracy": self.history_.history.get("val_accuracy", [None])[
                    -1
                ],
                "epochs_trained": len(self.history_.history["loss"]),
            }
        )

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")

        # Preprocess data
        X_sequences, _ = self.preprocessor.transform(X)

        if len(X_sequences) == 0:
            raise ValueError("Not enough data to create sequences for prediction")

        # Make predictions
        predictions = self.model_.predict(X_sequences, verbose=0)

        # Convert to class labels
        if self.n_classes_ == 2:
            predicted_classes = (predictions > 0.5).astype(int).flatten()
        else:
            predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted class probabilities
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")

        # Preprocess data
        X_sequences, _ = self.preprocessor.transform(X)

        if len(X_sequences) == 0:
            raise ValueError("Not enough data to create sequences for prediction")

        # Make predictions
        probabilities = self.model_.predict(X_sequences, verbose=0)

        # Format probabilities
        if self.n_classes_ == 2:
            # Binary classification: return probabilities for both classes
            proba_neg = 1 - probabilities.flatten()
            proba_pos = probabilities.flatten()
            return np.column_stack([proba_neg, proba_pos])
        else:
            # Multiclass classification
            return probabilities

    def plot_training_history(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot training history.

        Args:
            figsize: Figure size
        """
        if self.history_ is None:
            raise ValueError("Model must be trained before plotting history")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot loss
        ax1.plot(self.history_.history["loss"], label="Training Loss")
        if "val_loss" in self.history_.history:
            ax1.plot(self.history_.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Plot accuracy
        ax2.plot(self.history_.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in self.history_.history:
            ax2.plot(self.history_.history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        return fig

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model_ is None:
            raise ValueError("Model must be built before getting summary")

        summary_lines = []
        self.model_.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)


@register_model("gru_regressor")
class QuantGRURegressor(BaseRegressor):
    """
    GRU Regressor optimized for financial time series regression.

    Implements GRU neural networks with financial-specific features including
    bidirectional processing and robust training procedures for return prediction.
    GRU is often faster than LSTM while maintaining similar performance.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        gru_units: List[int] = [50, 50],
        dense_units: List[int] = [25],
        dropout_rate: float = 0.2,
        recurrent_dropout: float = 0.2,
        bidirectional: bool = False,
        activation: str = "tanh",
        dense_activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        l1_reg: float = 0.0,
        l2_reg: float = 0.01,
        batch_norm: bool = True,
        feature_scaler: str = "standard",
        target_scaler: str = "standard",
        random_state: int = 42,
        verbose: int = 1,
        **kwargs,
    ):
        """
        Initialize GRU Regressor.

        Args:
            sequence_length: Number of time steps in each sequence
            gru_units: List of GRU layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate for regularization
            recurrent_dropout: Recurrent dropout rate
            bidirectional: Whether to use bidirectional GRU
            activation: GRU activation function
            dense_activation: Dense layer activation function
            optimizer: Optimizer type
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            validation_split: Validation data split ratio
            early_stopping_patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            batch_norm: Whether to use batch normalization
            feature_scaler: Feature scaling method
            target_scaler: Target scaling method
            random_state: Random state for reproducibility
            verbose: Verbosity level
        """
        super().__init__(**kwargs)

        # Model architecture parameters
        self.sequence_length = sequence_length
        self.gru_units = gru_units if isinstance(gru_units, list) else [gru_units]
        self.dense_units = (
            dense_units if isinstance(dense_units, list) else [dense_units]
        )
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        self.activation = activation
        self.dense_activation = dense_activation

        # Training parameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience

        # Regularization
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_norm = batch_norm

        # Data preprocessing
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.random_state = random_state
        self.verbose = verbose

        # Initialize components
        self.model_ = None
        self.preprocessor = None
        self.n_features_in_ = None
        self.history_ = None
        self.training_time = None

        # Override training_history from base class to be dictionary
        self.training_history = {}

        # Set random seeds for reproducibility
        self._set_random_seeds()

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build GRU model architecture.

        Args:
            input_shape: Shape of input data (timesteps, features)

        Returns:
            Compiled Keras model
        """
        model = Sequential()

        # Input layer
        model.add(Input(shape=input_shape))

        # GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1

            if self.bidirectional:
                gru_layer = Bidirectional(
                    GRU(
                        units,
                        return_sequences=return_sequences,
                        activation=self.activation,
                        recurrent_dropout=self.recurrent_dropout,
                        kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    )
                )
            else:
                gru_layer = GRU(
                    units,
                    return_sequences=return_sequences,
                    activation=self.activation,
                    recurrent_dropout=self.recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                )

            model.add(gru_layer)

            # Batch normalization
            if self.batch_norm:
                model.add(BatchNormalization())

            # Dropout
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        # Dense layers
        for units in self.dense_units:
            model.add(
                Dense(
                    units,
                    activation=self.dense_activation,
                    kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                )
            )

            if self.batch_norm:
                model.add(BatchNormalization())

            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        # Output layer (single output for regression)
        model.add(Dense(1))

        return model

    def _get_optimizer(self):
        """Get optimizer instance."""
        if self.optimizer.lower() == "adam":
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "rmsprop":
            return RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _get_callbacks(self):
        """Get training callbacks."""
        callbacks_list = []

        # Early stopping
        if self.early_stopping_patience > 0:
            callbacks_list.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                )
            )

        # Learning rate reduction
        if self.reduce_lr_patience > 0:
            callbacks_list.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=self.verbose,
                )
            )

        return callbacks_list

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "QuantGRURegressor":
        """
        Fit GRU regressor.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights (optional)
            **kwargs: Additional fit parameters

        Returns:
            Fitted regressor
        """
        start_time = datetime.now()

        # Validate inputs
        X, y = self._validate_input(X, y)

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Initialize preprocessor (reuse LSTM preprocessor)
        self.preprocessor = LSTMDataPreprocessor(
            sequence_length=self.sequence_length,
            feature_scaler=self.feature_scaler,
            target_scaler=self.target_scaler,
        )

        # Preprocess data
        X_sequences, y_sequences = self.preprocessor.fit_transform(X, y)

        if len(X_sequences) == 0:
            raise ValueError(
                f"Not enough data to create sequences. Need at least {self.sequence_length} samples."
            )

        # Build model
        input_shape = (self.sequence_length, self.n_features_in_)
        self.model_ = self._build_model(input_shape)

        # Compile model
        optimizer = self._get_optimizer()
        self.model_.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # Get callbacks
        callbacks = self._get_callbacks()

        # Train model
        self.history_ = self.model_.fit(
            X_sequences,
            y_sequences,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=self.verbose,
            **kwargs,
        )

        self.training_time = datetime.now() - start_time

        # Update training history
        self.training_history.update(
            {
                "training_time": self.training_time,
                "final_loss": self.history_.history["loss"][-1],
                "final_val_loss": self.history_.history.get("val_loss", [None])[-1],
                "final_mae": self.history_.history["mae"][-1],
                "final_val_mae": self.history_.history.get("val_mae", [None])[-1],
                "epochs_trained": len(self.history_.history["loss"]),
            }
        )

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")

        # Preprocess data
        X_sequences, _ = self.preprocessor.transform(X)

        if len(X_sequences) == 0:
            raise ValueError("Not enough data to create sequences for prediction")

        # Make predictions
        predictions_scaled = self.model_.predict(X_sequences, verbose=0).flatten()

        # Inverse transform predictions
        predictions = self.preprocessor.inverse_transform_target(predictions_scaled)

        return predictions

    def predict_with_uncertainty(
        self, X: Union[np.ndarray, pd.DataFrame], n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout.

        Args:
            X: Input features
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")

        # Preprocess data
        X_sequences, _ = self.preprocessor.transform(X)

        if len(X_sequences) == 0:
            raise ValueError("Not enough data to create sequences for prediction")

        # Enable training mode for dropout during inference
        predictions_samples = []

        for _ in range(n_samples):
            # Make prediction with dropout enabled
            pred = self.model_(X_sequences, training=True).numpy().flatten()
            predictions_samples.append(pred)

        predictions_samples = np.array(predictions_samples)

        # Calculate statistics
        mean_predictions = np.mean(predictions_samples, axis=0)
        std_predictions = np.std(predictions_samples, axis=0)

        # Inverse transform
        mean_predictions = self.preprocessor.inverse_transform_target(mean_predictions)
        std_predictions = self.preprocessor.inverse_transform_target(std_predictions)

        return mean_predictions, std_predictions

    def plot_training_history(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot training history.

        Args:
            figsize: Figure size
        """
        if self.history_ is None:
            raise ValueError("Model must be trained before plotting history")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot loss
        ax1.plot(self.history_.history["loss"], label="Training Loss")
        if "val_loss" in self.history_.history:
            ax1.plot(self.history_.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Plot MAE
        ax2.plot(self.history_.history["mae"], label="Training MAE")
        if "val_mae" in self.history_.history:
            ax2.plot(self.history_.history["val_mae"], label="Validation MAE")
        ax2.set_title("Model MAE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.legend()

        plt.tight_layout()
        return fig

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model_ is None:
            raise ValueError("Model must be built before getting summary")

        summary_lines = []
        self.model_.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
