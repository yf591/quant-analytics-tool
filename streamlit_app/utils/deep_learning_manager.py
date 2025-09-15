#!/usr/bin/env python3
"""
Deep Learning Model Training Manager
For use in the Deep Learning Models Lab
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")


class DeepLearningModelManager:
    """Deep learning model training manager for LSTM and GRU models"""

    def __init__(self):
        self.scaler = StandardScaler()

    def train_model(
        self,
        feature_key: str,
        model_config: Dict[str, Any],
        hyperparams: Dict[str, Any],
        training_config: Dict[str, Any],
        session_state: Any,
    ) -> Tuple[bool, str, str]:
        """
        Train a deep learning model with the given configuration

        Returns:
            success: Whether training was successful
            message: Success/error message
            model_id: Unique identifier for the trained model
        """

        try:
            # Debug: Print session state info
            print(f"Debug: feature_key = {feature_key}")
            print(
                f"Debug: feature_cache keys = {list(session_state.feature_cache.keys()) if hasattr(session_state, 'feature_cache') else 'No feature_cache'}"
            )

            # Get feature data
            if not hasattr(session_state, "feature_cache"):
                return (
                    False,
                    "❌ No feature_cache found in session_state. Please run Feature Engineering first.",
                    None,
                )

            if not session_state.feature_cache:
                return (
                    False,
                    "❌ Feature cache is empty. Please generate features in Feature Engineering page first.",
                    None,
                )

            if feature_key not in session_state.feature_cache:
                available_keys = list(session_state.feature_cache.keys())
                return (
                    False,
                    f"❌ Feature dataset '{feature_key}' not found. Available datasets: {available_keys}",
                    None,
                )

            dataset_info = session_state.feature_cache[feature_key]
            print(f"Debug: dataset_info type = {type(dataset_info)}")

            # Handle different data formats from feature engineering
            if isinstance(dataset_info, pd.DataFrame):
                # Direct DataFrame from feature engineering
                features_df = dataset_info
                print(f"Debug: Using DataFrame directly, shape: {features_df.shape}")
            elif isinstance(dataset_info, dict) and "features" in dataset_info:
                # Dictionary format with 'features' key
                features_df = dataset_info["features"]
                print(f"Debug: Using features from dict, shape: {features_df.shape}")
            else:
                # Unsupported format
                if isinstance(dataset_info, dict):
                    available_keys = list(dataset_info.keys())
                    return (
                        False,
                        f"❌ Dataset missing 'features' key. Available keys: {available_keys}",
                        None,
                    )
                else:
                    return (
                        False,
                        f"❌ Unsupported dataset format: {type(dataset_info)}",
                        None,
                    )

            # Create a simple target if not provided
            target = None
            if isinstance(dataset_info, dict) and "target" in dataset_info:
                target = dataset_info["target"]
                print(f"Debug: Using existing target from dict")
            else:
                # Create a simple target based on available data
                print(
                    f"Debug: Creating target from features, columns: {list(features_df.columns)}"
                )

                # Try to get original price data for target creation
                target = self._create_target_from_data(
                    features_df, feature_key, session_state
                )

                if target is None:
                    return (
                        False,
                        "❌ Could not create target variable. Please ensure your data contains price or indicator columns.",
                        None,
                    )

            # Prepare features (remove target if it exists in features)
            X = features_df.copy()
            if hasattr(target, "name") and target.name in X.columns:
                X = X.drop(columns=[target.name])

            # Handle target column selection
            target_column = training_config.get("target_column", "auto")
            if target_column != "auto" and target_column in X.columns:
                target = X[target_column].copy()
                X = X.drop(columns=[target_column])

            # Basic data cleaning
            X = X.fillna(X.mean()).fillna(0)
            target = target.fillna(target.mode()[0] if len(target.mode()) > 0 else 0)

            # Convert target for classification tasks
            if model_config["task_type"] == "classification":
                print(
                    f"Debug: Converting target for classification. Original dtype: {target.dtype}"
                )
                print(
                    f"Debug: Target unique values before conversion: {np.unique(target)}"
                )

                # Ensure target is integer for classification
                if target.dtype in ["float64", "float32"]:
                    # If target is continuous, convert to binary based on threshold
                    if (
                        len(np.unique(target)) > 10
                    ):  # Too many unique values for classification
                        # Convert to binary based on median or sign
                        if np.all(target >= 0):
                            # Use median as threshold for positive values
                            threshold = target.median()
                            target = (target > threshold).astype(int)
                        else:
                            # Use sign for values that can be negative/positive
                            target = (target > 0).astype(int)
                    else:
                        # Already discrete values, just convert to int
                        target = target.astype(int)
                else:
                    # Already integer, but ensure it's int type
                    target = target.astype(int)

                print(
                    f"Debug: Target unique values after conversion: {np.unique(target)}"
                )
                print(f"Debug: Target dtype after conversion: {target.dtype}")

            # Align indices
            common_idx = X.index.intersection(target.index)
            X = X.loc[common_idx]
            target = target.loc[common_idx]

            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Prepare sequences for time series models
            sequence_length = hyperparams.get("sequence_length", 60)
            X_sequences, y_sequences = self._prepare_sequences(
                X, target, sequence_length
            )

            if len(X_sequences) == 0:
                return (
                    False,
                    f"❌ Not enough data for sequences. Need at least {sequence_length} samples.",
                    None,
                )

            # Scale features if requested
            if training_config.get("scale_features", True):
                # Reshape for scaling
                n_samples, n_timesteps, n_features = X_sequences.shape
                X_reshaped = X_sequences.reshape(-1, n_features)
                X_scaled = self.scaler.fit_transform(X_reshaped)
                X_sequences = X_scaled.reshape(n_samples, n_timesteps, n_features)

            # Split data
            test_size = training_config.get("test_size", 0.2)
            random_state = training_config.get("random_state", 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences,
                y_sequences,
                test_size=test_size,
                random_state=random_state,
                stratify=(
                    y_sequences
                    if model_config["task_type"] == "classification"
                    and len(np.unique(y_sequences)) > 1
                    else None
                ),
            )

            # Initialize model
            model = self._initialize_model(model_config, hyperparams)

            # Train model with preprocessed sequences
            print(f"Debug: Training model with shape: {X_train.shape}")

            # Create a dummy preprocessor for the model if it expects one
            from src.models.deep_learning.lstm import LSTMDataPreprocessor

            dummy_preprocessor = LSTMDataPreprocessor(
                sequence_length=sequence_length,
                feature_scaler="none",  # We already scaled the data
                target_scaler="none",
            )
            dummy_preprocessor.is_fitted = True
            dummy_preprocessor.X_scaler = None
            dummy_preprocessor.y_scaler = None
            model.preprocessor = dummy_preprocessor

            # Set model attributes that would normally be set during preprocessing
            model.n_features_in_ = X_train.shape[2]
            if model_config["task_type"] == "classification":
                model.n_classes_ = len(np.unique(y_train))
                model.classes_ = np.unique(y_train)

            # Build the model architecture
            if model_config["task_type"] == "classification":
                n_classes = len(np.unique(y_train))
                model.model_ = model._build_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    n_classes=n_classes,
                )
            else:
                model.model_ = model._build_model(
                    input_shape=(X_train.shape[1], X_train.shape[2])
                )

            # Compile the model with appropriate loss function
            if model_config["task_type"] == "classification":
                n_classes = len(np.unique(y_train))
                # Choose appropriate loss function based on number of classes
                if n_classes == 2:
                    # Binary classification: use binary_crossentropy
                    loss = "binary_crossentropy"
                    metrics = ["accuracy"]
                else:
                    # Multi-class classification: use sparse_categorical_crossentropy
                    loss = "sparse_categorical_crossentropy"
                    metrics = ["accuracy"]
            else:
                # Regression
                loss = "mse"
                metrics = ["mae"]

            model.model_.compile(
                optimizer=model._get_optimizer(),
                loss=loss,
                metrics=metrics,
            )

            # Train the model directly
            history = model.model_.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=hyperparams.get("epochs", 100),
                batch_size=hyperparams.get("batch_size", 32),
                callbacks=model._get_callbacks(),
                verbose=hyperparams.get("verbose", 1),
            )

            # Store training history
            model.history_ = history

            # Evaluate model
            evaluation = self._evaluate_model(
                model, X_train, X_test, y_train, y_test, model_config["task_type"]
            )

            # Generate unique model ID
            model_id = str(uuid.uuid4())

            # Store model information
            model_info = {
                "model": model,
                "model_config": model_config,
                "hyperparams": hyperparams,
                "training_config": training_config,
                "evaluation": evaluation,
                "feature_names": list(X.columns),
                "target_name": target.name if hasattr(target, "name") else "target",
                "scaler": (
                    self.scaler if training_config.get("scale_features", True) else None
                ),
                "timestamp": datetime.now(),
                "data_shape": {
                    "n_samples": len(X_sequences),
                    "n_features": X_sequences.shape[2],
                    "n_timesteps": X_sequences.shape[1],
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                },
                "training_history": (
                    history.history if hasattr(history, "history") else None
                ),
            }

            # Store in session state
            if "dl_model_cache" not in session_state:
                session_state.dl_model_cache = {}
            session_state.dl_model_cache[model_id] = model_info

            return True, "✅ Deep Learning model trained successfully", model_id

        except Exception as e:
            print(f"Error in train_model: {str(e)}")

            error_details = traceback.format_exc()
            print(f"Training error details: {error_details}")
            return (
                False,
                f"❌ Training failed: {str(e)}\n\nDetails: Check terminal for full error trace",
                None,
            )

    def _prepare_sequences(
        self, X: pd.DataFrame, y: pd.Series, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series models"""

        try:
            X_sequences = []
            y_sequences = []

            # Convert to numpy arrays
            X_array = X.values
            y_array = y.values

            # Create sequences
            for i in range(sequence_length, len(X_array)):
                X_sequences.append(X_array[i - sequence_length : i])
                y_sequences.append(y_array[i])

            return np.array(X_sequences), np.array(y_sequences)

        except Exception as e:
            print(f"Error preparing sequences: {e}")
            return np.array([]), np.array([])

    def _initialize_model(
        self, model_config: Dict[str, Any], hyperparams: Dict[str, Any]
    ):
        """Initialize deep learning model based on configuration"""

        model_type = model_config["model_type"]
        task_type = model_config["task_type"]
        model_class = model_config["model_class"]

        # Create model instance
        model = model_class(**hyperparams)

        return model

    def _evaluate_model(
        self, model, X_train, X_test, y_train, y_test, task_type: str
    ) -> Dict[str, float]:
        """Evaluate trained deep learning model"""

        try:
            print(f"Debug: Starting evaluation for {task_type}")
            print(
                f"Debug: y_train shape: {y_train.shape}, y_test shape: {y_test.shape}"
            )
            print(
                f"Debug: y_train dtype: {y_train.dtype}, y_test dtype: {y_test.dtype}"
            )
            print(f"Debug: y_train unique values: {np.unique(y_train)}")
            print(f"Debug: y_test unique values: {np.unique(y_test)}")

            # Ensure targets are integers for classification
            if task_type == "classification":
                print(f"Debug: Ensuring classification targets are integers")
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)
                print(
                    f"Debug: After conversion - y_train dtype: {y_train.dtype}, y_test dtype: {y_test.dtype}"
                )

            # Make predictions using the compiled model
            y_train_pred = model.model_.predict(X_train)
            y_test_pred = model.model_.predict(X_test)

            print(f"Debug: y_train_pred shape: {y_train_pred.shape}")
            print(f"Debug: y_test_pred shape: {y_test_pred.shape}")

            # Handle prediction shapes
            if len(y_train_pred.shape) > 1 and y_train_pred.shape[1] > 1:
                # Multi-class classification - use argmax
                y_train_pred = np.argmax(y_train_pred, axis=1)
                y_test_pred = np.argmax(y_test_pred, axis=1)
                print(f"Debug: Using argmax for multi-class")
            elif len(y_train_pred.shape) > 1:
                # Binary classification or regression - flatten
                y_train_pred = y_train_pred.flatten()
                y_test_pred = y_test_pred.flatten()
                print(f"Debug: Flattening predictions")

            print(f"Debug: Final y_train_pred shape: {y_train_pred.shape}")
            print(f"Debug: Final y_test_pred shape: {y_test_pred.shape}")

            if task_type == "classification":
                # Convert probabilities to class predictions for binary classification
                if y_train_pred.dtype in [np.float32, np.float64, float]:
                    print(f"Debug: Converting probabilities to classes")
                    print(
                        f"Debug: y_train_pred range: {y_train_pred.min():.4f} to {y_train_pred.max():.4f}"
                    )
                    print(
                        f"Debug: y_train_pred dtype before conversion: {y_train_pred.dtype}"
                    )
                    y_train_pred = (y_train_pred > 0.5).astype(int)
                    y_test_pred = (y_test_pred > 0.5).astype(int)
                    print(
                        f"Debug: y_train_pred dtype after conversion: {y_train_pred.dtype}"
                    )
                    print(
                        f"Debug: y_test_pred dtype after conversion: {y_test_pred.dtype}"
                    )

                # Ensure predictions are integers (additional safety check)
                y_train_pred = y_train_pred.astype(int)
                y_test_pred = y_test_pred.astype(int)

                print(
                    f"Debug: Final predictions - train unique: {np.unique(y_train_pred)}, test unique: {np.unique(y_test_pred)}"
                )
                print(
                    f"Debug: Final target types - y_train type: {type(y_train[0])}, y_train_pred type: {type(y_train_pred[0])}"
                )
                print(
                    f"Debug: Final dtypes - y_train: {y_train.dtype}, y_train_pred: {y_train_pred.dtype}"
                )

                evaluation_result = {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "train_precision": precision_score(
                        y_train, y_train_pred, average="weighted", zero_division=0
                    ),
                    "test_precision": precision_score(
                        y_test, y_test_pred, average="weighted", zero_division=0
                    ),
                    "train_recall": recall_score(
                        y_train, y_train_pred, average="weighted", zero_division=0
                    ),
                    "test_recall": recall_score(
                        y_test, y_test_pred, average="weighted", zero_division=0
                    ),
                    "train_f1": f1_score(
                        y_train, y_train_pred, average="weighted", zero_division=0
                    ),
                    "test_f1": f1_score(
                        y_test, y_test_pred, average="weighted", zero_division=0
                    ),
                }

                print(f"Debug: Evaluation results: {evaluation_result}")
                return evaluation_result
            else:
                evaluation_result = {
                    "train_mse": mean_squared_error(y_train, y_train_pred),
                    "test_mse": mean_squared_error(y_test, y_test_pred),
                    "train_mae": mean_absolute_error(y_train, y_train_pred),
                    "test_mae": mean_absolute_error(y_test, y_test_pred),
                    "train_r2": r2_score(y_train, y_train_pred),
                    "test_r2": r2_score(y_test, y_test_pred),
                }

                print(f"Debug: Regression evaluation results: {evaluation_result}")
                return evaluation_result

        except Exception as e:
            # Return basic metrics if detailed evaluation fails
            print(f"Debug: Evaluation failed with error: {str(e)}")
            import traceback

            print(f"Debug: Full traceback: {traceback.format_exc()}")
            return {"error": str(e), "basic_score": 0.0}

    def _create_target_from_data(self, features_df, feature_key, session_state):
        """Create target variable from available data sources"""

        try:
            # Method 1: Try to get original price data from data_cache
            target = self._create_target_from_price_data(
                feature_key, session_state, features_df.index
            )
            if target is not None:
                print(f"Debug: Created target from price data, shape: {target.shape}")
                return target

            # Method 2: Create target from technical indicators
            target = self._create_target_from_indicators(features_df)
            if target is not None:
                print(f"Debug: Created target from indicators, shape: {target.shape}")
                return target

            # Method 3: Create target from momentum/trend indicators
            target = self._create_target_from_momentum(features_df)
            if target is not None:
                print(f"Debug: Created target from momentum, shape: {target.shape}")
                return target

            return None

        except Exception as e:
            print(f"Debug: Error creating target: {e}")
            return None

    def _create_target_from_price_data(self, feature_key, session_state, feature_index):
        """Try to create target from original price data in data_cache"""

        try:
            # Extract symbol from feature_key (e.g., "GOOGL_1y_1d_technical" -> "GOOGL")
            symbol = feature_key.split("_")[0]

            if not hasattr(session_state, "data_cache") or not session_state.data_cache:
                return None

            # Look for matching symbol in data_cache
            for data_key, data_info in session_state.data_cache.items():
                if symbol in data_key and isinstance(data_info, dict):
                    if "data" in data_info and isinstance(
                        data_info["data"], pd.DataFrame
                    ):
                        price_data = data_info["data"]
                        if "Close" in price_data.columns:
                            # Create future return target
                            aligned_close = price_data["Close"].reindex(
                                feature_index, method="ffill"
                            )
                            if len(aligned_close.dropna()) > 1:
                                returns = aligned_close.pct_change().shift(
                                    -1
                                )  # Next period return
                                # Convert to binary: 1 if positive return, 0 if negative
                                target = (returns > 0).astype(int)
                                return target.dropna()
            return None

        except Exception as e:
            print(f"Debug: Error in _create_target_from_price_data: {e}")
            return None

    def _create_target_from_indicators(self, features_df):
        """Create target from technical indicators like RSI or MACD"""

        try:
            # Strategy 1: RSI-based target (RSI > 50 indicates bullish)
            if "RSI" in features_df.columns:
                rsi_signal = (features_df["RSI"] > 50).astype(int)
                if len(rsi_signal.unique()) > 1:  # Ensure we have both classes
                    return rsi_signal

            # Strategy 2: MACD-based target
            if (
                "MACD_MACD" in features_df.columns
                and "MACD_Signal" in features_df.columns
            ):
                macd_signal = (
                    features_df["MACD_MACD"] > features_df["MACD_Signal"]
                ).astype(int)
                if len(macd_signal.unique()) > 1:
                    return macd_signal

            # Strategy 3: Moving Average Cross
            if "SMA_20" in features_df.columns and "SMA_50" in features_df.columns:
                ma_signal = (features_df["SMA_20"] > features_df["SMA_50"]).astype(int)
                if len(ma_signal.unique()) > 1:
                    return ma_signal

            return None

        except Exception as e:
            print(f"Debug: Error in _create_target_from_indicators: {e}")
            return None

    def _create_target_from_momentum(self, features_df):
        """Create target from momentum indicators"""

        try:
            # Strategy 1: Momentum-based target
            if "MOMENTUM" in features_df.columns:
                momentum_signal = (features_df["MOMENTUM"] > 0).astype(int)
                if len(momentum_signal.unique()) > 1:
                    return momentum_signal

            # Strategy 2: Stochastic-based target
            if "Stoch_%K" in features_df.columns:
                stoch_signal = (features_df["Stoch_%K"] > 50).astype(int)
                if len(stoch_signal.unique()) > 1:
                    return stoch_signal

            # Strategy 3: ATR-based volatility regime
            if "ATR" in features_df.columns:
                atr_median = features_df["ATR"].median()
                volatility_regime = (features_df["ATR"] > atr_median).astype(int)
                if len(volatility_regime.unique()) > 1:
                    return volatility_regime

            return None

        except Exception as e:
            print(f"Debug: Error in _create_target_from_momentum: {e}")
            return None
