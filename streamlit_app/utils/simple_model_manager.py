#!/usr/bin/env python3
"""
Simple Model Training Manager
For use in the Traditional ML Models Lab
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import uuid
import warnings

warnings.filterwarnings("ignore")


class SimpleModelTrainingManager:
    """Simple model training manager for traditional ML models"""

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
        Train a model with the given configuration

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
                        f"❌ Dataset has unexpected structure. Available keys: {available_keys}",
                        None,
                    )
                else:
                    return (
                        False,
                        f"❌ Dataset is not a DataFrame or dictionary. Type: {type(dataset_info)}",
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
                    available_cols = list(features_df.columns)
                    return (
                        False,
                        f"❌ No suitable target variable found. Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}",
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

            # Align indices
            common_idx = X.index.intersection(target.index)
            X = X.loc[common_idx]
            target = target.loc[common_idx]

            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Scale features if requested
            if training_config.get("scale_features", False):
                X_scaled = self.scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            # Split data
            test_size = training_config.get("test_size", 0.2)
            random_state = training_config.get("random_state", 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                target,
                test_size=test_size,
                random_state=random_state,
                stratify=(
                    target
                    if model_config["task_type"] == "classification"
                    and len(target.unique()) > 1
                    else None
                ),
            )

            # Initialize model
            model = self._initialize_model(model_config, hyperparams)

            # Train model
            model.fit(X_train, y_train)

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
                    self.scaler
                    if training_config.get("scale_features", False)
                    else None
                ),
                "timestamp": datetime.now(),
                "data_shape": {
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                },
            }

            # Add feature importance if available
            if hasattr(model, "feature_importances_"):
                importance_df = pd.Series(
                    model.feature_importances_, index=X.columns, name="importance"
                ).sort_values(ascending=False)
                model_info["feature_importance"] = importance_df

            # Store in session state
            if "model_cache" not in session_state:
                session_state.model_cache = {}
            session_state.model_cache[model_id] = model_info

            return True, "✅ Model trained successfully", model_id

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            print(f"Training error details: {error_details}")
            return (
                False,
                f"❌ Training failed: {str(e)}\n\nDetails: Check terminal for full error trace",
                None,
            )

    def _initialize_model(
        self, model_config: Dict[str, Any], hyperparams: Dict[str, Any]
    ):
        """Initialize model based on configuration"""

        model_type = model_config["model_type"]
        task_type = model_config["task_type"]

        # Remove non-sklearn parameters
        clean_params = {
            k: v for k, v in hyperparams.items() if k not in ["scale_features"]
        }

        if model_type == "RandomForest":
            if task_type == "classification":
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(**clean_params)
            else:
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(**clean_params)

        elif model_type == "XGBoost":
            try:
                import xgboost as xgb

                if task_type == "classification":
                    return xgb.XGBClassifier(**clean_params)
                else:
                    return xgb.XGBRegressor(**clean_params)
            except ImportError:
                # Fallback to sklearn GradientBoosting
                if task_type == "classification":
                    from sklearn.ensemble import GradientBoostingClassifier

                    gb_params = {
                        "n_estimators": clean_params.get("n_estimators", 100),
                        "max_depth": clean_params.get("max_depth", 6),
                        "learning_rate": clean_params.get("learning_rate", 0.1),
                        "random_state": clean_params.get("random_state", 42),
                    }
                    return GradientBoostingClassifier(**gb_params)
                else:
                    from sklearn.ensemble import GradientBoostingRegressor

                    gb_params = {
                        "n_estimators": clean_params.get("n_estimators", 100),
                        "max_depth": clean_params.get("max_depth", 6),
                        "learning_rate": clean_params.get("learning_rate", 0.1),
                        "random_state": clean_params.get("random_state", 42),
                    }
                    return GradientBoostingRegressor(**gb_params)

        elif model_type == "SVM":
            if task_type == "classification":
                from sklearn.svm import SVC

                return SVC(**clean_params)
            else:
                from sklearn.svm import SVR

                return SVR(**clean_params)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _evaluate_model(
        self, model, X_train, X_test, y_train, y_test, task_type: str
    ) -> Dict[str, float]:
        """Evaluate trained model"""

        try:
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if task_type == "classification":
                return {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "accuracy": accuracy_score(y_test, y_test_pred),
                    "precision": precision_score(
                        y_test, y_test_pred, average="weighted", zero_division=0
                    ),
                    "recall": recall_score(
                        y_test, y_test_pred, average="weighted", zero_division=0
                    ),
                    "f1_score": f1_score(
                        y_test, y_test_pred, average="weighted", zero_division=0
                    ),
                }
            else:
                return {
                    "train_mse": mean_squared_error(y_train, y_train_pred),
                    "test_mse": mean_squared_error(y_test, y_test_pred),
                    "mse": mean_squared_error(y_test, y_test_pred),
                    "mae": mean_absolute_error(y_test, y_test_pred),
                    "r2_score": r2_score(y_test, y_test_pred),
                    "train_r2": r2_score(y_train, y_train_pred),
                }
        except Exception as e:
            # Return basic metrics if detailed evaluation fails
            return {"error": str(e), "basic_score": 0.0}

    def _create_target_from_data(self, features_df, feature_key, session_state):
        """Create target variable from available data sources"""

        try:
            # Method 1: Try to get original price data from data_cache
            target = self._create_target_from_price_data(
                feature_key, session_state, features_df.index
            )
            if target is not None:
                print("Debug: Created target from original price data")
                return target

            # Method 2: Create target from technical indicators
            target = self._create_target_from_indicators(features_df)
            if target is not None:
                print(
                    f"Debug: Created target from technical indicators, shape: {target.shape}"
                )
                return target

            # Method 3: Create target from momentum/trend indicators
            target = self._create_target_from_momentum(features_df)
            if target is not None:
                print(
                    f"Debug: Created target from momentum indicators, shape: {target.shape}"
                )
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
                if symbol.upper() in data_key.upper():
                    if "data" in data_info:
                        price_data = data_info["data"]

                        # Check for close price column
                        close_col = None
                        for col in ["Close", "close", "CLOSE"]:
                            if col in price_data.columns:
                                close_col = col
                                break

                        if close_col is not None:
                            # Align with feature data index
                            common_idx = price_data.index.intersection(feature_index)
                            if len(common_idx) > 10:  # Need sufficient data
                                close_prices = price_data.loc[common_idx, close_col]
                                returns = close_prices.pct_change().dropna()
                                target = (returns > 0).astype(int)
                                target.name = "price_direction"
                                return target
            return None

        except Exception as e:
            print(f"Debug: Error in _create_target_from_price_data: {e}")
            return None

    def _create_target_from_indicators(self, features_df):
        """Create target from technical indicators like RSI or MACD"""

        try:
            # Strategy 1: RSI-based target (RSI > 50 indicates bullish)
            if "RSI" in features_df.columns:
                rsi = features_df["RSI"].dropna()
                if len(rsi) > 10:
                    target = (rsi > 50).astype(int)
                    target.name = "rsi_direction"
                    return target

            # Strategy 2: MACD-based target
            if (
                "MACD_MACD" in features_df.columns
                and "MACD_Signal" in features_df.columns
            ):
                macd = features_df["MACD_MACD"].dropna()
                signal = features_df["MACD_Signal"].dropna()
                common_idx = macd.index.intersection(signal.index)
                if len(common_idx) > 10:
                    macd_aligned = macd.loc[common_idx]
                    signal_aligned = signal.loc[common_idx]
                    target = (macd_aligned > signal_aligned).astype(int)
                    target.name = "macd_signal"
                    return target

            # Strategy 3: Moving Average Cross
            if "SMA_20" in features_df.columns and "SMA_50" in features_df.columns:
                sma20 = features_df["SMA_20"].dropna()
                sma50 = features_df["SMA_50"].dropna()
                common_idx = sma20.index.intersection(sma50.index)
                if len(common_idx) > 10:
                    sma20_aligned = sma20.loc[common_idx]
                    sma50_aligned = sma50.loc[common_idx]
                    target = (sma20_aligned > sma50_aligned).astype(int)
                    target.name = "sma_cross"
                    return target

            return None

        except Exception as e:
            print(f"Debug: Error in _create_target_from_indicators: {e}")
            return None

    def _create_target_from_momentum(self, features_df):
        """Create target from momentum indicators"""

        try:
            # Strategy 1: Momentum-based target
            if "MOMENTUM" in features_df.columns:
                momentum = features_df["MOMENTUM"].dropna()
                if len(momentum) > 10:
                    target = (momentum > 0).astype(int)
                    target.name = "momentum_direction"
                    return target

            # Strategy 2: Stochastic-based target
            if "Stoch_%K" in features_df.columns:
                stoch_k = features_df["Stoch_%K"].dropna()
                if len(stoch_k) > 10:
                    target = (stoch_k > 50).astype(int)
                    target.name = "stoch_direction"
                    return target

            # Strategy 3: ATR-based volatility regime
            if "ATR" in features_df.columns:
                atr = features_df["ATR"].dropna()
                if len(atr) > 10:
                    atr_median = atr.median()
                    target = (atr > atr_median).astype(int)
                    target.name = "volatility_regime"
                    return target

            return None

        except Exception as e:
            print(f"Debug: Error in _create_target_from_momentum: {e}")
            return None
