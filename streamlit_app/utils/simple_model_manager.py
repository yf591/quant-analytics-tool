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
            # Get feature data
            if feature_key not in session_state.feature_cache:
                return False, f"Feature dataset '{feature_key}' not found", None

            dataset_info = session_state.feature_cache[feature_key]

            # Extract features and target
            if "features" not in dataset_info:
                return False, "Dataset missing features data", None

            features_df = dataset_info["features"]

            # Create a simple target if not provided
            if "target" in dataset_info:
                target = dataset_info["target"]
            else:
                # Create a simple target based on returns
                if "returns" in features_df.columns:
                    target = (features_df["returns"] > 0).astype(int)
                    target.name = "direction"
                elif "close" in features_df.columns:
                    returns = features_df["close"].pct_change()
                    target = (returns > 0).astype(int)
                    target.name = "direction"
                else:
                    return False, "No suitable target variable found", None

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

            return True, "Model trained successfully", model_id

        except Exception as e:
            return False, f"Training failed: {str(e)}", None

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
