"""
Automated Model Training Pipeline

This module provides comprehensive automated training pipeline for financial ML models.
Based on AFML methodologies for systematic model development and evaluation.
"""

import os
import pickle
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ...features.pipeline import FeaturePipeline
from ..traditional.random_forest import (
    QuantRandomForestClassifier,
    QuantRandomForestRegressor,
)
from ..traditional.xgboost_model import QuantXGBoostClassifier, QuantXGBoostRegressor
from ..traditional.svm_model import QuantSVMClassifier, QuantSVMRegressor
from ..deep_learning.lstm import QuantLSTMClassifier, QuantLSTMRegressor
from ..deep_learning.gru import QuantGRUClassifier, QuantGRURegressor
from ..advanced.transformer import TransformerClassifier, TransformerRegressor
from ..advanced.ensemble import StackingEnsemble, VotingEnsemble
from ..base import BaseFinancialModel


class ModelTrainingConfig:
    """Configuration for model training pipeline"""

    def __init__(
        self,
        models_to_train: List[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        time_series_cv: bool = True,
        cv_splits: int = 5,
        scaler_type: str = "standard",
        feature_selection: bool = True,
        hyperparameter_tuning: bool = True,
        ensemble_models: bool = True,
        save_models: bool = True,
        model_save_dir: str = "models/trained",
        random_state: int = 42,
    ):
        self.models_to_train = models_to_train or [
            "random_forest",
            "xgboost",
            "svm",
            "lstm",
            "gru",
            "transformer",
        ]
        self.test_size = test_size
        self.validation_size = validation_size
        self.time_series_cv = time_series_cv
        self.cv_splits = cv_splits
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.hyperparameter_tuning = hyperparameter_tuning
        self.ensemble_models = ensemble_models
        self.save_models = save_models
        self.model_save_dir = model_save_dir
        self.random_state = random_state


class ModelTrainingPipeline:
    """
    Automated model training pipeline for financial ML models.

    Features:
    - Multi-model training and comparison
    - Time-series aware cross-validation
    - Automated hyperparameter tuning
    - Model persistence and versioning
    - Ensemble model creation
    - Performance evaluation and reporting
    """

    def __init__(self, config: ModelTrainingConfig = None):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration
        """
        self.config = config or ModelTrainingConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.feature_pipeline = FeaturePipeline()
        self.scaler = self._get_scaler()
        self.trained_models = {}
        self.model_performance = {}
        self.ensemble_models = {}

        # Ensure model save directory exists
        os.makedirs(self.config.model_save_dir, exist_ok=True)

    def _get_scaler(self):
        """Get data scaler based on configuration"""
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        return scalers.get(self.config.scaler_type, StandardScaler())

    def _get_model_instance(
        self, model_name: str, task_type: str = "classification"
    ) -> BaseFinancialModel:
        """
        Get model instance by name and task type.

        Args:
            model_name: Name of the model
            task_type: "classification" or "regression"

        Returns:
            Model instance
        """
        model_map = {
            "classification": {
                "random_forest": QuantRandomForestClassifier,
                "xgboost": QuantXGBoostClassifier,
                "svm": QuantSVMClassifier,
                "lstm": QuantLSTMClassifier,
                "gru": QuantGRUClassifier,
                "transformer": TransformerClassifier,
            },
            "regression": {
                "random_forest": QuantRandomForestRegressor,
                "xgboost": QuantXGBoostRegressor,
                "svm": QuantSVMRegressor,
                "lstm": QuantLSTMRegressor,
                "gru": QuantGRURegressor,
                "transformer": TransformerRegressor,
            },
        }

        if task_type not in model_map:
            raise ValueError(f"Unknown task type: {task_type}")

        if model_name not in model_map[task_type]:
            raise ValueError(f"Unknown model: {model_name} for task: {task_type}")

        return model_map[task_type][model_name]()

    def prepare_data(
        self, data: pd.DataFrame, target_column: str, feature_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for training.

        Args:
            data: Input data
            target_column: Target variable column
            feature_columns: Feature columns (if None, auto-generate)

        Returns:
            Tuple of (features, target, feature_names)
        """
        self.logger.info("Preparing data for training...")

        # Generate features if not provided
        if feature_columns is None:
            self.logger.info("Generating features using feature pipeline...")
            features_df = self.feature_pipeline.generate_features(data)

            # Remove target column from features if present
            if target_column in features_df.columns:
                features_df = features_df.drop(columns=[target_column])

            feature_columns = features_df.columns.tolist()
        else:
            features_df = data[feature_columns].copy()

        # Extract target
        target = data[target_column].copy()

        # Remove rows with missing target values
        valid_indices = target.notna()
        features_df = features_df[valid_indices]
        target = target[valid_indices]

        # Handle missing feature values
        features_df = features_df.fillna(method="ffill").fillna(method="bfill")

        self.logger.info(
            f"Data prepared: {len(features_df)} samples, {len(feature_columns)} features"
        )

        return features_df, target, feature_columns

    def split_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split data into train/validation/test sets.

        Args:
            features: Feature data
            target: Target data

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(features)

        # Calculate split indices for time series data
        test_start = int(n_samples * (1 - self.config.test_size))
        val_start = int(test_start * (1 - self.config.validation_size))

        # Split data
        X_train = features.iloc[:val_start]
        X_val = features.iloc[val_start:test_start]
        X_test = features.iloc[test_start:]

        y_train = target.iloc[:val_start]
        y_val = target.iloc[val_start:test_start]
        y_test = target.iloc[test_start:]

        self.logger.info(
            f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using fitted scaler.

        Args:
            X_train, X_val, X_test: Feature datasets

        Returns:
            Scaled feature datasets
        """
        # Fit scaler on training data
        self.scaler.fit(X_train)

        # Transform all datasets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train), index=X_train.index, columns=X_train.columns
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val), index=X_val.index, columns=X_val.columns
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), index=X_test.index, columns=X_test.columns
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Train a single model and evaluate performance.

        Args:
            model_name: Name of the model
            X_train, X_val, X_test: Feature datasets
            y_train, y_val, y_test: Target datasets
            task_type: "classification" or "regression"

        Returns:
            Dictionary containing model and performance metrics
        """
        self.logger.info(f"Training {model_name} model...")

        try:
            # Get model instance
            model = self._get_model_instance(model_name, task_type)

            # Train model
            if model_name in ["lstm", "gru", "transformer"]:
                # Deep learning models need validation data
                model.fit(X_train, y_train, validation_data=(X_val, y_val))
            else:
                # Traditional ML models
                model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            # Calculate performance metrics
            if task_type == "classification":
                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                )

                performance = {
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    "val_accuracy": accuracy_score(y_val, y_pred_val),
                    "test_accuracy": accuracy_score(y_test, y_pred_test),
                    "test_precision": precision_score(
                        y_test, y_pred_test, average="weighted"
                    ),
                    "test_recall": recall_score(
                        y_test, y_pred_test, average="weighted"
                    ),
                    "test_f1": f1_score(y_test, y_pred_test, average="weighted"),
                }
            else:
                performance = {
                    "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)),
                    "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    "test_r2": r2_score(y_test, y_pred_test),
                }

            # Save model if requested
            if self.config.save_models:
                model_path = self._save_model(model, model_name, task_type)
                performance["model_path"] = model_path

            self.logger.info(f"{model_name} training completed successfully")

            return {
                "model": model,
                "performance": performance,
                "predictions": {
                    "train": y_pred_train,
                    "val": y_pred_val,
                    "test": y_pred_test,
                },
            }

        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}")
            return {
                "model": None,
                "performance": {"error": str(e)},
                "predictions": None,
            }

    def _save_model(
        self, model: BaseFinancialModel, model_name: str, task_type: str
    ) -> str:
        """Save trained model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{task_type}_{timestamp}.pkl"
        filepath = os.path.join(self.config.model_save_dir, filename)

        try:
            if hasattr(model, "save"):
                # Deep learning models with custom save method
                model.save(filepath.replace(".pkl", ""))
            else:
                # Traditional ML models
                joblib.dump(model, filepath)

            self.logger.info(f"Model saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            return ""

    def train_all_models(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        feature_columns: List[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models.

        Args:
            data: Input data
            target_column: Target variable column
            task_type: "classification" or "regression"
            feature_columns: Feature columns

        Returns:
            Dictionary containing all trained models and their performance
        """
        self.logger.info("Starting automated model training pipeline...")

        # Prepare data
        features, target, feature_names = self.prepare_data(
            data, target_column, feature_columns
        )

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            features, target
        )

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )

        # Train models
        results = {}
        for model_name in self.config.models_to_train:
            self.logger.info(f"Training {model_name}...")

            # Use scaled data for traditional ML and SVM, original for deep learning
            if model_name in ["random_forest", "xgboost", "svm"]:
                X_tr, X_v, X_te = X_train_scaled, X_val_scaled, X_test_scaled
            else:
                X_tr, X_v, X_te = X_train, X_val, X_test

            result = self.train_single_model(
                model_name, X_tr, X_v, X_te, y_train, y_val, y_test, task_type
            )

            results[model_name] = result

            if result["model"] is not None:
                self.trained_models[model_name] = result["model"]
                self.model_performance[model_name] = result["performance"]

        # Create ensemble models if requested
        if self.config.ensemble_models and len(self.trained_models) > 1:
            self.logger.info("Creating ensemble models...")
            ensemble_results = self._create_ensemble_models(
                X_train_scaled,
                X_val_scaled,
                X_test_scaled,
                y_train,
                y_val,
                y_test,
                task_type,
            )
            results.update(ensemble_results)

        self.logger.info("Automated model training pipeline completed")

        return results

    def _create_ensemble_models(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        task_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Create ensemble models from trained base models"""
        ensemble_results = {}

        try:
            # Get base models (only traditional ML for ensemble)
            base_models = {
                name: model
                for name, model in self.trained_models.items()
                if name in ["random_forest", "xgboost", "svm"]
            }

            if len(base_models) < 2:
                self.logger.warning("Need at least 2 base models for ensemble")
                return ensemble_results

            # Voting Ensemble
            voting_ensemble = VotingEnsemble(
                models=list(base_models.values()), task_type=task_type
            )
            voting_ensemble.fit(X_train, y_train)

            # Evaluate voting ensemble
            y_pred_test = voting_ensemble.predict(X_test)

            if task_type == "classification":
                from sklearn.metrics import accuracy_score, f1_score

                voting_performance = {
                    "test_accuracy": accuracy_score(y_test, y_pred_test),
                    "test_f1": f1_score(y_test, y_pred_test, average="weighted"),
                }
            else:
                voting_performance = {
                    "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    "test_r2": r2_score(y_test, y_pred_test),
                }

            # Save ensemble model
            if self.config.save_models:
                model_path = self._save_model(
                    voting_ensemble, "voting_ensemble", task_type
                )
                voting_performance["model_path"] = model_path

            ensemble_results["voting_ensemble"] = {
                "model": voting_ensemble,
                "performance": voting_performance,
                "predictions": {"test": y_pred_test},
            }

            self.trained_models["voting_ensemble"] = voting_ensemble
            self.model_performance["voting_ensemble"] = voting_performance

            self.logger.info("Ensemble models created successfully")

        except Exception as e:
            self.logger.error(f"Error creating ensemble models: {str(e)}")

        return ensemble_results

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all trained models.

        Returns:
            DataFrame with model performance comparison
        """
        if not self.model_performance:
            return pd.DataFrame()

        comparison_data = []
        for model_name, performance in self.model_performance.items():
            row = {"model": model_name}
            row.update(performance)
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_model(
        self, metric: str = "test_accuracy"
    ) -> Tuple[str, BaseFinancialModel]:
        """
        Get the best performing model based on specified metric.

        Args:
            metric: Performance metric to optimize

        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.model_performance:
            return None, None

        best_score = (
            float("-inf") if "accuracy" in metric or "r2" in metric else float("inf")
        )
        best_model_name = None

        for model_name, performance in self.model_performance.items():
            if metric in performance and "error" not in performance:
                score = performance[metric]

                if "accuracy" in metric or "r2" in metric:
                    # Higher is better
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                else:
                    # Lower is better (e.g., RMSE)
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name

        if best_model_name:
            return best_model_name, self.trained_models.get(best_model_name)

        return None, None
