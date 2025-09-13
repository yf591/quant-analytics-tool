"""
Model Training Utilities

This module provides utility functions for model training workflows,
separated from UI components for better testability and maintainability.

Design Principles:
- Separation of Concerns: UI logic vs Business logic
- Testability: Pure Python functions without Streamlit dependencies
- Reusability: Functions can be used across different pages
- Maintainability: Easy to modify business logic without touching UI
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
import inspect
from pathlib import Path
import sys

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Week 7: Traditional ML Models
    from src.models import (
        ModelFactory,
        ModelEvaluator,
        QuantRandomForestClassifier,
        QuantRandomForestRegressor,
        QuantXGBoostClassifier,
        QuantXGBoostRegressor,
    )
    from src.models.traditional.svm_model import (
        QuantSVMClassifier,
        QuantSVMRegressor,
    )

    # Week 8: Deep Learning Models
    from src.models.deep_learning import (
        QuantLSTMClassifier,
        QuantLSTMRegressor,
        QuantGRUClassifier,
        QuantGRURegressor,
    )

    # Week 9: Advanced Models
    from src.models.advanced.ensemble import (
        FinancialRandomForest,
        StackingEnsemble,
        VotingEnsemble,
        TimeSeriesBagging,
    )
    from src.models.advanced.transformer import (
        TransformerClassifier,
        TransformerRegressor,
    )
    from src.models.advanced.attention import (
        MultiHeadAttention,
        TemporalAttention,
    )
    from src.models.advanced.meta_labeling import (
        MetaLabelingModel,
    )

    # Week 10: Model Pipeline
    from src.models.pipeline.training_pipeline import (
        ModelTrainingPipeline,
        ModelTrainingConfig,
    )
    from src.models.pipeline.model_registry import ModelRegistry
    from src.config import settings

except ImportError as e:
    # Handle import errors gracefully for testing
    print(f"Import warning in model_utils: {e}")

    # Define fallback classes to prevent runtime errors
    class DummyModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    # Assign dummy classes for missing imports
    StackingEnsemble = DummyModel
    VotingEnsemble = DummyModel
    TimeSeriesBagging = DummyModel
    MetaLabelingModel = DummyModel
    MultiHeadAttention = DummyModel
    TemporalAttention = DummyModel


class ModelTrainingManager:
    """Manager class for model training operations"""

    def __init__(self):
        self.model_factory = ModelFactory()
        self.model_evaluator = ModelEvaluator()
        self.model_registry = ModelRegistry()
        self.training_pipeline = ModelTrainingPipeline()

    def initialize_session_state(self, session_state: Dict) -> None:
        """Initialize session state for model training"""

        if "model_cache" not in session_state:
            session_state["model_cache"] = {}

        if "training_history" not in session_state:
            session_state["training_history"] = []

        if "model_registry" not in session_state:
            session_state["model_registry"] = ModelRegistry()

    def train_model(
        self,
        feature_key: str,
        model_config: Dict,
        hyperparams: Dict,
        training_config: Dict,
        session_state: Dict,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Train a model with the specified configuration

        Args:
            feature_key: Key for feature data in session state
            model_config: Model configuration
            hyperparams: Hyperparameters
            training_config: Training configuration
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message, model_id)
        """

        try:
            # Get and validate feature data
            feature_data = self._get_feature_data(feature_key, session_state)
            if feature_data is None:
                return False, "Feature data not found or invalid", None

            # Prepare data for training
            X, y = self._prepare_training_data(feature_data, training_config)
            if X is None or y is None:
                return False, "Failed to prepare training data", None

            # Create model instance
            model = self._get_model_instance(
                model_config["model_class"], model_config["task_type"], hyperparams
            )
            if model is None:
                return False, "Failed to create model instance", None

            # Split data
            train_test_split = self._split_data(X, y, training_config)
            if train_test_split is None:
                return False, "Failed to split data", None

            X_train, X_test, y_train, y_test = train_test_split

            # Train model
            training_start = datetime.now()

            # Update training status
            model_id = f"{model_config['model_class']}_{training_start.strftime('%Y%m%d_%H%M%S')}"

            session_state["current_training"] = {
                "status": "training",
                "model_id": model_id,
                "start_time": training_start,
                "model_config": model_config,
                "hyperparams": hyperparams,
                "training_config": training_config,
                "progress": 0.3,
            }

            # Fit model
            model.fit(X_train, y_train)

            # Evaluate model
            session_state["current_training"]["progress"] = 0.7
            session_state["current_training"]["status"] = "evaluating"

            evaluation_results = self._evaluate_model(
                model, X_train, X_test, y_train, y_test, model_config["task_type"]
            )

            # Calculate feature importance if available
            feature_importance = self._calculate_feature_importance(model, X.columns)

            # Store model results
            training_duration = datetime.now() - training_start

            model_info = {
                "model": model,
                "model_id": model_id,
                "model_class": model_config["model_class"],
                "task_type": model_config["task_type"],
                "hyperparameters": hyperparams,
                "training_config": training_config,
                "feature_key": feature_key,
                "evaluation": evaluation_results,
                "feature_importance": feature_importance,
                "training_duration": training_duration.total_seconds(),
                "trained_at": training_start,
                "data_shape": X.shape,
            }

            # Store in cache
            session_state["model_cache"][model_id] = model_info

            # Add to training history
            session_state["training_history"].append(
                {
                    "model_id": model_id,
                    "model_class": model_config["model_class"],
                    "task_type": model_config["task_type"],
                    "training_score": evaluation_results.get("train_score", 0),
                    "test_score": evaluation_results.get("test_score", 0),
                    "trained_at": training_start,
                    "duration": training_duration.total_seconds(),
                }
            )

            # Register model
            try:
                self.model_registry.register_model(
                    model=model, model_id=model_id, metadata=model_info
                )
            except Exception as e:
                print(f"Warning: Failed to register model: {e}")

            # Update final status
            session_state["current_training"]["status"] = "completed"
            session_state["current_training"]["progress"] = 1.0

            return (
                True,
                f"Model trained successfully! Test score: {evaluation_results.get('test_score', 0):.4f}",
                model_id,
            )

        except Exception as e:
            if "current_training" in session_state:
                session_state["current_training"]["status"] = "failed"
                session_state["current_training"]["error"] = str(e)

            return False, f"Training failed: {str(e)}", None

    def _get_feature_data(
        self, feature_key: str, session_state: Dict
    ) -> Optional[pd.DataFrame]:
        """Get feature data from session state"""

        try:
            if feature_key not in session_state.get("feature_cache", {}):
                return None

            feature_data = session_state["feature_cache"][feature_key]

            if isinstance(feature_data, pd.DataFrame):
                return feature_data
            elif isinstance(feature_data, dict):
                # Handle old format
                if "features" in feature_data:
                    features = feature_data["features"]
                    if isinstance(features, pd.DataFrame):
                        return features
                elif "data" in feature_data:
                    data = feature_data["data"]
                    if isinstance(data, pd.DataFrame):
                        return data

            return None

        except Exception:
            return None

    def _prepare_training_data(
        self, feature_data: pd.DataFrame, training_config: Dict
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for training"""

        try:
            # For financial data, create target variable
            # This is a simplified approach - in practice, you'd want more sophisticated target creation

            if "Close" in feature_data.columns:
                # Create price return targets
                price_col = "Close"
            elif "close" in feature_data.columns:
                price_col = "close"
            else:
                # Use first numeric column as price
                numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return None, None
                price_col = numeric_cols[0]

            # Create target variable (next period return)
            returns = feature_data[price_col].pct_change().shift(-1)

            # For classification: positive/negative returns
            # For regression: actual returns
            task_type = training_config.get("task_type", "classification")

            if task_type == "classification":
                target = (returns > 0).astype(int)
            else:
                target = returns

            # Features: exclude the price column and any other price-like columns
            price_like_cols = [
                col
                for col in feature_data.columns
                if any(
                    price_word in col.lower()
                    for price_word in ["close", "open", "high", "low", "price"]
                )
            ]

            feature_cols = [
                col for col in feature_data.columns if col not in price_like_cols
            ]

            if len(feature_cols) == 0:
                # If no features left, use all numeric columns except target
                feature_cols = feature_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                if price_col in feature_cols:
                    feature_cols.remove(price_col)

            X = feature_data[feature_cols]
            y = target

            # Remove NaN values
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]

            if len(X) < 100:  # Minimum data requirement
                return None, None

            return X, y

        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, training_config: Dict
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Split data into train/test sets"""

        try:
            from sklearn.model_selection import train_test_split

            test_size = training_config.get("test_size", 0.2)
            random_state = training_config.get("random_state", 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=(
                    y if training_config.get("task_type") == "classification" else None
                ),
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            print(f"Error splitting data: {e}")
            return None

    def _get_model_instance(
        self, model_class: str, task_type: str, hyperparams: Dict
    ) -> Optional[Any]:
        """Get an instance of the specified model class - Complete Phase 3 Week 7-10 Support"""

        try:
            # Complete model class mapping for Phase 3 Week 7-10
            model_classes = {
                # Week 7: Traditional ML Models
                "QuantRandomForestClassifier": QuantRandomForestClassifier,
                "QuantRandomForestRegressor": QuantRandomForestRegressor,
                "QuantXGBoostClassifier": QuantXGBoostClassifier,
                "QuantXGBoostRegressor": QuantXGBoostRegressor,
                "QuantSVMClassifier": QuantSVMClassifier,
                "QuantSVMRegressor": QuantSVMRegressor,
                # Week 8: Deep Learning Models
                "QuantLSTMClassifier": QuantLSTMClassifier,
                "QuantLSTMRegressor": QuantLSTMRegressor,
                "QuantGRUClassifier": QuantGRUClassifier,
                "QuantGRURegressor": QuantGRURegressor,
                # Week 9: Advanced Models
                "TransformerClassifier": TransformerClassifier,
                "TransformerRegressor": TransformerRegressor,
                "FinancialRandomForest": FinancialRandomForest,
                "StackingEnsemble": StackingEnsemble,
                "VotingEnsemble": VotingEnsemble,
                "TimeSeriesBagging": TimeSeriesBagging,
                "MetaLabelingModel": MetaLabelingModel,
                # Attention Models (will be used as components in other models)
                "MultiHeadAttention": MultiHeadAttention,
                "TemporalAttention": TemporalAttention,
            }

            # Special handling for complex models
            if model_class in [
                "FinancialRandomForest",
                "StackingEnsemble",
                "VotingEnsemble",
                "TimeSeriesBagging",
            ]:
                # These models handle both classification and regression internally
                ModelClass = model_classes.get(model_class)
                if ModelClass is None:
                    print(f"Model class {model_class} not found")
                    return None

                # For ensemble models, use task_type as parameter
                if "task_type" not in hyperparams and model_class in [
                    "StackingEnsemble",
                    "VotingEnsemble",
                ]:
                    hyperparams["task_type"] = task_type.lower()

            elif model_class == "MetaLabelingModel":
                # Meta-labeling requires special initialization
                from src.models.advanced.meta_labeling import MetaLabelingConfig

                config = MetaLabelingConfig()
                # Will implement proper initialization later
                return None  # For now, skip meta-labeling

            elif "Attention" in model_class:
                # Attention models are layers, not standalone models
                ModelClass = model_classes.get(model_class)
                if ModelClass is None:
                    print(f"Attention model {model_class} not found")
                    return None

                # Return attention layer directly for now
                return ModelClass(**hyperparams)

            else:
                # Standard classifier/regressor models
                ModelClass = model_classes.get(model_class)
                if ModelClass is None:
                    print(f"Model class {model_class} not found")
                    return None

            # Filter hyperparameters to only include valid ones for the model
            try:
                import inspect

                model_signature = inspect.signature(ModelClass.__init__)
                valid_params = list(model_signature.parameters.keys())
                valid_params.remove("self")  # Remove 'self' parameter

                filtered_hyperparams = {
                    k: v for k, v in hyperparams.items() if k in valid_params
                }

                # Create model instance
                return ModelClass(**filtered_hyperparams)

            except Exception as e:
                print(f"Error filtering hyperparameters for {model_class}: {e}")
                # Fallback: try with basic hyperparameters
                try:
                    basic_params = {}
                    if "random_state" in hyperparams:
                        basic_params["random_state"] = hyperparams["random_state"]
                    return ModelClass(**basic_params)
                except Exception as e2:
                    print(f"Fallback creation failed for {model_class}: {e2}")
                    return ModelClass()  # Default initialization

        except Exception as e:
            print(f"Error creating model instance for {model_class}: {e}")
            return None

    def _evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        task_type: str,
    ) -> Dict[str, float]:
        """Evaluate model performance"""

        try:
            from sklearn.metrics import (
                accuracy_score,
                r2_score,
                mean_squared_error,
                classification_report,
            )

            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            results = {}

            if task_type == "classification":
                results["train_score"] = accuracy_score(y_train, train_pred)
                results["test_score"] = accuracy_score(y_test, test_pred)
                results["metric_type"] = "accuracy"
            else:
                results["train_score"] = r2_score(y_train, train_pred)
                results["test_score"] = r2_score(y_test, test_pred)
                results["mse"] = mean_squared_error(y_test, test_pred)
                results["metric_type"] = "r2_score"

            return results

        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {"train_score": 0, "test_score": 0, "metric_type": "unknown"}

    def _calculate_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Optional[pd.Series]:
        """Calculate feature importance if available"""

        try:
            # Check if model has feature_importances_ attribute
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                return pd.Series(importance, index=feature_names)

            # Check if model has coef_ attribute (linear models)
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multi-class classification
                    importance = np.abs(coef).mean(axis=0)
                else:
                    importance = np.abs(coef)
                return pd.Series(importance, index=feature_names)

            return None

        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return None

    def get_model_comparison_data(self, session_state: Dict) -> List[Dict]:
        """Get data for model comparison"""

        comparison_data = []

        for model_id, model_info in session_state.get("model_cache", {}).items():
            if "evaluation" in model_info:
                evaluation = model_info["evaluation"]

                comparison_data.append(
                    {
                        "Model ID": model_id,
                        "Model Type": model_info.get("model_class", "Unknown"),
                        "Task Type": model_info.get("task_type", "Unknown"),
                        "Train Score": evaluation.get("train_score", 0),
                        "Test Score": evaluation.get("test_score", 0),
                        "Metric": evaluation.get("metric_type", "Unknown"),
                        "Training Duration (s)": model_info.get("training_duration", 0),
                        "Data Shape": str(model_info.get("data_shape", "Unknown")),
                        "Trained At": model_info.get(
                            "trained_at", datetime.now()
                        ).strftime("%Y-%m-%d %H:%M"),
                    }
                )

        return comparison_data

    def get_training_history_summary(self, session_state: Dict) -> Dict[str, Any]:
        """Get training history summary"""

        history = session_state.get("training_history", [])

        if not history:
            return {
                "total_models": 0,
                "avg_test_score": 0,
                "best_model": None,
                "recent_models": 0,
            }

        # Calculate statistics
        test_scores = [h.get("test_score", 0) for h in history]
        avg_test_score = np.mean(test_scores)
        best_model_idx = np.argmax(test_scores)
        best_model = history[best_model_idx]

        # Recent models (last 24 hours)
        recent_cutoff = datetime.now() - pd.Timedelta(hours=24)
        recent_models = sum(
            1 for h in history if h.get("trained_at", datetime.min) > recent_cutoff
        )

        return {
            "total_models": len(history),
            "avg_test_score": avg_test_score,
            "best_model": best_model,
            "recent_models": recent_models,
        }

    def export_model(self, model_id: str, session_state: Dict) -> Optional[Dict]:
        """Export model data"""

        if model_id not in session_state.get("model_cache", {}):
            return None

        model_info = session_state["model_cache"][model_id]

        # Create exportable version (exclude non-serializable objects)
        export_data = {
            "model_id": model_id,
            "model_class": model_info.get("model_class"),
            "task_type": model_info.get("task_type"),
            "hyperparameters": model_info.get("hyperparameters", {}),
            "training_config": model_info.get("training_config", {}),
            "evaluation": model_info.get("evaluation", {}),
            "feature_importance": (
                model_info.get("feature_importance").to_dict()
                if model_info.get("feature_importance") is not None
                else None
            ),
            "training_duration": model_info.get("training_duration"),
            "trained_at": (
                model_info.get("trained_at").isoformat()
                if model_info.get("trained_at")
                else None
            ),
            "data_shape": model_info.get("data_shape"),
        }

        return export_data

    def cleanup_training_session(self, session_state: Dict) -> None:
        """Clean up current training session"""

        if "current_training" in session_state:
            del session_state["current_training"]

    def get_default_hyperparams(self, model_type: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for a given model type.
        Complete implementation for Phase 3 Week 7-10 models.

        Args:
            model_type: Model type (e.g., "random_forest", "xgboost", etc.)

        Returns:
            Dictionary of default hyperparameters
        """

        hyperparams_map = {
            # Week 7: Traditional ML Models
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "random_state": 42,
            },
            "svm": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "degree": 3,
                "coef0": 0.0,
                "probability": True,
                "random_state": 42,
            },
            # Week 8: Deep Learning Models
            "lstm": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "bidirectional": False,
                "sequence_length": 60,
            },
            "gru": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "bidirectional": False,
                "sequence_length": 60,
            },
            "transformer": {
                "d_model": 64,
                "nhead": 8,
                "num_layers": 4,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "sequence_length": 60,
            },
            # Week 9: Advanced Models
            "financial_random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "bootstrap": True,
                "sample_weights": True,
                "feature_importance_method": "shap",
                "random_state": 42,
            },
            "stacking_ensemble": {
                "base_models": ["random_forest", "xgboost", "svm"],
                "meta_model": "logistic_regression",
                "cv_folds": 5,
                "use_probabilities": True,
                "random_state": 42,
            },
            "voting_ensemble": {
                "models": ["random_forest", "xgboost", "svm"],
                "voting": "soft",
                "weights": None,
                "n_jobs": -1,
            },
            "time_series_bagging": {
                "base_estimator": "random_forest",
                "n_estimators": 10,
                "max_samples": 1.0,
                "bootstrap": True,
                "bootstrap_features": False,
                "random_state": 42,
            },
            "meta_labeling": {
                "primary_model": "random_forest",
                "meta_model": "logistic_regression",
                "profit_target": 0.02,
                "stop_loss": 0.01,
                "time_horizon": 5,
                "use_sample_weights": True,
            },
            # Week 9: Attention Models
            "multi-head_attention": {
                "d_model": 64,
                "num_heads": 8,
                "dropout_rate": 0.1,
                "use_bias": True,
            },
            "temporal_attention": {
                "units": 64,
                "time_steps": 60,
                "return_sequences": True,
                "dropout_rate": 0.1,
            },
            # Ensemble fallback
            "ensemble": {"n_estimators": 100, "random_state": 42},
        }

        return hyperparams_map.get(model_type, {})
