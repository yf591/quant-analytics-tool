"""
Base Model Classes

This module defines abstract base classes for all machine learning models
used in the financial analytics system. Ensures consistent interface
and methodology across different model types.

Based on AFML Chapter 6: Ensemble Methods principles.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib
import pickle
from pathlib import Path
import logging


@dataclass
class ModelConfig:
    """Configuration container for models."""

    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    feature_config: Dict[str, Any]


@dataclass
class ModelResults:
    """Container for model training and evaluation results."""

    model: Any
    training_score: float
    validation_score: float
    test_score: Optional[float] = None
    feature_importance: Optional[pd.Series] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None


class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class for all financial ML models.

    Provides standardized interface for training, prediction, evaluation,
    and persistence across different model types.
    """

    def __init__(self, model_name: str = "BaseModel", random_state: int = 42, **kwargs):
        """
        Initialize base model.

        Args:
            model_name: Name identifier for the model
            random_state: Random state for reproducibility
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_history = []
        self.logger = logging.getLogger(f"{__name__}.{model_name}")

        # Model metadata
        self.metadata = {
            "model_type": self.__class__.__name__,
            "creation_time": pd.Timestamp.now(),
            "version": "1.0.0",
        }

    @abstractmethod
    def _build_model(self, **kwargs) -> Any:
        """Build the underlying model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> "BaseModel":
        """Train the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions. Must be implemented by subclasses."""
        pass

    def _validate_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and preprocess input data."""
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = np.array(X)

        if y is not None:
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = np.array(y)

            # Check for matching lengths
            if len(X_array) != len(y_array):
                raise ValueError(
                    f"X and y must have same length: {len(X_array)} vs {len(y_array)}"
                )

            return X_array, y_array

        return X_array, None

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            if self.feature_names:
                return pd.Series(
                    importance, index=self.feature_names, name="importance"
                )
            else:
                return pd.Series(importance, name="importance")
        else:
            self.logger.warning("Model does not support feature importance")
            return None

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        save_data = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "training_history": self.training_history,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(save_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, filepath: str) -> "BaseModel":
        """Load a trained model from disk."""
        try:
            load_data = joblib.load(filepath)

            self.model = load_data["model"]
            self.model_name = load_data["model_name"]
            self.feature_names = load_data["feature_names"]
            self.metadata = load_data["metadata"]
            self.training_history = load_data.get("training_history", [])
            self.is_fitted = True

            self.logger.info(f"Model loaded from {filepath}")
            return self

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters for sklearn compatibility."""
        return {"model_name": self.model_name, "random_state": self.random_state}

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class BaseClassifier(BaseModel, ClassifierMixin):
    """
    Base class for classification models.

    Extends BaseModel with classification-specific functionality
    including probability predictions and classification metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes_ = None

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_array)
        else:
            raise NotImplementedError("Model does not support probability predictions")

    def predict_log_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict log class probabilities."""
        if hasattr(self.model, "predict_log_proba"):
            X_array, _ = self._validate_input(X)
            return self.model.predict_log_proba(X_array)
        else:
            return np.log(self.predict_proba(X))

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute decision function values."""
        if hasattr(self.model, "decision_function"):
            X_array, _ = self._validate_input(X)
            return self.model.decision_function(X_array)
        else:
            raise NotImplementedError("Model does not support decision function")

    def score(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> float:
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        _, y_array = self._validate_input(X, y)
        return accuracy_score(y_array, predictions)


class BaseRegressor(BaseModel, RegressorMixin):
    """
    Base class for regression models.

    Extends BaseModel with regression-specific functionality
    including regression metrics and prediction intervals.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        predictions = self.predict(X)
        _, y_array = self._validate_input(X, y)
        return r2_score(y_array, predictions)

    def predict_std(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict standard deviation of predictions (if supported)."""
        if hasattr(self.model, "predict_std"):
            X_array, _ = self._validate_input(X)
            return self.model.predict_std(X_array)
        else:
            raise NotImplementedError(
                "Model does not support prediction standard deviation"
            )


class ModelFactory:
    """Factory class for creating models based on configuration."""

    _models = {}

    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """Register a new model type."""
        cls._models[model_type] = model_class

    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseModel:
        """Create a model based on configuration."""
        model_type = config.model_type

        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls._models[model_type]
        return model_class(**config.hyperparameters)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available model types."""
        return list(cls._models.keys())


# Decorator for model registration
def register_model(model_type: str):
    """Decorator for automatic model registration."""

    def decorator(model_class):
        ModelFactory.register_model(model_type, model_class)
        return model_class

    return decorator
