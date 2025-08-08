"""
XGBoost Models for Financial Analysis

This module implements XGBoost classifiers and regressors optimized for financial
time series prediction with advanced features compatible with XGBoost 3.0+.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import time
import logging

from ..base import BaseClassifier, BaseRegressor, register_model


@register_model("xgboost_classifier")
class QuantXGBoostClassifier(BaseClassifier):
    """
    XGBoost Classifier optimized for financial applications.

    Implements gradient boosting with financial-specific optimizations
    and compatibility with XGBoost 3.0+.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        objective: str = "binary:logistic",
        eval_metric: str = "logloss",
        random_state: int = 42,
        n_jobs: int = -1,
        verbosity: int = 0,
        **kwargs,
    ):
        """
        Initialize XGBoost Classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            colsample_bylevel: Subsample ratio of columns for each level
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            min_child_weight: Minimum sum of instance weight needed in a child
            gamma: Minimum loss reduction required to make split
            objective: Learning objective
            eval_metric: Evaluation metric
            random_state: Random state for reproducibility
            n_jobs: Number of parallel threads
            verbosity: Verbosity level
        """
        super().__init__(model_name="QuantXGBoostClassifier", **kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

        self.model = self._build_model()
        self.eval_results = {}

    def _build_model(self) -> xgb.XGBClassifier:
        """Build the XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        verbose: bool = False,
        **kwargs,
    ) -> "QuantXGBoostClassifier":
        """Train the XGBoost classifier."""
        start_time = time.time()

        X_array, y_array = self._validate_input(X, y)

        # Fit the model
        self.model.fit(X_array, y_array)
        self.classes_ = self.model.classes_
        self.is_fitted = True

        # Record training time
        training_time = time.time() - start_time
        self.metadata["training_time"] = training_time

        # Record training history
        self.training_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "training_samples": len(X_array),
                "training_time": training_time,
                "n_features": X_array.shape[1],
                "n_classes": len(self.classes_),
            }
        )

        self.logger.info(
            f"XGBoost trained in {training_time:.2f}s with {len(X_array)} samples"
        )

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)
        return self.model.predict_proba(X_array)

    def get_feature_importance(self, importance_type: str = "weight") -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_dict = self.model.get_booster().get_score(
            importance_type=importance_type
        )

        if self.feature_names:
            importance = pd.Series(
                [
                    importance_dict.get(f"f{i}", 0)
                    for i in range(len(self.feature_names))
                ],
                index=self.feature_names,
                name=f"importance_{importance_type}",
            )
        else:
            importance = pd.Series(
                importance_dict, name=f"importance_{importance_type}"
            )

        return importance.sort_values(ascending=False)


@register_model("xgboost_regressor")
class QuantXGBoostRegressor(BaseRegressor):
    """
    XGBoost Regressor optimized for financial applications.

    Implements gradient boosting for regression tasks with financial-specific
    optimizations and compatibility with XGBoost 3.0+.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        objective: str = "reg:squarederror",
        eval_metric: str = "rmse",
        random_state: int = 42,
        n_jobs: int = -1,
        verbosity: int = 0,
        **kwargs,
    ):
        """Initialize XGBoost Regressor."""
        super().__init__(model_name="QuantXGBoostRegressor", **kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

        self.model = self._build_model()
        self.eval_results = {}

    def _build_model(self) -> xgb.XGBRegressor:
        """Build the XGBoost model."""
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        verbose: bool = False,
        **kwargs,
    ) -> "QuantXGBoostRegressor":
        """Train the XGBoost regressor."""
        start_time = time.time()

        X_array, y_array = self._validate_input(X, y)

        # Fit the model
        self.model.fit(X_array, y_array)
        self.is_fitted = True

        # Record training time
        training_time = time.time() - start_time
        self.metadata["training_time"] = training_time

        # Record training history
        self.training_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "training_samples": len(X_array),
                "training_time": training_time,
                "n_features": X_array.shape[1],
            }
        )

        self.logger.info(
            f"XGBoost regressor trained in {training_time:.2f}s with {len(X_array)} samples"
        )

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)
        return self.model.predict(X_array)

    def get_feature_importance(self, importance_type: str = "weight") -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_dict = self.model.get_booster().get_score(
            importance_type=importance_type
        )

        if self.feature_names:
            importance = pd.Series(
                [
                    importance_dict.get(f"f{i}", 0)
                    for i in range(len(self.feature_names))
                ],
                index=self.feature_names,
                name=f"importance_{importance_type}",
            )
        else:
            importance = pd.Series(
                importance_dict, name=f"importance_{importance_type}"
            )

        return importance.sort_values(ascending=False)
