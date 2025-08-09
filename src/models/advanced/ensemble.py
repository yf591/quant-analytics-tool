"""
Ensemble Methods for Financial Machine Learning

This module implements advanced ensemble methods specifically designed
for financial time series, based on AFML Chapter 6 principles.

Features:
- Bootstrap Aggregation (Bagging) with financial data considerations
- Random Forest with temporal validation
- Voting classifiers and regressors
- Stacking ensemble with meta-learning
- Sequential ensemble methods
- Time-aware cross-validation for ensemble training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import warnings
from dataclasses import dataclass
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int = 42

    # Ensemble-specific parameters
    voting: str = "soft"  # 'hard' or 'soft'
    meta_model: str = "logistic"  # 'logistic', 'linear', 'tree'
    stack_method: str = "predict_proba"  # 'predict' or 'predict_proba'

    # Time series specific
    time_split_n_splits: int = 5
    purge_length: int = 0  # Number of observations to purge
    embargo_length: int = 0  # Number of observations for embargo


class FinancialRandomForest(BaseEstimator):
    """Random Forest with financial time series considerations."""

    def __init__(self, config: EnsembleConfig, task_type: str = "classification"):
        self.config = config
        self.task_type = task_type
        self.model = None
        self.feature_importance_ = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit Random Forest with optional sample weights.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Optional sample weights for temporal bias correction
        """
        if self.task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                bootstrap=self.config.bootstrap,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                bootstrap=self.config.bootstrap,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )

        self.model.fit(X, y, sample_weight=sample_weight)
        self.feature_importance_ = self.model.feature_importances_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)


class TimeSeriesBagging(BaseEstimator):
    """
    Bagging ensemble with time series awareness.

    Based on AFML Chapter 6 principles for handling temporal structure
    in financial data.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        config: EnsembleConfig,
        task_type: str = "classification",
    ):
        self.base_estimator = base_estimator
        self.config = config
        self.task_type = task_type
        self.estimators_ = []
        self.estimators_features_ = []

    def _create_bootstrap_sample(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create bootstrap sample respecting temporal structure."""
        # Apply purge and embargo if specified
        if self.config.purge_length > 0 or self.config.embargo_length > 0:
            # Remove observations within purge/embargo windows
            valid_indices = self._apply_purge_embargo(indices, len(X))
            bootstrap_indices = np.random.choice(
                valid_indices, size=len(valid_indices), replace=True
            )
        else:
            bootstrap_indices = np.random.choice(
                indices, size=len(indices), replace=True
            )

        return X[bootstrap_indices], y[bootstrap_indices]

    def _apply_purge_embargo(
        self, indices: np.ndarray, total_length: int
    ) -> np.ndarray:
        """Apply purge and embargo periods to avoid data leakage."""
        valid_indices = []

        for idx in indices:
            # Check if index is valid (not in purge/embargo window)
            is_valid = True

            # Apply purge: remove observations too close to validation set
            for other_idx in indices:
                if (
                    idx != other_idx
                    and abs(idx - other_idx) <= self.config.purge_length
                ):
                    is_valid = False
                    break

            # Apply embargo: remove observations immediately after training set
            if idx + self.config.embargo_length >= total_length:
                is_valid = False

            if is_valid:
                valid_indices.append(idx)

        return np.array(valid_indices)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        """Fit bagging ensemble with time series considerations."""
        np.random.seed(self.config.random_state)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        self.estimators_ = []
        self.estimators_features_ = []

        for i in range(self.config.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y, indices)

            # Feature sampling
            n_features = X.shape[1]
            if self.config.max_features == "sqrt":
                max_features = int(np.sqrt(n_features))
            elif self.config.max_features == "log2":
                max_features = int(np.log2(n_features))
            elif isinstance(self.config.max_features, float):
                max_features = int(self.config.max_features * n_features)
            else:
                max_features = n_features

            feature_indices = np.random.choice(
                n_features, size=min(max_features, n_features), replace=False
            )

            # Train estimator
            estimator = self.base_estimator.__class__(
                **self.base_estimator.get_params()
            )
            estimator.fit(X_bootstrap[:, feature_indices], y_bootstrap)

            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_indices)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.estimators_:
            raise ValueError("Ensemble must be fitted before making predictions")

        predictions = []
        for estimator, feature_indices in zip(
            self.estimators_, self.estimators_features_
        ):
            pred = estimator.predict(X[:, feature_indices])
            predictions.append(pred)

        predictions = np.array(predictions)

        if self.task_type == "classification":
            # Majority voting for classification
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
            )
        else:
            # Average for regression
            return np.mean(predictions, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if not self.estimators_:
            raise ValueError("Ensemble must be fitted before making predictions")

        probabilities = []
        for estimator, feature_indices in zip(
            self.estimators_, self.estimators_features_
        ):
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X[:, feature_indices])
                probabilities.append(proba)

        if not probabilities:
            raise ValueError("Base estimators do not support predict_proba")

        return np.mean(probabilities, axis=0)


class StackingEnsemble(BaseEstimator):
    """
    Stacking ensemble with meta-learning.

    Implements stacking as described in AFML Chapter 6, with
    time series cross-validation for meta-model training.
    """

    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        config: EnsembleConfig,
        task_type: str = "classification",
    ):
        self.base_estimators = base_estimators
        self.config = config
        self.task_type = task_type
        self.meta_model = None
        self.fitted_estimators_ = []

    def _create_meta_model(self):
        """Create meta-learning model."""
        if self.task_type == "classification":
            if self.config.meta_model == "logistic":
                return LogisticRegression(random_state=self.config.random_state)
            elif self.config.meta_model == "tree":
                return DecisionTreeClassifier(random_state=self.config.random_state)
        else:
            if self.config.meta_model == "linear":
                return LinearRegression()
            elif self.config.meta_model == "tree":
                return DecisionTreeRegressor(random_state=self.config.random_state)

        raise ValueError(f"Unknown meta_model: {self.config.meta_model}")

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        """Fit stacking ensemble."""
        # Fit base estimators and generate meta-features using time series CV
        tscv = TimeSeriesSplit(
            n_splits=self.config.time_split_n_splits, max_train_size=None
        )

        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for i, estimator in enumerate(self.base_estimators):
                # Fit estimator on training fold
                fitted_estimator = estimator.__class__(**estimator.get_params())
                fitted_estimator.fit(X_train, y_train)

                # Generate predictions for validation fold
                if (
                    self.task_type == "classification"
                    and self.config.stack_method == "predict_proba"
                    and hasattr(fitted_estimator, "predict_proba")
                ):
                    pred = fitted_estimator.predict_proba(X_val)[:, 1]  # Positive class
                else:
                    pred = fitted_estimator.predict(X_val)

                meta_features[val_idx, i] = pred

        # Fit base estimators on full dataset
        self.fitted_estimators_ = []
        for estimator in self.base_estimators:
            fitted_estimator = estimator.__class__(**estimator.get_params())
            fitted_estimator.fit(X, y)
            self.fitted_estimators_.append(fitted_estimator)

        # Fit meta-model
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(meta_features, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make stacking ensemble predictions."""
        if not self.fitted_estimators_ or self.meta_model is None:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Generate meta-features from base estimators
        meta_features = np.zeros((X.shape[0], len(self.fitted_estimators_)))

        for i, estimator in enumerate(self.fitted_estimators_):
            if (
                self.task_type == "classification"
                and self.config.stack_method == "predict_proba"
                and hasattr(estimator, "predict_proba")
            ):
                meta_features[:, i] = estimator.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = estimator.predict(X)

        # Meta-model prediction
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get stacking ensemble prediction probabilities (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if not self.fitted_estimators_ or self.meta_model is None:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Generate meta-features from base estimators
        meta_features = np.zeros((X.shape[0], len(self.fitted_estimators_)))

        for i, estimator in enumerate(self.fitted_estimators_):
            if hasattr(estimator, "predict_proba"):
                meta_features[:, i] = estimator.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = estimator.predict(X)

        # Meta-model prediction probabilities
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Convert predictions to probabilities if meta-model doesn't support predict_proba
            pred = self.meta_model.predict(meta_features)
            proba = np.zeros((len(pred), 2))
            proba[:, 1] = pred
            proba[:, 0] = 1 - pred
            return proba


class VotingEnsemble(BaseEstimator):
    """Enhanced voting ensemble with financial considerations."""

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        config: EnsembleConfig,
        task_type: str = "classification",
    ):
        self.estimators = estimators
        self.config = config
        self.task_type = task_type
        self.ensemble_model = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        """Fit voting ensemble."""
        if self.task_type == "classification":
            self.ensemble_model = VotingClassifier(
                estimators=self.estimators,
                voting=self.config.voting,
                n_jobs=self.config.n_jobs,
            )
        else:
            self.ensemble_model = VotingRegressor(
                estimators=self.estimators, n_jobs=self.config.n_jobs
            )

        self.ensemble_model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make voting ensemble predictions."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble must be fitted before making predictions")
        return self.ensemble_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get voting ensemble prediction probabilities (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        if self.ensemble_model is None:
            raise ValueError("Ensemble must be fitted before making predictions")
        return self.ensemble_model.predict_proba(X)


def evaluate_ensemble_performance(
    ensemble: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    task_type: str = "classification",
) -> Dict[str, float]:
    """
    Evaluate ensemble performance using time series cross-validation.

    Args:
        ensemble: Fitted ensemble model
        X: Features
        y: Targets
        cv_splits: Number of CV splits
        task_type: 'classification' or 'regression'

    Returns:
        Dictionary with performance metrics
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    if task_type == "classification":
        scores = cross_val_score(ensemble, X, y, cv=tscv, scoring="accuracy")
        return {
            "accuracy_mean": np.mean(scores),
            "accuracy_std": np.std(scores),
            "accuracy_scores": scores,
        }
    else:
        scores = cross_val_score(
            ensemble, X, y, cv=tscv, scoring="neg_mean_squared_error"
        )
        return {
            "mse_mean": -np.mean(scores),
            "mse_std": np.std(scores),
            "mse_scores": -scores,
        }


def create_ensemble_model(
    ensemble_type: str,
    base_estimators: Optional[List[BaseEstimator]] = None,
    config: Optional[EnsembleConfig] = None,
    task_type: str = "classification",
) -> BaseEstimator:
    """
    Factory function to create ensemble models.

    Args:
        ensemble_type: Type of ensemble ('random_forest', 'bagging', 'voting', 'stacking')
        base_estimators: List of base estimators for ensemble
        config: Ensemble configuration
        task_type: 'classification' or 'regression'

    Returns:
        Configured ensemble model
    """
    if config is None:
        config = EnsembleConfig()

    if ensemble_type == "random_forest":
        return FinancialRandomForest(config=config, task_type=task_type)

    elif ensemble_type == "bagging":
        if base_estimators is None:
            if task_type == "classification":
                base_estimator = DecisionTreeClassifier(
                    random_state=config.random_state
                )
            else:
                base_estimator = DecisionTreeRegressor(random_state=config.random_state)
        else:
            base_estimator = base_estimators[0]

        return TimeSeriesBagging(
            base_estimator=base_estimator, config=config, task_type=task_type
        )

    elif ensemble_type == "voting":
        if base_estimators is None:
            # Create default base estimators
            if task_type == "classification":
                estimators = [
                    ("rf", RandomForestClassifier(random_state=config.random_state)),
                    ("svc", SVC(probability=True, random_state=config.random_state)),
                    ("lr", LogisticRegression(random_state=config.random_state)),
                ]
            else:
                estimators = [
                    ("rf", RandomForestRegressor(random_state=config.random_state)),
                    ("svr", SVR()),
                    ("lr", LinearRegression()),
                ]
        else:
            estimators = [
                (f"estimator_{i}", est) for i, est in enumerate(base_estimators)
            ]

        return VotingEnsemble(estimators=estimators, config=config, task_type=task_type)

    elif ensemble_type == "stacking":
        if base_estimators is None:
            # Create default base estimators
            if task_type == "classification":
                base_estimators = [
                    RandomForestClassifier(random_state=config.random_state),
                    SVC(probability=True, random_state=config.random_state),
                    LogisticRegression(random_state=config.random_state),
                ]
            else:
                base_estimators = [
                    RandomForestRegressor(random_state=config.random_state),
                    SVR(),
                    LinearRegression(),
                ]

        return StackingEnsemble(
            base_estimators=base_estimators, config=config, task_type=task_type
        )

    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
