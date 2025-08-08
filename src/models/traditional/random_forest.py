"""
Random Forest Models for Financial Analysis

This module implements Random Forest classifiers and regressors optimized for financial
time series prediction with AFML-compliant features and enhancements.

Based on AFML Chapter 6: Ensemble Methods principles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import time
import logging

from ..base import BaseClassifier, BaseRegressor, register_model


@register_model("random_forest_classifier")
class QuantRandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier optimized for financial applications.

    Implements AFML-compliant ensemble methods with financial-specific
    optimizations including sample weighting and feature importance analysis.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        bootstrap: bool = True,
        class_weight: Optional[Union[str, dict]] = "balanced_subsample",
        criterion: str = "gini",
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Initialize Random Forest Classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            class_weight: Weights associated with classes
            criterion: Split quality criterion ('gini' or 'entropy')
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        super().__init__(model_name="QuantRandomForestClassifier", **kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.criterion = criterion
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.model = self._build_model()

    def _build_model(self) -> RandomForestClassifier:
        """Build the Random Forest model."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            criterion=self.criterion,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "QuantRandomForestClassifier":
        """
        Train the Random Forest classifier.

        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights for training

        Returns:
            Self for method chaining
        """
        start_time = time.time()

        # Validate input
        X_array, y_array = self._validate_input(X, y)

        # Fit the model
        self.model.fit(X_array, y_array, sample_weight=sample_weight)
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
            f"Random Forest trained in {training_time:.2f}s with {len(X_array)} samples"
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

    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = "accuracy",
        n_iter: int = 50,
        method: str = "random",
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search or random search.

        Args:
            X: Training features
            y: Training targets
            param_grid: Parameter grid for search
            cv: Cross-validation folds
            scoring: Scoring metric
            n_iter: Number of iterations for random search
            method: 'grid' or 'random' search

        Returns:
            Dictionary with best parameters and scores
        """
        X_array, y_array = self._validate_input(X, y)

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5],
            }

        # Choose search method
        if method == "grid":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
            )
        else:
            # Convert to random search parameters
            random_param_grid = {}
            for key, values in param_grid.items():
                if isinstance(values, list):
                    if all(isinstance(v, int) for v in values):
                        random_param_grid[key] = randint(min(values), max(values))
                    elif all(isinstance(v, float) for v in values):
                        random_param_grid[key] = uniform(
                            min(values), max(values) - min(values)
                        )
                    else:
                        random_param_grid[key] = values
                else:
                    random_param_grid[key] = values

            search = RandomizedSearchCV(
                self.model,
                random_param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=self.random_state,
            )

        # Perform search
        search.fit(X_array, y_array)

        # Update model with best parameters
        self.model = search.best_estimator_

        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_,
        }

    def get_feature_importance(self, plot: bool = False) -> pd.Series:
        """
        Get feature importance with optional visualization.

        Args:
            plot: Whether to create importance plot

        Returns:
            Series with feature importance values
        """
        importance = super().get_feature_importance()

        if plot and importance is not None:
            self.plot_feature_importance(importance)

        return importance

    def plot_feature_importance(
        self,
        importance: Optional[pd.Series] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Plot feature importance."""
        if importance is None:
            importance = self.get_feature_importance()

        if importance is None:
            raise ValueError("No feature importance available")

        # Get top N features
        top_features = importance.nlargest(top_n)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        top_features.plot(kind="barh", ax=ax)
        ax.set_title(f"Top {top_n} Feature Importance - Random Forest")
        ax.set_xlabel("Importance")
        plt.tight_layout()

        return fig

    def get_tree_depths(self) -> List[int]:
        """Get depths of all trees in the forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return [tree.tree_.max_depth for tree in self.model.estimators_]

    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score if available."""
        if hasattr(self.model, "oob_score_"):
            return self.model.oob_score_
        else:
            self.logger.warning(
                "OOB score not available. Set oob_score=True when creating model."
            )
            return None


@register_model("random_forest_regressor")
class QuantRandomForestRegressor(BaseRegressor):
    """
    Random Forest Regressor optimized for financial applications.

    Implements AFML-compliant ensemble methods for regression tasks
    with financial-specific optimizations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        bootstrap: bool = True,
        criterion: str = "squared_error",
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Initialize Random Forest Regressor.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            criterion: Split quality criterion
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        super().__init__(model_name="QuantRandomForestRegressor", **kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.model = self._build_model()

    def _build_model(self) -> RandomForestRegressor:
        """Build the Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            criterion=self.criterion,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "QuantRandomForestRegressor":
        """Train the Random Forest regressor."""
        start_time = time.time()

        # Validate input
        X_array, y_array = self._validate_input(X, y)

        # Fit the model
        self.model.fit(X_array, y_array, sample_weight=sample_weight)
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
            f"Random Forest regressor trained in {training_time:.2f}s with {len(X_array)} samples"
        )

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)
        return self.model.predict(X_array)

    def predict_quantiles(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> np.ndarray:
        """
        Predict quantiles using individual tree predictions.

        Args:
            X: Input features
            quantiles: List of quantiles to predict

        Returns:
            Array of quantile predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)

        # Get predictions from all trees
        tree_predictions = np.array(
            [tree.predict(X_array) for tree in self.model.estimators_]
        )

        # Calculate quantiles across trees for each sample
        quantile_predictions = np.quantile(tree_predictions, quantiles, axis=0)

        return quantile_predictions.T

    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = "r2",
        n_iter: int = 50,
        method: str = "random",
    ) -> Dict[str, Any]:
        """Tune hyperparameters for regressor."""
        X_array, y_array = self._validate_input(X, y)

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5, 1.0],
            }

        # Choose search method
        if method == "grid":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
            )
        else:
            # Convert to random search parameters
            random_param_grid = {}
            for key, values in param_grid.items():
                if isinstance(values, list):
                    if all(isinstance(v, int) for v in values):
                        random_param_grid[key] = randint(min(values), max(values))
                    elif all(isinstance(v, float) for v in values):
                        random_param_grid[key] = uniform(
                            min(values), max(values) - min(values)
                        )
                    else:
                        random_param_grid[key] = values
                else:
                    random_param_grid[key] = values

            search = RandomizedSearchCV(
                self.model,
                random_param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=self.random_state,
            )

        # Perform search
        search.fit(X_array, y_array)

        # Update model with best parameters
        self.model = search.best_estimator_

        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_,
        }
