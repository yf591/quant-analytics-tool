"""
Support Vector Machine Models for Financial Analysis

This module implements SVM classifiers and regressors optimized for financial
time series prediction with kernel methods and advanced regularization.

Based on financial machine learning best practices and AFML principles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform
import time
import logging

from ..base import BaseClassifier, BaseRegressor, register_model


@register_model("svm_classifier")
class QuantSVMClassifier(BaseClassifier):
    """
    Support Vector Machine Classifier optimized for financial applications.

    Implements SVM with different kernels and advanced preprocessing
    for financial time series classification.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200,
        class_weight: Optional[Union[dict, str]] = None,
        max_iter: int = -1,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
        random_state: int = 42,
        scale_features: bool = True,
        **kwargs,
    ):
        """
        Initialize SVM Classifier.

        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree of polynomial kernel function
            coef0: Independent term in kernel function
            shrinking: Whether to use shrinking heuristic
            probability: Whether to enable probability estimates
            tol: Tolerance for stopping criterion
            cache_size: Size of kernel cache (in MB)
            class_weight: Weights associated with classes
            max_iter: Hard limit on iterations within solver
            decision_function_shape: Decision function shape ('ovo', 'ovr')
            break_ties: Break ties in multiclass
            random_state: Random state for reproducibility
            scale_features: Whether to scale features automatically
        """
        super().__init__(model_name="QuantSVMClassifier", **kwargs)

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.scale_features = scale_features

        self.scaler = StandardScaler() if scale_features else None
        self.model = self._build_model()

    def _build_model(self) -> Union[SVC, Pipeline]:
        """Build the SVM model with optional scaling pipeline."""
        svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state,
        )

        if self.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("svm", svm)])
        else:
            return svm

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "QuantSVMClassifier":
        """
        Train the SVM classifier.

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

        # Prepare fit parameters
        fit_params = {}
        if sample_weight is not None:
            if self.scale_features:
                fit_params["svm__sample_weight"] = sample_weight
            else:
                fit_params["sample_weight"] = sample_weight

        # Fit the model
        self.model.fit(X_array, y_array, **fit_params)

        # Get classes from the underlying SVM
        if self.scale_features:
            self.classes_ = self.model.named_steps["svm"].classes_
        else:
            self.classes_ = self.model.classes_

        self.is_fitted = True

        # Record training time
        training_time = time.time() - start_time
        self.metadata["training_time"] = training_time
        self.metadata["n_support"] = self._get_n_support()

        # Record training history
        self.training_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "training_samples": len(X_array),
                "training_time": training_time,
                "n_features": X_array.shape[1],
                "n_classes": len(self.classes_),
                "n_support": self.metadata["n_support"],
            }
        )

        self.logger.info(
            f"SVM trained in {training_time:.2f}s with {len(X_array)} samples"
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

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.

        Returns:
            Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)
        return self.model.decision_function(X_array)

    def _get_n_support(self) -> Union[int, np.ndarray]:
        """Get number of support vectors."""
        if self.scale_features:
            return self.model.named_steps["svm"].n_support_
        else:
            return self.model.n_support_

    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting support vectors")

        if self.scale_features:
            return self.model.named_steps["svm"].support_vectors_
        else:
            return self.model.support_vectors_

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
            # Different grids for different kernels
            if self.kernel == "rbf":
                param_grid = {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                }
            elif self.kernel == "poly":
                param_grid = {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                    "degree": [2, 3, 4, 5],
                }
            elif self.kernel == "linear":
                param_grid = {"C": [0.1, 1, 10, 100]}
            else:
                param_grid = {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                }

        # Adjust parameter names for pipeline
        if self.scale_features:
            param_grid = {f"svm__{k}": v for k, v in param_grid.items()}

        # Choose search method
        if method == "grid":
            search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
            )
        else:
            # Convert to random search parameters
            random_param_grid = {}
            for key, values in param_grid.items():
                if isinstance(values, list) and all(
                    isinstance(v, (int, float)) for v in values
                ):
                    if all(isinstance(v, int) for v in values):
                        random_param_grid[key] = values
                    else:
                        min_val, max_val = min(values), max(values)
                        random_param_grid[key] = uniform(min_val, max_val - min_val)
                else:
                    random_param_grid[key] = values

            search = RandomizedSearchCV(
                self.model,
                random_param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
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

    def plot_decision_boundary(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_indices: Tuple[int, int] = (0, 1),
        figsize: Tuple[int, int] = (10, 8),
        resolution: int = 100,
    ) -> plt.Figure:
        """
        Plot decision boundary for 2D feature space.

        Args:
            X: Features for plotting
            y: Targets for plotting
            feature_indices: Indices of features to plot
            figsize: Figure size
            resolution: Grid resolution for boundary

        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting decision boundary")

        X_array, y_array = self._validate_input(X, y)

        # Select two features
        X_plot = X_array[:, list(feature_indices)]

        # Create meshgrid
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )

        # Create full feature array for prediction
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        if X_array.shape[1] > 2:
            # Use mean values for other features
            other_features = np.tile(X_array.mean(axis=0)[2:], (len(grid_points), 1))
            grid_full = np.column_stack([grid_points, other_features])
        else:
            grid_full = grid_points

        # Predict on grid
        Z = self.model.predict(grid_full)
        Z = Z.reshape(xx.shape)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Set3)

        # Plot data points
        scatter = ax.scatter(
            X_plot[:, 0], X_plot[:, 1], c=y_array, cmap=plt.cm.Set1, edgecolors="black"
        )

        # Plot support vectors
        if hasattr(self, "support_vectors_"):
            support_vectors = self.get_support_vectors()
            if support_vectors.shape[1] >= 2:
                ax.scatter(
                    support_vectors[:, feature_indices[0]],
                    support_vectors[:, feature_indices[1]],
                    s=100,
                    facecolors="none",
                    edgecolors="black",
                    linewidth=2,
                )

        ax.set_xlabel(f"Feature {feature_indices[0]}")
        ax.set_ylabel(f"Feature {feature_indices[1]}")
        ax.set_title(f"SVM Decision Boundary ({self.kernel} kernel)")
        plt.colorbar(scatter, ax=ax)

        return fig


@register_model("svm_regressor")
class QuantSVMRegressor(BaseRegressor):
    """
    Support Vector Machine Regressor optimized for financial applications.

    Implements SVM regression with different kernels and advanced preprocessing
    for financial time series prediction.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200,
        max_iter: int = -1,
        random_state: int = 42,
        scale_features: bool = True,
        **kwargs,
    ):
        """Initialize SVM Regressor."""
        super().__init__(model_name="QuantSVMRegressor", **kwargs)

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.scale_features = scale_features

        self.scaler = StandardScaler() if scale_features else None
        self.model = self._build_model()

    def _build_model(self) -> Union[SVR, Pipeline]:
        """Build the SVM model with optional scaling pipeline."""
        svr = SVR(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            tol=self.tol,
            cache_size=self.cache_size,
            max_iter=self.max_iter,
        )

        if self.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("svr", svr)])
        else:
            return svr

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "QuantSVMRegressor":
        """Train the SVM regressor."""
        start_time = time.time()

        # Validate input
        X_array, y_array = self._validate_input(X, y)

        # Prepare fit parameters
        fit_params = {}
        if sample_weight is not None:
            if self.scale_features:
                fit_params["svr__sample_weight"] = sample_weight
            else:
                fit_params["sample_weight"] = sample_weight

        # Fit the model
        self.model.fit(X_array, y_array, **fit_params)
        self.is_fitted = True

        # Record training time
        training_time = time.time() - start_time
        self.metadata["training_time"] = training_time
        self.metadata["n_support"] = self._get_n_support()

        # Record training history
        self.training_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "training_samples": len(X_array),
                "training_time": training_time,
                "n_features": X_array.shape[1],
                "n_support": self.metadata["n_support"],
            }
        )

        self.logger.info(
            f"SVM regressor trained in {training_time:.2f}s with {len(X_array)} samples"
        )

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _ = self._validate_input(X)
        return self.model.predict(X_array)

    def _get_n_support(self) -> int:
        """Get number of support vectors."""
        if self.scale_features:
            return len(self.model.named_steps["svr"].support_)
        else:
            return len(self.model.support_)

    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting support vectors")

        if self.scale_features:
            return self.model.named_steps["svr"].support_vectors_
        else:
            return self.model.support_vectors_

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
            # Different grids for different kernels
            if self.kernel == "rbf":
                param_grid = {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                    "epsilon": [0.01, 0.1, 0.2, 0.5],
                }
            elif self.kernel == "poly":
                param_grid = {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                    "degree": [2, 3, 4, 5],
                    "epsilon": [0.01, 0.1, 0.2, 0.5],
                }
            elif self.kernel == "linear":
                param_grid = {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.2, 0.5]}
            else:
                param_grid = {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                    "epsilon": [0.01, 0.1, 0.2, 0.5],
                }

        # Adjust parameter names for pipeline
        if self.scale_features:
            param_grid = {f"svr__{k}": v for k, v in param_grid.items()}

        # Choose search method
        if method == "grid":
            search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
            )
        else:
            # Convert to random search parameters
            random_param_grid = {}
            for key, values in param_grid.items():
                if isinstance(values, list) and all(
                    isinstance(v, (int, float)) for v in values
                ):
                    if all(isinstance(v, int) for v in values):
                        random_param_grid[key] = values
                    else:
                        min_val, max_val = min(values), max(values)
                        random_param_grid[key] = uniform(min_val, max_val - min_val)
                else:
                    random_param_grid[key] = values

            search = RandomizedSearchCV(
                self.model,
                random_param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
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
