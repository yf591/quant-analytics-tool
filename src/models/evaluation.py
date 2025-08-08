"""
Model Evaluation Framework

This module provides comprehensive evaluation capabilities for financial machine learning models,
including cross-validation, performance metrics, and model comparison tools.

Based on AFML Chapter 7: Cross-Validation in Finance principles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import warnings
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    TimeSeriesSplit,
    KFold,
    StratifiedKFold,
    GroupKFold,
)
from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    log_loss,
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    # Additional metrics
    make_scorer,
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import logging


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    log_loss: Optional[float] = None

    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    explained_variance: Optional[float] = None

    # Financial metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    information_ratio: Optional[float] = None

    # Additional metrics
    prediction_time: Optional[float] = None
    training_time: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CrossValidationResults:
    """Container for cross-validation results."""

    scores: np.ndarray
    mean_score: float
    std_score: float
    individual_metrics: List[PerformanceMetrics]
    fold_predictions: List[np.ndarray]
    fold_probabilities: Optional[List[np.ndarray]] = None
    feature_importance: Optional[List[pd.Series]] = None


class ModelEvaluator:
    """
    Comprehensive model evaluation system.

    Provides unified evaluation capabilities for both classification
    and regression models with financial-specific metrics.
    """

    def __init__(self, problem_type: str = "auto"):
        """
        Initialize model evaluator.

        Args:
            problem_type: 'classification', 'regression', or 'auto'
        """
        self.problem_type = problem_type
        self.logger = logging.getLogger(__name__)

    def evaluate_model(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        X_train: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_train: Optional[Union[pd.Series, np.ndarray]] = None,
        returns: Optional[pd.Series] = None,
    ) -> PerformanceMetrics:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional, for overfitting analysis)
            y_train: Training targets (optional, for overfitting analysis)
            returns: Price returns for financial metrics calculation

        Returns:
            PerformanceMetrics object with all computed metrics
        """
        start_time = time.time()

        # Make predictions
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time

        # Determine problem type
        if self.problem_type == "auto":
            problem_type = self._detect_problem_type(y_test)
        else:
            problem_type = self.problem_type

        metrics = PerformanceMetrics(prediction_time=prediction_time)

        # Calculate metrics based on problem type
        if problem_type == "classification":
            metrics = self._calculate_classification_metrics(
                y_test, y_pred, model, X_test, metrics
            )
        else:
            metrics = self._calculate_regression_metrics(y_test, y_pred, metrics)

        # Calculate financial metrics if returns provided
        if returns is not None:
            metrics = self._calculate_financial_metrics(
                y_test, y_pred, returns, metrics
            )

        return metrics

    def _detect_problem_type(self, y: Union[pd.Series, np.ndarray]) -> str:
        """Automatically detect if problem is classification or regression."""
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Check data type
        if np.issubdtype(y_array.dtype, np.integer):
            unique_values = len(np.unique(y_array))
            if unique_values <= 10:  # Heuristic for classification
                return "classification"

        # Check if values are binary
        unique_vals = np.unique(y_array)
        if len(unique_vals) == 2:
            return "classification"

        return "regression"

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        X_test: np.ndarray,
        metrics: PerformanceMetrics,
    ) -> PerformanceMetrics:
        """Calculate classification-specific metrics."""
        try:
            # Basic classification metrics
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics.recall = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics.f1_score = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # ROC AUC (if binary classification and probabilities available)
            if len(np.unique(y_true)) == 2 and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics.roc_auc = roc_auc_score(y_true, y_proba)
                    metrics.log_loss = log_loss(y_true, y_proba)
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {str(e)}")

        return metrics

    def _calculate_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate regression-specific metrics."""
        try:
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y_true, y_pred)
            metrics.r2 = r2_score(y_true, y_pred)

            # MAPE (avoid division by zero)
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                metrics.mape = (
                    np.mean(
                        np.abs(
                            (y_true[non_zero_mask] - y_pred[non_zero_mask])
                            / y_true[non_zero_mask]
                        )
                    )
                    * 100
                )

            metrics.explained_variance = explained_variance_score(y_true, y_pred)

        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {str(e)}")

        return metrics

    def _calculate_financial_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: pd.Series,
        metrics: PerformanceMetrics,
    ) -> PerformanceMetrics:
        """Calculate financial performance metrics."""
        try:
            # Align returns with predictions
            if len(returns) != len(y_pred):
                # Try to align by truncating
                min_len = min(len(returns), len(y_pred))
                returns = returns.iloc[-min_len:]
                y_pred = y_pred[-min_len:]
                y_true = y_true[-min_len:]

            # Generate trading signals from predictions
            # For regression: positive prediction = buy signal
            # For classification: assume 1 = buy, 0 = sell/hold
            if self._detect_problem_type(y_true) == "classification":
                signals = y_pred
            else:
                signals = np.where(y_pred > 0, 1, 0)

            # Calculate strategy returns
            strategy_returns = returns * signals

            # Sharpe ratio
            if strategy_returns.std() > 0:
                metrics.sharpe_ratio = (
                    strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                )

            # Sortino ratio (downside deviation)
            downside_returns = strategy_returns[strategy_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                metrics.sortino_ratio = (
                    strategy_returns.mean() / downside_returns.std() * np.sqrt(252)
                )

            # Maximum drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            metrics.max_drawdown = drawdowns.min()

            # Calmar ratio
            if metrics.max_drawdown < 0:
                annual_return = strategy_returns.mean() * 252
                metrics.calmar_ratio = annual_return / abs(metrics.max_drawdown)

            # Information ratio (vs buy-and-hold)
            excess_returns = strategy_returns - returns
            if excess_returns.std() > 0:
                metrics.information_ratio = (
                    excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                )

        except Exception as e:
            self.logger.warning(f"Could not calculate financial metrics: {str(e)}")

        return metrics

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compare multiple models and return results DataFrame."""
        results = {}

        for name, model in models.items():
            try:
                metrics = self.evaluate_model(model, X_test, y_test, returns=returns)
                results[name] = metrics.to_dict()
            except Exception as e:
                self.logger.error(f"Error evaluating model {name}: {str(e)}")
                results[name] = {}

        return pd.DataFrame(results).T

    def plot_performance_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """Create visualization of model performance comparison."""
        if metrics is None:
            # Select key metrics based on available columns
            available_metrics = comparison_df.columns.tolist()
            key_metrics = ["accuracy", "f1_score", "r2", "sharpe_ratio", "max_drawdown"]
            metrics = [m for m in key_metrics if m in available_metrics]

        n_metrics = len(metrics)
        if n_metrics == 0:
            raise ValueError("No valid metrics found for plotting")

        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                data = comparison_df[metric].dropna()
                data.plot(kind="bar", ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel("Models")
                axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig


class CrossValidator:
    """
    Cross-validation system optimized for financial time series.

    Implements time-aware cross-validation methods to avoid look-ahead bias
    and account for temporal dependencies in financial data.
    """

    def __init__(
        self,
        cv_method: str = "time_series",
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Initialize cross-validator.

        Args:
            cv_method: 'time_series', 'purged', 'blocked', or 'stratified'
            n_splits: Number of cross-validation splits
            test_size: Size of test set for time series split
            gap: Gap between train and test sets (to avoid look-ahead bias)
        """
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.logger = logging.getLogger(__name__)

    def cross_validate_model(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        scoring: Union[str, Callable] = None,
        returns: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None,
    ) -> CrossValidationResults:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to cross-validate
            X: Features
            y: Targets
            scoring: Scoring function or string
            returns: Price returns for financial metrics
            groups: Group labels for grouped CV

        Returns:
            CrossValidationResults object
        """
        # Get cross-validation splitter
        cv_splitter = self._get_cv_splitter(X, y, groups)

        scores = []
        individual_metrics = []
        fold_predictions = []
        fold_probabilities = []
        feature_importance = []

        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
            try:
                # Get train/test data for this fold
                X_train, X_test = self._get_fold_data(X, train_idx, test_idx)
                y_train, y_test = self._get_fold_data(y, train_idx, test_idx)

                # Train model on fold
                fold_model = self._clone_model(model)
                fold_model.fit(X_train, y_train)

                # Make predictions
                y_pred = fold_model.predict(X_test)
                fold_predictions.append(y_pred)

                # Get probabilities if available
                if hasattr(fold_model, "predict_proba"):
                    y_proba = fold_model.predict_proba(X_test)
                    fold_probabilities.append(y_proba)

                # Calculate score
                if scoring:
                    if isinstance(scoring, str):
                        score = self._calculate_score(
                            scoring, y_test, y_pred, fold_model, X_test
                        )
                    else:
                        score = scoring(fold_model, X_test, y_test)
                else:
                    score = fold_model.score(X_test, y_test)

                scores.append(score)

                # Calculate detailed metrics
                evaluator = ModelEvaluator()
                fold_returns = returns.iloc[test_idx] if returns is not None else None
                metrics = evaluator.evaluate_model(
                    fold_model, X_test, y_test, returns=fold_returns
                )
                individual_metrics.append(metrics)

                # Get feature importance
                if hasattr(fold_model, "feature_importances_"):
                    importance = pd.Series(fold_model.feature_importances_)
                    feature_importance.append(importance)

                self.logger.info(
                    f"Fold {fold + 1}/{self.n_splits} completed with score: {score:.4f}"
                )

            except Exception as e:
                self.logger.error(f"Error in fold {fold + 1}: {str(e)}")
                scores.append(np.nan)

        scores = np.array(scores)
        valid_scores = scores[~np.isnan(scores)]

        return CrossValidationResults(
            scores=valid_scores,
            mean_score=np.mean(valid_scores),
            std_score=np.std(valid_scores),
            individual_metrics=individual_metrics,
            fold_predictions=fold_predictions,
            fold_probabilities=fold_probabilities if fold_probabilities else None,
            feature_importance=feature_importance if feature_importance else None,
        )

    def _get_cv_splitter(self, X, y, groups):
        """Get appropriate cross-validation splitter."""
        if self.cv_method == "time_series":
            return TimeSeriesSplit(
                n_splits=self.n_splits, test_size=self.test_size, gap=self.gap
            )
        elif self.cv_method == "stratified":
            return StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=42
            )
        elif self.cv_method == "grouped":
            return GroupKFold(n_splits=self.n_splits)
        else:
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

    def _get_fold_data(self, data, train_idx, test_idx):
        """Extract train/test data for a fold."""
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.iloc[train_idx], data.iloc[test_idx]
        else:
            return data[train_idx], data[test_idx]

    def _clone_model(self, model):
        """Create a copy of the model for training."""
        if hasattr(model, "get_params") and hasattr(model, "set_params"):
            # sklearn-style model
            from sklearn.base import clone

            return clone(model)
        else:
            # Custom model - try to create new instance
            return model.__class__(**model.get_params())

    def _calculate_score(self, scoring, y_true, y_pred, model, X_test):
        """Calculate score based on scoring method."""
        if scoring == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif scoring == "f1":
            return f1_score(y_true, y_pred, average="weighted")
        elif scoring == "r2":
            return r2_score(y_true, y_pred)
        elif scoring == "neg_mean_squared_error":
            return -mean_squared_error(y_true, y_pred)
        else:
            return model.score(X_test, y_true)
