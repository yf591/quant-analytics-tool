"""
Meta-Labeling for Financial Machine Learning

This module implements meta-labeling techniques as described in AFML Chapter 3.
Meta-labeling is a powerful technique that helps determine when to act on a
primary model's prediction and what position size to take.

Features:
- Primary model prediction generation
- Secondary meta-model for sizing decisions
- Triple barrier labeling method
- Fixed-time horizon labeling
- Sample weight calculation based on label imbalance
- Meta-labeling evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MetaLabelingConfig:
    """Configuration for meta-labeling."""

    # Triple barrier parameters
    profit_target: float = 0.02  # Profit taking threshold
    stop_loss: float = 0.01  # Stop loss threshold
    max_holding_period: int = 5  # Maximum holding period in periods

    # Meta-model parameters
    meta_model_type: str = "random_forest"  # 'random_forest', 'logistic'
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2

    # Sample weighting
    weight_by_returns: bool = True
    weight_by_uniqueness: bool = True

    # Cross-validation
    cv_splits: int = 5
    purge_length: int = 0
    embargo_length: int = 0

    random_state: int = 42


class TripleBarrierLabeling:
    """
    Triple Barrier Labeling Method

    Based on AFML Chapter 3, this method creates labels by setting
    three barriers: profit target, stop loss, and time horizon.
    """

    def __init__(self, config: MetaLabelingConfig):
        self.config = config

    def apply_triple_barrier(
        self,
        prices: pd.Series,
        events: pd.Series,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        max_holding_period: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Apply triple barrier labeling to price series.

        Args:
            prices: Price series with datetime index
            events: Series of event times (prediction times)
            profit_target: Profit taking threshold (if None, use config)
            stop_loss: Stop loss threshold (if None, use config)
            max_holding_period: Max holding period (if None, use config)

        Returns:
            DataFrame with barrier information and labels
        """
        if profit_target is None:
            profit_target = self.config.profit_target
        if stop_loss is None:
            stop_loss = self.config.stop_loss
        if max_holding_period is None:
            max_holding_period = self.config.max_holding_period

        # Initialize results
        results = []

        for event_time in events.index:
            if event_time not in prices.index:
                continue

            # Get starting price
            start_price = prices.loc[event_time]

            # Define time horizon
            future_prices = prices.loc[event_time:].iloc[1 : max_holding_period + 1]

            if len(future_prices) == 0:
                continue

            # Calculate returns
            returns = (future_prices / start_price) - 1

            # Find first barrier hit
            profit_hits = returns >= profit_target
            loss_hits = returns <= -stop_loss

            # Determine exit time and reason
            exit_time = None
            exit_reason = "time"
            exit_return = returns.iloc[-1] if len(returns) > 0 else 0

            if profit_hits.any():
                profit_exit_time = profit_hits.idxmax()
                if not loss_hits.any() or profit_exit_time <= loss_hits.idxmax():
                    exit_time = profit_exit_time
                    exit_reason = "profit"
                    exit_return = returns.loc[exit_time]

            if loss_hits.any() and exit_reason != "profit":
                loss_exit_time = loss_hits.idxmax()
                exit_time = loss_exit_time
                exit_reason = "loss"
                exit_return = returns.loc[exit_time]

            if exit_time is None:
                exit_time = returns.index[-1]

            # Create label (1 for profit, -1 for loss, 0 for neutral/time)
            if exit_reason == "profit":
                label = 1
            elif exit_reason == "loss":
                label = -1
            else:
                # Time-based exit: use actual return sign
                label = 1 if exit_return > 0 else -1 if exit_return < 0 else 0

            results.append(
                {
                    "event_time": event_time,
                    "exit_time": exit_time,
                    "holding_period": (
                        (exit_time - event_time).total_seconds() / 3600
                        if isinstance(exit_time, pd.Timestamp)
                        else len(returns)
                    ),
                    "exit_reason": exit_reason,
                    "return": exit_return,
                    "label": label,
                }
            )

        return pd.DataFrame(results).set_index("event_time")


class MetaLabelingModel:
    """
    Meta-labeling model that combines primary predictions with secondary sizing decisions.

    The primary model generates directional predictions, while the meta-model
    determines when to act on these predictions and what size to take.
    """

    def __init__(self, primary_model: BaseEstimator, config: MetaLabelingConfig):
        self.primary_model = primary_model
        self.config = config
        self.meta_model = None
        self.barrier_labeler = TripleBarrierLabeling(config)
        self.is_fitted = False

    def _create_meta_model(self) -> BaseEstimator:
        """Create meta-learning model."""
        if self.config.meta_model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif self.config.meta_model_type == "logistic":
            return LogisticRegression(
                random_state=self.config.random_state, max_iter=1000
            )
        else:
            raise ValueError(f"Unknown meta_model_type: {self.config.meta_model_type}")

    def _calculate_sample_weights(
        self, labels: pd.Series, returns: pd.Series
    ) -> np.ndarray:
        """
        Calculate sample weights based on label imbalance and return magnitude.

        Args:
            labels: Label series
            returns: Return series

        Returns:
            Array of sample weights
        """
        weights = np.ones(len(labels))

        # Weight by label frequency (inverse frequency weighting)
        label_counts = labels.value_counts()
        for label, count in label_counts.items():
            mask = labels == label
            weights[mask] = 1.0 / count

        # Weight by return magnitude if configured
        if self.config.weight_by_returns:
            return_weights = np.abs(returns) / np.abs(returns).mean()
            weights *= return_weights

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        return weights

    def generate_meta_features(
        self, X: np.ndarray, primary_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Generate features for meta-model.

        Args:
            X: Original features
            primary_predictions: Primary model predictions

        Returns:
            Combined feature matrix for meta-model
        """
        # Combine original features with primary predictions
        if hasattr(self.primary_model, "predict_proba"):
            # Use prediction probabilities if available
            primary_proba = self.primary_model.predict_proba(X)
            if primary_proba.shape[1] == 2:
                # Binary classification - use positive class probability
                primary_features = primary_proba[:, 1].reshape(-1, 1)
            else:
                # Multi-class - use all probabilities
                primary_features = primary_proba
        else:
            # Use predictions directly
            primary_features = primary_predictions.reshape(-1, 1)

        # Add prediction confidence/magnitude
        if hasattr(self.primary_model, "decision_function"):
            decision_scores = self.primary_model.decision_function(X)
            confidence = np.abs(decision_scores).reshape(-1, 1)
            primary_features = np.hstack([primary_features, confidence])

        # Combine with original features (subset)
        # Use most important features only to avoid overfitting
        feature_importance_threshold = 0.01
        if hasattr(self.primary_model, "feature_importances_"):
            important_features = (
                self.primary_model.feature_importances_ > feature_importance_threshold
            )
            if important_features.any():
                selected_X = X[:, important_features]
            else:
                # If no features meet threshold, take top 10
                top_features = np.argsort(self.primary_model.feature_importances_)[-10:]
                selected_X = X[:, top_features]
        else:
            # Use top 10 features if importance not available
            selected_X = X[:, : min(10, X.shape[1])]

        meta_features = np.hstack([primary_features, selected_X])

        return meta_features

    def fit(
        self,
        X: np.ndarray,
        prices: pd.Series,
        events: pd.Series,
        y_primary: Optional[np.ndarray] = None,
    ) -> "MetaLabelingModel":
        """
        Fit meta-labeling model.

        Args:
            X: Feature matrix
            prices: Price series for barrier labeling
            events: Event times series
            y_primary: Optional primary model targets (if None, derived from barriers)

        Returns:
            Fitted meta-labeling model
        """
        # Fit primary model first if targets provided
        if y_primary is not None:
            self.primary_model.fit(X, y_primary)

        # Generate primary predictions
        primary_predictions = self.primary_model.predict(X)

        # Apply triple barrier labeling
        barrier_results = self.barrier_labeler.apply_triple_barrier(prices, events)

        # Align data
        common_index = barrier_results.index.intersection(events.index)
        if len(common_index) == 0:
            raise ValueError("No common timestamps between barriers and events")

        # Filter data to common timestamps
        barrier_subset = barrier_results.loc[common_index]
        event_positions = [list(events.index).index(idx) for idx in common_index]
        X_subset = X[event_positions]
        primary_pred_subset = primary_predictions[event_positions]

        # Create meta-labels (binary: act or not act)
        # Act when primary prediction direction matches barrier outcome
        meta_labels = (
            (primary_pred_subset > 0) & (barrier_subset["label"] > 0)
            | (primary_pred_subset < 0) & (barrier_subset["label"] < 0)
        ).astype(int)

        # Generate meta-features
        meta_features = self.generate_meta_features(X_subset, primary_pred_subset)

        # Calculate sample weights
        sample_weights = self._calculate_sample_weights(
            barrier_subset["label"], barrier_subset["return"]
        )

        # Fit meta-model
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(meta_features, meta_labels, sample_weight=sample_weights)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make meta-labeling predictions.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (primary_predictions, meta_predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Get primary predictions
        primary_predictions = self.primary_model.predict(X)

        # Generate meta-features
        meta_features = self.generate_meta_features(X, primary_predictions)

        # Get meta-predictions (probability of acting)
        if hasattr(self.meta_model, "predict_proba"):
            meta_predictions = self.meta_model.predict_proba(meta_features)[:, 1]
        else:
            meta_predictions = self.meta_model.predict(meta_features)

        return primary_predictions, meta_predictions

    def predict_position_size(
        self, X: np.ndarray, base_size: float = 1.0, confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict position sizes using meta-labeling.

        Args:
            X: Feature matrix
            base_size: Base position size
            confidence_threshold: Minimum confidence to take position

        Returns:
            Array of position sizes (positive/negative for long/short)
        """
        primary_pred, meta_pred = self.predict(X)

        # Position size = direction * confidence * base_size
        # Only take position if meta-prediction exceeds threshold
        position_sizes = np.where(
            meta_pred >= confidence_threshold,
            np.sign(primary_pred) * meta_pred * base_size,
            0.0,
        )

        return position_sizes


def evaluate_meta_labeling(
    model: MetaLabelingModel,
    X_test: np.ndarray,
    prices_test: pd.Series,
    events_test: pd.Series,
    confidence_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate meta-labeling model performance.

    Args:
        model: Fitted meta-labeling model
        X_test: Test features
        prices_test: Test prices
        events_test: Test events
        confidence_threshold: Confidence threshold for position taking

    Returns:
        Dictionary with evaluation metrics
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before evaluation")

    # Generate predictions
    primary_pred, meta_pred = model.predict(X_test)

    # Apply triple barrier labeling to test data
    barrier_results = model.barrier_labeler.apply_triple_barrier(
        prices_test, events_test
    )

    # Align predictions with barrier results
    common_index = barrier_results.index.intersection(events_test.index)
    barrier_subset = barrier_results.loc[common_index]
    event_positions = [list(events_test.index).index(idx) for idx in common_index]

    primary_pred_subset = primary_pred[event_positions]
    meta_pred_subset = meta_pred[event_positions]

    # Calculate metrics
    # Primary model accuracy
    primary_accuracy = accuracy_score(
        barrier_subset["label"] > 0, primary_pred_subset > 0
    )

    # Meta-model metrics (when to act)
    actual_profitable = (primary_pred_subset > 0) & (barrier_subset["label"] > 0) | (
        primary_pred_subset < 0
    ) & (barrier_subset["label"] < 0)

    meta_binary_pred = meta_pred_subset >= confidence_threshold

    meta_accuracy = accuracy_score(actual_profitable, meta_binary_pred)
    meta_precision = precision_score(
        actual_profitable, meta_binary_pred, zero_division=0
    )
    meta_recall = recall_score(actual_profitable, meta_binary_pred, zero_division=0)
    meta_f1 = f1_score(actual_profitable, meta_binary_pred, zero_division=0)

    # Strategy performance
    position_sizes = model.predict_position_size(
        X_test[event_positions], confidence_threshold=confidence_threshold
    )

    strategy_returns = position_sizes * barrier_subset["return"].values

    # Performance metrics
    total_return = np.sum(strategy_returns)
    avg_return = np.mean(strategy_returns)
    sharpe_ratio = (
        avg_return / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
    )
    hit_rate = np.mean(strategy_returns > 0)

    # Position statistics
    n_positions = np.sum(position_sizes != 0)
    avg_position_size = (
        np.mean(np.abs(position_sizes[position_sizes != 0])) if n_positions > 0 else 0
    )

    return {
        "primary_accuracy": primary_accuracy,
        "meta_accuracy": meta_accuracy,
        "meta_precision": meta_precision,
        "meta_recall": meta_recall,
        "meta_f1": meta_f1,
        "total_return": total_return,
        "avg_return": avg_return,
        "sharpe_ratio": sharpe_ratio,
        "hit_rate": hit_rate,
        "n_positions": n_positions,
        "avg_position_size": avg_position_size,
        "n_samples": len(barrier_subset),
    }


def create_meta_labeling_model(
    primary_model: BaseEstimator, config: Optional[MetaLabelingConfig] = None
) -> MetaLabelingModel:
    """
    Factory function to create meta-labeling model.

    Args:
        primary_model: Primary prediction model
        config: Meta-labeling configuration

    Returns:
        Meta-labeling model instance
    """
    if config is None:
        config = MetaLabelingConfig()

    return MetaLabelingModel(primary_model=primary_model, config=config)
