"""
Walk-Forward Analysis Implementation

This module implements walk-forward analysis for time series financial data,
based on methodologies from "Advances in Financial Machine Learning" Chapter 7.
Provides purged cross-validation to prevent data leakage in financial models.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

from ..models.base import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged Group Time Series Split for financial data.

    Implements the purging methodology from AFML Chapter 7 to prevent
    data leakage in time series cross-validation. Observations that overlap
    with the test set are purged from the training set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        test_size: float = 0.2,
        gap_size: int = 0,
    ):
        """
        Initialize Purged Group Time Series Split.

        Args:
            n_splits: Number of splits for cross-validation
            embargo_pct: Percentage of observations to embargo after test set
            test_size: Proportion of data to use for testing
            gap_size: Number of observations to skip between train and test
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.test_size = test_size
        self.gap_size = gap_size

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """
        Generate train/test splits with purging and embargo.

        Args:
            X: Feature matrix with datetime index
            y: Target series
            groups: Group labels (not used)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)

        # Calculate step size for walk-forward
        step_size = (n_samples - test_size_samples) // self.n_splits

        for i in range(self.n_splits):
            # Define test period
            test_start = i * step_size + test_size_samples
            test_end = min(test_start + test_size_samples, n_samples)

            if test_end >= n_samples:
                break

            # Test indices
            test_indices = np.arange(test_start, test_end)

            # Training indices before test period
            train_end = test_start - self.gap_size
            train_indices = np.arange(0, max(0, train_end))

            # Apply embargo after test period
            embargo_size = int(self.embargo_pct * n_samples)
            embargo_end = min(test_end + embargo_size, n_samples)

            # Add training data after embargo period
            if embargo_end < n_samples:
                post_embargo_indices = np.arange(embargo_end, n_samples)
                train_indices = np.concatenate([train_indices, post_embargo_indices])

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for time series financial models.

    Implements comprehensive walk-forward testing with purged cross-validation,
    performance tracking, and statistical analysis of out-of-sample results.
    """

    def __init__(
        self,
        window_size: int = 252,  # Trading days in a year
        step_size: int = 21,  # Monthly rebalancing
        min_train_size: int = 252,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ):
        """
        Initialize Walk-Forward Analyzer.

        Args:
            window_size: Size of training window in observations
            step_size: Step size for moving window
            min_train_size: Minimum training set size
            embargo_pct: Embargo percentage for avoiding look-ahead bias
            purge_pct: Purge percentage for removing overlapping observations
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

        self.results_: List[Dict[str, Any]] = []
        self.performance_metrics_: Dict[str, List[float]] = {}

        logger.info(
            f"Initialized WalkForwardAnalyzer with window_size={window_size}, "
            f"step_size={step_size}"
        )

    def purge_training_set(
        self, train_times: pd.Series, test_times: pd.Series
    ) -> pd.Index:
        """
        Purge training observations that overlap with test period.

        Based on AFML Chapter 7, SNIPPET 7.1.

        Args:
            train_times: Series with start/end times for training observations
            test_times: Series with start/end times for test observations

        Returns:
            Purged training set index
        """
        purged_train = train_times.copy()

        for test_start, test_end in test_times.items():
            # Remove training observations that overlap with test period
            overlap_mask = (
                ((purged_train.index >= test_start) & (purged_train.index <= test_end))
                | ((purged_train >= test_start) & (purged_train <= test_end))
                | ((purged_train.index <= test_start) & (purged_train >= test_end))
            )

            purged_train = purged_train[~overlap_mask]

        return purged_train.index

    def create_embargo_period(
        self, test_end: pd.Timestamp, data_index: pd.Index
    ) -> pd.Index:
        """
        Create embargo period after test set to prevent look-ahead bias.

        Args:
            test_end: End time of test period
            data_index: Full data index

        Returns:
            Index of embargoed observations
        """
        embargo_size = int(len(data_index) * self.embargo_pct)
        test_end_idx = data_index.get_loc(test_end)

        embargo_end_idx = min(test_end_idx + embargo_size, len(data_index) - 1)

        if embargo_end_idx > test_end_idx:
            return data_index[test_end_idx:embargo_end_idx]
        else:
            return pd.Index([])

    def walk_forward_split(self, data: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate walk-forward train/test splits.

        Args:
            data: DataFrame with datetime index

        Returns:
            List of (train_index, test_index) tuples
        """
        splits = []
        data_length = len(data)

        # Start from minimum training size
        start_idx = max(0, self.min_train_size)

        while start_idx + self.step_size < data_length:
            # Define test period
            test_start_idx = start_idx
            test_end_idx = min(test_start_idx + self.step_size, data_length)

            # Define training period (expanding window)
            train_start_idx = max(0, test_start_idx - self.window_size)
            train_end_idx = test_start_idx

            # Create indices
            train_index = data.index[train_start_idx:train_end_idx]
            test_index = data.index[test_start_idx:test_end_idx]

            # Apply purging (remove overlapping observations)
            purge_size = int(len(train_index) * self.purge_pct)
            if purge_size > 0:
                train_index = train_index[:-purge_size]

            if len(train_index) >= self.min_train_size and len(test_index) > 0:
                splits.append((train_index, test_index))

            start_idx += self.step_size

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def run_walk_forward_analysis(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        refit_frequency: int = 1,
    ) -> Dict[str, Any]:
        """
        Run comprehensive walk-forward analysis.

        Args:
            model: Model to test
            X: Feature matrix
            y: Target vector
            sample_weights: Optional sample weights
            refit_frequency: How often to refit model (1 = every period)

        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting walk-forward analysis")

        # Generate splits
        splits = self.walk_forward_split(X)

        if len(splits) == 0:
            raise ValueError(
                "No valid splits generated. Check data size and parameters."
            )

        # Initialize results storage
        all_predictions = []
        all_actuals = []
        all_dates = []
        period_results = []

        fitted_model = None

        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}")

            try:
                # Prepare training data
                X_train = X.loc[train_idx]
                y_train = y.loc[train_idx]

                # Prepare test data
                X_test = X.loc[test_idx]
                y_test = y.loc[test_idx]

                # Sample weights for training
                train_weights = None
                if sample_weights is not None:
                    train_weights = sample_weights.loc[train_idx]

                # Refit model if needed
                if fitted_model is None or i % refit_frequency == 0:
                    logger.debug(f"Refitting model at split {i+1}")
                    fitted_model = model.clone()
                    fitted_model.fit(X_train, y_train, sample_weight=train_weights)

                # Make predictions
                predictions = fitted_model.predict(X_test)

                # Calculate metrics
                if hasattr(fitted_model, "predict_proba"):
                    pred_proba = fitted_model.predict_proba(X_test)
                else:
                    pred_proba = None

                # Store results
                period_result = {
                    "split_id": i,
                    "train_start": train_idx[0],
                    "train_end": train_idx[-1],
                    "test_start": test_idx[0],
                    "test_end": test_idx[-1],
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "predictions": predictions,
                    "actuals": y_test.values,
                    "test_dates": test_idx,
                    "probabilities": pred_proba,
                }

                # Calculate period metrics
                if fitted_model.model_type == "classifier":
                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                    recall = recall_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                    f1 = f1_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )

                    period_result.update(
                        {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                        }
                    )
                else:
                    # Regression metrics
                    mse = np.mean((predictions - y_test) ** 2)
                    mae = np.mean(np.abs(predictions - y_test))
                    rmse = np.sqrt(mse)

                    period_result.update({"mse": mse, "mae": mae, "rmse": rmse})

                period_results.append(period_result)

                # Accumulate for overall metrics
                all_predictions.extend(predictions)
                all_actuals.extend(y_test.values)
                all_dates.extend(test_idx)

            except Exception as e:
                logger.error(f"Error in split {i+1}: {str(e)}")
                continue

        if len(period_results) == 0:
            raise ValueError("No successful splits completed")

        # Calculate overall performance
        overall_metrics = self._calculate_overall_metrics(
            all_predictions, all_actuals, fitted_model.model_type
        )

        # Store results
        self.results_ = period_results

        result_summary = {
            "total_splits": len(splits),
            "successful_splits": len(period_results),
            "overall_metrics": overall_metrics,
            "period_results": period_results,
            "predictions_series": pd.Series(all_predictions, index=all_dates),
            "actuals_series": pd.Series(all_actuals, index=all_dates),
        }

        logger.info(
            f"Walk-forward analysis completed. {len(period_results)}/{len(splits)} splits successful"
        )

        return result_summary

    def _calculate_overall_metrics(
        self, predictions: List[float], actuals: List[float], model_type: str
    ) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        if model_type == "classifier":
            return {
                "accuracy": accuracy_score(actuals, predictions),
                "precision": precision_score(
                    actuals, predictions, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    actuals, predictions, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    actuals, predictions, average="weighted", zero_division=0
                ),
            }
        else:
            mse = np.mean((predictions - actuals) ** 2)
            return {
                "mse": mse,
                "mae": np.mean(np.abs(predictions - actuals)),
                "rmse": np.sqrt(mse),
                "r2": 1
                - (
                    np.sum((actuals - predictions) ** 2)
                    / np.sum((actuals - np.mean(actuals)) ** 2)
                ),
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of walk-forward performance.

        Returns:
            Dictionary with performance statistics
        """
        if not self.results_:
            raise ValueError("No results available. Run walk_forward_analysis first.")

        # Extract metrics from all periods
        metrics_by_period = {}

        for result in self.results_:
            for metric_name, value in result.items():
                if isinstance(value, (int, float)) and metric_name not in [
                    "split_id",
                    "train_size",
                    "test_size",
                ]:
                    if metric_name not in metrics_by_period:
                        metrics_by_period[metric_name] = []
                    metrics_by_period[metric_name].append(value)

        # Calculate summary statistics
        summary = {}
        for metric_name, values in metrics_by_period.items():
            summary[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
            }

        # Add stability metrics
        summary["stability"] = self._calculate_stability_metrics()

        return summary

    def _calculate_stability_metrics(self) -> Dict[str, float]:
        """Calculate stability metrics for walk-forward performance."""
        if not self.results_:
            return {}

        # Extract primary metric (accuracy for classification, RMSE for regression)
        if "accuracy" in self.results_[0]:
            primary_values = [r["accuracy"] for r in self.results_]
            metric_name = "accuracy"
        elif "rmse" in self.results_[0]:
            primary_values = [r["rmse"] for r in self.results_]
            metric_name = "rmse"
        else:
            return {}

        # Calculate stability metrics
        mean_performance = np.mean(primary_values)
        std_performance = np.std(primary_values)

        # Coefficient of variation (lower is more stable)
        cv = std_performance / mean_performance if mean_performance != 0 else np.inf

        # Percentage of periods above/below mean
        above_mean = np.sum(np.array(primary_values) > mean_performance) / len(
            primary_values
        )

        # Maximum drawdown in performance
        cummax = np.maximum.accumulate(primary_values)
        drawdowns = (cummax - primary_values) / cummax
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        return {
            "coefficient_of_variation": cv,
            "percentage_above_mean": above_mean,
            "max_performance_drawdown": max_drawdown,
            "performance_trend": self._calculate_trend(primary_values),
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in performance over time."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression slope
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope

    def plot_walk_forward_results(self) -> Dict[str, Any]:
        """
        Generate visualization data for walk-forward results.

        Returns:
            Dictionary with plot data
        """
        if not self.results_:
            raise ValueError("No results available. Run walk_forward_analysis first.")

        # Prepare time series data
        dates = []
        performance_values = []

        # Determine primary metric
        primary_metric = "accuracy" if "accuracy" in self.results_[0] else "rmse"

        for result in self.results_:
            dates.append(result["test_end"])
            performance_values.append(result[primary_metric])

        plot_data = {
            "dates": dates,
            "performance": performance_values,
            "metric_name": primary_metric,
            "splits_info": [
                {
                    "split_id": r["split_id"],
                    "train_size": r["train_size"],
                    "test_size": r["test_size"],
                    "train_period": f"{r['train_start']} to {r['train_end']}",
                    "test_period": f"{r['test_start']} to {r['test_end']}",
                }
                for r in self.results_
            ],
        }

        return plot_data


def create_time_series_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    embargo_pct: float = 0.01,
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Create time series splits with purging and embargo.

    Convenience function for creating AFML-compliant time series splits.

    Args:
        data: DataFrame with datetime index
        n_splits: Number of splits
        test_size: Proportion for test set
        embargo_pct: Embargo percentage

    Returns:
        List of (train_index, test_index) tuples
    """
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=n_splits, test_size=test_size, embargo_pct=embargo_pct
    )

    return list(splitter.split(data))
