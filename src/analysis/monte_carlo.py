"""
Monte Carlo analysis implementation for quantitative finance.

This module implements Monte Carlo simulation techniques for backtesting,
stress testing, and risk analysis based on AFML (Advances in Financial
Machine Learning) methodologies.

Key features:
- Monte Carlo cross-validation for model validation
- Synthetic data generation for backtesting
- Bootstrap sampling with purging
- Scenario analysis and stress testing
- Performance distribution analysis

References:
- AFML Chapter 12: Cross-Validation through Monte Carlo
- AFML Chapter 13: Backtesting through Synthetic Data
- AFML Chapter 14: Backtest Statistics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import jarque_bera, normaltest
from sklearn.base import clone
from sklearn.utils import resample
import warnings

from ..models.base import BaseModel


logger = logging.getLogger(__name__)


class MonteCarloAnalyzer:
    """
    Comprehensive Monte Carlo analysis for financial models.

    Implements various Monte Carlo techniques including cross-validation,
    bootstrap sampling, synthetic data generation, and performance analysis.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_levels: List[float] = None,
        random_state: int = None,
        purging_enabled: bool = True,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize Monte Carlo analyzer.

        Args:
            n_simulations: Number of Monte Carlo simulations
            confidence_levels: List of confidence levels for analysis
            random_state: Random state for reproducibility
            purging_enabled: Whether to enable sample purging
            embargo_pct: Percentage of data to embargo after each sample
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.random_state = random_state
        self.purging_enabled = purging_enabled
        self.embargo_pct = embargo_pct

        # Results storage
        self.simulation_results_ = []
        self.bootstrap_results_ = []
        self.synthetic_results_ = []

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

    def monte_carlo_cross_validation(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        n_bootstrap: int = None,
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo cross-validation with optional purging.

        Based on AFML Chapter 12 methodology for proper cross-validation
        in financial time series to avoid data leakage.

        Args:
            model: Model to validate
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            n_bootstrap: Number of bootstrap samples (defaults to n_simulations)

        Returns:
            Dictionary containing validation results and statistics
        """
        logger.info(f"Starting Monte Carlo CV with {self.n_simulations} simulations")

        if n_bootstrap is None:
            n_bootstrap = self.n_simulations

        n_samples = len(X)
        test_samples = int(n_samples * test_size)

        results = {
            "scores": [],
            "predictions": [],
            "feature_importance": [],
            "model_params": [],
        }

        for sim in range(n_bootstrap):
            try:
                # Generate bootstrap sample with optional purging
                if self.purging_enabled:
                    train_idx, test_idx = self._purged_bootstrap_sample(
                        n_samples, test_samples
                    )
                else:
                    train_idx, test_idx = self._simple_bootstrap_sample(
                        n_samples, test_samples
                    )

                # Prepare data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Clone and train model
                model_clone = model.clone()
                model_clone.fit(X_train, y_train)

                # Make predictions
                y_pred = model_clone.predict(X_test)

                # Calculate scores
                score = self._calculate_model_score(model_clone, X_test, y_test)

                # Store results
                results["scores"].append(score)
                results["predictions"].append(
                    {"actual": y_test.values, "predicted": y_pred, "test_idx": test_idx}
                )

                # Store feature importance if available
                if hasattr(model_clone, "feature_importances_"):
                    results["feature_importance"].append(
                        model_clone.feature_importances_
                    )

                # Store model parameters if available
                if hasattr(model_clone, "get_params"):
                    results["model_params"].append(model_clone.get_params())

            except Exception as e:
                logger.warning(f"Simulation {sim} failed: {str(e)}")
                continue

        # Calculate statistics
        scores = np.array(results["scores"])
        cv_results = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "scores": scores,
            "confidence_intervals": self._calculate_confidence_intervals(scores),
            "score_distribution": self._analyze_score_distribution(scores),
            "n_successful_simulations": len(scores),
            "raw_results": results,
        }

        self.simulation_results_.append(cv_results)
        logger.info(f"Completed {len(scores)} successful simulations")

        return cv_results

    def bootstrap_performance_analysis(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        block_size: int = None,
    ) -> Dict[str, Any]:
        """
        Perform bootstrap analysis of performance metrics.

        Analyzes the distribution of performance metrics using bootstrap
        resampling to estimate confidence intervals and significance.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns for comparison
            block_size: Block size for block bootstrap (preserves autocorrelation)

        Returns:
            Dictionary containing bootstrap performance analysis
        """
        logger.info("Starting bootstrap performance analysis")

        if block_size is None:
            block_size = max(1, int(np.sqrt(len(returns))))

        bootstrap_metrics = {
            "sharpe_ratio": [],
            "sortino_ratio": [],
            "max_drawdown": [],
            "calmar_ratio": [],
            "var_95": [],
            "cvar_95": [],
            "skewness": [],
            "kurtosis": [],
        }

        if benchmark_returns is not None:
            bootstrap_metrics.update(
                {"alpha": [], "beta": [], "information_ratio": [], "tracking_error": []}
            )

        for sim in range(self.n_simulations):
            try:
                # Generate bootstrap sample
                if block_size > 1:
                    boot_returns = self._block_bootstrap_sample(returns, block_size)
                    if benchmark_returns is not None:
                        boot_benchmark = self._block_bootstrap_sample(
                            benchmark_returns, block_size
                        )
                else:
                    boot_returns = self._simple_bootstrap_resample(returns)
                    if benchmark_returns is not None:
                        boot_benchmark = self._simple_bootstrap_resample(
                            benchmark_returns
                        )

                # Calculate metrics for bootstrap sample
                metrics = self._calculate_performance_metrics(boot_returns)

                for metric, value in metrics.items():
                    if metric in bootstrap_metrics:
                        bootstrap_metrics[metric].append(value)

                # Calculate relative metrics if benchmark provided
                if benchmark_returns is not None:
                    relative_metrics = self._calculate_relative_metrics(
                        boot_returns, boot_benchmark
                    )
                    for metric, value in relative_metrics.items():
                        if metric in bootstrap_metrics:
                            bootstrap_metrics[metric].append(value)

            except Exception as e:
                logger.warning(f"Bootstrap simulation {sim} failed: {str(e)}")
                continue

        # Analyze bootstrap distributions
        analysis_results = {}
        for metric, values in bootstrap_metrics.items():
            if values:
                values = np.array(values)
                analysis_results[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "confidence_intervals": self._calculate_confidence_intervals(
                        values
                    ),
                    "distribution_stats": self._analyze_score_distribution(values),
                    "raw_values": values,
                }

        self.bootstrap_results_.append(analysis_results)
        logger.info("Completed bootstrap performance analysis")

        return analysis_results

    def synthetic_data_backtesting(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        n_synthetic_datasets: int = None,
    ) -> Dict[str, Any]:
        """
        Perform backtesting using synthetic data generation.

        Based on AFML Chapter 13 methodology for generating synthetic
        datasets that preserve statistical properties of original data.

        Args:
            model: Model to test
            X: Original feature matrix
            y: Original target variable
            n_synthetic_datasets: Number of synthetic datasets to generate

        Returns:
            Dictionary containing synthetic backtesting results
        """
        logger.info("Starting synthetic data backtesting")

        if n_synthetic_datasets is None:
            n_synthetic_datasets = min(100, self.n_simulations)

        synthetic_results = {
            "original_performance": None,
            "synthetic_performances": [],
            "feature_stability": [],
            "prediction_consistency": [],
        }

        # Train on original data and get baseline performance
        original_model = model.clone()
        original_model.fit(X, y)
        original_pred = original_model.predict(X)
        original_score = self._calculate_model_score(original_model, X, y)
        synthetic_results["original_performance"] = original_score

        for sim in range(n_synthetic_datasets):
            try:
                # Generate synthetic dataset
                X_synthetic, y_synthetic = self._generate_synthetic_data(X, y)

                # Train model on synthetic data
                synthetic_model = model.clone()
                synthetic_model.fit(X_synthetic, y_synthetic)

                # Test on original data
                synthetic_pred = synthetic_model.predict(X)
                synthetic_score = self._calculate_model_score(synthetic_model, X, y)

                synthetic_results["synthetic_performances"].append(synthetic_score)

                # Analyze feature importance consistency
                if hasattr(original_model, "feature_importances_") and hasattr(
                    synthetic_model, "feature_importances_"
                ):
                    importance_corr = np.corrcoef(
                        original_model.feature_importances_,
                        synthetic_model.feature_importances_,
                    )[0, 1]
                    synthetic_results["feature_stability"].append(importance_corr)

                # Analyze prediction consistency
                pred_corr = np.corrcoef(original_pred, synthetic_pred)[0, 1]
                synthetic_results["prediction_consistency"].append(pred_corr)

            except Exception as e:
                logger.warning(f"Synthetic simulation {sim} failed: {str(e)}")
                continue

        # Calculate summary statistics
        synthetic_scores = np.array(synthetic_results["synthetic_performances"])

        summary = {
            "original_score": original_score,
            "synthetic_mean_score": np.mean(synthetic_scores),
            "synthetic_std_score": np.std(synthetic_scores),
            "score_degradation": original_score - np.mean(synthetic_scores),
            "synthetic_confidence_intervals": self._calculate_confidence_intervals(
                synthetic_scores
            ),
            "feature_stability_mean": (
                np.mean(synthetic_results["feature_stability"])
                if synthetic_results["feature_stability"]
                else None
            ),
            "prediction_consistency_mean": (
                np.mean(synthetic_results["prediction_consistency"])
                if synthetic_results["prediction_consistency"]
                else None
            ),
            "raw_results": synthetic_results,
        }

        self.synthetic_results_.append(summary)
        logger.info("Completed synthetic data backtesting")

        return summary

    def scenario_analysis(
        self,
        returns: pd.Series,
        scenarios: Dict[str, Dict[str, float]],
        correlation_matrix: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo scenario analysis.

        Simulates portfolio performance under various market scenarios
        using Monte Carlo methods.

        Args:
            returns: Historical returns
            scenarios: Dictionary of scenarios with parameter modifications
            correlation_matrix: Asset correlation matrix

        Returns:
            Dictionary containing scenario analysis results
        """
        logger.info("Starting Monte Carlo scenario analysis")

        scenario_results = {}

        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Analyzing scenario: {scenario_name}")

            scenario_returns = []

            for sim in range(self.n_simulations):
                try:
                    # Generate scenario-specific returns
                    simulated_returns = self._simulate_scenario_returns(
                        returns, scenario_params, correlation_matrix
                    )

                    # Calculate performance metrics
                    performance = self._calculate_performance_metrics(simulated_returns)
                    scenario_returns.append(performance)

                except Exception as e:
                    logger.warning(
                        f"Scenario {scenario_name} sim {sim} failed: {str(e)}"
                    )
                    continue

            # Aggregate scenario results
            if scenario_returns:
                scenario_df = pd.DataFrame(scenario_returns)
                scenario_results[scenario_name] = {
                    "summary_stats": scenario_df.describe(),
                    "confidence_intervals": {
                        metric: self._calculate_confidence_intervals(values)
                        for metric, values in scenario_df.items()
                    },
                    "raw_simulations": scenario_returns,
                }

        return scenario_results

    def _purged_bootstrap_sample(
        self, n_samples: int, test_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate purged bootstrap sample to avoid data leakage."""
        # Check if we have enough samples
        if test_samples >= n_samples:
            raise ValueError(
                f"test_samples ({test_samples}) must be less than n_samples ({n_samples})"
            )

        # Select test indices
        max_start = n_samples - test_samples
        if max_start <= 0:
            raise ValueError("Not enough samples for the requested test size")

        test_start = np.random.randint(0, max_start)
        test_idx = np.arange(test_start, test_start + test_samples)

        # Calculate embargo period
        embargo_size = int(n_samples * self.embargo_pct)

        # Create training indices excluding test period and embargo
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False

        # Apply embargo before test period
        embargo_start = max(0, test_start - embargo_size)
        train_mask[embargo_start:test_start] = False

        # Apply embargo after test period
        embargo_end = min(n_samples, test_start + test_samples + embargo_size)
        train_mask[test_start + test_samples : embargo_end] = False

        train_idx = np.where(train_mask)[0]

        # Ensure we have some training data
        if len(train_idx) == 0:
            raise ValueError("No training samples available after purging and embargo")

        return train_idx, test_idx

    def _simple_bootstrap_sample(
        self, n_samples: int, test_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simple bootstrap sample without purging."""
        indices = np.arange(n_samples)
        test_idx = np.random.choice(indices, size=test_samples, replace=False)
        train_idx = np.setdiff1d(indices, test_idx)

        return train_idx, test_idx

    def _block_bootstrap_sample(self, data: pd.Series, block_size: int) -> pd.Series:
        """Generate block bootstrap sample preserving autocorrelation."""
        if len(data) < block_size:
            # If data is smaller than block size, just return resampled data
            return self._simple_bootstrap_resample(data)

        n_blocks = int(np.ceil(len(data) / block_size))
        bootstrap_data = []

        for _ in range(n_blocks):
            max_start = max(1, len(data) - block_size + 1)
            start_idx = np.random.randint(0, max_start)
            end_idx = min(start_idx + block_size, len(data))
            block = data.iloc[start_idx:end_idx]
            bootstrap_data.extend(block.values)

        # Trim to original length
        bootstrap_data = bootstrap_data[: len(data)]

        return pd.Series(bootstrap_data, index=data.index[: len(bootstrap_data)])

    def _simple_bootstrap_resample(self, data: pd.Series) -> pd.Series:
        """Generate simple bootstrap resample."""
        bootstrap_indices = np.random.choice(len(data), size=len(data), replace=True)
        return data.iloc[bootstrap_indices].reset_index(drop=True)

    def _generate_synthetic_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic dataset preserving statistical properties."""
        n_samples = len(X)

        # For features, use multivariate normal with estimated covariance
        feature_means = X.mean()
        feature_cov = X.cov()

        # Handle potential singular covariance matrix
        try:
            synthetic_features = np.random.multivariate_normal(
                feature_means, feature_cov, size=n_samples
            )
        except np.linalg.LinAlgError:
            # Fallback to independent features if covariance is singular
            synthetic_features = np.random.normal(
                feature_means.values.reshape(1, -1),
                X.std().values.reshape(1, -1),
                size=(n_samples, len(X.columns)),
            )

        X_synthetic = pd.DataFrame(synthetic_features, columns=X.columns, index=X.index)

        # For target, preserve distribution shape
        if y.dtype == "object" or len(y.unique()) < 10:
            # Categorical target
            y_synthetic = np.random.choice(y.values, size=n_samples)
        else:
            # Continuous target - sample from empirical distribution
            y_synthetic = np.random.choice(y.values, size=n_samples)

        y_synthetic = pd.Series(y_synthetic, index=y.index, name=y.name)

        return X_synthetic, y_synthetic

    def _simulate_scenario_returns(
        self,
        returns: pd.Series,
        scenario_params: Dict[str, float],
        correlation_matrix: pd.DataFrame = None,
    ) -> pd.Series:
        """Simulate returns under specific scenario parameters."""
        # Extract scenario parameters
        mean_adjustment = scenario_params.get("mean_adjustment", 0.0)
        vol_multiplier = scenario_params.get("volatility_multiplier", 1.0)
        skew_adjustment = scenario_params.get("skewness_adjustment", 0.0)

        # Calculate base statistics
        original_mean = returns.mean()
        original_std = returns.std()

        # Adjust parameters according to scenario
        scenario_mean = original_mean + mean_adjustment
        scenario_std = original_std * vol_multiplier

        # Generate scenario returns
        n_periods = len(returns)

        if abs(skew_adjustment) < 1e-6:
            # Normal distribution
            scenario_returns = np.random.normal(scenario_mean, scenario_std, n_periods)
        else:
            # Skewed distribution using skewed normal
            from scipy.stats import skewnorm

            scenario_returns = skewnorm.rvs(
                a=skew_adjustment, loc=scenario_mean, scale=scenario_std, size=n_periods
            )

        return pd.Series(scenario_returns, index=returns.index)

    def _calculate_model_score(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series
    ) -> float:
        """Calculate appropriate score for model type."""
        if hasattr(model, "model_type") and model.model_type == "regressor":
            y_pred = model.predict(X)
            return -np.mean((y - y_pred) ** 2)  # Negative MSE
        else:
            # Classification
            return model.score(X, y) if hasattr(model, "score") else 0.0

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        # Basic statistics
        metrics["total_return"] = (1 + returns).prod() - 1
        metrics["annualized_return"] = (1 + returns.mean()) ** 252 - 1
        metrics["volatility"] = returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        if metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = (
                metrics["annualized_return"] / metrics["volatility"]
            )
        else:
            metrics["sharpe_ratio"] = 0.0

        # Downside metrics
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std() * np.sqrt(252)
            if downside_std > 0:
                metrics["sortino_ratio"] = metrics["annualized_return"] / downside_std
            else:
                metrics["sortino_ratio"] = 0.0
        else:
            metrics["sortino_ratio"] = np.inf

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        metrics["max_drawdown"] = drawdown.min()
        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / abs(metrics["max_drawdown"])
            if metrics["max_drawdown"] != 0
            else 0.0
        )

        # Risk metrics
        metrics["var_95"] = returns.quantile(0.05)
        metrics["cvar_95"] = returns[returns <= metrics["var_95"]].mean()

        # Distribution metrics
        metrics["skewness"] = returns.skew()
        metrics["kurtosis"] = returns.kurtosis()

        return metrics

    def _calculate_relative_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate metrics relative to benchmark."""
        excess_returns = returns - benchmark_returns

        metrics = {}
        metrics["alpha"] = excess_returns.mean() * 252

        # Beta calculation
        if benchmark_returns.var() > 0:
            metrics["beta"] = (
                np.cov(returns, benchmark_returns)[0, 1] / benchmark_returns.var()
            )
        else:
            metrics["beta"] = 0.0

        # Information ratio
        tracking_error = excess_returns.std() * np.sqrt(252)
        metrics["tracking_error"] = tracking_error

        if tracking_error > 0:
            metrics["information_ratio"] = metrics["alpha"] / tracking_error
        else:
            metrics["information_ratio"] = 0.0

        return metrics

    def _calculate_confidence_intervals(
        self, values: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for given values."""
        intervals = {}

        for confidence in self.confidence_levels:
            alpha = 1 - confidence
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            intervals[f"{confidence:.0%}"] = (lower, upper)

        return intervals

    def _analyze_score_distribution(self, scores: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of scores."""
        analysis = {}

        # Normality tests
        try:
            jb_stat, jb_pvalue = jarque_bera(scores)
            analysis["jarque_bera"] = {"statistic": jb_stat, "p_value": jb_pvalue}

            nt_stat, nt_pvalue = normaltest(scores)
            analysis["normality_test"] = {"statistic": nt_stat, "p_value": nt_pvalue}
        except Exception:
            analysis["normality_tests"] = "Failed"

        # Percentiles
        analysis["percentiles"] = {
            "5%": np.percentile(scores, 5),
            "25%": np.percentile(scores, 25),
            "50%": np.percentile(scores, 50),
            "75%": np.percentile(scores, 75),
            "95%": np.percentile(scores, 95),
        }

        return analysis

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all Monte Carlo analyses."""
        if not any(
            [self.simulation_results_, self.bootstrap_results_, self.synthetic_results_]
        ):
            raise ValueError("No analysis results available. Run analyses first.")

        summary = {
            "monte_carlo_cv": self.simulation_results_,
            "bootstrap_analysis": self.bootstrap_results_,
            "synthetic_backtesting": self.synthetic_results_,
            "analysis_summary": {
                "total_simulations": self.n_simulations,
                "confidence_levels": self.confidence_levels,
                "purging_enabled": self.purging_enabled,
                "embargo_percentage": self.embargo_pct,
            },
        }

        return summary

    def plot_monte_carlo_results(self) -> Dict[str, Any]:
        """Generate plot data for Monte Carlo results visualization."""
        if not any(
            [self.simulation_results_, self.bootstrap_results_, self.synthetic_results_]
        ):
            raise ValueError("No results available for plotting")

        plot_data = {
            "simulation_scores": [],
            "bootstrap_distributions": {},
            "synthetic_comparisons": {},
        }

        # Extract simulation scores
        if self.simulation_results_:
            for result in self.simulation_results_:
                plot_data["simulation_scores"].extend(result["scores"])

        # Extract bootstrap distributions
        if self.bootstrap_results_:
            for result in self.bootstrap_results_:
                for metric, data in result.items():
                    if "raw_values" in data:
                        plot_data["bootstrap_distributions"][metric] = data[
                            "raw_values"
                        ]

        # Extract synthetic comparisons
        if self.synthetic_results_:
            for result in self.synthetic_results_:
                plot_data["synthetic_comparisons"] = {
                    "original_score": result["original_score"],
                    "synthetic_scores": result["raw_results"]["synthetic_performances"],
                    "feature_stability": result["raw_results"]["feature_stability"],
                    "prediction_consistency": result["raw_results"][
                        "prediction_consistency"
                    ],
                }

        return plot_data


def monte_carlo_permutation_test(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_permutations: int = 1000,
    metric: str = "accuracy",
) -> Dict[str, Any]:
    """
    Perform Monte Carlo permutation test for model significance.

    Tests the null hypothesis that the model has no predictive power
    by comparing performance on real vs. permuted targets.

    Args:
        model: Model to test
        X: Feature matrix
        y: Target variable
        n_permutations: Number of permutation tests
        metric: Metric to use for comparison

    Returns:
        Dictionary containing permutation test results
    """
    # Train on original data
    original_model = model.clone()
    original_model.fit(X, y)
    original_score = (
        original_model.score(X, y) if hasattr(original_model, "score") else 0.0
    )

    # Permutation scores
    permutation_scores = []

    for i in range(n_permutations):
        # Permute target
        y_permuted = y.sample(frac=1.0).reset_index(drop=True)
        y_permuted.index = y.index  # Restore original index

        # Train on permuted data
        perm_model = model.clone()
        perm_model.fit(X, y_permuted)
        perm_score = (
            perm_model.score(X, y_permuted) if hasattr(perm_model, "score") else 0.0
        )

        permutation_scores.append(perm_score)

    permutation_scores = np.array(permutation_scores)

    # Calculate p-value
    p_value = float(np.mean(permutation_scores >= original_score))

    return {
        "original_score": original_score,
        "permutation_scores": permutation_scores,
        "p_value": p_value,
        "is_significant": bool(p_value < 0.05),
        "permutation_mean": np.mean(permutation_scores),
        "permutation_std": np.std(permutation_scores),
    }
