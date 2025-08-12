"""
Sensitivity analysis implementation for quantitative finance.

This module provides comprehensive sensitivity analysis tools for financial
models and strategies, including parameter sensitivity, feature importance
analysis, and robustness testing.

Key features:
- Parameter sensitivity analysis
- Feature importance sensitivity
- Model robustness testing
- Scenario-based sensitivity
- Gradient-based sensitivity
- Portfolio sensitivity analysis

References:
- AFML methodologies for financial model analysis
- Risk management best practices
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid
import warnings

from ..models.base import BaseModel


logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for financial models and strategies.

    Provides various methods to analyze model sensitivity to parameters,
    features, and market conditions.
    """

    def __init__(
        self,
        perturbation_range: float = 0.1,
        n_permutations: int = 100,
        confidence_level: float = 0.95,
        random_state: int = None,
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            perturbation_range: Range for parameter perturbations (±percentage)
            n_permutations: Number of permutations for importance analysis
            confidence_level: Confidence level for statistical tests
            random_state: Random state for reproducibility
        """
        self.perturbation_range = perturbation_range
        self.n_permutations = n_permutations
        self.confidence_level = confidence_level
        self.random_state = random_state

        # Results storage
        self.parameter_sensitivity_ = {}
        self.feature_importance_ = {}
        self.robustness_results_ = {}

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

    def parameter_sensitivity_analysis(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        parameter_ranges: Dict[str, List[float]] = None,
        performance_metric: str = "score",
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to model parameters.

        Tests how changes in model hyperparameters affect performance.

        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            parameter_ranges: Dictionary of parameter names to value ranges
            performance_metric: Metric to measure sensitivity ('score', 'loss', etc.)

        Returns:
            Dictionary containing parameter sensitivity results
        """
        logger.info("Starting parameter sensitivity analysis")

        if parameter_ranges is None:
            parameter_ranges = self._get_default_parameter_ranges(model)

        # Get baseline performance
        baseline_model = model.clone()
        baseline_model.fit(X, y)
        baseline_score = self._calculate_performance(
            baseline_model, X, y, performance_metric
        )

        sensitivity_results = {
            "baseline_score": baseline_score,
            "parameter_effects": {},
            "parameter_rankings": [],
            "stability_metrics": {},
        }

        # Test each parameter
        for param_name, param_values in parameter_ranges.items():
            logger.info(f"Analyzing parameter: {param_name}")

            param_scores = []
            param_models = []

            for param_value in param_values:
                try:
                    # Create model with modified parameter
                    test_model = model.clone()

                    # Set parameter value
                    if hasattr(test_model, "set_params"):
                        test_model.set_params(**{param_name: param_value})
                    else:
                        setattr(test_model, param_name, param_value)

                    # Train and evaluate
                    test_model.fit(X, y)
                    score = self._calculate_performance(
                        test_model, X, y, performance_metric
                    )

                    param_scores.append(score)
                    param_models.append(test_model)

                except Exception as e:
                    logger.warning(
                        f"Failed to test {param_name}={param_value}: {str(e)}"
                    )
                    param_scores.append(np.nan)
                    param_models.append(None)

            # Calculate sensitivity metrics for this parameter
            param_sensitivity = self._calculate_parameter_sensitivity(
                param_values, param_scores, baseline_score
            )

            sensitivity_results["parameter_effects"][param_name] = {
                "values": param_values,
                "scores": param_scores,
                "sensitivity_metrics": param_sensitivity,
                "score_range": np.nanmax(param_scores) - np.nanmin(param_scores),
                "relative_impact": param_sensitivity.get("normalized_sensitivity", 0),
            }

        # Rank parameters by sensitivity
        sensitivity_results["parameter_rankings"] = (
            self._rank_parameters_by_sensitivity(
                sensitivity_results["parameter_effects"]
            )
        )

        # Calculate overall stability metrics
        sensitivity_results["stability_metrics"] = self._calculate_stability_metrics(
            sensitivity_results["parameter_effects"]
        )

        self.parameter_sensitivity_[performance_metric] = sensitivity_results
        logger.info("Completed parameter sensitivity analysis")

        return sensitivity_results

    def feature_importance_sensitivity(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        importance_methods: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of feature importance across different methods.

        Compares feature importance rankings from multiple methods to assess
        stability and reliability.

        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            importance_methods: List of importance methods to compare

        Returns:
            Dictionary containing feature importance sensitivity results
        """
        logger.info("Starting feature importance sensitivity analysis")

        if importance_methods is None:
            importance_methods = ["permutation", "coefficient", "builtin"]

        importance_results = {
            "methods": {},
            "consistency_metrics": {},
            "stable_features": [],
            "unstable_features": [],
        }

        # Train base model
        base_model = model.clone()
        base_model.fit(X, y)

        # Calculate importance using different methods
        for method in importance_methods:
            try:
                importance_scores = self._calculate_feature_importance(
                    base_model, X, y, method
                )

                importance_results["methods"][method] = {
                    "scores": importance_scores,
                    "ranking": self._rank_features_by_importance(importance_scores),
                    "top_features": self._get_top_features(importance_scores, n=10),
                }

            except Exception as e:
                logger.warning(f"Failed to calculate {method} importance: {str(e)}")
                continue

        # Analyze consistency across methods
        if len(importance_results["methods"]) > 1:
            consistency_metrics = self._analyze_importance_consistency(
                importance_results["methods"]
            )
            importance_results["consistency_metrics"] = consistency_metrics

            # Identify stable vs unstable features
            stable_features, unstable_features = self._identify_stable_features(
                importance_results["methods"]
            )
            importance_results["stable_features"] = stable_features
            importance_results["unstable_features"] = unstable_features

        # Perturbation-based sensitivity
        perturbation_sensitivity = self._feature_perturbation_analysis(base_model, X, y)
        importance_results["perturbation_sensitivity"] = perturbation_sensitivity

        self.feature_importance_["analysis"] = importance_results
        logger.info("Completed feature importance sensitivity analysis")

        return importance_results

    def robustness_testing(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        noise_levels: List[float] = None,
        sample_fractions: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Test model robustness to data quality issues.

        Evaluates how model performance degrades with noise, missing data,
        and reduced sample sizes.

        Args:
            model: Model to test
            X: Feature matrix
            y: Target variable
            noise_levels: List of noise levels to test (as standard deviations)
            sample_fractions: List of sample size fractions to test

        Returns:
            Dictionary containing robustness test results
        """
        logger.info("Starting robustness testing")

        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]

        if sample_fractions is None:
            sample_fractions = [1.0, 0.8, 0.6, 0.4, 0.2]

        # Get baseline performance
        baseline_model = model.clone()
        baseline_model.fit(X, y)
        baseline_score = (
            baseline_model.score(X, y) if hasattr(baseline_model, "score") else 0.0
        )

        robustness_results = {
            "baseline_score": baseline_score,
            "noise_sensitivity": {},
            "sample_size_sensitivity": {},
            "missing_data_sensitivity": {},
            "combined_stress_test": {},
        }

        # Test noise sensitivity
        noise_scores = []
        for noise_level in noise_levels:
            noise_score = self._test_noise_robustness(model, X, y, noise_level)
            noise_scores.append(noise_score)

        robustness_results["noise_sensitivity"] = {
            "noise_levels": noise_levels,
            "scores": noise_scores,
            "degradation_curve": self._calculate_degradation_curve(
                baseline_score, noise_scores
            ),
        }

        # Test sample size sensitivity
        sample_scores = []
        for fraction in sample_fractions:
            sample_score = self._test_sample_size_robustness(model, X, y, fraction)
            sample_scores.append(sample_score)

        robustness_results["sample_size_sensitivity"] = {
            "sample_fractions": sample_fractions,
            "scores": sample_scores,
            "degradation_curve": self._calculate_degradation_curve(
                baseline_score, sample_scores
            ),
        }

        # Test missing data sensitivity
        missing_data_results = self._test_missing_data_robustness(model, X, y)
        robustness_results["missing_data_sensitivity"] = missing_data_results

        # Combined stress test
        stress_test_results = self._combined_stress_test(model, X, y)
        robustness_results["combined_stress_test"] = stress_test_results

        self.robustness_results_["analysis"] = robustness_results
        logger.info("Completed robustness testing")

        return robustness_results

    def scenario_sensitivity_analysis(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        scenarios: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Analyze model sensitivity under different market scenarios.

        Tests model performance when market conditions change according
        to predefined scenarios.

        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            scenarios: Dictionary of scenario name to parameter modifications

        Returns:
            Dictionary containing scenario sensitivity results
        """
        logger.info("Starting scenario sensitivity analysis")

        # Get baseline performance
        baseline_model = model.clone()
        baseline_model.fit(X, y)
        baseline_score = (
            baseline_model.score(X, y) if hasattr(baseline_model, "score") else 0.0
        )

        scenario_results = {
            "baseline_score": baseline_score,
            "scenario_scores": {},
            "worst_case_scenario": None,
            "best_case_scenario": None,
            "scenario_rankings": [],
        }

        scenario_scores = {}

        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Testing scenario: {scenario_name}")

            try:
                # Modify data according to scenario
                X_scenario, y_scenario = self._apply_scenario_modifications(
                    X, y, scenario_params
                )

                # Train and evaluate model
                scenario_model = model.clone()
                scenario_model.fit(X_scenario, y_scenario)
                scenario_score = (
                    scenario_model.score(X_scenario, y_scenario)
                    if hasattr(scenario_model, "score")
                    else 0.0
                )

                scenario_scores[scenario_name] = scenario_score

                scenario_results["scenario_scores"][scenario_name] = {
                    "score": scenario_score,
                    "score_change": scenario_score - baseline_score,
                    "relative_change": (
                        (scenario_score - baseline_score) / baseline_score
                        if baseline_score != 0
                        else 0
                    ),
                    "parameters": scenario_params,
                }

            except Exception as e:
                logger.warning(f"Failed to test scenario {scenario_name}: {str(e)}")
                scenario_scores[scenario_name] = np.nan

        # Identify best and worst case scenarios
        valid_scores = {k: v for k, v in scenario_scores.items() if not np.isnan(v)}

        if valid_scores:
            worst_scenario = min(valid_scores.keys(), key=lambda k: valid_scores[k])
            best_scenario = max(valid_scores.keys(), key=lambda k: valid_scores[k])

            scenario_results["worst_case_scenario"] = {
                "name": worst_scenario,
                "score": valid_scores[worst_scenario],
                "parameters": scenarios[worst_scenario],
            }

            scenario_results["best_case_scenario"] = {
                "name": best_scenario,
                "score": valid_scores[best_scenario],
                "parameters": scenarios[best_scenario],
            }

            # Rank scenarios by performance
            scenario_results["scenario_rankings"] = sorted(
                valid_scores.items(), key=lambda x: x[1], reverse=True
            )

        logger.info("Completed scenario sensitivity analysis")
        return scenario_results

    def _get_default_parameter_ranges(self, model: BaseModel) -> Dict[str, List[float]]:
        """Get default parameter ranges for common model types."""
        # This would be expanded based on model type
        default_ranges = {}

        # Example ranges for common parameters
        if hasattr(model, "learning_rate"):
            default_ranges["learning_rate"] = [0.001, 0.01, 0.05, 0.1, 0.2]

        if hasattr(model, "max_depth"):
            default_ranges["max_depth"] = [3, 5, 7, 10, 15]

        if hasattr(model, "n_estimators"):
            default_ranges["n_estimators"] = [50, 100, 200, 300, 500]

        if hasattr(model, "regularization"):
            default_ranges["regularization"] = [0.001, 0.01, 0.1, 1.0, 10.0]

        return default_ranges

    def _calculate_performance(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series, metric: str
    ) -> float:
        """Calculate model performance using specified metric."""
        if metric == "score" and hasattr(model, "score"):
            return model.score(X, y)
        elif metric == "loss":
            y_pred = model.predict(X)
            return -np.mean((y - y_pred) ** 2)  # Negative MSE for minimization
        else:
            # Default to score
            return model.score(X, y) if hasattr(model, "score") else 0.0

    def _calculate_parameter_sensitivity(
        self, param_values: List[float], scores: List[float], baseline_score: float
    ) -> Dict[str, float]:
        """Calculate sensitivity metrics for a parameter."""
        valid_scores = [s for s in scores if not np.isnan(s)]

        if len(valid_scores) < 2:
            return {"sensitivity": 0.0, "normalized_sensitivity": 0.0}

        # Calculate range of scores
        score_range = max(valid_scores) - min(valid_scores)

        # Calculate parameter range
        param_range = max(param_values) - min(param_values)

        # Sensitivity as score range / parameter range
        sensitivity = score_range / param_range if param_range > 0 else 0.0

        # Normalized sensitivity (relative to baseline)
        normalized_sensitivity = (
            score_range / abs(baseline_score) if baseline_score != 0 else 0.0
        )

        return {
            "sensitivity": sensitivity,
            "normalized_sensitivity": normalized_sensitivity,
            "score_range": score_range,
            "param_range": param_range,
        }

    def _rank_parameters_by_sensitivity(
        self, parameter_effects: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Rank parameters by their sensitivity."""
        rankings = []

        for param_name, effects in parameter_effects.items():
            sensitivity = effects["sensitivity_metrics"].get(
                "normalized_sensitivity", 0
            )
            rankings.append((param_name, sensitivity))

        # Sort by sensitivity (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def _calculate_stability_metrics(
        self, parameter_effects: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate overall model stability metrics."""
        sensitivities = []

        for effects in parameter_effects.values():
            sensitivity = effects["sensitivity_metrics"].get(
                "normalized_sensitivity", 0
            )
            sensitivities.append(sensitivity)

        if not sensitivities:
            return {
                "mean_sensitivity": 0.0,
                "max_sensitivity": 0.0,
                "stability_score": 1.0,
            }

        mean_sensitivity = np.mean(sensitivities)
        max_sensitivity = np.max(sensitivities)

        # Stability score (inverse of sensitivity)
        stability_score = 1.0 / (1.0 + mean_sensitivity)

        return {
            "mean_sensitivity": mean_sensitivity,
            "max_sensitivity": max_sensitivity,
            "stability_score": stability_score,
            "n_parameters_tested": len(sensitivities),
        }

    def _calculate_feature_importance(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series, method: str
    ) -> np.ndarray:
        """Calculate feature importance using specified method."""
        if method == "permutation":
            # Use permutation importance
            result = permutation_importance(
                model,
                X,
                y,
                n_repeats=self.n_permutations,
                random_state=self.random_state,
            )
            return result.importances_mean

        elif method == "coefficient" and hasattr(model, "coef_"):
            # Linear model coefficients
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            return np.abs(coef)

        elif method == "builtin" and hasattr(model, "feature_importances_"):
            # Built-in feature importances (e.g., tree-based models)
            return model.feature_importances_

        else:
            # Fallback: uniform importance
            return np.ones(X.shape[1]) / X.shape[1]

    def _rank_features_by_importance(self, importance_scores: np.ndarray) -> List[int]:
        """Rank features by importance scores."""
        return np.argsort(importance_scores)[::-1].tolist()

    def _get_top_features(
        self, importance_scores: np.ndarray, n: int = 10
    ) -> List[int]:
        """Get indices of top N most important features."""
        return self._rank_features_by_importance(importance_scores)[:n]

    def _analyze_importance_consistency(
        self, importance_methods: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze consistency of feature importance across methods."""
        rankings = {}
        scores = {}

        for method, results in importance_methods.items():
            rankings[method] = results["ranking"]
            scores[method] = results["scores"]

        if len(rankings) < 2:
            return {"consistency_score": 1.0}

        # Calculate rank correlation between methods
        correlations = []
        method_names = list(rankings.keys())

        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]

                # Spearman correlation of rankings
                rank_corr = stats.spearmanr(
                    rankings[method1][:10], rankings[method2][:10]  # Top 10 features
                )[0]

                if not np.isnan(rank_corr):
                    correlations.append(rank_corr)

        consistency_score = np.mean(correlations) if correlations else 0.0

        return {
            "consistency_score": consistency_score,
            "n_comparisons": len(correlations),
            "pairwise_correlations": correlations,
        }

    def _identify_stable_features(
        self, importance_methods: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[int], List[int]]:
        """Identify stable vs unstable features across methods."""
        if len(importance_methods) < 2:
            return [], []

        # Count how often each feature appears in top 10
        feature_counts = {}
        n_methods = len(importance_methods)

        for method, results in importance_methods.items():
            top_features = results["top_features"]
            for feature_idx in top_features:
                feature_counts[feature_idx] = feature_counts.get(feature_idx, 0) + 1

        # Stable features appear in most methods
        stable_threshold = n_methods * 0.7  # 70% of methods

        stable_features = [
            feature_idx
            for feature_idx, count in feature_counts.items()
            if count >= stable_threshold
        ]

        unstable_features = [
            feature_idx
            for feature_idx, count in feature_counts.items()
            if count < stable_threshold and count > 0
        ]

        return stable_features, unstable_features

    def _feature_perturbation_analysis(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze sensitivity to individual feature perturbations."""
        baseline_score = model.score(X, y) if hasattr(model, "score") else 0.0

        perturbation_effects = {}

        for i, feature_name in enumerate(X.columns):
            # Perturb this feature
            X_perturbed = X.copy()

            # Add noise to feature
            noise_std = X.iloc[:, i].std() * self.perturbation_range
            noise = np.random.normal(0, noise_std, len(X))
            X_perturbed.iloc[:, i] += noise

            # Evaluate perturbed model
            perturbed_score = (
                model.score(X_perturbed, y) if hasattr(model, "score") else 0.0
            )

            perturbation_effects[feature_name] = {
                "original_score": baseline_score,
                "perturbed_score": perturbed_score,
                "score_change": perturbed_score - baseline_score,
                "sensitivity": (
                    abs(perturbed_score - baseline_score) / abs(baseline_score)
                    if baseline_score != 0
                    else 0
                ),
            }

        # Rank features by perturbation sensitivity
        sensitivity_ranking = sorted(
            perturbation_effects.items(),
            key=lambda x: x[1]["sensitivity"],
            reverse=True,
        )

        return {
            "feature_effects": perturbation_effects,
            "sensitivity_ranking": sensitivity_ranking,
            "most_sensitive_feature": (
                sensitivity_ranking[0][0] if sensitivity_ranking else None
            ),
        }

    def _test_noise_robustness(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series, noise_level: float
    ) -> float:
        """Test model robustness to feature noise."""
        # Add noise to features
        X_noisy = X.copy()
        for col in X.columns:
            noise_std = X[col].std() * noise_level
            noise = np.random.normal(0, noise_std, len(X))
            X_noisy[col] += noise

        # Train and evaluate model
        noisy_model = model.clone()
        noisy_model.fit(X_noisy, y)

        return noisy_model.score(X_noisy, y) if hasattr(noisy_model, "score") else 0.0

    def _test_sample_size_robustness(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series, sample_fraction: float
    ) -> float:
        """Test model robustness to reduced sample size."""
        # Sample data
        n_samples = int(len(X) * sample_fraction)
        sample_indices = np.random.choice(len(X), size=n_samples, replace=False)

        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]

        # Train and evaluate model
        sample_model = model.clone()
        sample_model.fit(X_sample, y_sample)

        return (
            sample_model.score(X_sample, y_sample)
            if hasattr(sample_model, "score")
            else 0.0
        )

    def _test_missing_data_robustness(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Test model robustness to missing data."""
        missing_data_results = {}
        missing_fractions = [0.05, 0.1, 0.15, 0.2]

        for missing_fraction in missing_fractions:
            # Introduce missing data
            X_missing = X.copy()
            n_missing = int(X.size * missing_fraction)

            # Randomly select positions to make missing
            missing_positions = np.random.choice(X.size, size=n_missing, replace=False)

            for pos in missing_positions:
                row_idx = pos // X.shape[1]
                col_idx = pos % X.shape[1]
                X_missing.iloc[row_idx, col_idx] = np.nan

            # Fill missing values (simple mean imputation)
            X_missing = X_missing.fillna(X_missing.mean())

            # Train and evaluate model
            missing_model = model.clone()
            missing_model.fit(X_missing, y)
            score = (
                missing_model.score(X_missing, y)
                if hasattr(missing_model, "score")
                else 0.0
            )

            missing_data_results[missing_fraction] = score

        return {
            "missing_fractions": missing_fractions,
            "scores": list(missing_data_results.values()),
            "missing_data_effects": missing_data_results,
        }

    def _combined_stress_test(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Combined stress test with multiple degradation factors."""
        stress_scenarios = {
            "mild_stress": {
                "noise_level": 0.05,
                "sample_fraction": 0.8,
                "missing_fraction": 0.05,
            },
            "moderate_stress": {
                "noise_level": 0.1,
                "sample_fraction": 0.6,
                "missing_fraction": 0.1,
            },
            "high_stress": {
                "noise_level": 0.2,
                "sample_fraction": 0.4,
                "missing_fraction": 0.15,
            },
        }

        stress_results = {}

        for scenario_name, params in stress_scenarios.items():
            try:
                # Apply all stress factors
                X_stressed = X.copy()

                # Add noise
                for col in X.columns:
                    noise_std = X[col].std() * params["noise_level"]
                    noise = np.random.normal(0, noise_std, len(X))
                    X_stressed[col] += noise

                # Reduce sample size
                n_samples = int(len(X) * params["sample_fraction"])
                sample_indices = np.random.choice(len(X), size=n_samples, replace=False)
                X_stressed = X_stressed.iloc[sample_indices]
                y_stressed = y.iloc[sample_indices]

                # Add missing data
                n_missing = int(X_stressed.size * params["missing_fraction"])
                missing_positions = np.random.choice(
                    X_stressed.size, size=n_missing, replace=False
                )

                for pos in missing_positions:
                    row_idx = pos // X_stressed.shape[1]
                    col_idx = pos % X_stressed.shape[1]
                    X_stressed.iloc[row_idx, col_idx] = np.nan

                # Fill missing values
                X_stressed = X_stressed.fillna(X_stressed.mean())

                # Train and evaluate
                stressed_model = model.clone()
                stressed_model.fit(X_stressed, y_stressed)
                score = (
                    stressed_model.score(X_stressed, y_stressed)
                    if hasattr(stressed_model, "score")
                    else 0.0
                )

                stress_results[scenario_name] = {"score": score, "parameters": params}

            except Exception as e:
                logger.warning(f"Stress test {scenario_name} failed: {str(e)}")
                stress_results[scenario_name] = {"score": np.nan, "parameters": params}

        return stress_results

    def _calculate_degradation_curve(
        self, baseline_score: float, test_scores: List[float]
    ) -> List[float]:
        """Calculate performance degradation curve."""
        return [
            (baseline_score - score) / baseline_score if baseline_score != 0 else 0
            for score in test_scores
        ]

    def _apply_scenario_modifications(
        self, X: pd.DataFrame, y: pd.Series, scenario_params: Dict[str, float]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply scenario-specific modifications to data."""
        X_modified = X.copy()
        y_modified = y.copy()

        # Apply feature scaling modifications
        if "feature_scale" in scenario_params:
            scale_factor = scenario_params["feature_scale"]
            X_modified = X_modified * scale_factor

        # Apply target shift
        if "target_shift" in scenario_params:
            shift = scenario_params["target_shift"]
            y_modified = y_modified + shift

        # Apply volatility changes
        if "volatility_multiplier" in scenario_params:
            vol_mult = scenario_params["volatility_multiplier"]
            for col in X.columns:
                col_std = X[col].std()
                # Ensure positive scale parameter for normal distribution
                noise_scale = abs(col_std * (vol_mult - 1))
                if noise_scale > 0:
                    noise = np.random.normal(0, noise_scale, len(X))
                    X_modified[col] += noise

        return X_modified, y_modified

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all sensitivity analyses."""
        if not any(
            [
                self.parameter_sensitivity_,
                self.feature_importance_,
                self.robustness_results_,
            ]
        ):
            raise ValueError(
                "No sensitivity analysis results available. Run analyses first."
            )

        summary = {
            "parameter_sensitivity": self.parameter_sensitivity_,
            "feature_importance": self.feature_importance_,
            "robustness_results": self.robustness_results_,
            "overall_assessment": self._generate_overall_assessment(),
        }

        return summary

    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall model sensitivity assessment."""
        assessment = {
            "model_stability": "unknown",
            "key_risk_factors": [],
            "recommendations": [],
        }

        # Assess parameter stability
        if self.parameter_sensitivity_:
            param_results = list(self.parameter_sensitivity_.values())[0]
            stability_score = param_results.get("stability_metrics", {}).get(
                "stability_score", 0.5
            )

            if stability_score > 0.8:
                assessment["model_stability"] = "high"
            elif stability_score > 0.6:
                assessment["model_stability"] = "moderate"
            else:
                assessment["model_stability"] = "low"

        # Identify key risk factors
        risk_factors = []

        if self.parameter_sensitivity_:
            param_results = list(self.parameter_sensitivity_.values())[0]
            top_sensitive_params = param_results.get("parameter_rankings", [])[:3]
            risk_factors.extend(
                [f"Parameter: {param}" for param, _ in top_sensitive_params]
            )

        if self.feature_importance_:
            unstable_features = self.feature_importance_.get("analysis", {}).get(
                "unstable_features", []
            )
            if unstable_features:
                risk_factors.append(
                    f"Unstable features: {len(unstable_features)} features"
                )

        assessment["key_risk_factors"] = risk_factors

        # Generate recommendations
        recommendations = []

        if assessment["model_stability"] == "low":
            recommendations.append("Consider regularization or ensemble methods")
            recommendations.append("Perform additional hyperparameter tuning")

        if risk_factors:
            recommendations.append("Monitor identified risk factors closely")
            recommendations.append("Consider robust validation strategies")

        assessment["recommendations"] = recommendations

        return assessment

    def plot_sensitivity_results(self) -> Dict[str, Any]:
        """Generate plot data for sensitivity analysis visualization."""
        if not any(
            [
                self.parameter_sensitivity_,
                self.feature_importance_,
                self.robustness_results_,
            ]
        ):
            raise ValueError("No results available for plotting")

        plot_data = {
            "parameter_sensitivity": {},
            "feature_importance": {},
            "robustness_curves": {},
        }

        # Parameter sensitivity plots
        if self.parameter_sensitivity_:
            param_results = list(self.parameter_sensitivity_.values())[0]
            plot_data["parameter_sensitivity"] = {
                "parameter_effects": param_results.get("parameter_effects", {}),
                "rankings": param_results.get("parameter_rankings", []),
            }

        # Feature importance plots
        if self.feature_importance_:
            plot_data["feature_importance"] = self.feature_importance_.get(
                "analysis", {}
            )

        # Robustness curves
        if self.robustness_results_:
            robustness_data = self.robustness_results_.get("analysis", {})
            plot_data["robustness_curves"] = {
                "noise_sensitivity": robustness_data.get("noise_sensitivity", {}),
                "sample_size_sensitivity": robustness_data.get(
                    "sample_size_sensitivity", {}
                ),
                "missing_data_sensitivity": robustness_data.get(
                    "missing_data_sensitivity", {}
                ),
            }

        return plot_data


def create_default_scenarios() -> Dict[str, Dict[str, float]]:
    """Create default market scenario definitions."""
    scenarios = {
        "bull_market": {
            "feature_scale": 1.2,
            "target_shift": 0.02,
            "volatility_multiplier": 0.8,
        },
        "bear_market": {
            "feature_scale": 0.8,
            "target_shift": -0.03,
            "volatility_multiplier": 1.5,
        },
        "high_volatility": {
            "feature_scale": 1.0,
            "target_shift": 0.0,
            "volatility_multiplier": 2.0,
        },
        "low_volatility": {
            "feature_scale": 1.0,
            "target_shift": 0.0,
            "volatility_multiplier": 0.5,
        },
        "market_crash": {
            "feature_scale": 0.6,
            "target_shift": -0.1,
            "volatility_multiplier": 3.0,
        },
    }

    return scenarios
