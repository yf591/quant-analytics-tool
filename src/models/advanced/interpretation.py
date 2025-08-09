"""
Model Interpretation Tools for Financial Machine Learning

This module provides comprehensive model interpretation tools specifically
designed for financial models, including feature importance analysis,
SHAP values, permutation importance, and financial-specific explanations.

Features:
- Feature importance analysis with financial context
- SHAP (SHapley Additive exPlanations) integration
- Permutation importance for robust feature evaluation
- Partial dependence plots for financial features
- Model decision boundary visualization
- Time series specific interpretation methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class InterpretationConfig:
    """Configuration for model interpretation."""

    # Feature importance parameters
    importance_threshold: float = 0.01
    max_features_display: int = 20

    # Permutation importance parameters
    n_repeats: int = 10
    random_state: int = 42

    # SHAP parameters
    max_shap_samples: int = 1000
    shap_explainer_type: str = "auto"  # 'tree', 'linear', 'kernel', 'auto'

    # Visualization parameters
    figure_size: Tuple[int, int] = (12, 8)
    color_palette: str = "viridis"

    # Financial context
    feature_categories: Dict[str, List[str]] = None


class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis for financial models."""

    def __init__(self, config: InterpretationConfig):
        self.config = config
        self.importance_results = {}
        self.feature_names = None

    def analyze_tree_importance(
        self, model: BaseEstimator, feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Analyze feature importance for tree-based models.

        Args:
            model: Fitted tree-based model
            feature_names: List of feature names

        Returns:
            Dictionary of feature importances
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")

        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))

        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    def analyze_permutation_importance(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        scoring: str = "accuracy",
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze permutation importance.

        Args:
            model: Fitted model
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            scoring: Scoring metric

        Returns:
            Dictionary with mean and std of permutation importances
        """
        perm_importance = permutation_importance(
            model,
            X,
            y,
            n_repeats=self.config.n_repeats,
            random_state=self.config.random_state,
            scoring=scoring,
            n_jobs=-1,
        )

        results = {}
        for i, feature in enumerate(feature_names):
            results[feature] = {
                "mean": perm_importance.importances_mean[i],
                "std": perm_importance.importances_std[i],
            }

        # Sort by mean importance
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)
        )

        return sorted_results

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        title: str = "Feature Importance",
        max_features: Optional[int] = None,
    ) -> None:
        """
        Plot feature importance.

        Args:
            importance_dict: Dictionary of feature importances
            title: Plot title
            max_features: Maximum number of features to display
        """
        if max_features is None:
            max_features = self.config.max_features_display

        # Get top features
        top_features = dict(list(importance_dict.items())[:max_features])

        # Create plot
        plt.figure(figsize=self.config.figure_size)

        features = list(top_features.keys())
        importances = list(top_features.values())

        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        plt.barh(
            y_pos, importances, color=plt.cm.viridis(np.linspace(0, 1, len(features)))
        )

        plt.yticks(y_pos, features)
        plt.xlabel("Importance")
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, importance in enumerate(importances):
            plt.text(
                importance + 0.001, i, f"{importance:.3f}", va="center", fontsize=9
            )

        plt.tight_layout()
        plt.show()

    def plot_permutation_importance(
        self,
        perm_importance_dict: Dict[str, Dict[str, float]],
        title: str = "Permutation Importance",
        max_features: Optional[int] = None,
    ) -> None:
        """
        Plot permutation importance with error bars.

        Args:
            perm_importance_dict: Dictionary of permutation importances
            title: Plot title
            max_features: Maximum number of features to display
        """
        if max_features is None:
            max_features = self.config.max_features_display

        # Get top features
        top_features = dict(list(perm_importance_dict.items())[:max_features])

        # Extract means and stds
        features = list(top_features.keys())
        means = [top_features[f]["mean"] for f in features]
        stds = [top_features[f]["std"] for f in features]

        # Create plot
        plt.figure(figsize=self.config.figure_size)

        y_pos = np.arange(len(features))
        plt.barh(
            y_pos,
            means,
            xerr=stds,
            color=plt.cm.viridis(np.linspace(0, 1, len(features))),
            alpha=0.7,
            capsize=5,
        )

        plt.yticks(y_pos, features)
        plt.xlabel("Permutation Importance")
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(
                mean + std + 0.001, i, f"{mean:.3f}±{std:.3f}", va="center", fontsize=9
            )

        plt.tight_layout()
        plt.show()


class SHAPAnalyzer:
    """SHAP (SHapley Additive exPlanations) analysis for model interpretation."""

    def __init__(self, config: InterpretationConfig):
        self.config = config
        self.explainer = None
        self.shap_values = None

    def create_explainer(self, model: BaseEstimator, X_background: np.ndarray):
        """
        Create SHAP explainer based on model type.

        Args:
            model: Fitted model
            X_background: Background dataset for explainer
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP library is required. Install with: pip install shap"
            )

        # Subsample background data if too large
        if len(X_background) > self.config.max_shap_samples:
            indices = np.random.choice(
                len(X_background), self.config.max_shap_samples, replace=False
            )
            X_background = X_background[indices]

        # Create appropriate explainer
        if self.config.shap_explainer_type == "auto":
            # Auto-detect explainer type
            if hasattr(model, "tree_"):
                self.explainer = shap.TreeExplainer(model)
            elif hasattr(model, "coef_"):
                self.explainer = shap.LinearExplainer(model, X_background)
            else:
                self.explainer = shap.KernelExplainer(model.predict, X_background)
        elif self.config.shap_explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif self.config.shap_explainer_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_background)
        elif self.config.shap_explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(model.predict, X_background)
        else:
            raise ValueError(
                f"Unknown SHAP explainer type: {self.config.shap_explainer_type}"
            )

    def calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for given data.

        Args:
            X: Data to explain

        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer must be created first")

        # Subsample if dataset is too large
        if len(X) > self.config.max_shap_samples:
            indices = np.random.choice(
                len(X), self.config.max_shap_samples, replace=False
            )
            X_sample = X[indices]
        else:
            X_sample = X

        self.shap_values = self.explainer.shap_values(X_sample)

        # Handle multi-class case (take positive class for binary)
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]

        return self.shap_values

    def plot_shap_summary(
        self,
        feature_names: List[str],
        X: Optional[np.ndarray] = None,
        plot_type: str = "dot",
    ) -> None:
        """
        Plot SHAP summary plot.

        Args:
            feature_names: List of feature names
            X: Optional feature values for colored plots
            plot_type: Type of plot ('dot', 'bar', 'violin')
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library is required")

        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first")

        # Create summary plot
        if plot_type == "dot":
            shap.summary_plot(
                self.shap_values, X, feature_names=feature_names, show=False
            )
        elif plot_type == "bar":
            shap.summary_plot(
                self.shap_values,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
        elif plot_type == "violin":
            shap.summary_plot(
                self.shap_values,
                X,
                feature_names=feature_names,
                plot_type="violin",
                show=False,
            )

        plt.tight_layout()
        plt.show()

    def plot_shap_waterfall(
        self, sample_idx: int, feature_names: List[str], X: np.ndarray
    ) -> None:
        """
        Plot SHAP waterfall plot for a specific sample.

        Args:
            sample_idx: Index of sample to explain
            feature_names: List of feature names
            X: Feature values
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library is required")

        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first")

        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=X[sample_idx],
                feature_names=feature_names,
            ),
            show=False,
        )

        plt.tight_layout()
        plt.show()


class PartialDependenceAnalyzer:
    """Partial dependence analysis for understanding feature effects."""

    def __init__(self, config: InterpretationConfig):
        self.config = config

    def plot_partial_dependence(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        features: Union[List[int], List[str]],
        feature_names: List[str],
        grid_resolution: int = 100,
    ) -> None:
        """
        Plot partial dependence plots.

        Args:
            model: Fitted model
            X: Feature matrix
            features: Features to plot (indices or names)
            feature_names: List of feature names
            grid_resolution: Resolution of the grid
        """
        # Convert feature names to indices if needed
        if isinstance(features[0], str):
            feature_indices = [feature_names.index(f) for f in features]
        else:
            feature_indices = features

        # Calculate partial dependence
        pd_results = partial_dependence(
            model, X, feature_indices, grid_resolution=grid_resolution, kind="average"
        )

        # Create subplots
        n_features = len(feature_indices)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, feature_idx in enumerate(feature_indices):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # Plot partial dependence
            ax.plot(pd_results["grid_values"][i], pd_results["average"][i])
            ax.set_xlabel(feature_names[feature_idx])
            ax.set_ylabel("Partial Dependence")
            ax.set_title(f"Partial Dependence: {feature_names[feature_idx]}")
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()


class FinancialModelInterpreter:
    """
    Comprehensive model interpretation for financial models.

    Combines multiple interpretation methods with financial context.
    """

    def __init__(self, config: Optional[InterpretationConfig] = None):
        if config is None:
            config = InterpretationConfig()

        self.config = config
        self.feature_analyzer = FeatureImportanceAnalyzer(config)
        self.shap_analyzer = SHAPAnalyzer(config)
        self.pd_analyzer = PartialDependenceAnalyzer(config)

        # Store analysis results
        self.results = {}

    def comprehensive_analysis(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model interpretation analysis.

        Args:
            model: Fitted model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            feature_names: List of feature names
            task_type: 'classification' or 'regression'

        Returns:
            Dictionary containing all analysis results
        """
        results = {}

        print("Starting comprehensive model interpretation analysis...")

        # 1. Feature Importance Analysis
        print("1. Analyzing feature importance...")

        # Tree-based importance (if available)
        if hasattr(model, "feature_importances_"):
            tree_importance = self.feature_analyzer.analyze_tree_importance(
                model, feature_names
            )
            results["tree_importance"] = tree_importance

            print("   - Tree-based importance calculated")
            self.feature_analyzer.plot_feature_importance(
                tree_importance, "Tree-based Feature Importance"
            )

        # Permutation importance
        scoring = (
            "accuracy" if task_type == "classification" else "neg_mean_squared_error"
        )
        perm_importance = self.feature_analyzer.analyze_permutation_importance(
            model, X_test, y_test, feature_names, scoring=scoring
        )
        results["permutation_importance"] = perm_importance

        print("   - Permutation importance calculated")
        self.feature_analyzer.plot_permutation_importance(
            perm_importance, "Permutation Feature Importance"
        )

        # 2. SHAP Analysis
        print("2. Performing SHAP analysis...")
        try:
            self.shap_analyzer.create_explainer(model, X_train)
            shap_values = self.shap_analyzer.calculate_shap_values(X_test)
            results["shap_values"] = shap_values

            print("   - SHAP values calculated")

            # SHAP summary plots
            self.shap_analyzer.plot_shap_summary(feature_names, X_test, "dot")
            self.shap_analyzer.plot_shap_summary(feature_names, plot_type="bar")

            # Individual prediction explanation
            if len(X_test) > 0:
                self.shap_analyzer.plot_shap_waterfall(0, feature_names, X_test)

        except ImportError:
            print("   - SHAP not available. Install with: pip install shap")
            results["shap_values"] = None
        except Exception as e:
            print(f"   - SHAP analysis failed: {str(e)}")
            results["shap_values"] = None

        # 3. Partial Dependence Analysis
        print("3. Analyzing partial dependence...")
        try:
            # Get top important features for PD plots
            top_features = list(perm_importance.keys())[:6]  # Top 6 features

            self.pd_analyzer.plot_partial_dependence(
                model, X_test, top_features, feature_names
            )

            print("   - Partial dependence plots created")

        except Exception as e:
            print(f"   - Partial dependence analysis failed: {str(e)}")

        # 4. Financial Context Analysis
        print("4. Analyzing financial context...")
        financial_insights = self._analyze_financial_context(results, feature_names)
        results["financial_insights"] = financial_insights

        print("Comprehensive analysis completed!")

        self.results = results
        return results

    def _analyze_financial_context(
        self, analysis_results: Dict[str, Any], feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze results in financial context.

        Args:
            analysis_results: Results from other analyses
            feature_names: List of feature names

        Returns:
            Financial insights dictionary
        """
        insights = {
            "technical_indicators": [],
            "fundamental_features": [],
            "market_regime_features": [],
            "risk_features": [],
            "other_features": [],
        }

        # Categorize features by financial type
        for feature in feature_names:
            feature_lower = feature.lower()

            if any(
                indicator in feature_lower
                for indicator in [
                    "rsi",
                    "macd",
                    "bollinger",
                    "ma_",
                    "ema_",
                    "sma_",
                    "stoch",
                    "atr",
                ]
            ):
                insights["technical_indicators"].append(feature)
            elif any(
                fundamental in feature_lower
                for fundamental in [
                    "pe_",
                    "pb_",
                    "roe",
                    "roa",
                    "debt",
                    "revenue",
                    "earnings",
                ]
            ):
                insights["fundamental_features"].append(feature)
            elif any(
                regime in feature_lower
                for regime in ["vix", "volatility", "volume", "regime", "trend"]
            ):
                insights["market_regime_features"].append(feature)
            elif any(
                risk in feature_lower
                for risk in ["var", "cvar", "drawdown", "beta", "correlation"]
            ):
                insights["risk_features"].append(feature)
            else:
                insights["other_features"].append(feature)

        # Analyze importance by category
        if "permutation_importance" in analysis_results:
            perm_imp = analysis_results["permutation_importance"]

            for category, features in insights.items():
                if features:
                    category_importance = {
                        f: perm_imp.get(f, {"mean": 0})["mean"]
                        for f in features
                        if f in perm_imp
                    }
                    insights[f"{category}_importance"] = category_importance

        return insights

    def generate_interpretation_report(
        self, model_name: str = "Financial Model"
    ) -> str:
        """
        Generate a comprehensive interpretation report.

        Args:
            model_name: Name of the model being interpreted

        Returns:
            Formatted interpretation report
        """
        if not self.results:
            return "No analysis results available. Run comprehensive_analysis first."

        report = f"\n{'='*60}\n"
        report += f"MODEL INTERPRETATION REPORT: {model_name}\n"
        report += f"{'='*60}\n\n"

        # Feature Importance Summary
        if "permutation_importance" in self.results:
            report += "TOP 10 MOST IMPORTANT FEATURES:\n"
            report += "-" * 40 + "\n"

            perm_imp = self.results["permutation_importance"]
            for i, (feature, scores) in enumerate(list(perm_imp.items())[:10]):
                report += f"{i+1:2d}. {feature:<25} {scores['mean']:.4f} ± {scores['std']:.4f}\n"
            report += "\n"

        # Financial Context
        if "financial_insights" in self.results:
            insights = self.results["financial_insights"]

            report += "FEATURE CATEGORIES:\n"
            report += "-" * 40 + "\n"

            for category, features in insights.items():
                if features and not category.endswith("_importance"):
                    report += f"{category.replace('_', ' ').title()}: {len(features)} features\n"
            report += "\n"

        # Key Insights
        report += "KEY INSIGHTS:\n"
        report += "-" * 40 + "\n"

        if "permutation_importance" in self.results:
            top_feature = list(self.results["permutation_importance"].keys())[0]
            report += f"• Most important feature: {top_feature}\n"

            # Count features by importance threshold
            significant_features = sum(
                1
                for f, scores in self.results["permutation_importance"].items()
                if scores["mean"] > self.config.importance_threshold
            )
            report += f"• Significant features (>{self.config.importance_threshold}): {significant_features}\n"

        report += "\n"

        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-" * 40 + "\n"
        report += "• Focus feature engineering on top 10 most important features\n"
        report += "• Consider removing features with very low importance\n"
        report += "• Monitor feature stability across different market regimes\n"
        report += "• Validate feature importance using different time periods\n"

        return report


def create_model_interpreter(
    config: Optional[InterpretationConfig] = None,
) -> FinancialModelInterpreter:
    """
    Factory function to create model interpreter.

    Args:
        config: Interpretation configuration

    Returns:
        Model interpreter instance
    """
    return FinancialModelInterpreter(config=config)
