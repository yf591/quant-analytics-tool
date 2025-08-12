"""
Performance Attribution Analysis Module

Implements comprehensive performance attribution methodologies based on
AFML Chapters 12-13 for analyzing factor contributions to strategy returns.

This module provides:
- Brinson performance attribution model
- Factor-based return decomposition
- Sector/style attribution analysis
- Risk-adjusted performance metrics
- Multi-period attribution analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Types of attribution methods available."""

    BRINSON = "brinson"
    FACTOR_MODEL = "factor_model"
    RISK_BASED = "risk_based"
    STYLE_ANALYSIS = "style_analysis"
    MULTI_PERIOD = "multi_period"


@dataclass
class AttributionResult:
    """Result container for performance attribution analysis."""

    method: str
    period: str
    total_return: float
    attribution_components: Dict[str, float]
    selection_effect: Optional[float] = None
    allocation_effect: Optional[float] = None
    interaction_effect: Optional[float] = None
    active_return: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None


@dataclass
class FactorExposure:
    """Factor exposure data for attribution analysis."""

    factor_name: str
    exposure: float
    factor_return: float
    contribution: float
    significance: Optional[float] = None


class PerformanceAttributionAnalyzer:
    """
    Comprehensive performance attribution analyzer.

    Implements various attribution methodologies including Brinson model,
    factor-based attribution, and risk-based decomposition based on
    AFML methodologies.
    """

    def __init__(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        attribution_period: str = "monthly",
    ):
        """
        Initialize the Performance Attribution Analyzer.

        Args:
            benchmark_returns: Benchmark returns series
            risk_free_rate: Risk-free rate for calculations
            attribution_period: Attribution analysis period ('daily', 'monthly', 'quarterly')
        """
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.attribution_period = attribution_period

        # Results storage
        self.attribution_results_ = {}
        self.factor_exposures_ = {}
        self.performance_metrics_ = {}

    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        period_dates: Optional[List[str]] = None,
    ) -> Dict[str, AttributionResult]:
        """
        Perform Brinson performance attribution analysis.

        Based on Brinson, Hood, and Beebower methodology, decomposes
        active return into allocation, selection, and interaction effects.

        Args:
            portfolio_weights: Portfolio weights by sector/asset
            portfolio_returns: Portfolio returns by sector/asset
            benchmark_weights: Benchmark weights by sector/asset
            benchmark_returns: Benchmark returns by sector/asset
            period_dates: Specific periods for analysis

        Returns:
            Dictionary of AttributionResult objects by period
        """
        results = {}

        if period_dates is None:
            # Use actual index values instead of strftime conversion
            period_dates = portfolio_returns.index.tolist()

        for period in period_dates:
            try:
                # Get period data - handle both string and datetime indices
                if period in portfolio_returns.index:
                    p_returns = portfolio_returns.loc[period]
                    b_returns = benchmark_returns.loc[period]
                    p_weights = (
                        portfolio_weights.loc[period]
                        if period in portfolio_weights.index
                        else portfolio_weights.iloc[-1]
                    )
                    b_weights = (
                        benchmark_weights.loc[period]
                        if period in benchmark_weights.index
                        else benchmark_weights.iloc[-1]
                    )
                else:
                    continue

                # Calculate Brinson attribution components
                allocation_effect = self._calculate_allocation_effect(
                    p_weights, b_weights, b_returns
                )
                selection_effect = self._calculate_selection_effect(
                    p_weights, p_returns, b_returns
                )
                interaction_effect = self._calculate_interaction_effect(
                    p_weights, b_weights, p_returns, b_returns
                )

                # Total active return
                portfolio_total_return = float((p_weights * p_returns).sum())
                benchmark_total_return = float((b_weights * b_returns).sum())
                active_return = portfolio_total_return - benchmark_total_return

                # Attribution components breakdown
                attribution_components = {
                    "allocation_effect": float(allocation_effect),
                    "selection_effect": float(selection_effect),
                    "interaction_effect": float(interaction_effect),
                    "total_attribution": float(
                        allocation_effect + selection_effect + interaction_effect
                    ),
                }

                # Additional metrics
                additional_metrics = {
                    "portfolio_return": portfolio_total_return,
                    "benchmark_return": benchmark_total_return,
                    "attribution_residual": active_return
                    - (
                        float(allocation_effect)
                        + float(selection_effect)
                        + float(interaction_effect)
                    ),
                }

                result = AttributionResult(
                    method="brinson",
                    period=str(period),
                    total_return=portfolio_total_return,
                    attribution_components=attribution_components,
                    selection_effect=float(selection_effect),
                    allocation_effect=float(allocation_effect),
                    interaction_effect=float(interaction_effect),
                    active_return=active_return,
                    additional_metrics=additional_metrics,
                )

                results[str(period)] = result

            except Exception as e:
                logger.warning(
                    f"Error in Brinson attribution for period {period}: {str(e)}"
                )
                continue

        self.attribution_results_["brinson"] = results
        return results

    def factor_based_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        rolling_window: int = 252,
        method: str = "regression",
    ) -> Dict[str, Any]:
        """
        Perform factor-based performance attribution using multi-factor models.

        Args:
            strategy_returns: Strategy returns series
            factor_returns: Factor returns DataFrame
            rolling_window: Rolling window for dynamic attribution
            method: Attribution method ('regression', 'style_analysis')

        Returns:
            Dictionary containing factor attribution results
        """
        results = {
            "factor_exposures": {},
            "factor_contributions": {},
            "attribution_quality": {},
            "rolling_attribution": {},
            "static_attribution": {},
        }

        # Align data
        if len(strategy_returns) == 0 or len(factor_returns.columns) == 0:
            logger.warning("Empty data provided for factor attribution")
            return results

        aligned_data = pd.concat(
            [strategy_returns, factor_returns], axis=1, join="inner"
        )
        strategy_returns_aligned = aligned_data.iloc[:, 0]
        factor_returns_aligned = aligned_data.iloc[:, 1:]

        if len(strategy_returns_aligned) == 0:
            logger.warning("No aligned data for factor attribution")
            return results

        # Static factor attribution (full period)
        static_result = self._calculate_static_factor_attribution(
            strategy_returns_aligned, factor_returns_aligned, method
        )
        results["static_attribution"] = static_result

        # Rolling factor attribution
        if len(strategy_returns_aligned) >= rolling_window:
            rolling_results = self._calculate_rolling_factor_attribution(
                strategy_returns_aligned, factor_returns_aligned, rolling_window, method
            )
            results["rolling_attribution"] = rolling_results

        # Factor exposure analysis
        results["factor_exposures"] = self._analyze_factor_exposures(
            strategy_returns_aligned, factor_returns_aligned
        )

        # Attribution quality metrics
        results["attribution_quality"] = self._calculate_attribution_quality(
            strategy_returns_aligned, factor_returns_aligned, static_result
        )

        self.attribution_results_["factor_based"] = results
        return results

    def risk_based_attribution(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Perform risk-based attribution analysis.

        Decomposes tracking error into systematic and idiosyncratic components.

        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            factor_returns: Factor returns for risk model

        Returns:
            Dictionary containing risk attribution results
        """
        results = {
            "tracking_error_decomposition": {},
            "systematic_risk": {},
            "idiosyncratic_risk": {},
            "risk_attribution": {},
            "var_attribution": {},
        }

        # Calculate active returns
        active_returns = strategy_returns - benchmark_returns
        active_returns = active_returns.dropna()

        if len(active_returns) == 0:
            logger.warning("No data for risk attribution")
            return results

        # Tracking error decomposition
        tracking_error = active_returns.std() * np.sqrt(252)
        results["tracking_error_decomposition"]["total_tracking_error"] = tracking_error

        # Factor model for systematic risk
        factor_model_result = self._calculate_factor_model_risk(
            active_returns, factor_returns
        )
        results["systematic_risk"] = factor_model_result["systematic"]
        results["idiosyncratic_risk"] = factor_model_result["idiosyncratic"]

        # Risk attribution by factor
        results["risk_attribution"] = self._calculate_risk_attribution_by_factor(
            active_returns, factor_returns
        )

        # VaR attribution
        results["var_attribution"] = self._calculate_var_attribution(
            active_returns, factor_returns
        )

        self.attribution_results_["risk_based"] = results
        return results

    def style_analysis_attribution(
        self,
        strategy_returns: pd.Series,
        style_benchmarks: pd.DataFrame,
        rolling_window: int = 252,
    ) -> Dict[str, Any]:
        """
        Perform return-based style analysis attribution.

        Based on Sharpe's style analysis methodology.

        Args:
            strategy_returns: Strategy returns series
            style_benchmarks: Style benchmark returns DataFrame
            rolling_window: Rolling window for dynamic style analysis

        Returns:
            Dictionary containing style attribution results
        """
        results = {
            "style_exposures": {},
            "style_contributions": {},
            "selection_return": 0.0,
            "timing_return": 0.0,
            "rolling_style_analysis": {},
            "style_consistency": {},
        }

        # Align data
        aligned_data = pd.concat(
            [strategy_returns, style_benchmarks], axis=1, join="inner"
        )
        if len(aligned_data) == 0:
            return results

        strategy_aligned = aligned_data.iloc[:, 0]
        benchmarks_aligned = aligned_data.iloc[:, 1:]

        # Static style analysis
        static_exposures = self._calculate_style_exposures(
            strategy_aligned, benchmarks_aligned
        )
        results["style_exposures"] = static_exposures

        # Calculate style contributions
        style_contributions = {}
        for style, exposure in static_exposures.items():
            if style in benchmarks_aligned.columns:
                contribution = exposure * benchmarks_aligned[style].mean() * 252
                style_contributions[style] = contribution

        results["style_contributions"] = style_contributions

        # Selection return (alpha)
        predicted_returns = sum(
            static_exposures.get(style, 0) * benchmarks_aligned[style]
            for style in benchmarks_aligned.columns
        )
        selection_returns = strategy_aligned - predicted_returns
        results["selection_return"] = selection_returns.mean() * 252

        # Rolling style analysis
        if len(strategy_aligned) >= rolling_window:
            rolling_results = self._calculate_rolling_style_analysis(
                strategy_aligned, benchmarks_aligned, rolling_window
            )
            results["rolling_style_analysis"] = rolling_results

            # Style timing analysis
            results["timing_return"] = self._calculate_style_timing_return(
                rolling_results
            )

        # Style consistency metrics
        results["style_consistency"] = self._calculate_style_consistency(
            results.get("rolling_style_analysis", {})
        )

        self.attribution_results_["style_analysis"] = results
        return results

    def multi_period_attribution(
        self,
        attribution_results: Dict[str, AttributionResult],
        compounding_method: str = "geometric",
    ) -> Dict[str, Any]:
        """
        Perform multi-period attribution analysis.

        Aggregates single-period attribution results across multiple periods.

        Args:
            attribution_results: Dictionary of single-period attribution results
            compounding_method: Method for aggregating returns ('geometric', 'arithmetic')

        Returns:
            Dictionary containing multi-period attribution results
        """
        if not attribution_results:
            return {}

        results = {
            "aggregated_attribution": {},
            "period_contributions": {},
            "attribution_stability": {},
            "cumulative_effects": {},
        }

        # Aggregate attribution components
        component_series = {}
        for period, attr_result in attribution_results.items():
            for component, value in attr_result.attribution_components.items():
                if component not in component_series:
                    component_series[component] = []
                component_series[component].append(value)

        # Calculate aggregated effects
        for component, values in component_series.items():
            if compounding_method == "geometric":
                # Geometric aggregation for returns
                aggregated = (np.prod([1 + v for v in values]) - 1) if values else 0.0
            else:
                # Arithmetic aggregation
                aggregated = sum(values)

            results["aggregated_attribution"][component] = aggregated

        # Period contribution analysis
        results["period_contributions"] = self._analyze_period_contributions(
            attribution_results
        )

        # Attribution stability metrics
        results["attribution_stability"] = self._calculate_attribution_stability(
            component_series
        )

        # Cumulative effects
        results["cumulative_effects"] = self._calculate_cumulative_attribution_effects(
            attribution_results
        )

        self.attribution_results_["multi_period"] = results
        return results

    def comprehensive_attribution_report(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive attribution report combining all methods.

        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            factor_returns: Factor returns DataFrame
            sector_data: Sector allocation and returns data

        Returns:
            Dictionary containing comprehensive attribution analysis
        """
        report = {
            "summary_metrics": {},
            "attribution_methods": {},
            "key_insights": {},
            "recommendations": {},
        }

        # Basic performance metrics
        report["summary_metrics"] = self._calculate_summary_performance_metrics(
            strategy_returns, benchmark_returns
        )

        # Factor-based attribution if factor data available
        if factor_returns is not None:
            try:
                factor_attribution = self.factor_based_attribution(
                    strategy_returns, factor_returns
                )
                report["attribution_methods"]["factor_based"] = factor_attribution
            except Exception as e:
                logger.warning(f"Factor attribution failed: {str(e)}")

        # Risk-based attribution if benchmark available
        if benchmark_returns is not None:
            try:
                risk_attribution = self.risk_based_attribution(
                    strategy_returns,
                    benchmark_returns,
                    factor_returns if factor_returns is not None else pd.DataFrame(),
                )
                report["attribution_methods"]["risk_based"] = risk_attribution
            except Exception as e:
                logger.warning(f"Risk attribution failed: {str(e)}")

        # Brinson attribution if sector data available
        if sector_data is not None:
            try:
                brinson_results = self.brinson_attribution(**sector_data)
                report["attribution_methods"]["brinson"] = brinson_results
            except Exception as e:
                logger.warning(f"Brinson attribution failed: {str(e)}")

        # Generate insights and recommendations
        report["key_insights"] = self._generate_attribution_insights(report)
        report["recommendations"] = self._generate_attribution_recommendations(report)

        return report

    # Helper methods for calculation

    def _calculate_allocation_effect(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate allocation effect in Brinson attribution."""
        weight_diff = portfolio_weights - benchmark_weights
        return (weight_diff * benchmark_returns).sum()

    def _calculate_selection_effect(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate selection effect in Brinson attribution."""
        return_diff = portfolio_returns - benchmark_returns
        return (portfolio_weights * return_diff).sum()

    def _calculate_interaction_effect(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate interaction effect in Brinson attribution."""
        weight_diff = portfolio_weights - benchmark_weights
        return_diff = portfolio_returns - benchmark_returns
        return (weight_diff * return_diff).sum()

    def _calculate_static_factor_attribution(
        self, strategy_returns: pd.Series, factor_returns: pd.DataFrame, method: str
    ) -> Dict[str, Any]:
        """Calculate static factor attribution using regression."""
        result = {
            "factor_loadings": {},
            "factor_contributions": {},
            "alpha": 0.0,
            "r_squared": 0.0,
            "residual_volatility": 0.0,
        }

        try:
            if method == "regression":
                # OLS regression
                X = factor_returns.values
                y = strategy_returns.values

                model = LinearRegression(fit_intercept=True)
                model.fit(X, y)

                # Extract results
                result["alpha"] = model.intercept_ * 252  # Annualized
                factor_loadings = dict(zip(factor_returns.columns, model.coef_))
                result["factor_loadings"] = factor_loadings

                # Calculate factor contributions
                factor_contributions = {}
                for factor, loading in factor_loadings.items():
                    contribution = loading * factor_returns[factor].mean() * 252
                    factor_contributions[factor] = contribution
                result["factor_contributions"] = factor_contributions

                # Model fit statistics
                predictions = model.predict(X)
                residuals = y - predictions
                result["r_squared"] = model.score(X, y)
                result["residual_volatility"] = np.std(residuals) * np.sqrt(252)

            elif method == "style_analysis":
                # Constrained regression (weights sum to 1, non-negative)
                result = self._constrained_style_regression(
                    strategy_returns, factor_returns
                )

        except Exception as e:
            logger.warning(f"Error in static factor attribution: {str(e)}")

        return result

    def _calculate_rolling_factor_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int,
        method: str,
    ) -> Dict[str, pd.Series]:
        """Calculate rolling factor attribution."""
        rolling_results = {
            "alpha": [],
            "r_squared": [],
            "factor_loadings": {factor: [] for factor in factor_returns.columns},
        }

        dates = []

        for i in range(window, len(strategy_returns)):
            window_start = i - window
            window_end = i

            window_strategy = strategy_returns.iloc[window_start:window_end]
            window_factors = factor_returns.iloc[window_start:window_end]

            try:
                static_result = self._calculate_static_factor_attribution(
                    window_strategy, window_factors, method
                )

                rolling_results["alpha"].append(static_result["alpha"])
                rolling_results["r_squared"].append(static_result["r_squared"])

                for factor in factor_returns.columns:
                    loading = static_result["factor_loadings"].get(factor, 0.0)
                    rolling_results["factor_loadings"][factor].append(loading)

                dates.append(strategy_returns.index[window_end - 1])

            except Exception as e:
                logger.warning(f"Error in rolling attribution at index {i}: {str(e)}")
                continue

        # Convert to pandas Series
        for key in rolling_results:
            if key == "factor_loadings":
                for factor in rolling_results[key]:
                    rolling_results[key][factor] = pd.Series(
                        rolling_results[key][factor], index=dates
                    )
            else:
                rolling_results[key] = pd.Series(rolling_results[key], index=dates)

        return rolling_results

    def _analyze_factor_exposures(
        self, strategy_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> Dict[str, FactorExposure]:
        """Analyze factor exposures and their significance."""
        exposures = {}

        try:
            static_attribution = self._calculate_static_factor_attribution(
                strategy_returns, factor_returns, "regression"
            )

            for factor in factor_returns.columns:
                loading = static_attribution["factor_loadings"].get(factor, 0.0)
                factor_return = factor_returns[factor].mean() * 252
                contribution = static_attribution["factor_contributions"].get(
                    factor, 0.0
                )

                # Calculate statistical significance (t-statistic)
                factor_vol = factor_returns[factor].std() * np.sqrt(252)
                significance = abs(loading) / factor_vol if factor_vol > 0 else 0.0

                exposure = FactorExposure(
                    factor_name=factor,
                    exposure=loading,
                    factor_return=factor_return,
                    contribution=contribution,
                    significance=significance,
                )

                exposures[factor] = exposure

        except Exception as e:
            logger.warning(f"Error analyzing factor exposures: {str(e)}")

        return exposures

    def _calculate_attribution_quality(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        static_result: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate quality metrics for attribution analysis."""
        quality_metrics = {
            "r_squared": static_result.get("r_squared", 0.0),
            "tracking_error_explained": 0.0,
            "factor_significance_ratio": 0.0,
            "attribution_completeness": 0.0,
        }

        try:
            # Tracking error explanation
            total_vol = strategy_returns.std() * np.sqrt(252)
            residual_vol = static_result.get("residual_volatility", total_vol)
            explained_vol = np.sqrt(max(0, total_vol**2 - residual_vol**2))

            quality_metrics["tracking_error_explained"] = (
                explained_vol / total_vol if total_vol > 0 else 0.0
            )

            # Factor significance ratio
            factor_loadings = static_result.get("factor_loadings", {})
            if factor_loadings:
                significant_factors = sum(
                    1 for loading in factor_loadings.values() if abs(loading) > 0.1
                )
                quality_metrics["factor_significance_ratio"] = (
                    significant_factors / len(factor_loadings)
                )

            # Attribution completeness
            total_return = strategy_returns.mean() * 252
            attributed_return = sum(
                static_result.get("factor_contributions", {}).values()
            )
            alpha = static_result.get("alpha", 0.0)

            if abs(total_return) > 1e-6:
                quality_metrics["attribution_completeness"] = (
                    attributed_return + alpha
                ) / total_return

        except Exception as e:
            logger.warning(f"Error calculating attribution quality: {str(e)}")

        return quality_metrics

    def _calculate_factor_model_risk(
        self, active_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate systematic and idiosyncratic risk components."""
        result = {
            "systematic": {"variance": 0.0, "volatility": 0.0, "contribution": 0.0},
            "idiosyncratic": {"variance": 0.0, "volatility": 0.0, "contribution": 0.0},
        }

        try:
            if len(factor_returns) == 0:
                # No factors, all risk is idiosyncratic
                total_var = active_returns.var()
                result["idiosyncratic"]["variance"] = total_var
                result["idiosyncratic"]["volatility"] = np.sqrt(total_var) * np.sqrt(
                    252
                )
                result["idiosyncratic"]["contribution"] = 1.0
                return result

            # Factor model regression
            X = factor_returns.values
            y = active_returns.values

            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)

            # Systematic risk (explained by factors)
            predictions = model.predict(X)
            systematic_var = np.var(predictions)

            # Idiosyncratic risk (residual)
            residuals = y - predictions
            idiosyncratic_var = np.var(residuals)

            # Total variance
            total_var = systematic_var + idiosyncratic_var

            result["systematic"]["variance"] = systematic_var
            result["systematic"]["volatility"] = np.sqrt(systematic_var) * np.sqrt(252)
            result["systematic"]["contribution"] = (
                systematic_var / total_var if total_var > 0 else 0.0
            )

            result["idiosyncratic"]["variance"] = idiosyncratic_var
            result["idiosyncratic"]["volatility"] = np.sqrt(
                idiosyncratic_var
            ) * np.sqrt(252)
            result["idiosyncratic"]["contribution"] = (
                idiosyncratic_var / total_var if total_var > 0 else 1.0
            )

        except Exception as e:
            logger.warning(f"Error in factor model risk calculation: {str(e)}")

        return result

    def _calculate_risk_attribution_by_factor(
        self, active_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate risk attribution by individual factors."""
        risk_attribution = {}

        try:
            if len(factor_returns) == 0:
                return risk_attribution

            # Factor model
            X = factor_returns.values
            y = active_returns.values

            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)

            factor_loadings = model.coef_

            # Calculate factor contribution to total variance
            total_var = active_returns.var()

            for i, factor in enumerate(factor_returns.columns):
                # Factor contribution to variance
                loading = factor_loadings[i]
                factor_var = factor_returns[factor].var()
                factor_contribution = (
                    (loading**2 * factor_var) / total_var if total_var > 0 else 0.0
                )

                risk_attribution[factor] = factor_contribution

        except Exception as e:
            logger.warning(f"Error in risk attribution by factor: {str(e)}")

        return risk_attribution

    def _calculate_var_attribution(
        self,
        active_returns: pd.Series,
        factor_returns: pd.DataFrame,
        confidence_level: float = 0.05,
    ) -> Dict[str, float]:
        """Calculate VaR attribution by factors."""
        var_attribution = {}

        try:
            # Portfolio VaR
            portfolio_var = np.percentile(active_returns, confidence_level * 100)

            if len(factor_returns) == 0:
                return {"total_var": portfolio_var}

            # Component VaR calculation using marginal VaR
            for factor in factor_returns.columns:
                # Calculate marginal VaR for this factor
                correlation = active_returns.corr(factor_returns[factor])
                factor_vol = factor_returns[factor].std()

                # Marginal VaR approximation
                marginal_var = (
                    correlation * factor_vol * stats.norm.ppf(confidence_level)
                )

                var_attribution[factor] = marginal_var

            var_attribution["total_var"] = portfolio_var

        except Exception as e:
            logger.warning(f"Error in VaR attribution: {str(e)}")

        return var_attribution

    def _calculate_style_exposures(
        self, strategy_returns: pd.Series, style_benchmarks: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate style exposures using constrained regression."""
        exposures = {}

        try:
            # Constrained optimization to ensure weights sum to 1 and are non-negative
            from scipy.optimize import minimize

            def objective(weights):
                portfolio_return = (weights * style_benchmarks).sum(axis=1)
                tracking_error = ((strategy_returns - portfolio_return) ** 2).sum()
                return tracking_error

            # Constraints: weights sum to 1, all weights >= 0
            n_styles = len(style_benchmarks.columns)
            constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
            bounds = [(0, 1) for _ in range(n_styles)]

            # Initial guess: equal weights
            x0 = np.ones(n_styles) / n_styles

            result = minimize(objective, x0, bounds=bounds, constraints=constraints)

            if result.success:
                for i, style in enumerate(style_benchmarks.columns):
                    exposures[style] = result.x[i]
            else:
                # Fallback to unconstrained regression
                for style in style_benchmarks.columns:
                    correlation = strategy_returns.corr(style_benchmarks[style])
                    exposures[style] = max(0, correlation)

                # Normalize to sum to 1
                total_exposure = sum(exposures.values())
                if total_exposure > 0:
                    exposures = {k: v / total_exposure for k, v in exposures.items()}

        except Exception as e:
            logger.warning(f"Error calculating style exposures: {str(e)}")
            # Simple correlation-based fallback
            for style in style_benchmarks.columns:
                correlation = strategy_returns.corr(style_benchmarks[style])
                exposures[style] = max(0, correlation)

        return exposures

    def _calculate_rolling_style_analysis(
        self, strategy_returns: pd.Series, style_benchmarks: pd.DataFrame, window: int
    ) -> Dict[str, pd.Series]:
        """Calculate rolling style analysis."""
        rolling_exposures = {style: [] for style in style_benchmarks.columns}
        dates = []

        for i in range(window, len(strategy_returns)):
            window_start = i - window
            window_end = i

            window_strategy = strategy_returns.iloc[window_start:window_end]
            window_styles = style_benchmarks.iloc[window_start:window_end]

            try:
                exposures = self._calculate_style_exposures(
                    window_strategy, window_styles
                )

                for style in style_benchmarks.columns:
                    rolling_exposures[style].append(exposures.get(style, 0.0))

                dates.append(strategy_returns.index[window_end - 1])

            except Exception as e:
                logger.warning(
                    f"Error in rolling style analysis at index {i}: {str(e)}"
                )
                continue

        # Convert to pandas Series
        for style in rolling_exposures:
            rolling_exposures[style] = pd.Series(rolling_exposures[style], index=dates)

        return rolling_exposures

    def _calculate_style_timing_return(
        self, rolling_results: Dict[str, pd.Series]
    ) -> float:
        """Calculate return from style timing decisions."""
        timing_return = 0.0

        try:
            if not rolling_results:
                return timing_return

            # Simple timing return calculation based on exposure changes
            for style, exposures in rolling_results.items():
                if len(exposures) > 1:
                    exposure_changes = exposures.diff().dropna()
                    # Positive timing return if exposures increase when style performs well
                    timing_component = exposure_changes.sum() / len(exposure_changes)
                    timing_return += timing_component

        except Exception as e:
            logger.warning(f"Error calculating style timing return: {str(e)}")

        return timing_return

    def _calculate_style_consistency(
        self, rolling_results: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """Calculate style consistency metrics."""
        consistency_metrics = {}

        try:
            for style, exposures in rolling_results.items():
                if len(exposures) > 1:
                    # Coefficient of variation as consistency measure
                    mean_exposure = exposures.mean()
                    exposure_vol = exposures.std()

                    consistency = (
                        1 - (exposure_vol / abs(mean_exposure))
                        if abs(mean_exposure) > 0
                        else 0
                    )
                    consistency_metrics[style] = max(0, min(1, consistency))

        except Exception as e:
            logger.warning(f"Error calculating style consistency: {str(e)}")

        return consistency_metrics

    def _analyze_period_contributions(
        self, attribution_results: Dict[str, AttributionResult]
    ) -> Dict[str, Any]:
        """Analyze contribution of each period to total attribution."""
        period_analysis = {
            "period_weights": {},
            "period_impacts": {},
            "outlier_periods": [],
        }

        try:
            if not attribution_results:
                return period_analysis

            # Calculate period weights based on absolute returns
            total_abs_return = sum(
                abs(result.total_return) for result in attribution_results.values()
            )

            for period, result in attribution_results.items():
                weight = (
                    abs(result.total_return) / total_abs_return
                    if total_abs_return > 0
                    else 0
                )
                period_analysis["period_weights"][period] = weight

                # Period impact on attribution
                impact_score = abs(result.active_return or 0) * weight
                period_analysis["period_impacts"][period] = impact_score

            # Identify outlier periods (top 10% by impact)
            impacts = list(period_analysis["period_impacts"].values())
            if impacts:
                outlier_threshold = np.percentile(impacts, 90)
                outliers = [
                    period
                    for period, impact in period_analysis["period_impacts"].items()
                    if impact >= outlier_threshold
                ]
                period_analysis["outlier_periods"] = outliers

        except Exception as e:
            logger.warning(f"Error analyzing period contributions: {str(e)}")

        return period_analysis

    def _calculate_attribution_stability(
        self, component_series: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate stability metrics for attribution components."""
        stability_metrics = {}

        try:
            for component, values in component_series.items():
                if len(values) > 1:
                    # Coefficient of variation
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = abs(std_val / mean_val) if abs(mean_val) > 0 else float("inf")

                    # Stability score (inverse of CV, capped at 1)
                    stability = 1 / (1 + cv)
                    stability_metrics[component] = stability

        except Exception as e:
            logger.warning(f"Error calculating attribution stability: {str(e)}")

        return stability_metrics

    def _calculate_cumulative_attribution_effects(
        self, attribution_results: Dict[str, AttributionResult]
    ) -> Dict[str, List[float]]:
        """Calculate cumulative attribution effects over time."""
        cumulative_effects = {}

        try:
            # Sort periods chronologically
            sorted_periods = sorted(attribution_results.keys())

            for component in [
                "allocation_effect",
                "selection_effect",
                "interaction_effect",
            ]:
                cumulative_values = []
                cumulative_sum = 0.0

                for period in sorted_periods:
                    result = attribution_results[period]
                    if hasattr(result, component):
                        value = getattr(result, component) or 0.0
                        cumulative_sum += value
                        cumulative_values.append(cumulative_sum)
                    else:
                        cumulative_values.append(cumulative_sum)

                cumulative_effects[component] = cumulative_values

        except Exception as e:
            logger.warning(
                f"Error calculating cumulative attribution effects: {str(e)}"
            )

        return cumulative_effects

    def _calculate_summary_performance_metrics(
        self, strategy_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate summary performance metrics."""
        metrics = {}

        try:
            # Basic metrics
            metrics["total_return"] = (1 + strategy_returns).prod() - 1
            metrics["annualized_return"] = strategy_returns.mean() * 252
            metrics["volatility"] = strategy_returns.std() * np.sqrt(252)

            # Risk-adjusted metrics
            excess_returns = strategy_returns - self.risk_free_rate / 252
            metrics["sharpe_ratio"] = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0
                else 0.0
            )

            # Downside metrics
            negative_returns = strategy_returns[strategy_returns < 0]
            metrics["downside_deviation"] = (
                negative_returns.std() * np.sqrt(252)
                if len(negative_returns) > 0
                else 0.0
            )

            if metrics["downside_deviation"] > 0:
                metrics["sortino_ratio"] = (
                    metrics["annualized_return"] - self.risk_free_rate
                ) / metrics["downside_deviation"]
            else:
                metrics["sortino_ratio"] = 0.0

            # Maximum drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics["max_drawdown"] = drawdowns.min()

            # Active metrics vs benchmark
            if benchmark_returns is not None:
                active_returns = strategy_returns - benchmark_returns
                metrics["active_return"] = active_returns.mean() * 252
                metrics["tracking_error"] = active_returns.std() * np.sqrt(252)

                if metrics["tracking_error"] > 0:
                    metrics["information_ratio"] = (
                        metrics["active_return"] / metrics["tracking_error"]
                    )
                else:
                    metrics["information_ratio"] = 0.0

                # Beta
                if benchmark_returns.std() > 0:
                    metrics["beta"] = (
                        strategy_returns.cov(benchmark_returns)
                        / benchmark_returns.var()
                    )
                else:
                    metrics["beta"] = 0.0

        except Exception as e:
            logger.warning(f"Error calculating summary metrics: {str(e)}")

        return metrics

    def _constrained_style_regression(
        self, strategy_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform constrained style regression analysis."""
        # Placeholder for constrained regression implementation
        # In practice, would use scipy.optimize with constraints
        return self._calculate_static_factor_attribution(
            strategy_returns, factor_returns, "regression"
        )

    def _generate_attribution_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate key insights from attribution analysis."""
        insights = []

        try:
            summary_metrics = report.get("summary_metrics", {})
            attribution_methods = report.get("attribution_methods", {})

            # Performance insights
            sharpe_ratio = summary_metrics.get("sharpe_ratio", 0)
            if sharpe_ratio > 1.0:
                insights.append(
                    "Strategy demonstrates strong risk-adjusted performance"
                )
            elif sharpe_ratio < 0:
                insights.append("Strategy shows negative risk-adjusted returns")

            # Factor attribution insights
            if "factor_based" in attribution_methods:
                factor_results = attribution_methods["factor_based"].get(
                    "static_attribution", {}
                )
                alpha = factor_results.get("alpha", 0)
                r_squared = factor_results.get("r_squared", 0)

                if alpha > 0.02:  # 2% annual alpha
                    insights.append("Strategy generates significant positive alpha")

                if r_squared > 0.8:
                    insights.append("Returns are well explained by systematic factors")
                elif r_squared < 0.3:
                    insights.append("Returns show high idiosyncratic component")

            # Risk attribution insights
            if "risk_based" in attribution_methods:
                risk_results = attribution_methods["risk_based"]
                systematic_contribution = risk_results.get("systematic_risk", {}).get(
                    "contribution", 0
                )

                if systematic_contribution > 0.7:
                    insights.append("Risk is primarily systematic/factor-driven")
                else:
                    insights.append("Significant idiosyncratic risk component")

        except Exception as e:
            logger.warning(f"Error generating insights: {str(e)}")

        return insights

    def _generate_attribution_recommendations(
        self, report: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on attribution analysis."""
        recommendations = []

        try:
            summary_metrics = report.get("summary_metrics", {})
            attribution_methods = report.get("attribution_methods", {})

            # Performance-based recommendations
            information_ratio = summary_metrics.get("information_ratio", 0)
            if information_ratio < 0.5:
                recommendations.append(
                    "Consider improving active return generation or reducing tracking error"
                )

            max_drawdown = summary_metrics.get("max_drawdown", 0)
            if max_drawdown < -0.2:  # More than 20% drawdown
                recommendations.append(
                    "Implement enhanced risk management to reduce drawdown risk"
                )

            # Factor-based recommendations
            if "factor_based" in attribution_methods:
                factor_results = attribution_methods["factor_based"]
                quality_metrics = factor_results.get("attribution_quality", {})
                r_squared = quality_metrics.get("r_squared", 0)

                if r_squared < 0.5:
                    recommendations.append(
                        "Consider expanding factor model to better explain returns"
                    )

            # Risk-based recommendations
            if "risk_based" in attribution_methods:
                risk_results = attribution_methods["risk_based"]
                tracking_error = risk_results.get(
                    "tracking_error_decomposition", {}
                ).get("total_tracking_error", 0)

                if tracking_error > 0.15:  # 15% tracking error
                    recommendations.append(
                        "High tracking error - consider position size management"
                    )

        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")

        return recommendations

    def get_attribution_summary(self) -> Dict[str, Any]:
        """Get summary of all attribution analyses performed."""
        summary = {
            "available_methods": list(self.attribution_results_.keys()),
            "performance_metrics": self.performance_metrics_,
            "factor_exposures": self.factor_exposures_,
            "attribution_results": self.attribution_results_,
        }

        return summary
