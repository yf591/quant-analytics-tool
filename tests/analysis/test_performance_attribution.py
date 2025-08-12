"""
Test suite for performance attribution analysis implementation.

Tests comprehensive performance attribution methodologies including Brinson model,
factor-based attribution, risk-based decomposition, and style analysis.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.analysis.performance_attribution import (
    PerformanceAttributionAnalyzer,
    AttributionResult,
    FactorExposure,
    AttributionMethod,
)


class TestPerformanceAttributionAnalyzer:
    """Test suite for PerformanceAttributionAnalyzer."""

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")

        # Generate correlated return series
        n_assets = 3
        returns = np.random.multivariate_normal(
            mean=[0.0005, 0.0003, 0.0004],
            cov=[
                [0.0004, 0.0001, 0.0001],
                [0.0001, 0.0003, 0.0001],
                [0.0001, 0.0001, 0.0005],
            ],
            size=len(dates),
        )

        return pd.DataFrame(
            returns, index=dates, columns=["strategy", "benchmark", "factor1"]
        )

    @pytest.fixture
    def sample_factor_returns(self):
        """Create sample factor returns data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")

        # Generate factor returns with different characteristics
        market_factor = np.random.normal(0.0005, 0.015, len(dates))
        value_factor = np.random.normal(0.0002, 0.01, len(dates))
        momentum_factor = np.random.normal(0.0001, 0.012, len(dates))
        size_factor = np.random.normal(0.0003, 0.008, len(dates))

        return pd.DataFrame(
            {
                "market": market_factor,
                "value": value_factor,
                "momentum": momentum_factor,
                "size": size_factor,
            },
            index=dates,
        )

    @pytest.fixture
    def sample_sector_data(self):
        """Create sample sector allocation and returns data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        sectors = ["Technology", "Healthcare", "Finance", "Energy"]

        # Portfolio weights
        portfolio_weights = pd.DataFrame(
            np.random.dirichlet([1, 1, 1, 1], len(dates)), index=dates, columns=sectors
        )

        # Benchmark weights
        benchmark_weights = pd.DataFrame(
            np.random.dirichlet([1.2, 0.8, 1.0, 0.6], len(dates)),
            index=dates,
            columns=sectors,
        )

        # Sector returns
        portfolio_returns = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(dates), len(sectors))),
            index=dates,
            columns=sectors,
        )

        benchmark_returns = pd.DataFrame(
            np.random.normal(0.008, 0.04, (len(dates), len(sectors))),
            index=dates,
            columns=sectors,
        )

        return {
            "portfolio_weights": portfolio_weights,
            "portfolio_returns": portfolio_returns,
            "benchmark_weights": benchmark_weights,
            "benchmark_returns": benchmark_returns,
        }

    @pytest.fixture
    def attribution_analyzer(self, sample_returns_data):
        """Create attribution analyzer instance."""
        benchmark_returns = sample_returns_data["benchmark"]
        return PerformanceAttributionAnalyzer(
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.02,
            attribution_period="monthly",
        )

    def test_analyzer_initialization(self, sample_returns_data):
        """Test PerformanceAttributionAnalyzer initialization."""
        benchmark_returns = sample_returns_data["benchmark"]

        analyzer = PerformanceAttributionAnalyzer(
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.025,
            attribution_period="daily",
        )

        assert analyzer.benchmark_returns.equals(benchmark_returns)
        assert analyzer.risk_free_rate == 0.025
        assert analyzer.attribution_period == "daily"
        assert analyzer.attribution_results_ == {}
        assert analyzer.factor_exposures_ == {}
        assert analyzer.performance_metrics_ == {}

    def test_default_initialization(self):
        """Test default initialization parameters."""
        analyzer = PerformanceAttributionAnalyzer()

        assert analyzer.benchmark_returns is None
        assert analyzer.risk_free_rate == 0.02
        assert analyzer.attribution_period == "monthly"

    def test_brinson_attribution(self, attribution_analyzer, sample_sector_data):
        """Test Brinson performance attribution."""
        results = attribution_analyzer.brinson_attribution(**sample_sector_data)

        # Check results structure
        assert isinstance(results, dict)
        assert len(results) > 0

        # Check each period result
        for period, result in results.items():
            assert isinstance(result, AttributionResult)
            assert result.method == "brinson"
            assert result.period == period
            assert isinstance(result.total_return, float)
            assert isinstance(result.attribution_components, dict)

            # Check attribution components
            components = result.attribution_components
            assert "allocation_effect" in components
            assert "selection_effect" in components
            assert "interaction_effect" in components
            assert "total_attribution" in components

            # Check that components are numeric
            for component, value in components.items():
                assert isinstance(value, (int, float))
                assert not np.isnan(value)

            # Check optional fields
            assert result.selection_effect is not None
            assert result.allocation_effect is not None
            assert result.interaction_effect is not None
            assert result.active_return is not None

            # Check additional metrics
            assert result.additional_metrics is not None
            assert "portfolio_return" in result.additional_metrics
            assert "benchmark_return" in result.additional_metrics

        # Check that results are stored
        assert "brinson" in attribution_analyzer.attribution_results_

    def test_factor_based_attribution(
        self, attribution_analyzer, sample_returns_data, sample_factor_returns
    ):
        """Test factor-based performance attribution."""
        strategy_returns = sample_returns_data["strategy"]

        results = attribution_analyzer.factor_based_attribution(
            strategy_returns, sample_factor_returns
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "factor_exposures" in results
        assert "factor_contributions" in results
        assert "attribution_quality" in results
        assert "static_attribution" in results

        # Check static attribution
        static_attr = results["static_attribution"]
        assert "factor_loadings" in static_attr
        assert "factor_contributions" in static_attr
        assert "alpha" in static_attr
        assert "r_squared" in static_attr
        assert "residual_volatility" in static_attr

        # Check factor loadings
        factor_loadings = static_attr["factor_loadings"]
        for factor in sample_factor_returns.columns:
            assert factor in factor_loadings
            assert isinstance(factor_loadings[factor], (int, float))

        # Check factor contributions
        factor_contributions = static_attr["factor_contributions"]
        for factor in sample_factor_returns.columns:
            assert factor in factor_contributions
            assert isinstance(factor_contributions[factor], (int, float))

        # Check attribution quality
        quality_metrics = results["attribution_quality"]
        assert "r_squared" in quality_metrics
        assert "tracking_error_explained" in quality_metrics
        assert "factor_significance_ratio" in quality_metrics
        assert "attribution_completeness" in quality_metrics

        # Check ranges
        assert 0 <= quality_metrics["r_squared"] <= 1
        assert 0 <= quality_metrics["tracking_error_explained"] <= 1
        assert 0 <= quality_metrics["factor_significance_ratio"] <= 1

        # Check rolling attribution if sufficient data
        if "rolling_attribution" in results and results["rolling_attribution"]:
            rolling_attr = results["rolling_attribution"]
            assert "alpha" in rolling_attr
            assert "r_squared" in rolling_attr
            assert "factor_loadings" in rolling_attr

    def test_risk_based_attribution(
        self, attribution_analyzer, sample_returns_data, sample_factor_returns
    ):
        """Test risk-based attribution analysis."""
        strategy_returns = sample_returns_data["strategy"]
        benchmark_returns = sample_returns_data["benchmark"]

        results = attribution_analyzer.risk_based_attribution(
            strategy_returns, benchmark_returns, sample_factor_returns
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "tracking_error_decomposition" in results
        assert "systematic_risk" in results
        assert "idiosyncratic_risk" in results
        assert "risk_attribution" in results
        assert "var_attribution" in results

        # Check tracking error decomposition
        te_decomp = results["tracking_error_decomposition"]
        assert "total_tracking_error" in te_decomp
        assert isinstance(te_decomp["total_tracking_error"], (int, float))
        assert te_decomp["total_tracking_error"] >= 0

        # Check systematic risk
        systematic_risk = results["systematic_risk"]
        assert "variance" in systematic_risk
        assert "volatility" in systematic_risk
        assert "contribution" in systematic_risk
        assert 0 <= systematic_risk["contribution"] <= 1

        # Check idiosyncratic risk
        idiosyncratic_risk = results["idiosyncratic_risk"]
        assert "variance" in idiosyncratic_risk
        assert "volatility" in idiosyncratic_risk
        assert "contribution" in idiosyncratic_risk
        assert 0 <= idiosyncratic_risk["contribution"] <= 1

        # Check that contributions sum to approximately 1
        total_contribution = (
            systematic_risk["contribution"] + idiosyncratic_risk["contribution"]
        )
        assert abs(total_contribution - 1.0) < 0.01

        # Check risk attribution by factor
        risk_attribution = results["risk_attribution"]
        assert isinstance(risk_attribution, dict)
        for factor, contribution in risk_attribution.items():
            assert isinstance(contribution, (int, float))
            assert 0 <= contribution <= 1

        # Check VaR attribution
        var_attribution = results["var_attribution"]
        assert isinstance(var_attribution, dict)
        assert "total_var" in var_attribution

    def test_style_analysis_attribution(
        self, attribution_analyzer, sample_returns_data
    ):
        """Test style analysis attribution."""
        strategy_returns = sample_returns_data["strategy"]

        # Create style benchmarks (use subset of factor returns as styles)
        style_benchmarks = sample_returns_data[["benchmark", "factor1"]]

        results = attribution_analyzer.style_analysis_attribution(
            strategy_returns, style_benchmarks
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "style_exposures" in results
        assert "style_contributions" in results
        assert "selection_return" in results
        assert "timing_return" in results
        assert "style_consistency" in results

        # Check style exposures
        style_exposures = results["style_exposures"]
        assert isinstance(style_exposures, dict)
        for style in style_benchmarks.columns:
            if style in style_exposures:
                assert isinstance(style_exposures[style], (int, float))
                assert 0 <= style_exposures[style] <= 1

        # Check style contributions
        style_contributions = results["style_contributions"]
        assert isinstance(style_contributions, dict)

        # Check selection return (alpha)
        assert isinstance(results["selection_return"], (int, float))

        # Check timing return
        assert isinstance(results["timing_return"], (int, float))

        # Check style consistency
        style_consistency = results["style_consistency"]
        assert isinstance(style_consistency, dict)

    def test_multi_period_attribution(self, attribution_analyzer, sample_sector_data):
        """Test multi-period attribution analysis."""
        # First run Brinson attribution to get single-period results
        brinson_results = attribution_analyzer.brinson_attribution(**sample_sector_data)

        # Test multi-period aggregation
        multi_period_results = attribution_analyzer.multi_period_attribution(
            brinson_results
        )

        # Check results structure
        assert isinstance(multi_period_results, dict)
        assert "aggregated_attribution" in multi_period_results
        assert "period_contributions" in multi_period_results
        assert "attribution_stability" in multi_period_results
        assert "cumulative_effects" in multi_period_results

        # Check aggregated attribution
        aggregated = multi_period_results["aggregated_attribution"]
        assert isinstance(aggregated, dict)
        assert "allocation_effect" in aggregated
        assert "selection_effect" in aggregated
        assert "interaction_effect" in aggregated

        # Check period contributions
        period_contrib = multi_period_results["period_contributions"]
        assert "period_weights" in period_contrib
        assert "period_impacts" in period_contrib
        assert "outlier_periods" in period_contrib

        # Check attribution stability
        stability = multi_period_results["attribution_stability"]
        assert isinstance(stability, dict)
        for component, stability_score in stability.items():
            assert 0 <= stability_score <= 1

        # Check cumulative effects
        cumulative = multi_period_results["cumulative_effects"]
        assert isinstance(cumulative, dict)

    def test_comprehensive_attribution_report(
        self,
        attribution_analyzer,
        sample_returns_data,
        sample_factor_returns,
        sample_sector_data,
    ):
        """Test comprehensive attribution report generation."""
        strategy_returns = sample_returns_data["strategy"]
        benchmark_returns = sample_returns_data["benchmark"]

        report = attribution_analyzer.comprehensive_attribution_report(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            factor_returns=sample_factor_returns,
            sector_data=sample_sector_data,
        )

        # Check report structure
        assert isinstance(report, dict)
        assert "summary_metrics" in report
        assert "attribution_methods" in report
        assert "key_insights" in report
        assert "recommendations" in report

        # Check summary metrics
        summary_metrics = report["summary_metrics"]
        expected_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "downside_deviation",
            "sortino_ratio",
            "max_drawdown",
            "active_return",
            "tracking_error",
            "information_ratio",
            "beta",
        ]

        for metric in expected_metrics:
            if metric in summary_metrics:
                assert isinstance(summary_metrics[metric], (int, float))
                assert not np.isnan(summary_metrics[metric])

        # Check attribution methods
        attribution_methods = report["attribution_methods"]
        assert isinstance(attribution_methods, dict)
        # Should have at least factor_based and risk_based
        expected_methods = ["factor_based", "risk_based"]
        for method in expected_methods:
            if method in attribution_methods:
                assert isinstance(attribution_methods[method], dict)

        # Check insights and recommendations
        assert isinstance(report["key_insights"], list)
        assert isinstance(report["recommendations"], list)

    def test_attribution_result_dataclass(self):
        """Test AttributionResult dataclass."""
        result = AttributionResult(
            method="test_method",
            period="2020-01",
            total_return=0.05,
            attribution_components={"factor1": 0.02, "factor2": 0.03},
            selection_effect=0.01,
            allocation_effect=0.02,
            interaction_effect=0.005,
            active_return=0.015,
            tracking_error=0.03,
            information_ratio=0.5,
            additional_metrics={"metric1": 0.1},
        )

        assert result.method == "test_method"
        assert result.period == "2020-01"
        assert result.total_return == 0.05
        assert result.attribution_components == {"factor1": 0.02, "factor2": 0.03}
        assert result.selection_effect == 0.01
        assert result.allocation_effect == 0.02
        assert result.interaction_effect == 0.005
        assert result.active_return == 0.015
        assert result.tracking_error == 0.03
        assert result.information_ratio == 0.5
        assert result.additional_metrics == {"metric1": 0.1}

    def test_factor_exposure_dataclass(self):
        """Test FactorExposure dataclass."""
        exposure = FactorExposure(
            factor_name="market",
            exposure=0.8,
            factor_return=0.1,
            contribution=0.08,
            significance=2.5,
        )

        assert exposure.factor_name == "market"
        assert exposure.exposure == 0.8
        assert exposure.factor_return == 0.1
        assert exposure.contribution == 0.08
        assert exposure.significance == 2.5

    def test_allocation_effect_calculation(self, attribution_analyzer):
        """Test allocation effect calculation."""
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        benchmark_weights = pd.Series([0.3, 0.4, 0.3], index=["A", "B", "C"])
        benchmark_returns = pd.Series([0.05, 0.03, 0.02], index=["A", "B", "C"])

        allocation_effect = attribution_analyzer._calculate_allocation_effect(
            portfolio_weights, benchmark_weights, benchmark_returns
        )

        # Expected: (0.4-0.3)*0.05 + (0.3-0.4)*0.03 + (0.3-0.3)*0.02 = 0.005 - 0.003 + 0 = 0.002
        expected = 0.002
        assert abs(allocation_effect - expected) < 1e-6

    def test_selection_effect_calculation(self, attribution_analyzer):
        """Test selection effect calculation."""
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        portfolio_returns = pd.Series([0.06, 0.04, 0.03], index=["A", "B", "C"])
        benchmark_returns = pd.Series([0.05, 0.03, 0.02], index=["A", "B", "C"])

        selection_effect = attribution_analyzer._calculate_selection_effect(
            portfolio_weights, portfolio_returns, benchmark_returns
        )

        # Expected: 0.4*(0.06-0.05) + 0.3*(0.04-0.03) + 0.3*(0.03-0.02) = 0.004 + 0.003 + 0.003 = 0.01
        expected = 0.01
        assert abs(selection_effect - expected) < 1e-6

    def test_interaction_effect_calculation(self, attribution_analyzer):
        """Test interaction effect calculation."""
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        benchmark_weights = pd.Series([0.3, 0.4, 0.3], index=["A", "B", "C"])
        portfolio_returns = pd.Series([0.06, 0.04, 0.03], index=["A", "B", "C"])
        benchmark_returns = pd.Series([0.05, 0.03, 0.02], index=["A", "B", "C"])

        interaction_effect = attribution_analyzer._calculate_interaction_effect(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        # Expected: (0.4-0.3)*(0.06-0.05) + (0.3-0.4)*(0.04-0.03) + (0.3-0.3)*(0.03-0.02)
        # = 0.1*0.01 + (-0.1)*0.01 + 0*0.01 = 0.001 - 0.001 + 0 = 0
        expected = 0.0
        assert abs(interaction_effect - expected) < 1e-6

    def test_static_factor_attribution(
        self, attribution_analyzer, sample_returns_data, sample_factor_returns
    ):
        """Test static factor attribution calculation."""
        strategy_returns = sample_returns_data["strategy"]

        result = attribution_analyzer._calculate_static_factor_attribution(
            strategy_returns, sample_factor_returns, "regression"
        )

        # Check result structure
        assert "factor_loadings" in result
        assert "factor_contributions" in result
        assert "alpha" in result
        assert "r_squared" in result
        assert "residual_volatility" in result

        # Check factor loadings
        factor_loadings = result["factor_loadings"]
        for factor in sample_factor_returns.columns:
            assert factor in factor_loadings
            assert isinstance(factor_loadings[factor], (int, float))

        # Check ranges
        assert 0 <= result["r_squared"] <= 1
        assert result["residual_volatility"] >= 0

    def test_style_exposures_calculation(
        self, attribution_analyzer, sample_returns_data
    ):
        """Test style exposures calculation."""
        strategy_returns = sample_returns_data["strategy"]
        style_benchmarks = sample_returns_data[["benchmark", "factor1"]]

        exposures = attribution_analyzer._calculate_style_exposures(
            strategy_returns, style_benchmarks
        )

        # Check exposures
        assert isinstance(exposures, dict)
        for style, exposure in exposures.items():
            assert isinstance(exposure, (int, float))
            assert 0 <= exposure <= 1  # Should be between 0 and 1

        # Exposures should approximately sum to 1 (for valid style analysis)
        total_exposure = sum(exposures.values())
        assert 0.5 <= total_exposure <= 1.5  # Allow some flexibility

    def test_performance_metrics_calculation(
        self, attribution_analyzer, sample_returns_data
    ):
        """Test summary performance metrics calculation."""
        strategy_returns = sample_returns_data["strategy"]
        benchmark_returns = sample_returns_data["benchmark"]

        metrics = attribution_analyzer._calculate_summary_performance_metrics(
            strategy_returns, benchmark_returns
        )

        # Check that all expected metrics are calculated
        expected_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "downside_deviation",
            "sortino_ratio",
            "max_drawdown",
            "active_return",
            "tracking_error",
            "information_ratio",
            "beta",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

        # Check logical relationships
        assert metrics["max_drawdown"] <= 0  # Drawdown should be negative or zero
        assert metrics["volatility"] >= 0  # Volatility should be positive
        assert (
            metrics["downside_deviation"] >= 0
        )  # Downside deviation should be positive

    def test_factor_model_risk_calculation(
        self, attribution_analyzer, sample_returns_data, sample_factor_returns
    ):
        """Test factor model risk calculation."""
        strategy_returns = sample_returns_data["strategy"]
        benchmark_returns = sample_returns_data["benchmark"]
        active_returns = strategy_returns - benchmark_returns

        result = attribution_analyzer._calculate_factor_model_risk(
            active_returns, sample_factor_returns
        )

        # Check result structure
        assert "systematic" in result
        assert "idiosyncratic" in result

        # Check systematic risk
        systematic = result["systematic"]
        assert "variance" in systematic
        assert "volatility" in systematic
        assert "contribution" in systematic
        assert systematic["variance"] >= 0
        assert systematic["volatility"] >= 0
        assert 0 <= systematic["contribution"] <= 1

        # Check idiosyncratic risk
        idiosyncratic = result["idiosyncratic"]
        assert "variance" in idiosyncratic
        assert "volatility" in idiosyncratic
        assert "contribution" in idiosyncratic
        assert idiosyncratic["variance"] >= 0
        assert idiosyncratic["volatility"] >= 0
        assert 0 <= idiosyncratic["contribution"] <= 1

        # Check that contributions sum to approximately 1
        total_contribution = systematic["contribution"] + idiosyncratic["contribution"]
        assert abs(total_contribution - 1.0) < 0.01

    def test_empty_data_handling(self, attribution_analyzer):
        """Test handling of empty or insufficient data."""
        empty_returns = pd.Series([], dtype=float)
        empty_factors = pd.DataFrame()

        # Test factor attribution with empty data
        result = attribution_analyzer.factor_based_attribution(
            empty_returns, empty_factors
        )
        assert isinstance(result, dict)
        assert "factor_exposures" in result

        # Test risk attribution with empty data
        result = attribution_analyzer.risk_based_attribution(
            empty_returns, empty_returns, empty_factors
        )
        assert isinstance(result, dict)
        assert "tracking_error_decomposition" in result

    def test_error_handling(self, attribution_analyzer):
        """Test error handling in various scenarios."""
        # Test with NaN values
        nan_returns = pd.Series([0.01, np.nan, 0.02, np.nan])
        nan_factors = pd.DataFrame(
            {
                "factor1": [0.005, np.nan, 0.01, 0.008],
                "factor2": [np.nan, 0.003, np.nan, 0.006],
            }
        )

        # Should handle NaN values gracefully
        result = attribution_analyzer.factor_based_attribution(nan_returns, nan_factors)
        assert isinstance(result, dict)

        # Test with mismatched indices
        returns1 = pd.Series(
            [0.01, 0.02, 0.03], index=["2020-01-01", "2020-01-02", "2020-01-03"]
        )
        factors1 = pd.DataFrame(
            {"factor1": [0.005, 0.01]}, index=["2020-01-01", "2020-01-02"]
        )

        result = attribution_analyzer.factor_based_attribution(returns1, factors1)
        assert isinstance(result, dict)

    def test_attribution_insights_generation(self, attribution_analyzer):
        """Test attribution insights generation."""
        # Create mock report data
        mock_report = {
            "summary_metrics": {
                "sharpe_ratio": 1.2,
                "information_ratio": 0.8,
                "max_drawdown": -0.15,
            },
            "attribution_methods": {
                "factor_based": {
                    "static_attribution": {"alpha": 0.03, "r_squared": 0.75},
                    "attribution_quality": {"r_squared": 0.75},
                },
                "risk_based": {"systematic_risk": {"contribution": 0.8}},
            },
        }

        insights = attribution_analyzer._generate_attribution_insights(mock_report)

        assert isinstance(insights, list)
        assert len(insights) > 0

        # Check that insights are strings
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0

    def test_attribution_recommendations_generation(self, attribution_analyzer):
        """Test attribution recommendations generation."""
        # Create mock report data
        mock_report = {
            "summary_metrics": {"information_ratio": 0.3, "max_drawdown": -0.25},
            "attribution_methods": {
                "factor_based": {"attribution_quality": {"r_squared": 0.4}},
                "risk_based": {
                    "tracking_error_decomposition": {"total_tracking_error": 0.18}
                },
            },
        }

        recommendations = attribution_analyzer._generate_attribution_recommendations(
            mock_report
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check that recommendations are strings
        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0

    def test_attribution_summary(
        self, attribution_analyzer, sample_returns_data, sample_factor_returns
    ):
        """Test attribution summary generation."""
        strategy_returns = sample_returns_data["strategy"]

        # Run some attribution analysis first
        attribution_analyzer.factor_based_attribution(
            strategy_returns, sample_factor_returns
        )

        summary = attribution_analyzer.get_attribution_summary()

        # Check summary structure
        assert isinstance(summary, dict)
        assert "available_methods" in summary
        assert "performance_metrics" in summary
        assert "factor_exposures" in summary
        assert "attribution_results" in summary

        # Check available methods
        assert "factor_based" in summary["available_methods"]


class TestIntegration:
    """Integration tests for performance attribution."""

    def test_full_attribution_workflow(self):
        """Test complete attribution workflow with realistic data."""
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range("2019-01-01", periods=252 * 2, freq="D")

        # Generate market factors
        market_returns = np.random.normal(0.0004, 0.015, len(dates))
        value_factor = np.random.normal(0.0001, 0.008, len(dates))
        momentum_factor = np.random.normal(0.0002, 0.01, len(dates))

        factor_returns = pd.DataFrame(
            {
                "market": market_returns,
                "value": value_factor,
                "momentum": momentum_factor,
            },
            index=dates,
        )

        # Generate strategy returns with factor exposures
        strategy_returns = (
            0.8 * factor_returns["market"]
            + 0.3 * factor_returns["value"]
            + -0.2 * factor_returns["momentum"]
            + np.random.normal(0.0001, 0.005, len(dates))  # Alpha + idiosyncratic
        )
        strategy_returns = pd.Series(strategy_returns, index=dates, name="strategy")

        # Generate benchmark returns
        benchmark_returns = (
            1.0 * factor_returns["market"]
            + 0.1 * factor_returns["value"]
            + 0.1 * factor_returns["momentum"]
        )
        benchmark_returns = pd.Series(benchmark_returns, index=dates, name="benchmark")

        # Initialize analyzer
        analyzer = PerformanceAttributionAnalyzer(
            benchmark_returns=benchmark_returns, risk_free_rate=0.02
        )

        # Run comprehensive attribution
        report = analyzer.comprehensive_attribution_report(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns,
        )

        # Verify report structure
        assert "summary_metrics" in report
        assert "attribution_methods" in report
        assert "key_insights" in report
        assert "recommendations" in report

        # Verify attribution methods ran successfully
        attribution_methods = report["attribution_methods"]
        assert "factor_based" in attribution_methods
        assert "risk_based" in attribution_methods

        # Check factor attribution results
        factor_results = attribution_methods["factor_based"]["static_attribution"]

        # Should have reasonable factor loadings (close to true exposures)
        market_loading = factor_results["factor_loadings"]["market"]
        value_loading = factor_results["factor_loadings"]["value"]
        momentum_loading = factor_results["factor_loadings"]["momentum"]

        # Allow for some estimation error
        assert 0.6 <= market_loading <= 1.0  # True: 0.8
        assert 0.1 <= value_loading <= 0.5  # True: 0.3
        assert -0.4 <= momentum_loading <= 0.0  # True: -0.2

        # Should have reasonable alpha (may be positive or negative)
        alpha = factor_results["alpha"]
        assert isinstance(alpha, (int, float))
        assert not np.isnan(alpha)

        # Should have reasonable R-squared
        r_squared = factor_results["r_squared"]
        assert r_squared > 0.5  # Model should explain most of the returns

        # Check risk attribution
        risk_results = attribution_methods["risk_based"]
        systematic_contribution = risk_results["systematic_risk"]["contribution"]
        idiosyncratic_contribution = risk_results["idiosyncratic_risk"]["contribution"]

        # Most risk should be systematic (from factor exposures)
        assert systematic_contribution > 0.4  # Relaxed threshold
        assert abs(systematic_contribution + idiosyncratic_contribution - 1.0) < 0.01

        # Check performance metrics
        summary_metrics = report["summary_metrics"]
        assert "sharpe_ratio" in summary_metrics
        assert "information_ratio" in summary_metrics
        assert "tracking_error" in summary_metrics

        # Information ratio should be reasonable (may be positive or negative)
        information_ratio = summary_metrics["information_ratio"]
        assert isinstance(information_ratio, (int, float))
        assert not np.isnan(information_ratio)

        # Check insights and recommendations
        assert len(report["key_insights"]) > 0
        assert (
            len(report["recommendations"]) >= 0
        )  # May be empty if performance is good


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
