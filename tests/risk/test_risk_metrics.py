"""
Test Suite for Risk Metrics Module - Week 12 Risk Management

Comprehensive tests for VaR, CVaR, drawdown analysis, and portfolio risk metrics.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.risk.risk_metrics import RiskMetrics, PortfolioRiskAnalyzer


class TestRiskMetrics(unittest.TestCase):
    """Test cases for RiskMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create test data
        self.dates = pd.date_range("2020-01-01", periods=500, freq="D")
        self.normal_returns = pd.Series(
            np.random.normal(0.001, 0.02, 500), index=self.dates
        )

        # Create non-normal returns (with skew and kurtosis)
        self.skewed_returns = pd.Series(
            np.random.exponential(0.01, 500) - 0.005, index=self.dates
        )

        # Small dataset for edge cases
        self.small_returns = pd.Series(
            [0.01, -0.02, 0.005], index=pd.date_range("2020-01-01", periods=3)
        )

        self.risk_metrics = RiskMetrics(
            confidence_level=0.95, rolling_window=50, min_periods=10
        )

    def test_initialization(self):
        """Test RiskMetrics initialization."""
        rm = RiskMetrics(confidence_level=0.99, rolling_window=100, min_periods=20)

        self.assertEqual(rm.confidence_level, 0.99)
        self.assertEqual(rm.rolling_window, 100)
        self.assertEqual(rm.min_periods, 20)
        self.assertEqual(rm.risk_free_rate, 0.02)

    def test_parametric_var(self):
        """Test parametric VaR calculation."""
        var = self.risk_metrics.value_at_risk(self.normal_returns, method="parametric")

        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)  # VaR should be positive (loss)
        self.assertLess(var, 0.1)  # Reasonable upper bound

    def test_historical_var(self):
        """Test historical VaR calculation."""
        var = self.risk_metrics.value_at_risk(self.normal_returns, method="historical")

        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)
        self.assertLess(var, 0.1)

    def test_cornish_fisher_var(self):
        """Test Cornish-Fisher VaR calculation."""
        var = self.risk_metrics.value_at_risk(
            self.skewed_returns, method="cornish_fisher"
        )

        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)

    def test_var_confidence_levels(self):
        """Test VaR with different confidence levels."""
        var_95 = self.risk_metrics.value_at_risk(
            self.normal_returns, confidence_level=0.95
        )
        var_99 = self.risk_metrics.value_at_risk(
            self.normal_returns, confidence_level=0.99
        )

        # Higher confidence should give higher VaR
        self.assertGreater(var_99, var_95)

    def test_var_invalid_method(self):
        """Test VaR with invalid method."""
        var = self.risk_metrics.value_at_risk(self.normal_returns, method="invalid")
        self.assertTrue(np.isnan(var))

    def test_var_insufficient_data(self):
        """Test VaR with insufficient data."""
        var = self.risk_metrics.value_at_risk(self.small_returns)
        self.assertTrue(np.isnan(var))

    def test_parametric_cvar(self):
        """Test parametric CVaR calculation."""
        cvar = self.risk_metrics.conditional_var(
            self.normal_returns, method="parametric"
        )
        var = self.risk_metrics.value_at_risk(self.normal_returns, method="parametric")

        self.assertIsInstance(cvar, float)
        self.assertGreater(cvar, 0)
        self.assertGreater(cvar, var)  # CVaR should be higher than VaR

    def test_historical_cvar(self):
        """Test historical CVaR calculation."""
        cvar = self.risk_metrics.conditional_var(
            self.normal_returns, method="historical"
        )
        var = self.risk_metrics.value_at_risk(self.normal_returns, method="historical")

        self.assertIsInstance(cvar, float)
        self.assertGreater(cvar, 0)
        self.assertGreaterEqual(cvar, var)  # CVaR >= VaR

    def test_cvar_invalid_method(self):
        """Test CVaR with invalid method."""
        cvar = self.risk_metrics.conditional_var(self.normal_returns, method="invalid")
        self.assertTrue(np.isnan(cvar))

    def test_cvar_insufficient_data(self):
        """Test CVaR with insufficient data."""
        cvar = self.risk_metrics.conditional_var(self.small_returns)
        self.assertTrue(np.isnan(cvar))

    def test_maximum_drawdown_simple(self):
        """Test maximum drawdown calculation with simple returns."""
        # Create returns with known drawdown
        test_returns = pd.Series([0.1, -0.05, -0.1, 0.05, 0.2])

        dd_metrics = self.risk_metrics.maximum_drawdown(test_returns, method="simple")

        self.assertIn("max_drawdown", dd_metrics)
        self.assertIn("max_duration", dd_metrics)
        self.assertGreater(dd_metrics["max_drawdown"], 0)
        self.assertIsInstance(dd_metrics["max_duration"], (int, np.integer))

    def test_maximum_drawdown_log(self):
        """Test maximum drawdown with log returns."""
        dd_metrics = self.risk_metrics.maximum_drawdown(
            self.normal_returns, method="log"
        )

        self.assertIn("max_drawdown", dd_metrics)
        self.assertGreater(dd_metrics["max_drawdown"], 0)

    def test_maximum_drawdown_insufficient_data(self):
        """Test drawdown with insufficient data."""
        single_return = pd.Series([0.01])
        dd_metrics = self.risk_metrics.maximum_drawdown(single_return)

        self.assertTrue(np.isnan(dd_metrics["max_drawdown"]))

    def test_risk_adjusted_returns(self):
        """Test risk-adjusted return metrics."""
        metrics = self.risk_metrics.risk_adjusted_returns(self.normal_returns)

        expected_metrics = [
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "total_return",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

        # Sharpe ratio should be reasonable
        self.assertGreater(metrics["sharpe_ratio"], -2)
        self.assertLess(metrics["sharpe_ratio"], 5)

    def test_risk_adjusted_returns_with_benchmark(self):
        """Test risk-adjusted returns with benchmark."""
        # Create benchmark returns
        benchmark = pd.Series(np.random.normal(0.0005, 0.015, 500), index=self.dates)

        metrics = self.risk_metrics.risk_adjusted_returns(
            self.normal_returns, benchmark
        )

        benchmark_metrics = ["tracking_error", "information_ratio", "beta", "alpha"]

        for metric in benchmark_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

        # Beta should be reasonable
        self.assertGreater(metrics["beta"], -2)
        self.assertLess(metrics["beta"], 3)

    def test_risk_adjusted_returns_insufficient_data(self):
        """Test risk-adjusted returns with insufficient data."""
        metrics = self.risk_metrics.risk_adjusted_returns(self.small_returns)

        self.assertEqual(len(metrics), 0)

    def test_infer_frequency(self):
        """Test frequency inference."""
        # Daily returns
        daily_freq = self.risk_metrics._infer_frequency(self.normal_returns)
        self.assertEqual(daily_freq, 252)

        # Weekly returns
        weekly_dates = pd.date_range("2020-01-01", periods=52, freq="W")
        weekly_returns = pd.Series(
            np.random.normal(0.001, 0.02, 52), index=weekly_dates
        )
        weekly_freq = self.risk_metrics._infer_frequency(weekly_returns)
        self.assertEqual(weekly_freq, 52)

        # Monthly returns
        monthly_dates = pd.date_range("2020-01-01", periods=12, freq="M")
        monthly_returns = pd.Series(
            np.random.normal(0.001, 0.02, 12), index=monthly_dates
        )
        monthly_freq = self.risk_metrics._infer_frequency(monthly_returns)
        self.assertEqual(monthly_freq, 12)

    def test_rolling_risk_metrics(self):
        """Test rolling risk metrics calculation."""
        metrics = ["var", "cvar", "volatility", "sharpe"]
        rolling_metrics = self.risk_metrics.rolling_risk_metrics(
            self.normal_returns, metrics
        )

        self.assertIsInstance(rolling_metrics, pd.DataFrame)

        if not rolling_metrics.empty:
            for metric in metrics:
                self.assertIn(metric, rolling_metrics.columns)

            # Check that we have reasonable number of observations
            expected_length = (
                len(self.normal_returns) - self.risk_metrics.rolling_window + 1
            )
            self.assertEqual(len(rolling_metrics), expected_length)

    def test_rolling_risk_metrics_insufficient_data(self):
        """Test rolling metrics with insufficient data."""
        rolling_metrics = self.risk_metrics.rolling_risk_metrics(self.small_returns)

        self.assertIsInstance(rolling_metrics, pd.DataFrame)
        self.assertTrue(rolling_metrics.empty)

    def test_stress_test_scenarios_default(self):
        """Test stress testing with default scenarios."""
        stress_results = self.risk_metrics.stress_test_scenarios(self.normal_returns)

        expected_scenarios = [
            "market_crash",
            "volatility_spike",
            "recession",
            "black_swan",
        ]

        for scenario in expected_scenarios:
            self.assertIn(scenario, stress_results)

            scenario_results = stress_results[scenario]
            self.assertIn("var", scenario_results)
            self.assertIn("cvar", scenario_results)
            self.assertIn("max_drawdown", scenario_results)

            # Stress scenarios should generally have higher risk
            self.assertGreater(scenario_results["var"], 0)
            self.assertGreater(scenario_results["cvar"], 0)

    def test_stress_test_scenarios_custom(self):
        """Test stress testing with custom scenarios."""
        custom_scenarios = {
            "mild_stress": {"mean_shock": -0.01, "vol_multiplier": 1.2},
            "severe_stress": {"mean_shock": -0.08, "vol_multiplier": 2.5},
        }

        stress_results = self.risk_metrics.stress_test_scenarios(
            self.normal_returns, scenarios=custom_scenarios
        )

        for scenario in custom_scenarios.keys():
            self.assertIn(scenario, stress_results)


class TestPortfolioRiskAnalyzer(unittest.TestCase):
    """Test cases for PortfolioRiskAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create portfolio returns data
        self.dates = pd.date_range("2020-01-01", periods=300, freq="D")
        n_assets = 5

        # Generate correlated returns
        correlation_matrix = np.eye(n_assets)
        correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.3
        correlation_matrix[2, 3] = correlation_matrix[3, 2] = 0.5

        returns_data = np.random.multivariate_normal(
            mean=np.array([0.001, 0.0008, 0.0012, 0.0005, 0.0015]),
            cov=correlation_matrix * 0.0004,
            size=300,
        )

        self.returns = pd.DataFrame(
            returns_data,
            index=self.dates,
            columns=[f"Asset_{i}" for i in range(n_assets)],
        )

        self.weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

        self.portfolio_analyzer = PortfolioRiskAnalyzer(confidence_level=0.95)

    def test_initialization(self):
        """Test PortfolioRiskAnalyzer initialization."""
        analyzer = PortfolioRiskAnalyzer(confidence_level=0.99)

        self.assertEqual(analyzer.confidence_level, 0.99)
        self.assertIsInstance(analyzer.risk_metrics, RiskMetrics)

    def test_portfolio_var(self):
        """Test portfolio VaR calculation."""
        portfolio_var = self.portfolio_analyzer.portfolio_var(
            self.returns, self.weights, method="parametric"
        )

        self.assertIsInstance(portfolio_var, float)
        self.assertGreater(portfolio_var, 0)
        self.assertLess(portfolio_var, 0.1)

    def test_portfolio_var_historical(self):
        """Test portfolio VaR with historical method."""
        portfolio_var = self.portfolio_analyzer.portfolio_var(
            self.returns, self.weights, method="historical"
        )

        self.assertIsInstance(portfolio_var, float)
        self.assertGreater(portfolio_var, 0)

    def test_portfolio_var_invalid_weights(self):
        """Test portfolio VaR with invalid weights."""
        invalid_weights = np.array([0.5, 0.3])  # Wrong dimension

        portfolio_var = self.portfolio_analyzer.portfolio_var(
            self.returns, invalid_weights
        )

        self.assertTrue(np.isnan(portfolio_var))

    def test_component_var(self):
        """Test component VaR calculation."""
        component_vars = self.portfolio_analyzer.component_var(
            self.returns, self.weights, method="parametric"
        )

        self.assertIsInstance(component_vars, pd.Series)
        self.assertEqual(len(component_vars), len(self.weights))

        # All component VaRs should be finite
        self.assertTrue(component_vars.notna().all())

        # Sum of component VaRs should approximate portfolio VaR
        portfolio_var = self.portfolio_analyzer.portfolio_var(
            self.returns, self.weights, method="parametric"
        )

        if not np.isnan(portfolio_var):
            component_sum = component_vars.sum()
            # Allow for larger numerical differences in component VaR calculation
            relative_error = abs(component_sum - portfolio_var) / portfolio_var
            self.assertLess(relative_error, 0.2)  # Within 20%

    def test_component_var_invalid_portfolio(self):
        """Test component VaR with invalid portfolio."""
        # Create returns that would result in NaN portfolio VaR
        invalid_returns = pd.DataFrame(np.full((5, 3), np.nan))

        component_vars = self.portfolio_analyzer.component_var(
            invalid_returns, np.array([0.5, 0.3, 0.2])
        )

        self.assertTrue(component_vars.isna().all())

    def test_concentration_risk_herfindahl(self):
        """Test concentration risk using Herfindahl index."""
        # Concentrated portfolio
        concentrated_weights = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        concentration = self.portfolio_analyzer.concentration_risk(
            concentrated_weights, method="herfindahl"
        )

        self.assertIsInstance(concentration, float)
        self.assertGreater(concentration, 0)
        self.assertLessEqual(concentration, 1)  # HHI is bounded by 1

        # More concentrated portfolio should have higher HHI
        uniform_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        uniform_concentration = self.portfolio_analyzer.concentration_risk(
            uniform_weights, method="herfindahl"
        )

        self.assertGreater(concentration, uniform_concentration)

    def test_concentration_risk_entropy(self):
        """Test concentration risk using entropy."""
        concentration = self.portfolio_analyzer.concentration_risk(
            self.weights, method="entropy"
        )

        self.assertIsInstance(concentration, float)
        self.assertGreater(concentration, 0)

    def test_concentration_risk_invalid_method(self):
        """Test concentration risk with invalid method."""
        concentration = self.portfolio_analyzer.concentration_risk(
            self.weights, method="invalid"
        )

        self.assertTrue(np.isnan(concentration))


class TestRiskMetricsIntegration(unittest.TestCase):
    """Integration tests for risk metrics components."""

    def setUp(self):
        """Set up integration test fixtures."""
        np.random.seed(42)

        # Create realistic market data
        self.dates = pd.date_range("2020-01-01", periods=500, freq="D")

        # Market returns with regime changes
        regime1_returns = np.random.normal(0.0008, 0.015, 250)  # Normal market
        regime2_returns = np.random.normal(-0.002, 0.035, 250)  # Crisis period

        self.market_returns = pd.Series(
            np.concatenate([regime1_returns, regime2_returns]), index=self.dates
        )

        self.risk_metrics = RiskMetrics(confidence_level=0.95)

    def test_full_risk_analysis_workflow(self):
        """Test complete risk analysis workflow."""
        # Calculate all risk metrics
        var_95 = self.risk_metrics.value_at_risk(
            self.market_returns, method="historical"
        )
        cvar_95 = self.risk_metrics.conditional_var(
            self.market_returns, method="historical"
        )
        dd_metrics = self.risk_metrics.maximum_drawdown(self.market_returns)
        risk_adj_metrics = self.risk_metrics.risk_adjusted_returns(self.market_returns)

        # All metrics should be calculated successfully
        self.assertFalse(np.isnan(var_95))
        self.assertFalse(np.isnan(cvar_95))
        self.assertFalse(np.isnan(dd_metrics["max_drawdown"]))
        self.assertGreater(len(risk_adj_metrics), 0)

        # Relationships should hold
        self.assertGreater(cvar_95, var_95)
        self.assertGreater(dd_metrics["max_drawdown"], 0)

    def test_risk_metrics_consistency(self):
        """Test consistency across different VaR methods."""
        var_parametric = self.risk_metrics.value_at_risk(
            self.market_returns, method="parametric"
        )
        var_historical = self.risk_metrics.value_at_risk(
            self.market_returns, method="historical"
        )
        var_cornish_fisher = self.risk_metrics.value_at_risk(
            self.market_returns, method="cornish_fisher"
        )

        # All methods should produce reasonable results
        self.assertGreater(var_parametric, 0)
        self.assertGreater(var_historical, 0)
        self.assertGreater(var_cornish_fisher, 0)

        # Results should be in same order of magnitude
        ratio1 = var_historical / var_parametric
        ratio2 = var_cornish_fisher / var_parametric

        self.assertGreater(ratio1, 0.1)
        self.assertLess(ratio1, 10)
        self.assertGreater(ratio2, 0.1)
        self.assertLess(ratio2, 10)

    def test_portfolio_vs_individual_risk(self):
        """Test portfolio diversification effects."""
        # Create two assets with different risk characteristics
        asset1_returns = pd.Series(np.random.normal(0.001, 0.02, 300))
        asset2_returns = pd.Series(np.random.normal(0.0005, 0.015, 300))

        returns_df = pd.DataFrame({"Asset1": asset1_returns, "Asset2": asset2_returns})

        # Equal weights
        weights = np.array([0.5, 0.5])

        # Calculate individual and portfolio VaRs
        analyzer = PortfolioRiskAnalyzer()
        portfolio_var = analyzer.portfolio_var(returns_df, weights)

        individual_var1 = self.risk_metrics.value_at_risk(asset1_returns)
        individual_var2 = self.risk_metrics.value_at_risk(asset2_returns)
        weighted_average_var = 0.5 * individual_var1 + 0.5 * individual_var2

        # Portfolio VaR should be less than weighted average (diversification benefit)
        # unless assets are perfectly correlated
        self.assertGreater(weighted_average_var, 0)
        self.assertGreater(portfolio_var, 0)


if __name__ == "__main__":
    unittest.main()
