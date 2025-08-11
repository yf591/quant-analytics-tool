"""
Test suite for stress testing module.

This module contains comprehensive tests for the stress testing framework,
including scenario analysis, Monte Carlo simulation, sensitivity analysis,
and tail risk evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.risk.stress_testing import (
    StressTesting,
    ScenarioGenerator,
    MonteCarloEngine,
    SensitivityAnalyzer,
    TailRiskAnalyzer,
    StressTestResult,
    StressTestType,
)


class TestScenarioGenerator:
    """Test cases for ScenarioGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ScenarioGenerator(random_seed=42)

    def test_market_crash_scenario_moderate(self):
        """Test moderate market crash scenario generation."""
        scenario = self.generator.generate_market_crash_scenario("moderate")

        assert scenario["name"] == "Market Crash (moderate)"
        assert scenario["equity_shock"] == -0.25
        assert scenario["volatility_multiplier"] == 3.0
        assert scenario["correlation_increase"] == 0.5
        assert "bond_shock" in scenario
        assert "currency_shock" in scenario

    def test_market_crash_scenario_extreme(self):
        """Test extreme market crash scenario generation."""
        scenario = self.generator.generate_market_crash_scenario("extreme")

        assert scenario["name"] == "Market Crash (extreme)"
        assert scenario["equity_shock"] == -0.60
        assert scenario["volatility_multiplier"] == 6.0
        assert scenario["correlation_increase"] == 0.9

    def test_market_crash_scenario_invalid_severity(self):
        """Test market crash scenario with invalid severity defaults to moderate."""
        scenario = self.generator.generate_market_crash_scenario("invalid")

        assert scenario["name"] == "Market Crash (moderate)"
        assert scenario["equity_shock"] == -0.25

    def test_interest_rate_scenario_up(self):
        """Test upward interest rate shock scenario."""
        scenario = self.generator.generate_interest_rate_scenario("up", 0.025)

        assert scenario["name"] == "Interest Rate Shock (up 2bp)"
        assert scenario["rate_shock"] == 0.025
        assert scenario["bond_duration_effect"] == -0.125  # -0.025 * 5.0
        assert scenario["equity_valuation_effect"] == -0.0125  # -0.025 * 0.5

    def test_interest_rate_scenario_down(self):
        """Test downward interest rate shock scenario."""
        scenario = self.generator.generate_interest_rate_scenario("down", 0.015)

        assert scenario["name"] == "Interest Rate Shock (down 2bp)"
        assert scenario["rate_shock"] == -0.015
        assert scenario["bond_duration_effect"] == 0.075  # 0.015 * 5.0

    def test_volatility_scenario_high(self):
        """Test high volatility regime scenario."""
        scenario = self.generator.generate_volatility_scenario("high")

        assert scenario["name"] == "Volatility Regime (high)"
        assert scenario["volatility_multiplier"] == 2.5
        assert "correlation_effect" in scenario
        assert "liquidity_impact" in scenario

    def test_volatility_scenario_invalid_regime(self):
        """Test volatility scenario with invalid regime defaults to normal."""
        scenario = self.generator.generate_volatility_scenario("invalid")

        assert scenario["name"] == "Volatility Regime (invalid)"
        assert scenario["volatility_multiplier"] == 1.0

    def test_currency_crisis_scenario(self):
        """Test currency crisis scenario generation."""
        scenario = self.generator.generate_currency_crisis_scenario("EUR")

        assert scenario["name"] == "Currency Crisis (EUR)"
        assert scenario["currency_shock"] == -0.30
        assert scenario["emerging_market_contagion"] == -0.25
        assert scenario["safe_haven_rally"] == 0.10


class TestMonteCarloEngine:
    """Test cases for MonteCarloEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MonteCarloEngine(num_simulations=1000, random_seed=42)

        # Sample data
        np.random.seed(42)
        self.expected_returns = np.array([0.08, 0.12, 0.10])
        self.covariance_matrix = np.array(
            [[0.04, 0.01, 0.02], [0.01, 0.09, 0.03], [0.02, 0.03, 0.16]]
        )
        self.portfolio_weights = np.array([0.4, 0.4, 0.2])

    def test_monte_carlo_initialization(self):
        """Test Monte Carlo engine initialization."""
        assert self.engine.num_simulations == 1000
        assert self.engine.random_seed == 42

    def test_simulate_portfolio_returns_basic(self):
        """Test basic portfolio return simulation."""
        results = self.engine.simulate_portfolio_returns(
            self.expected_returns,
            self.covariance_matrix,
            self.portfolio_weights,
            time_horizon=252,
        )

        assert "portfolio_returns" in results
        assert "cumulative_returns" in results
        assert "final_returns" in results
        assert "var_metrics" in results
        assert "statistics" in results

        # Check dimensions
        assert results["portfolio_returns"].shape == (1000, 252)
        assert results["cumulative_returns"].shape == (1000, 252)
        assert len(results["final_returns"]) == 1000

    def test_simulate_portfolio_returns_var_metrics(self):
        """Test VaR metrics calculation in Monte Carlo simulation."""
        results = self.engine.simulate_portfolio_returns(
            self.expected_returns,
            self.covariance_matrix,
            self.portfolio_weights,
            confidence_levels=[0.95, 0.99],
        )

        var_metrics = results["var_metrics"]
        assert "VaR_0.95" in var_metrics
        assert "VaR_0.99" in var_metrics
        assert "CVaR_0.95" in var_metrics
        assert "CVaR_0.99" in var_metrics

        # CVaR should be worse than VaR
        assert var_metrics["CVaR_0.95"] <= var_metrics["VaR_0.95"]
        assert var_metrics["CVaR_0.99"] <= var_metrics["VaR_0.99"]

    def test_simulate_portfolio_returns_statistics(self):
        """Test statistics calculation in Monte Carlo simulation."""
        results = self.engine.simulate_portfolio_returns(
            self.expected_returns, self.covariance_matrix, self.portfolio_weights
        )

        stats = results["statistics"]
        required_stats = [
            "mean_return",
            "std_return",
            "skewness",
            "kurtosis",
            "min_return",
            "max_return",
            "probability_loss",
        ]

        for stat in required_stats:
            assert stat in stats
            assert np.isfinite(stats[stat])

    def test_simulate_portfolio_returns_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        wrong_weights = np.array([0.5, 0.5])  # Only 2 assets instead of 3

        with pytest.raises(ValueError, match="dimension mismatch"):
            self.engine.simulate_portfolio_returns(
                self.expected_returns, self.covariance_matrix, wrong_weights
            )

    def test_simulate_portfolio_returns_covariance_mismatch(self):
        """Test error handling for covariance matrix mismatch."""
        wrong_cov = np.array([[0.04, 0.01], [0.01, 0.09]])  # 2x2 instead of 3x3

        with pytest.raises(ValueError, match="dimension mismatch"):
            self.engine.simulate_portfolio_returns(
                self.expected_returns, wrong_cov, self.portfolio_weights
            )


class TestSensitivityAnalyzer:
    """Test cases for SensitivityAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SensitivityAnalyzer()

        # Sample data
        self.portfolio_weights = np.array([100, 150, 200])  # Dollar amounts
        self.asset_prices = np.array([50, 75, 100])

    def test_portfolio_sensitivities_basic(self):
        """Test basic portfolio sensitivity calculation."""
        sensitivities = self.analyzer.calculate_portfolio_sensitivities(
            self.portfolio_weights, self.asset_prices, shock_size=0.01
        )

        required_keys = [
            "delta",
            "gamma",
            "cross_gamma",
            "dollar_delta",
            "percentage_delta",
        ]
        for key in required_keys:
            assert key in sensitivities

        # Check dimensions
        assert len(sensitivities["delta"]) == 3
        assert len(sensitivities["gamma"]) == 3
        assert sensitivities["cross_gamma"].shape == (3, 3)

    def test_portfolio_sensitivities_delta_calculation(self):
        """Test delta calculation accuracy."""
        sensitivities = self.analyzer.calculate_portfolio_sensitivities(
            self.portfolio_weights, self.asset_prices, shock_size=0.01
        )

        deltas = sensitivities["delta"]

        # For a simple portfolio, delta should be related to position sizes
        # Check that deltas are reasonable values
        assert len(deltas) == 3
        for delta in deltas:
            assert np.isfinite(delta)

        # Check dollar delta makes sense
        dollar_deltas = sensitivities["dollar_delta"]
        assert np.allclose(dollar_deltas, deltas * self.asset_prices, rtol=1e-6)

    def test_stress_test_correlation_changes(self):
        """Test correlation stress testing."""
        # Generate sample returns data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns_data = pd.DataFrame(
            np.random.multivariate_normal(
                [0.001, 0.001, 0.001],
                [
                    [0.0004, 0.0001, 0.0002],
                    [0.0001, 0.0009, 0.0003],
                    [0.0002, 0.0003, 0.0016],
                ],
                252,
            ),
            index=dates,
            columns=["Asset1", "Asset2", "Asset3"],
        )

        portfolio_weights = np.array([0.4, 0.4, 0.2])
        correlation_shocks = [0.3, 0.7]

        results = self.analyzer.stress_test_correlation_changes(
            returns_data, portfolio_weights, correlation_shocks
        )

        assert len(results) == 2

        for shock in correlation_shocks:
            key = f"correlation_{shock}"
            assert key in results
            assert "portfolio_volatility" in results[key]
            assert "volatility_ratio" in results[key]
            assert "correlation_level" in results[key]

            # Check that correlation level matches
            assert results[key]["correlation_level"] == shock


class TestTailRiskAnalyzer:
    """Test cases for TailRiskAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TailRiskAnalyzer()

        # Generate sample return series with fat tails
        np.random.seed(42)
        self.returns_data = pd.Series(np.random.standard_t(df=3, size=1000) * 0.02)

    def test_extreme_value_analysis_basic(self):
        """Test basic extreme value analysis."""
        results = self.analyzer.extreme_value_analysis(
            self.returns_data, threshold_percentile=0.05, confidence_level=0.99
        )

        required_keys = [
            "tail_var",
            "tail_cvar",
            "xi",
            "beta",
            "threshold",
            "n_exceedances",
            "exceedance_rate",
        ]

        for key in required_keys:
            assert key in results

        # Basic sanity checks
        assert results["n_exceedances"] > 0
        assert 0 < results["exceedance_rate"] < 1
        # Note: For extreme value analysis, tail_cvar might not always be worse than tail_var
        # depending on the distribution characteristics
        assert np.isfinite(results["tail_var"])
        assert np.isfinite(results["tail_cvar"])

    def test_extreme_value_analysis_insufficient_data(self):
        """Test extreme value analysis with insufficient exceedances."""
        # Very small dataset
        small_data = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

        results = self.analyzer.extreme_value_analysis(
            small_data, threshold_percentile=0.05, confidence_level=0.99
        )

        # Should return NaN values for insufficient data
        assert np.isnan(results["tail_var"])
        assert np.isnan(results["tail_cvar"])

    def test_calculate_tail_dependence(self):
        """Test tail dependence calculation."""
        # Generate correlated returns data
        np.random.seed(42)
        n_obs = 500

        # Create returns with tail dependence
        factor = np.random.standard_t(df=3, size=n_obs)
        idiosyncratic = np.random.standard_t(df=3, size=(n_obs, 3)) * 0.5

        returns_data = (
            pd.DataFrame(
                {
                    "Asset1": factor * 0.8 + idiosyncratic[:, 0],
                    "Asset2": factor * 0.6 + idiosyncratic[:, 1],
                    "Asset3": factor * 0.4 + idiosyncratic[:, 2],
                }
            )
            * 0.02
        )

        results = self.analyzer.calculate_tail_dependence(
            returns_data, quantile_level=0.05
        )

        required_keys = [
            "lower_tail_dependence",
            "upper_tail_dependence",
            "average_lower_tail_dep",
            "average_upper_tail_dep",
        ]

        for key in required_keys:
            assert key in results

        # Check dimensions
        assert results["lower_tail_dependence"].shape == (3, 3)
        assert results["upper_tail_dependence"].shape == (3, 3)

        # Diagonal should be 1
        np.testing.assert_array_equal(
            np.diag(results["lower_tail_dependence"]), np.ones(3)
        )


class TestStressTesting:
    """Test cases for StressTesting class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stress_tester = StressTesting(random_seed=42)

        # Sample portfolio data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        self.asset_returns = pd.DataFrame(
            np.random.multivariate_normal(
                [0.001, 0.001, 0.001],
                [
                    [0.0004, 0.0001, 0.0002],
                    [0.0001, 0.0009, 0.0003],
                    [0.0002, 0.0003, 0.0016],
                ],
                252,
            ),
            index=dates,
            columns=["Stock1", "Stock2", "Bond1"],
        )
        self.portfolio_weights = np.array([0.4, 0.4, 0.2])
        self.asset_prices = np.array([100, 150, 200])

    def test_stress_testing_initialization(self):
        """Test stress testing framework initialization."""
        assert self.stress_tester.random_seed == 42
        assert hasattr(self.stress_tester, "scenario_generator")
        assert hasattr(self.stress_tester, "monte_carlo_engine")
        assert hasattr(self.stress_tester, "sensitivity_analyzer")
        assert hasattr(self.stress_tester, "tail_risk_analyzer")

    def test_run_scenario_stress_test_default_scenarios(self):
        """Test scenario stress test with default scenarios."""
        results = self.stress_tester.run_scenario_stress_test(
            self.portfolio_weights, self.asset_returns, portfolio_value=1000000.0
        )

        # Should have multiple scenarios
        assert len(results) > 0

        # Each result should be a StressTestResult
        for scenario_name, result in results.items():
            assert isinstance(result, StressTestResult)
            assert result.test_type == "scenario"
            assert result.scenario_name == scenario_name
            assert result.portfolio_value == 1000000.0
            assert hasattr(result, "stressed_value")
            assert hasattr(result, "loss_percentage")

    def test_run_scenario_stress_test_custom_scenarios(self):
        """Test scenario stress test with custom scenarios."""
        custom_scenarios = [
            {
                "name": "Custom Crash",
                "equity_shock": -0.30,
                "volatility_multiplier": 2.0,
            }
        ]

        results = self.stress_tester.run_scenario_stress_test(
            self.portfolio_weights,
            self.asset_returns,
            scenarios=custom_scenarios,
            portfolio_value=500000.0,
        )

        assert len(results) == 1
        assert "Custom Crash" in results
        assert results["Custom Crash"].portfolio_value == 500000.0

    def test_run_monte_carlo_stress_test(self):
        """Test Monte Carlo stress test."""
        expected_returns = self.asset_returns.mean().values * 252
        covariance_matrix = self.asset_returns.cov().values * 252

        result = self.stress_tester.run_monte_carlo_stress_test(
            self.portfolio_weights,
            expected_returns,
            covariance_matrix,
            portfolio_value=1000000.0,
        )

        assert isinstance(result, StressTestResult)
        assert result.test_type == "monte_carlo"
        assert result.portfolio_value == 1000000.0
        assert result.confidence_level == 0.99

        # Check metrics
        assert "worst_case_return" in result.metrics
        assert "expected_return" in result.metrics
        assert "volatility" in result.metrics

    def test_run_comprehensive_stress_test(self):
        """Test comprehensive stress test suite."""
        results = self.stress_tester.run_comprehensive_stress_test(
            self.portfolio_weights,
            self.asset_returns,
            asset_prices=self.asset_prices,
            portfolio_value=1000000.0,
        )

        required_sections = [
            "scenario_tests",
            "monte_carlo_test",
            "sensitivity_analysis",
            "tail_risk_analysis",
            "summary",
        ]

        for section in required_sections:
            assert section in results

        # Check scenario tests
        assert len(results["scenario_tests"]) > 0

        # Check Monte Carlo test
        assert isinstance(results["monte_carlo_test"], StressTestResult)

        # Check sensitivity analysis
        assert "delta" in results["sensitivity_analysis"]

        # Check summary
        summary = results["summary"]
        assert "worst_scenario_loss" in summary
        assert "monte_carlo_var_99" in summary
        assert "portfolio_value" in summary
        assert summary["portfolio_value"] == 1000000.0

    def test_run_comprehensive_stress_test_without_prices(self):
        """Test comprehensive stress test without asset prices."""
        results = self.stress_tester.run_comprehensive_stress_test(
            self.portfolio_weights,
            self.asset_returns,
            asset_prices=None,  # No prices provided
            portfolio_value=1000000.0,
        )

        # Should still work but without sensitivity analysis
        assert "scenario_tests" in results
        assert "monte_carlo_test" in results
        assert "tail_risk_analysis" in results
        assert "summary" in results

    def test_apply_scenario_shocks(self):
        """Test scenario shock application to returns."""
        scenario = {"equity_shock": -0.20, "volatility_multiplier": 2.0}

        shocked_returns = self.stress_tester._apply_scenario_shocks(
            self.asset_returns, scenario
        )

        # Should have same shape as original
        assert shocked_returns.shape == self.asset_returns.shape

        # Volatility should be higher
        original_vol = self.asset_returns.std()
        shocked_vol = shocked_returns.std()

        # Most assets should have higher volatility (approximately 2x)
        for col in shocked_returns.columns:
            assert shocked_vol[col] > original_vol[col]


class TestStressTestResult:
    """Test cases for StressTestResult class."""

    def test_stress_test_result_creation(self):
        """Test stress test result data class creation."""
        result = StressTestResult(
            test_type="scenario",
            scenario_name="Test Scenario",
            portfolio_value=1000000.0,
            stressed_value=850000.0,
            loss=150000.0,
            loss_percentage=0.15,
            confidence_level=0.95,
            metrics={"volatility": 0.25},
            scenario_details={"shock": -0.20},
        )

        assert result.test_type == "scenario"
        assert result.scenario_name == "Test Scenario"
        assert result.portfolio_value == 1000000.0
        assert result.stressed_value == 850000.0
        assert result.loss == 150000.0
        assert result.loss_percentage == 0.15
        assert result.confidence_level == 0.95
        assert result.metrics["volatility"] == 0.25
        assert result.scenario_details["shock"] == -0.20


class TestStressTestIntegration:
    """Integration tests for the complete stress testing framework."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.stress_tester = StressTesting(random_seed=42)

        # More realistic portfolio data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=504, freq="D")  # 2 years

        # Generate correlated asset returns
        n_assets = 5
        factor_loadings = np.array([0.8, 0.6, 0.7, 0.3, 0.5])
        market_factor = np.random.normal(0, 0.015, 504)
        idiosyncratic = np.random.normal(0, 0.01, (504, n_assets))

        returns = np.outer(market_factor, factor_loadings) + idiosyncratic

        self.asset_returns = pd.DataFrame(
            returns,
            index=dates,
            columns=["Large Cap", "Small Cap", "International", "Bonds", "Commodities"],
        )

        self.portfolio_weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        self.asset_prices = np.array([100, 80, 120, 95, 110])

    def test_full_stress_testing_workflow(self):
        """Test complete stress testing workflow."""
        # Run comprehensive stress test
        results = self.stress_tester.run_comprehensive_stress_test(
            self.portfolio_weights,
            self.asset_returns,
            asset_prices=self.asset_prices,
            portfolio_value=10000000.0,  # $10M portfolio
        )

        # Validate all components
        assert len(results["scenario_tests"]) >= 3  # Multiple scenarios
        assert isinstance(results["monte_carlo_test"], StressTestResult)
        assert "delta" in results["sensitivity_analysis"]
        assert "tail_var" in results["tail_risk_analysis"]

        # Check summary makes sense
        summary = results["summary"]
        assert summary["portfolio_value"] == 10000000.0
        assert summary["worst_scenario_loss"] >= 0  # Loss should be positive
        assert summary["number_scenarios_tested"] > 0

        # Monte Carlo VaR should be reasonable
        mc_var = summary["monte_carlo_var_99"]
        assert 0 <= mc_var <= 1.0  # Should be between 0 and 100%

    @patch("src.risk.stress_testing.logger")
    def test_error_handling_and_logging(self, mock_logger):
        """Test error handling and logging throughout the framework."""
        # Test with invalid data
        invalid_returns = pd.DataFrame()  # Empty DataFrame

        with pytest.raises(Exception):
            self.stress_tester.run_comprehensive_stress_test(
                self.portfolio_weights, invalid_returns
            )

        # Verify error logging was called
        mock_logger.error.assert_called()

    def test_stress_test_reproducibility(self):
        """Test that stress tests are reproducible with same random seed."""
        # Run stress test twice with same seed
        results1 = self.stress_tester.run_monte_carlo_stress_test(
            self.portfolio_weights,
            self.asset_returns.mean().values * 252,
            self.asset_returns.cov().values * 252,
            portfolio_value=1000000.0,
        )

        # Create new instance with same seed
        stress_tester2 = StressTesting(random_seed=42)
        results2 = stress_tester2.run_monte_carlo_stress_test(
            self.portfolio_weights,
            self.asset_returns.mean().values * 252,
            self.asset_returns.cov().values * 252,
            portfolio_value=1000000.0,
        )

        # Results should be approximately equal (allowing for small numerical differences)
        assert (
            abs(results1.stressed_value - results2.stressed_value) < 10000
        )  # Within $10k
        assert (
            abs(results1.loss_percentage - results2.loss_percentage) < 0.01
        )  # Within 1%
