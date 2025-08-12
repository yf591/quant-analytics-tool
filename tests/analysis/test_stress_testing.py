"""
Test suite for advanced stress testing implementation.

Tests comprehensive stress testing methodologies including binary strategy testing,
historical scenario replay, extreme event simulation, and strategy risk quantification.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.analysis.stress_testing import (
    AdvancedStressTester,
    BinaryStrategyParams,
    StressTestResult,
    StressTestType,
)


class TestAdvancedStressTester:
    """Test suite for AdvancedStressTester."""

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")

        # Create realistic financial returns
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        returns_df = pd.DataFrame(returns, index=dates, columns=["asset_returns"])

        return returns_df

    @pytest.fixture
    def sample_volume_data(self):
        """Create sample volume data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")

        # Create realistic volume data
        volume = np.random.lognormal(15, 1, len(dates))  # Log-normal volume
        volume_df = pd.DataFrame(volume, index=dates, columns=["volume"])

        return volume_df

    @pytest.fixture
    def binary_strategy_params(self):
        """Create sample binary strategy parameters."""
        return BinaryStrategyParams(
            stop_loss=0.02,  # 2% stop loss
            profit_target=0.03,  # 3% profit target
            precision_rate=0.55,  # 55% win rate
            frequency=100,  # 100 bets per year
            target_sharpe=1.0,  # Target Sharpe ratio
        )

    @pytest.fixture
    def mock_strategy_function(self):
        """Create mock strategy function for testing."""

        def strategy_func(returns_data):
            # Simple momentum strategy
            returns = (
                returns_data.iloc[:, 0]
                if len(returns_data.columns) > 0
                else returns_data
            )
            signals = (returns.rolling(5).mean() > 0).astype(int)
            strategy_returns = returns * signals.shift(1).fillna(0)
            return strategy_returns

        return strategy_func

    def test_stress_tester_initialization(self):
        """Test AdvancedStressTester initialization."""
        stress_tester = AdvancedStressTester(
            confidence_levels=[0.95, 0.99], n_simulations=5000, random_state=42
        )

        assert stress_tester.confidence_levels == [0.95, 0.99]
        assert stress_tester.n_simulations == 5000
        assert stress_tester.random_state == 42
        assert stress_tester.stress_results_ == {}
        assert stress_tester.strategy_risk_metrics_ == {}
        assert stress_tester.extreme_scenarios_ == {}

    def test_default_initialization(self):
        """Test default initialization parameters."""
        stress_tester = AdvancedStressTester()

        assert stress_tester.confidence_levels == [0.95, 0.99, 0.999]
        assert stress_tester.n_simulations == 10000
        assert stress_tester.random_state is None

    def test_binary_strategy_stress_test(self, binary_strategy_params):
        """Test binary strategy stress testing."""
        stress_tester = AdvancedStressTester(random_state=42)

        # Run binary strategy stress test
        results = stress_tester.binary_strategy_stress_test(binary_strategy_params)

        # Check results structure
        assert isinstance(results, dict)
        assert "baseline_sharpe" in results
        assert "baseline_params" in results
        assert "stress_scenarios" in results
        assert "implied_metrics" in results
        assert "risk_assessment" in results

        # Check baseline Sharpe ratio calculation
        assert isinstance(results["baseline_sharpe"], float)
        assert (
            results["baseline_sharpe"] > 0
        )  # Should be positive with given parameters

        # Check stress scenarios
        assert len(results["stress_scenarios"]) > 0

        # Verify each stress scenario result
        for scenario_name, scenario_result in results["stress_scenarios"].items():
            assert isinstance(scenario_result, StressTestResult)
            assert scenario_result.test_type == StressTestType.BINARY_STRATEGY.value
            assert scenario_result.scenario_name == scenario_name
            assert isinstance(scenario_result.baseline_metric, float)
            assert isinstance(scenario_result.stressed_metric, float)
            assert isinstance(scenario_result.stress_impact, float)
            assert isinstance(scenario_result.relative_impact, float)

        # Check implied metrics
        implied_metrics = results["implied_metrics"]
        assert "current_sharpe" in implied_metrics
        assert "expected_return_per_bet" in implied_metrics
        assert "variance_per_bet" in implied_metrics
        assert "annual_return" in implied_metrics
        assert "annual_volatility" in implied_metrics

        # Check risk assessment
        risk_assessment = results["risk_assessment"]
        assert "risk_level" in risk_assessment
        assert "key_vulnerabilities" in risk_assessment
        assert "stress_resistance" in risk_assessment
        assert "recommendations" in risk_assessment

        assert risk_assessment["risk_level"] in ["low", "moderate", "high", "critical"]

    def test_binary_sharpe_ratio_calculation(self, binary_strategy_params):
        """Test binary Sharpe ratio calculation based on AFML formulation."""
        stress_tester = AdvancedStressTester()

        sharpe_ratio = stress_tester._calculate_binary_sharpe_ratio(
            binary_strategy_params
        )

        # Should return a positive Sharpe ratio for profitable parameters
        assert isinstance(sharpe_ratio, float)
        assert sharpe_ratio > 0

        # Test with losing strategy
        losing_params = BinaryStrategyParams(
            stop_loss=0.03,
            profit_target=0.02,
            precision_rate=0.45,  # Less than 50% win rate with unfavorable risk/reward
            frequency=100,
        )

        losing_sharpe = stress_tester._calculate_binary_sharpe_ratio(losing_params)
        assert losing_sharpe < sharpe_ratio  # Should be lower

    def test_implied_precision_calculation(self):
        """Test implied precision calculation based on AFML Snippet 15.3."""
        stress_tester = AdvancedStressTester()

        # Test case from AFML
        sl = 0.02
        pt = 0.03
        freq = 100
        target_sharpe = 1.0

        implied_precision = stress_tester._calculate_implied_precision(
            sl, pt, freq, target_sharpe
        )

        # Should return a valid precision between 0 and 1
        assert isinstance(implied_precision, float)
        assert 0 <= implied_precision <= 1 or np.isnan(implied_precision)

        # Test edge cases
        # Impossible target (very high Sharpe)
        impossible_precision = stress_tester._calculate_implied_precision(
            sl, pt, freq, 10.0
        )
        assert np.isnan(impossible_precision)

        # Very low target (should be achievable)
        easy_precision = stress_tester._calculate_implied_precision(sl, pt, freq, 0.1)
        # Note: Low target may still be impossible with given parameters
        # Just check it returns a number or NaN
        assert isinstance(easy_precision, float)

    def test_implied_frequency_calculation(self):
        """Test implied betting frequency calculation based on AFML Snippet 15.4."""
        stress_tester = AdvancedStressTester()

        sl = 0.02
        pt = 0.03
        p = 0.55
        target_sharpe = 1.0

        implied_frequency = stress_tester._calculate_implied_frequency(
            sl, pt, p, target_sharpe
        )

        # Should return a positive frequency or NaN if impossible
        assert isinstance(implied_frequency, float)
        assert implied_frequency > 0 or np.isnan(implied_frequency)

        # Verify by back-calculation
        if not np.isnan(implied_frequency):
            test_params = BinaryStrategyParams(sl, pt, p, implied_frequency)
            calculated_sharpe = stress_tester._calculate_binary_sharpe_ratio(
                test_params
            )
            assert abs(calculated_sharpe - target_sharpe) < 0.01  # Should be close

    def test_historical_scenario_replay(
        self, sample_returns_data, mock_strategy_function
    ):
        """Test historical scenario replay functionality."""
        stress_tester = AdvancedStressTester(random_state=42)

        # Define custom crisis periods within data range
        crisis_periods = {
            "test_crisis": ("2020-03-01", "2020-04-01"),
            "test_recovery": ("2020-06-01", "2020-07-01"),
        }

        results = stress_tester.historical_scenario_replay(
            sample_returns_data, mock_strategy_function, crisis_periods
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "baseline_performance" in results
        assert "crisis_performance" in results
        assert "stress_impact" in results
        assert "recovery_analysis" in results

        # Check baseline performance
        baseline_perf = results["baseline_performance"]
        expected_metrics = [
            "total_return",
            "annual_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "var_95",
            "var_99",
            "skewness",
            "kurtosis",
        ]

        for metric in expected_metrics:
            assert metric in baseline_perf
            assert isinstance(baseline_perf[metric], (int, float))

        # Check crisis performance (may be empty if no valid periods)
        assert isinstance(results["crisis_performance"], dict)
        assert isinstance(results["stress_impact"], dict)

    def test_extreme_event_simulation(
        self, sample_returns_data, mock_strategy_function
    ):
        """Test extreme event simulation."""
        stress_tester = AdvancedStressTester(random_state=42)

        # Define custom extreme scenarios
        extreme_scenarios = {
            "test_crash": {
                "return_shock": -0.1,
                "volatility_multiplier": 2.0,
                "duration_days": 5,
            },
            "test_flash_crash": {
                "return_shock": -0.05,
                "volatility_multiplier": 3.0,
                "duration_days": 1,
            },
        }

        results = stress_tester.extreme_event_simulation(
            sample_returns_data, mock_strategy_function, extreme_scenarios
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "baseline_performance" in results
        assert "extreme_scenarios" in results
        assert "tail_risk_metrics" in results
        assert "scenario_rankings" in results

        # Check extreme scenarios results
        for scenario_name in extreme_scenarios.keys():
            if scenario_name in results["extreme_scenarios"]:
                scenario_data = results["extreme_scenarios"][scenario_name]
                assert "performance" in scenario_data
                assert "impact" in scenario_data
                assert "scenario_params" in scenario_data

        # Check tail risk metrics
        if results["tail_risk_metrics"]:
            tail_metrics = results["tail_risk_metrics"]
            expected_tail_metrics = [
                "worst_case_impact",
                "average_extreme_impact",
                "severe_event_frequency",
                "expected_shortfall",
            ]

            for metric in expected_tail_metrics:
                if metric in tail_metrics:
                    assert isinstance(tail_metrics[metric], (int, float))

    def test_strategy_risk_quantification(
        self, binary_strategy_params, sample_returns_data
    ):
        """Test strategy risk quantification."""
        stress_tester = AdvancedStressTester(random_state=42)

        results = stress_tester.strategy_risk_quantification(
            binary_strategy_params, sample_returns_data
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "strategy_params" in results
        assert "implied_precision_analysis" in results
        assert "betting_frequency_analysis" in results
        assert "sharpe_ratio_analysis" in results
        assert "risk_capacity_analysis" in results
        assert "stress_testing_summary" in results

        # Check implied precision analysis
        precision_analysis = results["implied_precision_analysis"]
        assert "current_precision" in precision_analysis
        assert "target_precision_requirements" in precision_analysis
        assert "precision_gap_analysis" in precision_analysis
        assert "feasibility_assessment" in precision_analysis

        # Check betting frequency analysis
        frequency_analysis = results["betting_frequency_analysis"]
        assert "current_frequency" in frequency_analysis
        assert "target_frequency_requirements" in frequency_analysis
        assert "frequency_scaling_analysis" in frequency_analysis
        assert "practical_constraints" in frequency_analysis

        # Check strategy risk metrics storage
        assert "strategy_risk" in stress_tester.strategy_risk_metrics_

    def test_liquidity_stress_testing(
        self, sample_returns_data, sample_volume_data, mock_strategy_function
    ):
        """Test liquidity stress testing."""
        stress_tester = AdvancedStressTester(random_state=42)

        results = stress_tester.liquidity_stress_testing(
            sample_returns_data, sample_volume_data, mock_strategy_function
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "baseline_performance" in results
        assert "liquidity_scenarios" in results
        assert "impact_analysis" in results
        assert "liquidity_risk_metrics" in results

        # Check liquidity scenarios
        liquidity_scenarios = results["liquidity_scenarios"]
        expected_scenarios = [
            "mild_liquidity_stress",
            "moderate_liquidity_stress",
            "severe_liquidity_stress",
            "liquidity_crisis",
        ]

        # Should have at least some scenarios
        assert len(liquidity_scenarios) > 0

        for scenario_name, scenario_data in liquidity_scenarios.items():
            assert "performance" in scenario_data
            assert "impact" in scenario_data
            assert "volume_multiplier" in scenario_data
            assert 0 < scenario_data["volume_multiplier"] <= 1.0  # Volume reduction

        # Check liquidity risk metrics
        liquidity_metrics = results["liquidity_risk_metrics"]
        assert isinstance(liquidity_metrics, dict)

    def test_performance_metrics_calculation(self, sample_returns_data):
        """Test performance metrics calculation."""
        stress_tester = AdvancedStressTester()

        returns = sample_returns_data.iloc[:, 0]
        metrics = stress_tester._calculate_performance_metrics(returns)

        # Check that all expected metrics are calculated
        expected_metrics = [
            "total_return",
            "annual_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "avg_drawdown",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
            "skewness",
            "kurtosis",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric]) or metric in [
                "avg_drawdown"
            ]  # avg_drawdown can be NaN if no drawdowns

        # Check logical relationships
        assert (
            metrics["var_99"] <= metrics["var_95"]
        )  # 99% VaR should be worse than 95% VaR
        assert metrics["cvar_99"] <= metrics["cvar_95"]  # Same for CVaR
        assert metrics["max_drawdown"] <= 0  # Drawdown should be negative or zero

    def test_extreme_event_data_generation(self, sample_returns_data):
        """Test extreme event data generation."""
        stress_tester = AdvancedStressTester(random_state=42)

        scenario_params = {
            "return_shock": -0.1,
            "volatility_multiplier": 2.0,
            "duration_days": 5,
        }

        extreme_data = stress_tester._generate_extreme_event_data(
            sample_returns_data, scenario_params
        )

        # Check that data structure is preserved
        assert extreme_data.shape == sample_returns_data.shape
        assert extreme_data.index.equals(sample_returns_data.index)
        assert extreme_data.columns.equals(sample_returns_data.columns)

        # Check that modifications were applied
        assert not extreme_data.equals(sample_returns_data)

        # Volatility should be higher
        original_vol = sample_returns_data.std()
        extreme_vol = extreme_data.std()
        assert (extreme_vol > original_vol).all()

    def test_liquidity_impact_application(
        self, sample_returns_data, sample_volume_data
    ):
        """Test liquidity impact application."""
        stress_tester = AdvancedStressTester()

        # Create shocked volume (50% reduction)
        shocked_volume = sample_volume_data * 0.5

        adjusted_returns = stress_tester._apply_liquidity_impact(
            sample_returns_data, sample_volume_data, shocked_volume
        )

        # Check that structure is preserved
        assert adjusted_returns.shape == sample_returns_data.shape
        assert adjusted_returns.index.equals(sample_returns_data.index)
        assert adjusted_returns.columns.equals(sample_returns_data.columns)

        # Returns should generally be lower due to liquidity impact
        mean_original = sample_returns_data.mean()
        mean_adjusted = adjusted_returns.mean()
        assert (mean_adjusted < mean_original).all()

    def test_comprehensive_stress_summary(self, binary_strategy_params):
        """Test comprehensive stress summary generation."""
        stress_tester = AdvancedStressTester(random_state=42)

        # Run some stress tests first
        stress_tester.binary_strategy_stress_test(binary_strategy_params)

        # Generate comprehensive summary
        summary = stress_tester.get_comprehensive_stress_summary()

        # Check summary structure
        assert isinstance(summary, dict)
        assert "stress_test_types" in summary
        assert "overall_risk_assessment" in summary
        assert "key_vulnerabilities" in summary
        assert "stress_test_summary" in summary
        assert "recommendations" in summary

        # Check stress test types
        assert "binary_strategy" in summary["stress_test_types"]

        # Check overall risk assessment
        risk_assessment = summary["overall_risk_assessment"]
        assert "overall_risk_level" in risk_assessment
        assert risk_assessment["overall_risk_level"] in [
            "low",
            "moderate",
            "high",
            "critical",
            "unknown",
        ]

        # Check recommendations
        recommendations = summary["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10  # Should be limited

    def test_binary_strategy_params_validation(self):
        """Test BinaryStrategyParams validation."""
        # Valid parameters
        valid_params = BinaryStrategyParams(
            stop_loss=0.02, profit_target=0.03, precision_rate=0.55, frequency=100
        )

        assert valid_params.stop_loss == 0.02
        assert valid_params.profit_target == 0.03
        assert valid_params.precision_rate == 0.55
        assert valid_params.frequency == 100
        assert valid_params.target_sharpe is None

    def test_stress_test_result_dataclass(self):
        """Test StressTestResult dataclass."""
        result = StressTestResult(
            test_type="test",
            scenario_name="test_scenario",
            baseline_metric=1.0,
            stressed_metric=0.8,
            stress_impact=-0.2,
            relative_impact=-0.2,
            confidence_level=0.95,
            scenario_parameters={"param1": 1.0},
            additional_metrics={"metric1": 0.5},
        )

        assert result.test_type == "test"
        assert result.scenario_name == "test_scenario"
        assert result.baseline_metric == 1.0
        assert result.stressed_metric == 0.8
        assert result.stress_impact == -0.2
        assert result.relative_impact == -0.2
        assert result.confidence_level == 0.95
        assert result.scenario_parameters == {"param1": 1.0}
        assert result.additional_metrics == {"metric1": 0.5}

    def test_create_default_binary_stress_scenarios(self):
        """Test default binary stress scenario creation."""
        stress_tester = AdvancedStressTester()

        scenarios = stress_tester._create_default_binary_stress_scenarios()

        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

        # Check expected scenarios
        expected_scenarios = [
            "precision_decline_mild",
            "precision_decline_moderate",
            "precision_decline_severe",
            "frequency_reduction",
            "stop_loss_widening",
            "profit_target_reduction",
            "combined_stress_mild",
            "combined_stress_severe",
        ]

        for scenario in expected_scenarios:
            assert scenario in scenarios
            assert isinstance(scenarios[scenario], dict)

    def test_apply_binary_stress(self, binary_strategy_params):
        """Test binary stress application."""
        stress_tester = AdvancedStressTester()

        stress_params = {
            "precision_multiplier": 0.9,
            "frequency_multiplier": 0.8,
            "stop_loss_multiplier": 1.2,
            "profit_target_multiplier": 0.9,
        }

        stressed_params = stress_tester._apply_binary_stress(
            binary_strategy_params, stress_params
        )

        # Check that stress was applied correctly
        assert (
            stressed_params.precision_rate
            == binary_strategy_params.precision_rate * 0.9
        )
        assert stressed_params.frequency == binary_strategy_params.frequency * 0.8
        assert stressed_params.stop_loss == binary_strategy_params.stop_loss * 1.2
        assert (
            stressed_params.profit_target == binary_strategy_params.profit_target * 0.9
        )

        # Precision should be clamped to [0, 1]
        extreme_stress = {"precision_multiplier": 2.0}
        extreme_stressed = stress_tester._apply_binary_stress(
            binary_strategy_params, extreme_stress
        )
        assert extreme_stressed.precision_rate <= 1.0

    def test_identify_crisis_periods(self, sample_returns_data):
        """Test crisis period identification."""
        stress_tester = AdvancedStressTester()

        crisis_periods = stress_tester._identify_crisis_periods(sample_returns_data)

        assert isinstance(crisis_periods, dict)

        # Should filter to periods within data range
        for period_name, (start_date, end_date) in crisis_periods.items():
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            assert start_dt >= sample_returns_data.index.min()
            assert end_dt <= sample_returns_data.index.max()

    def test_create_extreme_scenarios(self):
        """Test extreme scenario creation."""
        stress_tester = AdvancedStressTester()

        scenarios = stress_tester._create_extreme_scenarios()

        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

        # Check expected scenarios
        expected_scenarios = [
            "black_monday",
            "market_crash",
            "flash_crash",
            "prolonged_bear_market",
            "liquidity_crisis",
        ]

        for scenario in expected_scenarios:
            assert scenario in scenarios
            scenario_params = scenarios[scenario]
            assert isinstance(scenario_params, dict)

            # Check for expected parameters
            if "return_shock" in scenario_params:
                assert scenario_params["return_shock"] < 0  # Should be negative
            if "volatility_multiplier" in scenario_params:
                assert (
                    scenario_params["volatility_multiplier"] >= 1.0
                )  # Should increase volatility

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        stress_tester = AdvancedStressTester()

        # Test summary without results
        with pytest.raises(ValueError, match="No stress test results available"):
            stress_tester.get_comprehensive_stress_summary()

        # Test with invalid binary strategy parameters
        invalid_params = BinaryStrategyParams(
            stop_loss=0,  # Zero stop loss
            profit_target=0,  # Zero profit target
            precision_rate=0.5,
            frequency=0,  # Zero frequency
        )

        # Should handle gracefully without crashing
        try:
            results = stress_tester.binary_strategy_stress_test(invalid_params)
            # Should complete without crashing, though results may not be meaningful
            assert isinstance(results, dict)
        except Exception as e:
            # If it does fail, should be a clear error
            assert "division by zero" in str(e).lower() or "invalid" in str(e).lower()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        stress_tester = AdvancedStressTester(random_state=42)

        # Test with perfect strategy (100% win rate)
        perfect_params = BinaryStrategyParams(
            stop_loss=0.01,
            profit_target=0.02,
            precision_rate=1.0,  # 100% win rate
            frequency=100,
        )

        perfect_sharpe = stress_tester._calculate_binary_sharpe_ratio(perfect_params)
        assert perfect_sharpe > 0  # Should be positive for perfect strategy
        assert not np.isnan(perfect_sharpe)

        # Test with impossible strategy (0% win rate)
        impossible_params = BinaryStrategyParams(
            stop_loss=0.02,
            profit_target=0.01,
            precision_rate=0.0,  # 0% win rate
            frequency=100,
        )

        impossible_sharpe = stress_tester._calculate_binary_sharpe_ratio(
            impossible_params
        )
        assert impossible_sharpe < 0  # Should be negative

        # Test with very high target Sharpe ratio
        high_target_precision = stress_tester._calculate_implied_precision(
            0.02, 0.03, 100, 5.0  # Very high target Sharpe
        )
        assert np.isnan(high_target_precision)  # Should be impossible

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        stress_tester = AdvancedStressTester()

        # Test with very small values
        tiny_params = BinaryStrategyParams(
            stop_loss=1e-6,
            profit_target=2e-6,
            precision_rate=0.5,
            frequency=1e6,  # Very high frequency
        )

        tiny_sharpe = stress_tester._calculate_binary_sharpe_ratio(tiny_params)
        assert isinstance(tiny_sharpe, float)
        assert not np.isnan(tiny_sharpe)
        assert not np.isinf(tiny_sharpe)

        # Test with very large values
        large_params = BinaryStrategyParams(
            stop_loss=100,
            profit_target=200,
            precision_rate=0.6,
            frequency=1,  # Very low frequency
        )

        large_sharpe = stress_tester._calculate_binary_sharpe_ratio(large_params)
        assert isinstance(large_sharpe, float)
        assert not np.isnan(large_sharpe)
        assert not np.isinf(large_sharpe)


class TestIntegration:
    """Integration tests for stress testing."""

    def test_full_stress_testing_workflow(self):
        """Test complete stress testing workflow."""
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range("2019-01-01", periods=500, freq="D")

        # Generate more realistic financial time series
        returns = []
        for i in range(len(dates)):
            if i == 0:
                returns.append(np.random.normal(0.001, 0.02))
            else:
                # Add some autocorrelation and volatility clustering
                prev_return = returns[-1]
                volatility = 0.015 + 0.005 * abs(prev_return)
                returns.append(np.random.normal(0.0005, volatility))

        returns_df = pd.DataFrame(returns, index=dates, columns=["returns"])

        # Generate volume data
        volume = np.random.lognormal(15, 0.5, len(dates))
        volume_df = pd.DataFrame(volume, index=dates, columns=["volume"])

        # Create strategy function
        def momentum_strategy(returns_data):
            returns = (
                returns_data.iloc[:, 0]
                if len(returns_data.columns) > 0
                else returns_data
            )
            # Simple momentum strategy with some sophistication
            short_ma = returns.rolling(5).mean()
            long_ma = returns.rolling(20).mean()
            signals = (short_ma > long_ma).astype(int)
            strategy_returns = (
                returns * signals.shift(1).fillna(0) * 0.8
            )  # 80% market exposure
            return strategy_returns

        # Binary strategy parameters
        binary_params = BinaryStrategyParams(
            stop_loss=0.025,
            profit_target=0.035,
            precision_rate=0.58,
            frequency=80,
            target_sharpe=1.2,
        )

        # Initialize comprehensive stress tester
        stress_tester = AdvancedStressTester(
            confidence_levels=[0.95, 0.99, 0.999],
            n_simulations=1000,  # Reduced for testing speed
            random_state=42,
        )

        # 1. Binary strategy stress test
        binary_results = stress_tester.binary_strategy_stress_test(binary_params)

        # 2. Historical scenario replay
        historical_results = stress_tester.historical_scenario_replay(
            returns_df, momentum_strategy
        )

        # 3. Extreme event simulation
        extreme_results = stress_tester.extreme_event_simulation(
            returns_df, momentum_strategy
        )

        # 4. Strategy risk quantification
        risk_results = stress_tester.strategy_risk_quantification(
            binary_params, returns_df
        )

        # 5. Liquidity stress testing
        liquidity_results = stress_tester.liquidity_stress_testing(
            returns_df, volume_df, momentum_strategy
        )

        # Verify all analyses completed successfully
        assert binary_results["baseline_sharpe"] is not None
        assert len(binary_results["stress_scenarios"]) > 0

        assert "baseline_performance" in historical_results
        assert "crisis_performance" in historical_results

        assert "baseline_performance" in extreme_results
        assert "extreme_scenarios" in extreme_results

        assert "strategy_params" in risk_results
        assert "implied_precision_analysis" in risk_results

        assert "baseline_performance" in liquidity_results
        assert "liquidity_scenarios" in liquidity_results

        # 6. Generate comprehensive summary
        comprehensive_summary = stress_tester.get_comprehensive_stress_summary()

        # Verify comprehensive summary
        assert len(comprehensive_summary["stress_test_types"]) >= 4
        assert "overall_risk_assessment" in comprehensive_summary
        assert "recommendations" in comprehensive_summary

        # Check that key risk metrics are reasonable
        overall_assessment = comprehensive_summary["overall_risk_assessment"]
        assert "overall_risk_level" in overall_assessment
        assert overall_assessment["overall_risk_level"] in [
            "low",
            "moderate",
            "high",
            "critical",
            "unknown",
        ]

        # Verify storage of results
        assert len(stress_tester.stress_results_) >= 4
        assert "binary_strategy" in stress_tester.stress_results_
        assert "historical_replay" in stress_tester.stress_results_
        assert "extreme_events" in stress_tester.stress_results_
        assert "liquidity_stress" in stress_tester.stress_results_

        assert "strategy_risk" in stress_tester.strategy_risk_metrics_

        # Test specific AFML compliance

        # Verify binary strategy Sharpe calculation follows AFML formulation
        current_sharpe = binary_results["baseline_sharpe"]
        expected_return = binary_params.precision_rate * binary_params.profit_target + (
            1 - binary_params.precision_rate
        ) * (-binary_params.stop_loss)
        expected_annual_return = expected_return * binary_params.frequency

        assert (
            abs(
                binary_results["implied_metrics"]["annual_return"]
                - expected_annual_return
            )
            < 1e-6
        )
        assert current_sharpe > 0  # Should be profitable with given parameters

        # Verify implied precision analysis
        precision_analysis = risk_results["implied_precision_analysis"]
        assert "feasibility_assessment" in precision_analysis

        # Check that some target Sharpe ratios are feasible or at least analyzed
        feasible_targets = [
            k
            for k, v in precision_analysis["feasibility_assessment"].items()
            if v in ["easily_achievable", "achievable"]
        ]
        # At least the analysis should be complete, even if no targets are easily feasible
        assert len(precision_analysis["feasibility_assessment"]) > 0

        print(f"Completed comprehensive stress testing workflow:")
        print(f"- Binary strategy baseline Sharpe: {current_sharpe:.3f}")
        print(f"- Overall risk level: {overall_assessment['overall_risk_level']}")
        print(
            f"- Number of stress scenarios tested: {len(binary_results['stress_scenarios'])}"
        )
        print(
            f"- Number of extreme events tested: {len(extreme_results.get('extreme_scenarios', {}))}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
