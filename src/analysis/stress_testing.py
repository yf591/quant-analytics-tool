"""
Advanced Stress Testing Module for Quantitative Analysis

Implements comprehensive stress testing methodologies based on AFML Chapter 15
"Understanding Strategy Risk" with advanced scenario analysis, tail risk evaluation,
and strategy risk assessment.

This module focuses on strategy-level stress testing including:
- Binary outcome strategy testing
- Implied precision and betting frequency analysis
- Historical scenario replay
- Extreme event simulation
- Strategy risk quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from scipy import stats
from scipy.optimize import minimize_scalar, brentq
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """Types of stress tests available."""

    BINARY_STRATEGY = "binary_strategy"
    HISTORICAL_REPLAY = "historical_replay"
    EXTREME_EVENTS = "extreme_events"
    TAIL_RISK = "tail_risk"
    STRATEGY_RISK = "strategy_risk"
    LIQUIDITY_STRESS = "liquidity_stress"


@dataclass
class BinaryStrategyParams:
    """Parameters for binary strategy stress testing."""

    stop_loss: float
    profit_target: float
    precision_rate: float
    frequency: float  # bets per year
    target_sharpe: Optional[float] = None


@dataclass
class StressTestResult:
    """Result container for stress test analysis."""

    test_type: str
    scenario_name: str
    baseline_metric: float
    stressed_metric: float
    stress_impact: float
    relative_impact: float
    confidence_level: float
    scenario_parameters: Dict[str, Any]
    additional_metrics: Dict[str, float]


class AdvancedStressTester:
    """
    Advanced stress testing analyzer implementing AFML methodologies.

    Based on Chapter 15: Understanding Strategy Risk, this class provides
    comprehensive stress testing capabilities for quantitative strategies.
    """

    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99, 0.999],
        n_simulations: int = 10000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the Advanced Stress Tester.

        Args:
            confidence_levels: List of confidence levels for stress testing
            n_simulations: Number of Monte Carlo simulations
            random_state: Random state for reproducibility
        """
        self.confidence_levels = confidence_levels
        self.n_simulations = n_simulations
        self.random_state = random_state

        # Results storage
        self.stress_results_ = {}
        self.strategy_risk_metrics_ = {}
        self.extreme_scenarios_ = {}

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

    def binary_strategy_stress_test(
        self,
        strategy_params: BinaryStrategyParams,
        stress_scenarios: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Perform stress testing on binary outcome strategies.

        Based on AFML Snippet 15.3-15.5, tests strategy robustness under
        various parameter perturbations and market conditions.

        Args:
            strategy_params: Binary strategy parameters
            stress_scenarios: Custom stress scenarios

        Returns:
            Dictionary containing stress test results
        """
        if stress_scenarios is None:
            stress_scenarios = self._create_default_binary_stress_scenarios()

        results = {
            "baseline_sharpe": self._calculate_binary_sharpe_ratio(strategy_params),
            "baseline_params": strategy_params,
            "stress_scenarios": {},
            "implied_metrics": {},
            "risk_assessment": {},
        }

        # Calculate baseline implied metrics
        results["implied_metrics"] = self._calculate_implied_metrics(strategy_params)

        # Test each stress scenario
        for scenario_name, scenario_params in stress_scenarios.items():
            try:
                # Create stressed strategy parameters
                stressed_params = self._apply_binary_stress(
                    strategy_params, scenario_params
                )

                # Calculate stressed metrics
                stressed_sharpe = self._calculate_binary_sharpe_ratio(stressed_params)

                # Calculate impact
                stress_impact = stressed_sharpe - results["baseline_sharpe"]
                relative_impact = (
                    stress_impact / results["baseline_sharpe"]
                    if results["baseline_sharpe"] != 0
                    else 0
                )

                results["stress_scenarios"][scenario_name] = StressTestResult(
                    test_type=StressTestType.BINARY_STRATEGY.value,
                    scenario_name=scenario_name,
                    baseline_metric=results["baseline_sharpe"],
                    stressed_metric=stressed_sharpe,
                    stress_impact=stress_impact,
                    relative_impact=relative_impact,
                    confidence_level=0.95,  # Default confidence level
                    scenario_parameters=scenario_params,
                    additional_metrics={
                        "stressed_precision": stressed_params.precision_rate,
                        "stressed_frequency": stressed_params.frequency,
                        "stressed_stop_loss": stressed_params.stop_loss,
                        "stressed_profit_target": stressed_params.profit_target,
                    },
                )

            except Exception as e:
                logger.warning(
                    f"Failed to process stress scenario {scenario_name}: {str(e)}"
                )
                continue

        # Calculate risk assessment
        results["risk_assessment"] = self._assess_binary_strategy_risk(results)

        # Store results
        self.stress_results_["binary_strategy"] = results

        return results

    def historical_scenario_replay(
        self,
        returns_data: pd.DataFrame,
        strategy_function: Callable,
        crisis_periods: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Replay historical crisis scenarios to test strategy performance.

        Args:
            returns_data: Historical returns data
            strategy_function: Function that takes returns and produces strategy returns
            crisis_periods: List of (start_date, end_date) tuples for crisis periods

        Returns:
            Dictionary containing historical replay results
        """
        if crisis_periods is None:
            crisis_periods = self._identify_crisis_periods(returns_data)

        results = {
            "baseline_performance": {},
            "crisis_performance": {},
            "stress_impact": {},
            "recovery_analysis": {},
        }

        # Calculate baseline performance
        baseline_strategy_returns = strategy_function(returns_data)
        results["baseline_performance"] = self._calculate_performance_metrics(
            baseline_strategy_returns
        )

        # Test each crisis period
        for period_name, (start_date, end_date) in crisis_periods.items():
            try:
                # Extract crisis period data
                crisis_mask = (returns_data.index >= start_date) & (
                    returns_data.index <= end_date
                )
                crisis_returns = returns_data[crisis_mask]

                if len(crisis_returns) == 0:
                    logger.warning(f"No data found for crisis period {period_name}")
                    continue

                # Calculate strategy performance during crisis
                crisis_strategy_returns = strategy_function(crisis_returns)
                crisis_performance = self._calculate_performance_metrics(
                    crisis_strategy_returns
                )

                # Calculate stress impact
                performance_impact = {}
                for metric, baseline_value in results["baseline_performance"].items():
                    crisis_value = crisis_performance.get(metric, 0)
                    impact = crisis_value - baseline_value
                    relative_impact = (
                        impact / baseline_value if baseline_value != 0 else 0
                    )

                    performance_impact[metric] = {
                        "absolute_impact": impact,
                        "relative_impact": relative_impact,
                        "crisis_value": crisis_value,
                        "baseline_value": baseline_value,
                    }

                results["crisis_performance"][period_name] = crisis_performance
                results["stress_impact"][period_name] = performance_impact

                # Analyze recovery period (6 months after crisis)
                recovery_end = pd.to_datetime(end_date) + timedelta(days=180)
                recovery_mask = (returns_data.index > end_date) & (
                    returns_data.index <= recovery_end
                )

                if recovery_mask.any():
                    recovery_returns = returns_data[recovery_mask]
                    recovery_strategy_returns = strategy_function(recovery_returns)
                    recovery_performance = self._calculate_performance_metrics(
                        recovery_strategy_returns
                    )
                    results["recovery_analysis"][period_name] = recovery_performance

            except Exception as e:
                logger.warning(
                    f"Failed to process crisis period {period_name}: {str(e)}"
                )
                continue

        # Store results
        self.stress_results_["historical_replay"] = results

        return results

    def extreme_event_simulation(
        self,
        returns_data: pd.DataFrame,
        strategy_function: Callable,
        extreme_scenarios: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate extreme market events and test strategy performance.

        Args:
            returns_data: Historical returns data
            strategy_function: Function that takes returns and produces strategy returns
            extreme_scenarios: Custom extreme event scenarios

        Returns:
            Dictionary containing extreme event simulation results
        """
        if extreme_scenarios is None:
            extreme_scenarios = self._create_extreme_scenarios()

        results = {
            "baseline_performance": {},
            "extreme_scenarios": {},
            "tail_risk_metrics": {},
            "scenario_rankings": {},
        }

        # Calculate baseline performance
        baseline_strategy_returns = strategy_function(returns_data)
        results["baseline_performance"] = self._calculate_performance_metrics(
            baseline_strategy_returns
        )

        # Test each extreme scenario
        scenario_impacts = {}

        for scenario_name, scenario_params in extreme_scenarios.items():
            try:
                # Generate extreme event data
                extreme_returns = self._generate_extreme_event_data(
                    returns_data, scenario_params
                )

                # Calculate strategy performance under extreme conditions
                extreme_strategy_returns = strategy_function(extreme_returns)
                extreme_performance = self._calculate_performance_metrics(
                    extreme_strategy_returns
                )

                # Calculate impact
                performance_impact = {}
                for metric, baseline_value in results["baseline_performance"].items():
                    extreme_value = extreme_performance.get(metric, 0)
                    impact = extreme_value - baseline_value
                    relative_impact = (
                        impact / baseline_value if baseline_value != 0 else 0
                    )

                    performance_impact[metric] = {
                        "absolute_impact": impact,
                        "relative_impact": relative_impact,
                        "extreme_value": extreme_value,
                        "baseline_value": baseline_value,
                    }

                results["extreme_scenarios"][scenario_name] = {
                    "performance": extreme_performance,
                    "impact": performance_impact,
                    "scenario_params": scenario_params,
                }

                # Store overall impact for ranking
                overall_impact = performance_impact.get("total_return", {}).get(
                    "relative_impact", 0
                )
                scenario_impacts[scenario_name] = overall_impact

            except Exception as e:
                logger.warning(
                    f"Failed to process extreme scenario {scenario_name}: {str(e)}"
                )
                continue

        # Rank scenarios by impact
        results["scenario_rankings"] = dict(
            sorted(scenario_impacts.items(), key=lambda x: x[1])
        )

        # Calculate tail risk metrics
        if results["extreme_scenarios"]:
            results["tail_risk_metrics"] = self._calculate_tail_risk_metrics(results)

        # Store results
        self.stress_results_["extreme_events"] = results

        return results

    def strategy_risk_quantification(
        self, strategy_params: BinaryStrategyParams, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Quantify strategy risk using AFML methodologies.

        Based on AFML Chapter 15, calculates implied precision requirements,
        betting frequency implications, and strategy risk metrics.

        Args:
            strategy_params: Strategy parameters
            market_data: Historical market data

        Returns:
            Dictionary containing strategy risk quantification
        """
        results = {
            "strategy_params": strategy_params,
            "implied_precision_analysis": {},
            "betting_frequency_analysis": {},
            "sharpe_ratio_analysis": {},
            "risk_capacity_analysis": {},
            "stress_testing_summary": {},
        }

        # Calculate implied precision requirements
        results["implied_precision_analysis"] = self._analyze_implied_precision(
            strategy_params
        )

        # Calculate betting frequency implications
        results["betting_frequency_analysis"] = self._analyze_betting_frequency(
            strategy_params
        )

        # Sharpe ratio analysis
        results["sharpe_ratio_analysis"] = self._analyze_sharpe_ratio_requirements(
            strategy_params
        )

        # Risk capacity analysis
        results["risk_capacity_analysis"] = self._analyze_risk_capacity(
            strategy_params, market_data
        )

        # Generate summary
        results["stress_testing_summary"] = self._generate_strategy_risk_summary(
            results
        )

        # Store results
        self.strategy_risk_metrics_["strategy_risk"] = results

        return results

    def liquidity_stress_testing(
        self,
        returns_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        strategy_function: Callable,
        liquidity_shocks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Test strategy performance under liquidity stress conditions.

        Args:
            returns_data: Historical returns data
            volume_data: Historical volume data
            strategy_function: Strategy function
            liquidity_shocks: Custom liquidity shock scenarios

        Returns:
            Dictionary containing liquidity stress test results
        """
        if liquidity_shocks is None:
            liquidity_shocks = {
                "mild_liquidity_stress": 0.8,  # 20% volume reduction
                "moderate_liquidity_stress": 0.5,  # 50% volume reduction
                "severe_liquidity_stress": 0.2,  # 80% volume reduction
                "liquidity_crisis": 0.05,  # 95% volume reduction
            }

        results = {
            "baseline_performance": {},
            "liquidity_scenarios": {},
            "impact_analysis": {},
            "liquidity_risk_metrics": {},
        }

        # Calculate baseline performance
        baseline_strategy_returns = strategy_function(returns_data)
        results["baseline_performance"] = self._calculate_performance_metrics(
            baseline_strategy_returns
        )

        # Test each liquidity scenario
        for scenario_name, volume_multiplier in liquidity_shocks.items():
            try:
                # Apply liquidity shock
                shocked_volume = volume_data * volume_multiplier

                # Adjust returns for liquidity impact (simplified model)
                liquidity_adjusted_returns = self._apply_liquidity_impact(
                    returns_data, volume_data, shocked_volume
                )

                # Calculate strategy performance under liquidity stress
                stressed_strategy_returns = strategy_function(
                    liquidity_adjusted_returns
                )
                stressed_performance = self._calculate_performance_metrics(
                    stressed_strategy_returns
                )

                # Calculate impact
                performance_impact = {}
                for metric, baseline_value in results["baseline_performance"].items():
                    stressed_value = stressed_performance.get(metric, 0)
                    impact = stressed_value - baseline_value
                    relative_impact = (
                        impact / baseline_value if baseline_value != 0 else 0
                    )

                    performance_impact[metric] = {
                        "absolute_impact": impact,
                        "relative_impact": relative_impact,
                        "stressed_value": stressed_value,
                        "baseline_value": baseline_value,
                    }

                results["liquidity_scenarios"][scenario_name] = {
                    "performance": stressed_performance,
                    "impact": performance_impact,
                    "volume_multiplier": volume_multiplier,
                }

            except Exception as e:
                logger.warning(
                    f"Failed to process liquidity scenario {scenario_name}: {str(e)}"
                )
                continue

        # Calculate liquidity risk metrics
        results["liquidity_risk_metrics"] = self._calculate_liquidity_risk_metrics(
            results
        )

        # Store results
        self.stress_results_["liquidity_stress"] = results

        return results

    def get_comprehensive_stress_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of all stress testing results.

        Returns:
            Dictionary containing comprehensive stress test summary
        """
        if not self.stress_results_:
            raise ValueError(
                "No stress test results available. Run stress tests first."
            )

        summary = {
            "stress_test_types": list(self.stress_results_.keys()),
            "overall_risk_assessment": {},
            "key_vulnerabilities": [],
            "stress_test_summary": {},
            "recommendations": [],
        }

        # Overall risk assessment
        summary["overall_risk_assessment"] = self._generate_overall_risk_assessment()

        # Identify key vulnerabilities
        summary["key_vulnerabilities"] = self._identify_key_vulnerabilities()

        # Summarize each stress test
        for test_type, results in self.stress_results_.items():
            summary["stress_test_summary"][test_type] = self._summarize_stress_test(
                test_type, results
            )

        # Generate recommendations
        summary["recommendations"] = self._generate_stress_test_recommendations()

        return summary

    # Helper methods for binary strategy analysis

    def _calculate_binary_sharpe_ratio(self, params: BinaryStrategyParams) -> float:
        """Calculate Sharpe ratio for binary strategy based on AFML formulation."""
        try:
            # AFML Snippet 15.1 implementation
            p = params.precision_rate
            sl = params.stop_loss
            pt = params.profit_target
            freq = params.frequency

            # Handle edge cases
            if freq <= 0 or sl <= 0 or pt <= 0:
                return 0.0

            # Expected return per bet
            expected_return = p * pt + (1 - p) * (-sl)

            # Variance per bet
            expected_return_squared = p * (pt**2) + (1 - p) * (sl**2)
            variance = expected_return_squared - (expected_return**2)

            # Handle zero variance (perfect strategy case)
            if variance <= 0:
                # For perfect strategies, use a simplified calculation
                if expected_return > 0:
                    return 100.0  # Very high Sharpe for perfect strategy
                elif expected_return < 0:
                    return -100.0  # Very low Sharpe for perfectly losing strategy
                else:
                    return 0.0

            # Annual Sharpe ratio
            annual_return = expected_return * freq
            annual_std = np.sqrt(variance * freq)

            sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0.0

            return sharpe_ratio

        except Exception as e:
            logger.warning(f"Error calculating binary Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_implied_precision(
        self, sl: float, pt: float, freq: float, target_sharpe: float
    ) -> float:
        """Calculate implied precision based on AFML Snippet 15.3."""
        try:
            # Handle edge cases
            if sl <= 0 or pt <= 0 or freq <= 0 or target_sharpe <= 0:
                return np.nan

            if pt <= sl:  # Profit target must be greater than stop loss
                return np.nan

            # AFML exact implementation
            a = (freq + target_sharpe**2) * (pt - sl) ** 2
            b = (2 * freq * sl - target_sharpe**2 * (pt - sl)) * (pt - sl)
            c = freq * sl**2

            if a == 0:  # Avoid division by zero
                return np.nan

            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return np.nan

            # AFML uses positive root only
            p = (-b + np.sqrt(discriminant)) / (2.0 * a)

            # Validate precision is between 0 and 1
            if 0 <= p <= 1:
                return p
            else:
                return np.nan

        except Exception as e:
            logger.warning(f"Error calculating implied precision: {str(e)}")
            return np.nan

    def _calculate_implied_frequency(
        self, sl: float, pt: float, p: float, target_sharpe: float
    ) -> float:
        """Calculate implied betting frequency based on AFML Snippet 15.4."""
        try:
            numerator = (target_sharpe * (pt - sl)) ** 2 * p * (1 - p)
            denominator = ((pt - sl) * p + sl) ** 2

            freq = numerator / denominator if denominator > 0 else np.nan

            # Validate by checking if it produces the target Sharpe ratio
            test_params = BinaryStrategyParams(sl, pt, p, freq)
            calculated_sharpe = self._calculate_binary_sharpe_ratio(test_params)

            if abs(calculated_sharpe - target_sharpe) < 1e-6:
                return freq
            else:
                return np.nan

        except Exception as e:
            logger.warning(f"Error calculating implied frequency: {str(e)}")
            return np.nan

    def _calculate_implied_metrics(
        self, params: BinaryStrategyParams
    ) -> Dict[str, float]:
        """Calculate various implied metrics for binary strategy."""
        metrics = {}

        # Current Sharpe ratio
        metrics["current_sharpe"] = self._calculate_binary_sharpe_ratio(params)

        # Implied precision for different target Sharpe ratios
        target_sharpes = [0.5, 1.0, 1.5, 2.0]
        for target_sharpe in target_sharpes:
            implied_p = self._calculate_implied_precision(
                params.stop_loss, params.profit_target, params.frequency, target_sharpe
            )
            metrics[f"implied_precision_sr_{target_sharpe}"] = implied_p

        # Implied frequency for different target Sharpe ratios
        for target_sharpe in target_sharpes:
            implied_freq = self._calculate_implied_frequency(
                params.stop_loss,
                params.profit_target,
                params.precision_rate,
                target_sharpe,
            )
            metrics[f"implied_frequency_sr_{target_sharpe}"] = implied_freq

        # Risk metrics
        metrics["expected_return_per_bet"] = (
            params.precision_rate * params.profit_target
            + (1 - params.precision_rate) * (-params.stop_loss)
        )

        expected_return_squared = params.precision_rate * (
            params.profit_target**2
        ) + (1 - params.precision_rate) * (params.stop_loss**2)

        metrics["variance_per_bet"] = expected_return_squared - (
            metrics["expected_return_per_bet"] ** 2
        )

        metrics["annual_return"] = metrics["expected_return_per_bet"] * params.frequency
        metrics["annual_volatility"] = np.sqrt(
            metrics["variance_per_bet"] * params.frequency
        )

        return metrics

    def _create_default_binary_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Create default stress scenarios for binary strategies."""
        return {
            "precision_decline_mild": {"precision_multiplier": 0.95},
            "precision_decline_moderate": {"precision_multiplier": 0.9},
            "precision_decline_severe": {"precision_multiplier": 0.8},
            "frequency_reduction": {"frequency_multiplier": 0.7},
            "stop_loss_widening": {"stop_loss_multiplier": 1.2},
            "profit_target_reduction": {"profit_target_multiplier": 0.9},
            "combined_stress_mild": {
                "precision_multiplier": 0.95,
                "frequency_multiplier": 0.9,
                "stop_loss_multiplier": 1.1,
            },
            "combined_stress_severe": {
                "precision_multiplier": 0.85,
                "frequency_multiplier": 0.7,
                "stop_loss_multiplier": 1.3,
                "profit_target_multiplier": 0.8,
            },
        }

    def _apply_binary_stress(
        self, params: BinaryStrategyParams, stress: Dict[str, float]
    ) -> BinaryStrategyParams:
        """Apply stress scenario to binary strategy parameters."""
        stressed_params = BinaryStrategyParams(
            stop_loss=params.stop_loss * stress.get("stop_loss_multiplier", 1.0),
            profit_target=params.profit_target
            * stress.get("profit_target_multiplier", 1.0),
            precision_rate=min(
                1.0,
                max(
                    0.0, params.precision_rate * stress.get("precision_multiplier", 1.0)
                ),
            ),
            frequency=params.frequency * stress.get("frequency_multiplier", 1.0),
            target_sharpe=params.target_sharpe,
        )

        return stressed_params

    def _assess_binary_strategy_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk of binary strategy based on stress test results."""
        assessment = {
            "risk_level": "unknown",
            "key_vulnerabilities": [],
            "stress_resistance": {},
            "recommendations": [],
        }

        if not results.get("stress_scenarios"):
            return assessment

        # Calculate stress impacts
        impacts = []
        vulnerability_count = 0

        for scenario_name, scenario_result in results["stress_scenarios"].items():
            relative_impact = abs(scenario_result.relative_impact)
            impacts.append(relative_impact)

            # Check for significant vulnerabilities (>10% impact)
            if relative_impact > 0.1:
                vulnerability_count += 1
                assessment["key_vulnerabilities"].append(
                    {
                        "scenario": scenario_name,
                        "impact": relative_impact,
                        "description": f"Strategy vulnerable to {scenario_name} with {relative_impact:.1%} impact",
                    }
                )

        # Assess overall risk level
        max_impact = max(impacts) if impacts else 0
        avg_impact = np.mean(impacts) if impacts else 0

        if max_impact < 0.05:
            assessment["risk_level"] = "low"
        elif max_impact < 0.15:
            assessment["risk_level"] = "moderate"
        elif max_impact < 0.3:
            assessment["risk_level"] = "high"
        else:
            assessment["risk_level"] = "critical"

        # Stress resistance metrics
        assessment["stress_resistance"] = {
            "maximum_impact": max_impact,
            "average_impact": avg_impact,
            "vulnerability_count": vulnerability_count,
            "total_scenarios_tested": len(results["stress_scenarios"]),
        }

        # Generate recommendations
        if assessment["risk_level"] in ["high", "critical"]:
            assessment["recommendations"].append("Consider reducing position sizes")
            assessment["recommendations"].append("Implement additional risk controls")

        if vulnerability_count > len(results["stress_scenarios"]) * 0.5:
            assessment["recommendations"].append(
                "Strategy shows high sensitivity to parameter changes"
            )

        return assessment

    def _identify_crisis_periods(
        self, returns_data: pd.DataFrame
    ) -> Dict[str, Tuple[str, str]]:
        """Identify historical crisis periods in the data."""
        # Default crisis periods (can be expanded)
        crisis_periods = {
            "dotcom_crash": ("2000-03-01", "2002-10-01"),
            "financial_crisis": ("2007-10-01", "2009-03-01"),
            "covid_crash": ("2020-02-01", "2020-04-01"),
        }

        # Filter to periods that exist in the data
        available_periods = {}
        for name, (start, end) in crisis_periods.items():
            if (
                pd.to_datetime(start) >= returns_data.index.min()
                and pd.to_datetime(end) <= returns_data.index.max()
            ):
                available_periods[name] = (start, end)

        return available_periods

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return {}

        metrics = {}

        # Basic return metrics
        metrics["total_return"] = (1 + returns).prod() - 1
        metrics["annual_return"] = (
            1 + returns.mean()
        ) ** 252 - 1  # Assuming daily data
        metrics["volatility"] = returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        if metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = metrics["annual_return"] / metrics["volatility"]
        else:
            metrics["sharpe_ratio"] = 0.0

        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        metrics["max_drawdown"] = drawdown.min()
        metrics["avg_drawdown"] = (
            drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        )

        # Tail risk metrics
        metrics["var_95"] = returns.quantile(0.05)
        metrics["var_99"] = returns.quantile(0.01)
        metrics["cvar_95"] = returns[returns <= metrics["var_95"]].mean()
        metrics["cvar_99"] = returns[returns <= metrics["var_99"]].mean()

        # Higher moments
        metrics["skewness"] = returns.skew()
        metrics["kurtosis"] = returns.kurtosis()

        return metrics

    def _create_extreme_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Create extreme event scenarios for testing."""
        return {
            "black_monday": {
                "return_shock": -0.2,  # 20% single day drop
                "volatility_multiplier": 3.0,
                "duration_days": 1,
            },
            "market_crash": {
                "return_shock": -0.35,  # 35% decline over period
                "volatility_multiplier": 2.5,
                "duration_days": 30,
            },
            "flash_crash": {
                "return_shock": -0.1,  # 10% intraday drop
                "volatility_multiplier": 5.0,
                "duration_days": 1,
                "recovery_factor": 0.7,  # Partial recovery
            },
            "prolonged_bear_market": {
                "return_shock": -0.5,  # 50% decline
                "volatility_multiplier": 1.8,
                "duration_days": 250,
            },
            "liquidity_crisis": {
                "return_shock": -0.15,
                "volatility_multiplier": 2.0,
                "liquidity_factor": 0.3,  # 70% liquidity reduction
                "duration_days": 14,
            },
        }

    def _generate_extreme_event_data(
        self, returns_data: pd.DataFrame, scenario_params: Dict[str, float]
    ) -> pd.DataFrame:
        """Generate synthetic data for extreme event scenario."""
        extreme_returns = returns_data.copy()

        # Apply return shock
        if "return_shock" in scenario_params:
            shock_magnitude = scenario_params["return_shock"]
            duration = scenario_params.get("duration_days", 1)

            # Distribute shock over duration
            daily_shock = shock_magnitude / duration

            # Apply shock to random period
            start_idx = np.random.randint(0, len(extreme_returns) - duration)
            end_idx = start_idx + duration

            extreme_returns.iloc[start_idx:end_idx] += daily_shock

        # Apply volatility multiplier
        if "volatility_multiplier" in scenario_params:
            vol_mult = scenario_params["volatility_multiplier"]
            mean_returns = extreme_returns.mean()
            extreme_returns = (extreme_returns - mean_returns) * vol_mult + mean_returns

        # Apply recovery factor if applicable
        if "recovery_factor" in scenario_params:
            recovery_factor = scenario_params["recovery_factor"]
            # Apply partial recovery in subsequent periods
            recovery_adjustment = shock_magnitude * (1 - recovery_factor)
            extreme_returns.iloc[end_idx : end_idx + 5] += recovery_adjustment / 5

        return extreme_returns

    def _calculate_tail_risk_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate tail risk metrics from extreme scenario results."""
        metrics = {}

        # Extract performance impacts
        impacts = []
        for scenario_name, scenario_data in results["extreme_scenarios"].items():
            if "impact" in scenario_data and "total_return" in scenario_data["impact"]:
                impact = scenario_data["impact"]["total_return"]["relative_impact"]
                impacts.append(impact)

        if impacts:
            impacts = np.array(impacts)

            # Tail risk metrics
            metrics["worst_case_impact"] = np.min(impacts)
            metrics["average_extreme_impact"] = np.mean(impacts)
            metrics["tail_var_95"] = np.percentile(impacts, 5)
            metrics["tail_var_99"] = np.percentile(impacts, 1)

            # Extreme event frequency (scenarios with >10% impact)
            severe_events = impacts[impacts < -0.1]
            metrics["severe_event_frequency"] = len(severe_events) / len(impacts)

            # Expected shortfall
            if len(severe_events) > 0:
                metrics["expected_shortfall"] = np.mean(severe_events)
            else:
                metrics["expected_shortfall"] = 0.0

        return metrics

    def _analyze_implied_precision(
        self, params: BinaryStrategyParams
    ) -> Dict[str, Any]:
        """Analyze implied precision requirements for different target Sharpe ratios."""
        analysis = {
            "current_precision": params.precision_rate,
            "target_precision_requirements": {},
            "precision_gap_analysis": {},
            "feasibility_assessment": {},
        }

        target_sharpes = [0.5, 1.0, 1.5, 2.0, 2.5]

        for target_sharpe in target_sharpes:
            implied_precision = self._calculate_implied_precision(
                params.stop_loss, params.profit_target, params.frequency, target_sharpe
            )

            analysis["target_precision_requirements"][
                f"sharpe_{target_sharpe}"
            ] = implied_precision

            if not np.isnan(implied_precision):
                precision_gap = implied_precision - params.precision_rate
                analysis["precision_gap_analysis"][f"sharpe_{target_sharpe}"] = {
                    "required_precision": implied_precision,
                    "current_precision": params.precision_rate,
                    "precision_gap": precision_gap,
                    "relative_improvement_needed": (
                        precision_gap / params.precision_rate
                        if params.precision_rate > 0
                        else np.inf
                    ),
                }

                # Feasibility assessment
                if implied_precision <= 1.0:
                    if precision_gap <= 0.05:
                        feasibility = "easily_achievable"
                    elif precision_gap <= 0.1:
                        feasibility = "achievable"
                    elif precision_gap <= 0.2:
                        feasibility = "challenging"
                    else:
                        feasibility = "very_difficult"
                else:
                    feasibility = "impossible"

                analysis["feasibility_assessment"][
                    f"sharpe_{target_sharpe}"
                ] = feasibility

        return analysis

    def _analyze_betting_frequency(
        self, params: BinaryStrategyParams
    ) -> Dict[str, Any]:
        """Analyze betting frequency implications for different target Sharpe ratios."""
        analysis = {
            "current_frequency": params.frequency,
            "target_frequency_requirements": {},
            "frequency_scaling_analysis": {},
            "practical_constraints": {},
        }

        target_sharpes = [0.5, 1.0, 1.5, 2.0, 2.5]

        for target_sharpe in target_sharpes:
            implied_frequency = self._calculate_implied_frequency(
                params.stop_loss,
                params.profit_target,
                params.precision_rate,
                target_sharpe,
            )

            analysis["target_frequency_requirements"][
                f"sharpe_{target_sharpe}"
            ] = implied_frequency

            if not np.isnan(implied_frequency):
                frequency_ratio = (
                    implied_frequency / params.frequency
                    if params.frequency > 0
                    else np.inf
                )

                analysis["frequency_scaling_analysis"][f"sharpe_{target_sharpe}"] = {
                    "required_frequency": implied_frequency,
                    "current_frequency": params.frequency,
                    "frequency_ratio": frequency_ratio,
                    "additional_bets_needed": implied_frequency - params.frequency,
                }

                # Practical constraints assessment
                if implied_frequency <= 50:  # ~1 bet per week
                    constraint_level = "no_constraint"
                elif implied_frequency <= 250:  # ~1 bet per day
                    constraint_level = "manageable"
                elif implied_frequency <= 1000:  # ~4 bets per day
                    constraint_level = "challenging"
                else:
                    constraint_level = "impractical"

                analysis["practical_constraints"][
                    f"sharpe_{target_sharpe}"
                ] = constraint_level

        return analysis

    def _analyze_sharpe_ratio_requirements(
        self, params: BinaryStrategyParams
    ) -> Dict[str, Any]:
        """Analyze Sharpe ratio implications and requirements."""
        analysis = {
            "current_sharpe": self._calculate_binary_sharpe_ratio(params),
            "sharpe_sensitivity": {},
            "parameter_impact_on_sharpe": {},
            "optimization_insights": {},
        }

        # Sensitivity analysis - how Sharpe ratio changes with parameter variations
        parameter_variations = {
            "precision_rate": np.linspace(
                max(0.01, params.precision_rate - 0.1),
                min(0.99, params.precision_rate + 0.1),
                11,
            ),
            "stop_loss": np.linspace(
                params.stop_loss * 0.5, params.stop_loss * 1.5, 11
            ),
            "profit_target": np.linspace(
                params.profit_target * 0.5, params.profit_target * 1.5, 11
            ),
            "frequency": np.linspace(
                params.frequency * 0.5, params.frequency * 2.0, 11
            ),
        }

        for param_name, param_values in parameter_variations.items():
            sharpe_values = []

            for param_value in param_values:
                test_params = BinaryStrategyParams(
                    stop_loss=(
                        param_value if param_name == "stop_loss" else params.stop_loss
                    ),
                    profit_target=(
                        param_value
                        if param_name == "profit_target"
                        else params.profit_target
                    ),
                    precision_rate=(
                        param_value
                        if param_name == "precision_rate"
                        else params.precision_rate
                    ),
                    frequency=(
                        param_value if param_name == "frequency" else params.frequency
                    ),
                )

                sharpe = self._calculate_binary_sharpe_ratio(test_params)
                sharpe_values.append(sharpe)

            # Calculate sensitivity metrics
            sharpe_range = max(sharpe_values) - min(sharpe_values)
            param_range = max(param_values) - min(param_values)
            sensitivity = sharpe_range / param_range if param_range > 0 else 0

            analysis["sharpe_sensitivity"][param_name] = {
                "sensitivity_coefficient": sensitivity,
                "sharpe_range": sharpe_range,
                "optimal_value": param_values[np.argmax(sharpe_values)],
                "optimal_sharpe": max(sharpe_values),
            }

        return analysis

    def _analyze_risk_capacity(
        self, params: BinaryStrategyParams, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze risk capacity and capital requirements."""
        analysis = {
            "strategy_risk_metrics": {},
            "capital_requirements": {},
            "capacity_constraints": {},
            "scalability_analysis": {},
        }

        # Calculate strategy risk metrics
        current_sharpe = self._calculate_binary_sharpe_ratio(params)
        implied_metrics = self._calculate_implied_metrics(params)

        analysis["strategy_risk_metrics"] = {
            "current_sharpe_ratio": current_sharpe,
            "annual_return": implied_metrics["annual_return"],
            "annual_volatility": implied_metrics["annual_volatility"],
            "expected_return_per_bet": implied_metrics["expected_return_per_bet"],
            "variance_per_bet": implied_metrics["variance_per_bet"],
        }

        # Capital requirements analysis
        # Estimate capital needed for different risk levels
        risk_levels = [0.01, 0.02, 0.05, 0.1]  # 1%, 2%, 5%, 10% risk per bet

        for risk_level in risk_levels:
            # Estimate position size based on Kelly criterion (simplified)
            kelly_fraction = (
                implied_metrics["expected_return_per_bet"]
                / implied_metrics["variance_per_bet"]
                if implied_metrics["variance_per_bet"] > 0
                else 0
            )

            # Conservative position sizing (fraction of Kelly)
            conservative_fraction = min(kelly_fraction * 0.25, risk_level)

            # Estimate required capital
            bet_size = conservative_fraction
            min_capital_per_bet = (
                abs(params.stop_loss) / bet_size if bet_size > 0 else np.inf
            )

            analysis["capital_requirements"][f"risk_{risk_level:.1%}"] = {
                "kelly_fraction": kelly_fraction,
                "conservative_fraction": conservative_fraction,
                "min_capital_per_bet": min_capital_per_bet,
                "annual_capital_requirement": min_capital_per_bet * params.frequency,
            }

        return analysis

    def _apply_liquidity_impact(
        self,
        returns_data: pd.DataFrame,
        original_volume: pd.DataFrame,
        shocked_volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply liquidity impact to returns based on volume changes."""
        # Simplified liquidity impact model
        # In practice, this would be much more sophisticated

        # Ensure volumes have same structure as returns
        volume_ratio = shocked_volume.values / original_volume.values

        # Market impact model: impact proportional to volume reduction
        impact_factor = 1 - volume_ratio

        # Apply impact (reduces returns when liquidity decreases)
        # Create impact DataFrame with same structure as returns
        impact_df = pd.DataFrame(
            impact_factor * 0.01,  # 1% impact per full volume reduction
            index=returns_data.index,
            columns=returns_data.columns,
        )

        liquidity_adjusted_returns = returns_data - impact_df

        return liquidity_adjusted_returns

    def _calculate_liquidity_risk_metrics(
        self, results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate liquidity risk metrics."""
        metrics = {}

        if "liquidity_scenarios" in results:
            impacts = []

            for scenario_name, scenario_data in results["liquidity_scenarios"].items():
                if (
                    "impact" in scenario_data
                    and "total_return" in scenario_data["impact"]
                ):
                    impact = scenario_data["impact"]["total_return"]["relative_impact"]
                    impacts.append(impact)

            if impacts:
                impacts = np.array(impacts)

                metrics["max_liquidity_impact"] = np.min(
                    impacts
                )  # Most negative impact
                metrics["average_liquidity_impact"] = np.mean(impacts)
                metrics["liquidity_var_95"] = np.percentile(impacts, 5)

                # Liquidity stress resistance
                severe_impacts = impacts[impacts < -0.05]  # >5% impact
                metrics["liquidity_stress_frequency"] = (
                    len(severe_impacts) / len(impacts) if len(impacts) > 0 else 0
                )

        return metrics

    def _generate_overall_risk_assessment(self) -> Dict[str, Any]:
        """Generate overall risk assessment across all stress tests."""
        assessment = {
            "overall_risk_level": "unknown",
            "key_risk_factors": [],
            "stress_test_performance": {},
            "vulnerability_summary": {},
        }

        # Collect impacts from all stress tests
        all_impacts = []
        risk_factors = []

        for test_type, results in self.stress_results_.items():
            if test_type == "binary_strategy" and "stress_scenarios" in results:
                for scenario_name, scenario_result in results[
                    "stress_scenarios"
                ].items():
                    impact = abs(scenario_result.relative_impact)
                    all_impacts.append(impact)

                    if impact > 0.1:  # 10% threshold
                        risk_factors.append(
                            {
                                "test_type": test_type,
                                "scenario": scenario_name,
                                "impact": impact,
                            }
                        )

            elif "extreme_scenarios" in results:
                for scenario_name, scenario_data in results[
                    "extreme_scenarios"
                ].items():
                    if (
                        "impact" in scenario_data
                        and "total_return" in scenario_data["impact"]
                    ):
                        impact = abs(
                            scenario_data["impact"]["total_return"]["relative_impact"]
                        )
                        all_impacts.append(impact)

                        if impact > 0.1:
                            risk_factors.append(
                                {
                                    "test_type": test_type,
                                    "scenario": scenario_name,
                                    "impact": impact,
                                }
                            )

        # Overall risk level assessment
        if all_impacts:
            max_impact = max(all_impacts)
            avg_impact = np.mean(all_impacts)

            if max_impact < 0.05:
                assessment["overall_risk_level"] = "low"
            elif max_impact < 0.15:
                assessment["overall_risk_level"] = "moderate"
            elif max_impact < 0.3:
                assessment["overall_risk_level"] = "high"
            else:
                assessment["overall_risk_level"] = "critical"

            assessment["stress_test_performance"] = {
                "max_impact": max_impact,
                "average_impact": avg_impact,
                "number_of_tests": len(all_impacts),
                "high_risk_scenarios": len([i for i in all_impacts if i > 0.1]),
            }

        assessment["key_risk_factors"] = sorted(
            risk_factors, key=lambda x: x["impact"], reverse=True
        )[:5]

        return assessment

    def _identify_key_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Identify key vulnerabilities across all stress tests."""
        vulnerabilities = []

        for test_type, results in self.stress_results_.items():
            if test_type == "binary_strategy":
                if (
                    "risk_assessment" in results
                    and "key_vulnerabilities" in results["risk_assessment"]
                ):
                    for vuln in results["risk_assessment"]["key_vulnerabilities"]:
                        vulnerabilities.append(
                            {
                                "test_type": test_type,
                                "vulnerability": vuln,
                                "severity": (
                                    "high"
                                    if vuln["impact"] > 0.2
                                    else "moderate" if vuln["impact"] > 0.1 else "low"
                                ),
                            }
                        )

        return sorted(
            vulnerabilities, key=lambda x: x["vulnerability"]["impact"], reverse=True
        )

    def _summarize_stress_test(
        self, test_type: str, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarize individual stress test results."""
        summary = {
            "test_type": test_type,
            "scenarios_tested": 0,
            "key_findings": [],
            "worst_case_scenario": None,
            "recommendations": [],
        }

        if test_type == "binary_strategy" and "stress_scenarios" in results:
            summary["scenarios_tested"] = len(results["stress_scenarios"])

            # Find worst case scenario
            worst_impact = 0
            worst_scenario = None

            for scenario_name, scenario_result in results["stress_scenarios"].items():
                impact = abs(scenario_result.relative_impact)
                if impact > worst_impact:
                    worst_impact = impact
                    worst_scenario = {
                        "name": scenario_name,
                        "impact": impact,
                        "details": scenario_result.scenario_parameters,
                    }

            summary["worst_case_scenario"] = worst_scenario

            # Key findings
            if worst_impact > 0.2:
                summary["key_findings"].append(
                    f"Strategy shows high sensitivity to parameter changes (max impact: {worst_impact:.1%})"
                )

            if "risk_assessment" in results:
                risk_level = results["risk_assessment"].get("risk_level", "unknown")
                summary["key_findings"].append(
                    f"Overall risk level assessed as: {risk_level}"
                )

        elif "extreme_scenarios" in results:
            summary["scenarios_tested"] = len(results["extreme_scenarios"])

            # Extract key findings for extreme events
            if "tail_risk_metrics" in results:
                tail_metrics = results["tail_risk_metrics"]
                worst_case = tail_metrics.get("worst_case_impact", 0)
                summary["key_findings"].append(
                    f"Worst case extreme event impact: {worst_case:.1%}"
                )

        return summary

    def _generate_stress_test_recommendations(self) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []

        # Overall risk assessment
        overall_assessment = self._generate_overall_risk_assessment()
        risk_level = overall_assessment["overall_risk_level"]

        if risk_level in ["high", "critical"]:
            recommendations.append(
                "Consider reducing position sizes to mitigate high risk exposure"
            )
            recommendations.append("Implement additional risk controls and monitoring")
            recommendations.append("Review and potentially adjust strategy parameters")

        # Specific recommendations based on vulnerabilities
        vulnerabilities = self._identify_key_vulnerabilities()

        if len(vulnerabilities) > 3:
            recommendations.append(
                "Strategy shows multiple vulnerabilities - consider diversification"
            )

        # Parameter-specific recommendations
        for test_type, results in self.stress_results_.items():
            if test_type == "binary_strategy" and "implied_metrics" in results:
                current_sharpe = results["implied_metrics"].get("current_sharpe", 0)
                if current_sharpe < 0.5:
                    recommendations.append(
                        "Low Sharpe ratio indicates need for strategy improvement"
                    )
                elif current_sharpe > 2.0:
                    recommendations.append(
                        "High Sharpe ratio may indicate unrealistic expectations - validate assumptions"
                    )

        return recommendations[:10]  # Limit to top 10 recommendations

    def _generate_strategy_risk_summary(
        self, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of strategy risk quantification."""
        summary = {
            "overall_assessment": "unknown",
            "key_metrics": {},
            "critical_insights": [],
            "action_items": [],
        }

        # Extract key metrics
        if "sharpe_ratio_analysis" in results:
            current_sharpe = results["sharpe_ratio_analysis"].get("current_sharpe", 0)
            summary["key_metrics"]["current_sharpe_ratio"] = current_sharpe

        if "implied_precision_analysis" in results:
            precision_data = results["implied_precision_analysis"]
            current_precision = precision_data.get("current_precision", 0)
            summary["key_metrics"]["current_precision"] = current_precision

            # Check feasibility for target Sharpe of 1.0
            feasibility_1_0 = precision_data.get("feasibility_assessment", {}).get(
                "sharpe_1.0", "unknown"
            )
            if feasibility_1_0 == "impossible":
                summary["critical_insights"].append(
                    "Target Sharpe ratio of 1.0 is impossible with current parameters"
                )
            elif feasibility_1_0 == "very_difficult":
                summary["critical_insights"].append(
                    "Target Sharpe ratio of 1.0 requires significant precision improvement"
                )

        # Overall assessment
        if current_sharpe < 0.5:
            summary["overall_assessment"] = "needs_improvement"
            summary["action_items"].append(
                "Focus on improving strategy precision or adjusting parameters"
            )
        elif current_sharpe < 1.0:
            summary["overall_assessment"] = "acceptable"
            summary["action_items"].append("Consider optimization opportunities")
        else:
            summary["overall_assessment"] = "strong"
            summary["action_items"].append(
                "Monitor for parameter drift and validate assumptions"
            )

        return summary
