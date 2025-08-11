"""
Stress Testing Module for Portfolio Risk Management

This module implements comprehensive stress testing methodologies for portfolio
risk assessment, including scenario analysis, Monte Carlo simulation,
sensitivity analysis, and tail risk evaluation.

Based on risk management best practices and quantitative finance literature.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.optimize import minimize
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """Enumeration of stress test types."""

    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    SCENARIO = "scenario"
    SENSITIVITY = "sensitivity"
    TAIL_RISK = "tail_risk"


@dataclass
class StressTestResult:
    """Data class for stress test results."""

    test_type: str
    scenario_name: str
    portfolio_value: float
    stressed_value: float
    loss: float
    loss_percentage: float
    confidence_level: float
    metrics: Dict[str, float]
    scenario_details: Dict[str, Any]


class ScenarioGenerator:
    """Generate stress test scenarios for portfolio analysis."""

    def __init__(self, random_seed: int = 42):
        """Initialize scenario generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_market_crash_scenario(
        self, severity: str = "moderate"
    ) -> Dict[str, float]:
        """Generate market crash scenario parameters.

        Args:
            severity: Crash severity ('mild', 'moderate', 'severe', 'extreme')

        Returns:
            Dictionary containing scenario parameters
        """
        crash_parameters = {
            "mild": {
                "equity_shock": -0.15,
                "vol_spike": 2.0,
                "correlation_increase": 0.3,
            },
            "moderate": {
                "equity_shock": -0.25,
                "vol_spike": 3.0,
                "correlation_increase": 0.5,
            },
            "severe": {
                "equity_shock": -0.40,
                "vol_spike": 4.0,
                "correlation_increase": 0.7,
            },
            "extreme": {
                "equity_shock": -0.60,
                "vol_spike": 6.0,
                "correlation_increase": 0.9,
            },
        }

        if severity not in crash_parameters:
            severity = "moderate"

        params = crash_parameters[severity]

        return {
            "name": f"Market Crash ({severity})",
            "equity_shock": params["equity_shock"],
            "bond_shock": params["equity_shock"] * 0.3,  # Bonds less affected
            "volatility_multiplier": params["vol_spike"],
            "correlation_increase": params["correlation_increase"],
            "currency_shock": params["equity_shock"] * 0.5,
            "commodity_shock": params["equity_shock"] * 0.7,
        }

    def generate_interest_rate_scenario(
        self, direction: str = "up", magnitude: float = 0.02
    ) -> Dict[str, float]:
        """Generate interest rate shock scenario.

        Args:
            direction: Rate change direction ('up' or 'down')
            magnitude: Rate change magnitude (absolute value)

        Returns:
            Dictionary containing scenario parameters
        """
        multiplier = 1 if direction == "up" else -1

        return {
            "name": f"Interest Rate Shock ({direction} {magnitude*100:.0f}bp)",
            "rate_shock": magnitude * multiplier,
            "bond_duration_effect": -magnitude * multiplier * 5.0,  # Typical duration
            "equity_valuation_effect": -magnitude * multiplier * 0.5,
            "currency_effect": magnitude * multiplier * 0.3,
            "real_estate_effect": -magnitude * multiplier * 0.8,
        }

    def generate_volatility_scenario(
        self, vol_regime: str = "high"
    ) -> Dict[str, float]:
        """Generate volatility regime scenario.

        Args:
            vol_regime: Volatility regime ('low', 'normal', 'high', 'extreme')

        Returns:
            Dictionary containing scenario parameters
        """
        vol_multipliers = {"low": 0.5, "normal": 1.0, "high": 2.5, "extreme": 4.0}

        multiplier = vol_multipliers.get(vol_regime, 1.0)

        return {
            "name": f"Volatility Regime ({vol_regime})",
            "volatility_multiplier": multiplier,
            "correlation_effect": min(0.9, 0.3 + multiplier * 0.2),
            "liquidity_impact": (multiplier - 1.0) * 0.1,
            "bid_ask_widening": (multiplier - 1.0) * 0.05,
        }

    def generate_currency_crisis_scenario(
        self, affected_currency: str = "USD"
    ) -> Dict[str, float]:
        """Generate currency crisis scenario.

        Args:
            affected_currency: Currency under stress

        Returns:
            Dictionary containing scenario parameters
        """
        return {
            "name": f"Currency Crisis ({affected_currency})",
            "currency_shock": -0.30,
            "emerging_market_contagion": -0.25,
            "commodity_effect": -0.15,
            "safe_haven_rally": 0.10,  # USD, CHF, JPY benefit
            "volatility_spike": 3.0,
        }


class MonteCarloEngine:
    """Monte Carlo simulation engine for stress testing."""

    def __init__(self, num_simulations: int = 10000, random_seed: int = 42):
        """Initialize Monte Carlo engine.

        Args:
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def simulate_portfolio_returns(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_weights: np.ndarray,
        time_horizon: int = 252,
        confidence_levels: List[float] = [0.95, 0.99, 0.999],
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """Simulate portfolio returns using Monte Carlo.

        Args:
            expected_returns: Expected asset returns
            covariance_matrix: Asset covariance matrix
            portfolio_weights: Portfolio weights
            time_horizon: Simulation horizon in days
            confidence_levels: VaR confidence levels

        Returns:
            Dictionary containing simulation results
        """
        try:
            # Validate inputs
            if len(expected_returns) != len(portfolio_weights):
                raise ValueError("Expected returns and weights dimension mismatch")

            if covariance_matrix.shape[0] != len(portfolio_weights):
                raise ValueError("Covariance matrix and weights dimension mismatch")

            # Convert to daily parameters
            daily_returns = expected_returns / 252
            daily_cov = covariance_matrix / 252

            # Portfolio expected return and volatility
            portfolio_return = np.dot(portfolio_weights, daily_returns)
            portfolio_variance = np.dot(
                portfolio_weights, np.dot(daily_cov, portfolio_weights)
            )
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Generate random returns
            random_returns = np.random.multivariate_normal(
                daily_returns, daily_cov, (self.num_simulations, time_horizon)
            )

            # Calculate portfolio returns for each simulation
            portfolio_returns = np.dot(random_returns, portfolio_weights)

            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1) - 1
            final_returns = cumulative_returns[:, -1]

            # Calculate risk metrics
            var_metrics = {}
            for conf in confidence_levels:
                var_metrics[f"VaR_{conf}"] = np.percentile(
                    final_returns, (1 - conf) * 100
                )
                cvar_mask = final_returns <= var_metrics[f"VaR_{conf}"]
                if np.sum(cvar_mask) > 0:
                    var_metrics[f"CVaR_{conf}"] = np.mean(final_returns[cvar_mask])
                else:
                    var_metrics[f"CVaR_{conf}"] = var_metrics[f"VaR_{conf}"]

            # Additional statistics
            statistics = {
                "mean_return": np.mean(final_returns),
                "std_return": np.std(final_returns),
                "skewness": stats.skew(final_returns),
                "kurtosis": stats.kurtosis(final_returns),
                "min_return": np.min(final_returns),
                "max_return": np.max(final_returns),
                "probability_loss": np.mean(final_returns < 0),
                "expected_shortfall_5pct": np.mean(
                    final_returns[final_returns <= np.percentile(final_returns, 5)]
                ),
            }

            logger.info(
                f"Monte Carlo simulation completed: {self.num_simulations} simulations"
            )

            return {
                "portfolio_returns": portfolio_returns,
                "cumulative_returns": cumulative_returns,
                "final_returns": final_returns,
                "var_metrics": var_metrics,
                "statistics": statistics,
                "portfolio_params": {
                    "expected_return": portfolio_return * 252,
                    "volatility": portfolio_volatility * np.sqrt(252),
                },
            }

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            raise


class SensitivityAnalyzer:
    """Sensitivity analysis for portfolio stress testing."""

    def __init__(self):
        """Initialize sensitivity analyzer."""
        pass

    def calculate_portfolio_sensitivities(
        self,
        portfolio_weights: np.ndarray,
        asset_prices: np.ndarray,
        shock_size: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """Calculate portfolio sensitivities to asset price shocks.

        Args:
            portfolio_weights: Portfolio weights
            asset_prices: Current asset prices
            shock_size: Relative shock size (e.g., 0.01 for 1%)

        Returns:
            Dictionary containing sensitivity measures
        """
        try:
            n_assets = len(portfolio_weights)
            sensitivities = {}

            # Calculate portfolio value
            portfolio_value = np.sum(portfolio_weights * asset_prices)

            # Delta (first-order sensitivity)
            deltas = np.zeros(n_assets)
            for i in range(n_assets):
                shocked_prices = asset_prices.copy()
                shocked_prices[i] *= 1 + shock_size
                shocked_value = np.sum(portfolio_weights * shocked_prices)
                deltas[i] = (shocked_value - portfolio_value) / (
                    shock_size * asset_prices[i]
                )

            # Gamma (second-order sensitivity)
            gammas = np.zeros(n_assets)
            for i in range(n_assets):
                # Up shock
                shocked_prices_up = asset_prices.copy()
                shocked_prices_up[i] *= 1 + shock_size
                value_up = np.sum(portfolio_weights * shocked_prices_up)

                # Down shock
                shocked_prices_down = asset_prices.copy()
                shocked_prices_down[i] *= 1 - shock_size
                value_down = np.sum(portfolio_weights * shocked_prices_down)

                # Second derivative approximation
                gammas[i] = (value_up - 2 * portfolio_value + value_down) / (
                    shock_size * asset_prices[i]
                ) ** 2

            # Cross-sensitivities (simplified)
            cross_sensitivities = np.zeros((n_assets, n_assets))
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    shocked_prices = asset_prices.copy()
                    shocked_prices[i] *= 1 + shock_size
                    shocked_prices[j] *= 1 + shock_size

                    shocked_value = np.sum(portfolio_weights * shocked_prices)
                    base_effect = (
                        deltas[i] * shock_size * asset_prices[i]
                        + deltas[j] * shock_size * asset_prices[j]
                    )

                    cross_effect = shocked_value - portfolio_value - base_effect
                    cross_sensitivities[i, j] = cross_sensitivities[j, i] = (
                        cross_effect
                        / (shock_size**2 * asset_prices[i] * asset_prices[j])
                    )

            sensitivities = {
                "delta": deltas,
                "gamma": gammas,
                "cross_gamma": cross_sensitivities,
                "dollar_delta": deltas * asset_prices,
                "percentage_delta": deltas * asset_prices / portfolio_value,
            }

            logger.info("Portfolio sensitivity analysis completed")

            return sensitivities

        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            raise

    def stress_test_correlation_changes(
        self,
        returns_data: pd.DataFrame,
        portfolio_weights: np.ndarray,
        correlation_shocks: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    ) -> Dict[str, Dict[str, float]]:
        """Stress test portfolio under different correlation scenarios.

        Args:
            returns_data: Historical returns data
            portfolio_weights: Portfolio weights
            correlation_shocks: List of correlation levels to test

        Returns:
            Dictionary containing correlation stress test results
        """
        try:
            base_cov = returns_data.cov().values
            base_corr = returns_data.corr().values
            base_vol = np.sqrt(np.diag(base_cov))

            # Base portfolio volatility
            base_portfolio_vol = np.sqrt(
                np.dot(portfolio_weights, np.dot(base_cov, portfolio_weights))
            )

            results = {}

            for shock_corr in correlation_shocks:
                # Create stressed correlation matrix
                stressed_corr = np.full_like(base_corr, shock_corr)
                np.fill_diagonal(stressed_corr, 1.0)

                # Convert back to covariance matrix
                stressed_cov = np.outer(base_vol, base_vol) * stressed_corr

                # Calculate stressed portfolio volatility
                stressed_vol = np.sqrt(
                    np.dot(portfolio_weights, np.dot(stressed_cov, portfolio_weights))
                )

                # Calculate VaR impact
                vol_ratio = stressed_vol / base_portfolio_vol

                results[f"correlation_{shock_corr}"] = {
                    "portfolio_volatility": stressed_vol,
                    "volatility_ratio": vol_ratio,
                    "var_impact": vol_ratio - 1.0,
                    "correlation_level": shock_corr,
                }

            logger.info(
                f"Correlation stress test completed for {len(correlation_shocks)} scenarios"
            )

            return results

        except Exception as e:
            logger.error(f"Error in correlation stress test: {e}")
            raise


class TailRiskAnalyzer:
    """Tail risk analysis for extreme event assessment."""

    def __init__(self):
        """Initialize tail risk analyzer."""
        pass

    def extreme_value_analysis(
        self,
        returns_data: pd.Series,
        threshold_percentile: float = 0.05,
        confidence_level: float = 0.99,
    ) -> Dict[str, float]:
        """Perform extreme value analysis using Peaks over Threshold method.

        Args:
            returns_data: Return series
            threshold_percentile: Threshold percentile for tail analysis
            confidence_level: Confidence level for tail VaR

        Returns:
            Dictionary containing extreme value statistics
        """
        try:
            # Calculate threshold
            threshold = returns_data.quantile(threshold_percentile)

            # Extract exceedances
            exceedances = returns_data[returns_data <= threshold] - threshold

            if len(exceedances) < 10:
                logger.warning(
                    "Insufficient exceedances for robust extreme value analysis"
                )
                return {
                    "tail_var": float("nan"),
                    "tail_cvar": float("nan"),
                    "xi": float("nan"),
                    "beta": float("nan"),
                    "n_exceedances": len(exceedances),
                }

            # Fit Generalized Pareto Distribution
            # Simple method of moments estimation
            exceedances_abs = np.abs(exceedances)
            mean_exc = np.mean(exceedances_abs)
            var_exc = np.var(exceedances_abs)

            # Shape parameter (xi) and scale parameter (beta)
            xi = 0.5 * (mean_exc**2 / var_exc - 1)
            beta = 0.5 * mean_exc * (mean_exc**2 / var_exc + 1)

            # Tail VaR and CVaR estimation
            n_total = len(returns_data)
            n_exceedances = len(exceedances)
            exceedance_rate = n_exceedances / n_total

            # Tail VaR
            if xi != 0:
                tail_quantile = (1 - confidence_level) / exceedance_rate
                tail_var = threshold + (beta / xi) * (tail_quantile ** (-xi) - 1)
            else:
                tail_var = threshold + beta * np.log(
                    (1 - confidence_level) / exceedance_rate
                )

            # Tail CVaR
            if xi != 0 and xi < 1:
                tail_cvar = tail_var / (1 - xi) + (beta - xi * threshold) / (1 - xi)
            else:
                tail_cvar = tail_var  # Approximation

            results = {
                "tail_var": tail_var,
                "tail_cvar": tail_cvar,
                "xi": xi,  # Shape parameter
                "beta": beta,  # Scale parameter
                "threshold": threshold,
                "n_exceedances": n_exceedances,
                "exceedance_rate": exceedance_rate,
            }

            logger.info("Extreme value analysis completed")

            return results

        except Exception as e:
            logger.error(f"Error in extreme value analysis: {e}")
            raise

    def calculate_tail_dependence(
        self, returns_data: pd.DataFrame, quantile_level: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """Calculate tail dependence between assets.

        Args:
            returns_data: Multi-asset returns data
            quantile_level: Quantile level for tail analysis

        Returns:
            Dictionary containing tail dependence measures
        """
        try:
            n_assets = returns_data.shape[1]

            # Lower and upper tail dependence
            lower_tail_dep = np.zeros((n_assets, n_assets))
            upper_tail_dep = np.zeros((n_assets, n_assets))

            for i in range(n_assets):
                for j in range(i, n_assets):
                    if i == j:
                        lower_tail_dep[i, j] = upper_tail_dep[i, j] = 1.0
                    else:
                        # Lower tail dependence
                        lower_threshold_i = returns_data.iloc[:, i].quantile(
                            quantile_level
                        )
                        lower_threshold_j = returns_data.iloc[:, j].quantile(
                            quantile_level
                        )

                        both_lower = (returns_data.iloc[:, i] <= lower_threshold_i) & (
                            returns_data.iloc[:, j] <= lower_threshold_j
                        )
                        either_lower = (
                            returns_data.iloc[:, i] <= lower_threshold_i
                        ) | (returns_data.iloc[:, j] <= lower_threshold_j)

                        if np.sum(either_lower) > 0:
                            lower_tail_dep[i, j] = lower_tail_dep[j, i] = np.sum(
                                both_lower
                            ) / np.sum(either_lower)

                        # Upper tail dependence
                        upper_threshold_i = returns_data.iloc[:, i].quantile(
                            1 - quantile_level
                        )
                        upper_threshold_j = returns_data.iloc[:, j].quantile(
                            1 - quantile_level
                        )

                        both_upper = (returns_data.iloc[:, i] >= upper_threshold_i) & (
                            returns_data.iloc[:, j] >= upper_threshold_j
                        )
                        either_upper = (
                            returns_data.iloc[:, i] >= upper_threshold_i
                        ) | (returns_data.iloc[:, j] >= upper_threshold_j)

                        if np.sum(either_upper) > 0:
                            upper_tail_dep[i, j] = upper_tail_dep[j, i] = np.sum(
                                both_upper
                            ) / np.sum(either_upper)

            logger.info("Tail dependence analysis completed")

            return {
                "lower_tail_dependence": lower_tail_dep,
                "upper_tail_dependence": upper_tail_dep,
                "average_lower_tail_dep": np.mean(
                    lower_tail_dep[lower_tail_dep != 1.0]
                ),
                "average_upper_tail_dep": np.mean(
                    upper_tail_dep[upper_tail_dep != 1.0]
                ),
            }

        except Exception as e:
            logger.error(f"Error in tail dependence analysis: {e}")
            raise


class StressTesting:
    """Comprehensive stress testing framework for portfolio risk management."""

    def __init__(self, random_seed: int = 42):
        """Initialize stress testing framework.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.scenario_generator = ScenarioGenerator(random_seed)
        self.monte_carlo_engine = MonteCarloEngine(random_seed=random_seed)
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.tail_risk_analyzer = TailRiskAnalyzer()

        logger.info("Stress Testing framework initialized")

    def run_scenario_stress_test(
        self,
        portfolio_weights: np.ndarray,
        asset_returns: pd.DataFrame,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        portfolio_value: float = 1000000.0,
    ) -> Dict[str, StressTestResult]:
        """Run scenario-based stress tests.

        Args:
            portfolio_weights: Portfolio weights
            asset_returns: Historical asset returns
            scenarios: List of stress scenarios (if None, uses default scenarios)
            portfolio_value: Current portfolio value

        Returns:
            Dictionary containing stress test results for each scenario
        """
        try:
            if scenarios is None:
                scenarios = [
                    self.scenario_generator.generate_market_crash_scenario("moderate"),
                    self.scenario_generator.generate_market_crash_scenario("severe"),
                    self.scenario_generator.generate_interest_rate_scenario("up", 0.02),
                    self.scenario_generator.generate_volatility_scenario("high"),
                    self.scenario_generator.generate_currency_crisis_scenario(),
                ]

            results = {}

            for scenario in scenarios:
                # Apply scenario shocks to historical data
                shocked_returns = self._apply_scenario_shocks(asset_returns, scenario)

                # Calculate portfolio return under stress
                portfolio_return = np.dot(
                    shocked_returns.mean().values, portfolio_weights
                )
                stressed_value = portfolio_value * (1 + portfolio_return)
                loss = portfolio_value - stressed_value
                loss_percentage = loss / portfolio_value

                # Calculate additional metrics
                portfolio_vol = np.sqrt(
                    np.dot(
                        portfolio_weights,
                        np.dot(shocked_returns.cov().values, portfolio_weights),
                    )
                )

                result = StressTestResult(
                    test_type="scenario",
                    scenario_name=scenario["name"],
                    portfolio_value=portfolio_value,
                    stressed_value=stressed_value,
                    loss=loss,
                    loss_percentage=loss_percentage,
                    confidence_level=0.0,  # Not applicable for scenario tests
                    metrics={
                        "portfolio_return": portfolio_return,
                        "portfolio_volatility": portfolio_vol * np.sqrt(252),
                        "sharpe_ratio": (
                            portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                        ),
                    },
                    scenario_details=scenario,
                )

                results[scenario["name"]] = result

            logger.info(
                f"Scenario stress test completed for {len(scenarios)} scenarios"
            )

            return results

        except Exception as e:
            logger.error(f"Error in scenario stress test: {e}")
            raise

    def run_monte_carlo_stress_test(
        self,
        portfolio_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_value: float = 1000000.0,
        time_horizon: int = 252,
        confidence_levels: List[float] = [0.95, 0.99, 0.999],
    ) -> StressTestResult:
        """Run Monte Carlo stress test.

        Args:
            portfolio_weights: Portfolio weights
            expected_returns: Expected asset returns
            covariance_matrix: Asset covariance matrix
            portfolio_value: Current portfolio value
            time_horizon: Simulation horizon in days
            confidence_levels: VaR confidence levels

        Returns:
            Stress test result
        """
        try:
            # Run Monte Carlo simulation
            mc_results = self.monte_carlo_engine.simulate_portfolio_returns(
                expected_returns,
                covariance_matrix,
                portfolio_weights,
                time_horizon,
                confidence_levels,
            )

            # Extract worst-case scenarios
            final_returns = mc_results["final_returns"]
            worst_case_return = np.min(final_returns)
            worst_case_value = portfolio_value * (1 + worst_case_return)

            # VaR metrics
            var_99 = mc_results["var_metrics"].get("VaR_0.99", worst_case_return)
            var_99_value = portfolio_value * (1 + var_99)

            result = StressTestResult(
                test_type="monte_carlo",
                scenario_name=f"Monte Carlo ({self.monte_carlo_engine.num_simulations} simulations)",
                portfolio_value=portfolio_value,
                stressed_value=var_99_value,
                loss=portfolio_value - var_99_value,
                loss_percentage=-var_99,
                confidence_level=0.99,
                metrics={
                    "worst_case_return": worst_case_return,
                    "worst_case_value": worst_case_value,
                    "expected_return": mc_results["statistics"]["mean_return"],
                    "volatility": mc_results["statistics"]["std_return"],
                    "skewness": mc_results["statistics"]["skewness"],
                    "kurtosis": mc_results["statistics"]["kurtosis"],
                    "probability_loss": mc_results["statistics"]["probability_loss"],
                },
                scenario_details={
                    "simulation_params": mc_results["portfolio_params"],
                    "var_metrics": mc_results["var_metrics"],
                    "statistics": mc_results["statistics"],
                },
            )

            logger.info("Monte Carlo stress test completed")

            return result

        except Exception as e:
            logger.error(f"Error in Monte Carlo stress test: {e}")
            raise

    def run_comprehensive_stress_test(
        self,
        portfolio_weights: np.ndarray,
        asset_returns: pd.DataFrame,
        asset_prices: Optional[np.ndarray] = None,
        portfolio_value: float = 1000000.0,
    ) -> Dict[str, Any]:
        """Run comprehensive stress testing suite.

        Args:
            portfolio_weights: Portfolio weights
            asset_returns: Historical asset returns
            asset_prices: Current asset prices (optional)
            portfolio_value: Current portfolio value

        Returns:
            Dictionary containing all stress test results
        """
        try:
            logger.info("Starting comprehensive stress test suite")

            results = {
                "scenario_tests": {},
                "monte_carlo_test": None,
                "sensitivity_analysis": {},
                "tail_risk_analysis": {},
                "summary": {},
            }

            # Expected returns and covariance
            expected_returns = asset_returns.mean().values * 252
            covariance_matrix = asset_returns.cov().values * 252

            # 1. Scenario stress tests
            results["scenario_tests"] = self.run_scenario_stress_test(
                portfolio_weights, asset_returns, portfolio_value=portfolio_value
            )

            # 2. Monte Carlo stress test
            results["monte_carlo_test"] = self.run_monte_carlo_stress_test(
                portfolio_weights,
                expected_returns,
                covariance_matrix,
                portfolio_value=portfolio_value,
            )

            # 3. Sensitivity analysis (if prices available)
            if asset_prices is not None:
                results["sensitivity_analysis"] = (
                    self.sensitivity_analyzer.calculate_portfolio_sensitivities(
                        portfolio_weights, asset_prices
                    )
                )

                # Correlation stress test
                results["correlation_stress"] = (
                    self.sensitivity_analyzer.stress_test_correlation_changes(
                        asset_returns, portfolio_weights
                    )
                )

            # 4. Tail risk analysis
            portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
            results["tail_risk_analysis"] = (
                self.tail_risk_analyzer.extreme_value_analysis(portfolio_returns)
            )

            # 5. Summary statistics
            scenario_losses = [
                result.loss_percentage for result in results["scenario_tests"].values()
            ]
            mc_loss = results["monte_carlo_test"].loss_percentage

            results["summary"] = {
                "worst_scenario_loss": max(scenario_losses) if scenario_losses else 0,
                "average_scenario_loss": (
                    np.mean(scenario_losses) if scenario_losses else 0
                ),
                "monte_carlo_var_99": mc_loss,
                "tail_var_99": results["tail_risk_analysis"].get(
                    "tail_var", float("nan")
                ),
                "number_scenarios_tested": len(results["scenario_tests"]),
                "portfolio_value": portfolio_value,
            }

            logger.info("Comprehensive stress test suite completed successfully")

            return results

        except Exception as e:
            logger.error(f"Error in comprehensive stress test: {e}")
            raise

    def _apply_scenario_shocks(
        self, asset_returns: pd.DataFrame, scenario: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply scenario shocks to asset returns.

        Args:
            asset_returns: Historical asset returns
            scenario: Scenario parameters

        Returns:
            Shocked returns DataFrame
        """
        shocked_returns = asset_returns.copy()

        # Apply equity shock if present
        if "equity_shock" in scenario:
            shocked_returns = shocked_returns + scenario["equity_shock"] / len(
                shocked_returns
            )

        # Apply volatility multiplier
        if "volatility_multiplier" in scenario:
            mean_returns = shocked_returns.mean()
            shocked_returns = (shocked_returns - mean_returns) * scenario[
                "volatility_multiplier"
            ] + mean_returns

        return shocked_returns
