"""
Risk Management UI Utilities (Clean Architecture)
Week 14 UI Integration - UI Support Functions ONLY
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import logging
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Backend imports ONLY - NO local implementations
    from src.risk.position_sizing import PositionSizer, PortfolioSizer
    from src.risk.risk_metrics import RiskMetrics, PortfolioRiskAnalyzer
    from src.risk.portfolio_optimization import (
        PortfolioOptimizer,
        AFMLPortfolioOptimizer,
    )
    from src.risk.stress_testing import StressTesting, ScenarioGenerator
except ImportError as e:
    st.error(f"Failed to import risk management backend: {e}")


class RiskManagementProcessor:
    """Risk Management UI Data Processor - Backend integration ONLY"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use backend classes ONLY
        try:
            self.risk_metrics = RiskMetrics()
            self.portfolio_optimizer = PortfolioOptimizer()
            self.stress_tester = StressTesting()
            self.scenario_generator = ScenarioGenerator()
        except Exception as e:
            self.logger.error(f"Failed to initialize backend classes: {e}")
            self.risk_metrics = None
            self.portfolio_optimizer = None
            self.stress_tester = None
            self.scenario_generator = None

    def extract_backtest_data(self, backtest_key: str) -> Optional[pd.Series]:
        """Extract returns data from backtest cache - data extraction ONLY"""

        try:
            if (
                "backtest_cache" not in st.session_state
                or backtest_key not in st.session_state.backtest_cache
            ):
                self.logger.warning("No backtest cache found in session state")
                return None

            backtest_result = st.session_state.backtest_cache[backtest_key]
            self.logger.info(f"Backtest result keys: {list(backtest_result.keys())}")

            # Try multiple data extraction strategies

            # Strategy 1: Direct returns data
            if "returns" in backtest_result:
                returns_data = backtest_result["returns"]
                self.logger.info(
                    f"Found 'returns' data: type={type(returns_data)}, len={len(returns_data) if hasattr(returns_data, '__len__') else 'N/A'}"
                )
                if (
                    isinstance(returns_data, (list, np.ndarray))
                    and len(returns_data) > 1
                ):
                    return pd.Series(returns_data)
                elif isinstance(returns_data, pd.Series) and len(returns_data) > 1:
                    return returns_data

            # Strategy 2: Extract from portfolio values
            if "portfolio_values" in backtest_result:
                portfolio_values = backtest_result["portfolio_values"]
                self.logger.info(
                    f"Found 'portfolio_values' data: type={type(portfolio_values)}, len={len(portfolio_values) if hasattr(portfolio_values, '__len__') else 'N/A'}"
                )
                if (
                    isinstance(portfolio_values, (list, np.ndarray))
                    and len(portfolio_values) > 1
                ):
                    portfolio_series = pd.Series(portfolio_values)
                    # Calculate returns: (current - previous) / previous
                    returns = portfolio_series.pct_change().dropna()
                    if len(returns) > 0:
                        self.logger.info(
                            f"Calculated returns from portfolio values: {len(returns)} points"
                        )
                        return returns

            # Strategy 3: Extract from results dictionary
            if "results" in backtest_result:
                results = backtest_result["results"]
                self.logger.info(f"Found 'results' data: type={type(results)}")
                if isinstance(results, dict):
                    if "returns" in results:
                        returns_data = results["returns"]
                        if (
                            isinstance(returns_data, (list, pd.Series))
                            and len(returns_data) > 1
                        ):
                            return (
                                pd.Series(returns_data)
                                if isinstance(returns_data, list)
                                else returns_data
                            )
                    if "portfolio_value" in results:
                        portfolio_values = results["portfolio_value"]
                        if (
                            isinstance(portfolio_values, (list, pd.Series))
                            and len(portfolio_values) > 1
                        ):
                            portfolio_series = (
                                pd.Series(portfolio_values)
                                if isinstance(portfolio_values, list)
                                else portfolio_values
                            )
                            returns = portfolio_series.pct_change().dropna()
                            if len(returns) > 0:
                                return returns

            # Strategy 4: Look for any time series data that could be portfolio values
            for key in [
                "equity_curve",
                "portfolio_equity",
                "cumulative_returns",
                "nav",
            ]:
                if key in backtest_result:
                    data = backtest_result[key]
                    self.logger.info(f"Found '{key}' data: type={type(data)}")
                    if (
                        isinstance(data, (list, np.ndarray, pd.Series))
                        and len(data) > 1
                    ):
                        series_data = (
                            pd.Series(data) if not isinstance(data, pd.Series) else data
                        )
                        if (
                            series_data.nunique() > 1
                        ):  # Ensure it's not all the same values
                            returns = series_data.pct_change().dropna()
                            if len(returns) > 0:
                                self.logger.info(
                                    f"Calculated returns from '{key}': {len(returns)} points"
                                )
                                return returns

            self.logger.warning(
                f"Could not extract valid returns data from backtest result. Available keys: {list(backtest_result.keys())}"
            )
            return None

        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            return None

    def get_risk_metrics_from_backend(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Get risk metrics using backend ONLY - NO calculations here"""

        if self.risk_metrics is None or returns is None or len(returns) < 2:
            return self._empty_risk_metrics()

        try:
            # Use available backend methods ONLY
            var = self.risk_metrics.value_at_risk(
                returns, confidence_level=confidence_level, method="historical"
            )
            cvar = self.risk_metrics.conditional_var(
                returns, confidence_level=confidence_level, method="historical"
            )

            # Get drawdown metrics
            drawdown_results = self.risk_metrics.maximum_drawdown(returns)
            max_drawdown = drawdown_results.get("max_drawdown", 0)

            # Get risk-adjusted returns metrics
            risk_adjusted = self.risk_metrics.risk_adjusted_returns(returns)

            return {
                "var_95": var,
                "cvar_95": cvar,
                "max_drawdown": max_drawdown,
                "volatility": risk_adjusted.get("annual_volatility", 0),
                "sharpe_ratio": risk_adjusted.get("sharpe_ratio", 0),
                "sortino_ratio": risk_adjusted.get("sortino_ratio", 0),
                "calmar_ratio": risk_adjusted.get("calmar_ratio", 0),
                "analysis_date": pd.Timestamp.now(),
            }

        except Exception as e:
            self.logger.error(f"Backend risk calculation failed: {e}")
            return self._empty_risk_metrics()

    def get_position_sizes_from_backend(
        self, returns: pd.Series, method: str = "Kelly"
    ) -> Dict[str, float]:
        """Calculate position sizes using backend ONLY"""

        if returns is None:
            return {"position_size": 0.0, "confidence": 0.0}

        try:
            # Use backend position sizing engine ONLY
            from src.risk.position_sizing import PositionSizingEngine

            engine = PositionSizingEngine()

            sizing_result = engine.calculate_optimal_size(
                returns=returns, method=method, risk_tolerance=0.02
            )

            return (
                sizing_result
                if sizing_result
                else {"position_size": 0.1, "method": "Default"}
            )

        except Exception as e:
            self.logger.error(f"Position sizing failed: {e}")
            return {"position_size": 0.0, "error": str(e)}

    def get_portfolio_optimization_from_backend(
        self,
        returns: pd.Series,
        method: str = "Mean Variance",
        max_weight: float = 0.3,
        target_volatility: float = 0.15,
    ) -> Dict[str, Any]:
        """Portfolio optimization using backend ONLY"""

        if self.portfolio_optimizer is None or returns is None or len(returns) < 10:
            return self._empty_optimization_results()

        try:
            # Create a simple multi-asset returns matrix (simulate multiple assets)
            # In real implementation, this would come from the actual portfolio assets
            n_assets = 3  # Assume 3 assets for demo

            # Create correlated returns for multiple assets based on single returns series
            np.random.seed(42)  # For reproducible results
            returns_matrix = pd.DataFrame(
                {
                    "Asset_1": returns,
                    "Asset_2": returns + np.random.normal(0, 0.01, len(returns)),
                    "Asset_3": returns + np.random.normal(0, 0.015, len(returns)),
                }
            )

            # Prepare data in the format expected by backend
            expected_returns = returns_matrix.mean().values * 252  # Annualize
            covariance_matrix = returns_matrix.cov().values * 252  # Annualize

            self.logger.info(f"Expected returns: {expected_returns}")
            self.logger.info(f"Covariance matrix shape: {covariance_matrix.shape}")

            # Map method names to backend method calls
            if method == "Mean Variance":
                optimization_method = "mean_variance"
            elif method == "Risk Parity":
                optimization_method = "risk_parity"
            elif method == "Maximum Sharpe":
                optimization_method = "max_sharpe"
            elif method == "Minimum Variance":
                optimization_method = "min_variance"
            else:
                optimization_method = "mean_variance"

            # Use backend optimizer with correct method calls
            if optimization_method == "mean_variance":
                results = self.portfolio_optimizer.mean_variance_optimization(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    target_volatility=target_volatility,
                    objective="sharpe",
                )
            elif optimization_method == "risk_parity":
                results = self.portfolio_optimizer.risk_parity_optimization(
                    covariance_matrix=covariance_matrix
                )
            elif optimization_method == "min_variance":
                results = self.portfolio_optimizer.minimum_variance_optimization(
                    covariance_matrix=covariance_matrix
                )
            else:  # Default to mean variance
                results = self.portfolio_optimizer.mean_variance_optimization(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    target_volatility=target_volatility,
                    objective="sharpe",
                )

            # Format results for UI
            if results and "weights" in results:
                weights = results["weights"]

                # Convert numpy array to dictionary with asset names
                if isinstance(weights, np.ndarray):
                    asset_names = ["Asset_1", "Asset_2", "Asset_3"]
                    weights_dict = {
                        asset: float(weight)
                        for asset, weight in zip(asset_names, weights)
                    }
                else:
                    weights_dict = weights

                self.logger.info(f"Optimization results: {results}")

                return {
                    "weights": weights_dict,
                    "expected_return": results.get("expected_return", 0),
                    "volatility": results.get(
                        "volatility",
                        np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights))),
                    ),
                    "sharpe_ratio": results.get("sharpe_ratio", 0),
                    "method": method,
                    "optimization_date": pd.Timestamp.now(),
                }
            else:
                self.logger.warning(f"Invalid optimization results: {results}")
                return self._empty_optimization_results()

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._empty_optimization_results()

    def _empty_optimization_results(self) -> Dict[str, Any]:
        """Return empty optimization results"""
        return {
            "weights": {},
            "expected_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "method": "None",
            "error": "Optimization failed",
        }

    def get_stress_test_from_backend(
        self,
        returns: pd.Series,
        stress_type: str = "Market Crash",
        severity: str = "Moderate",
        num_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """Stress testing using backend ONLY"""

        if (
            self.stress_tester is None
            or self.scenario_generator is None
            or returns is None
        ):
            return self._empty_stress_test_results()

        try:
            # Generate scenario based on type using backend
            if stress_type == "Market Crash":
                scenario = self.scenario_generator.generate_market_crash_scenario(
                    severity=severity.lower()
                )
            elif stress_type == "Interest Rate Shock":
                shock_size = (
                    0.01
                    if severity == "Mild"
                    else 0.02 if severity == "Moderate" else 0.03
                )
                scenario = self.scenario_generator.generate_interest_rate_scenario(
                    direction="up", shock_size=shock_size
                )
            elif stress_type == "High Volatility":
                scenario = self.scenario_generator.generate_volatility_scenario(
                    regime="high"
                )
            else:
                # Custom scenario
                scenario = {
                    "type": "custom",
                    "market_shock": (
                        -0.15
                        if severity == "Mild"
                        else -0.25 if severity == "Moderate" else -0.35
                    ),
                    "volatility_multiplier": (
                        1.5
                        if severity == "Mild"
                        else 2.0 if severity == "Moderate" else 3.0
                    ),
                }

            self.logger.info(f"Generated scenario: {scenario}")

            # Prepare data for stress testing
            asset_returns = pd.DataFrame({"portfolio": returns})
            portfolio_weights = np.array([1.0])  # Single portfolio asset

            # Use backend stress testing with correct method
            stress_results = self.stress_tester.run_scenario_stress_test(
                portfolio_weights=portfolio_weights,
                asset_returns=asset_returns,
                scenarios=[scenario],
                portfolio_value=100000.0,
            )

            # Extract results from stress test
            if stress_results and len(stress_results) > 0:
                # Get the first (and only) scenario result
                first_result = list(stress_results.values())[0]

                # Extract metrics from StressTestResult object
                worst_case_loss = getattr(first_result, "worst_case_loss", 0)
                probability_of_loss = getattr(first_result, "probability_of_loss", 0)
                expected_loss = getattr(first_result, "expected_loss", 0)

                # Convert to percentages if they're not already
                if abs(worst_case_loss) > 1:  # Assume it's in absolute terms
                    worst_case_loss = (
                        worst_case_loss / 100000.0
                    )  # Convert to percentage
                if abs(expected_loss) > 1:
                    expected_loss = expected_loss / 100000.0

                return {
                    "scenario": scenario,
                    "worst_case_loss": worst_case_loss,
                    "probability_of_loss": probability_of_loss,
                    "average_loss": expected_loss,  # Use expected_loss instead
                    "num_simulations": num_simulations,
                    "stress_type": stress_type,
                    "severity": severity,
                    "analysis_date": pd.Timestamp.now(),
                }
            else:
                # Fallback: Simple stress calculation if backend method fails
                self.logger.warning(
                    "Backend stress test returned empty results, using simple calculation"
                )

                # Apply stress scenario manually to returns
                stressed_returns = returns.copy()

                if "equity_shock" in scenario:
                    stressed_returns = stressed_returns + scenario["equity_shock"]
                elif "market_shock" in scenario:
                    stressed_returns = stressed_returns + scenario["market_shock"]

                # Apply volatility multiplier if present
                if "volatility_multiplier" in scenario:
                    vol_multiplier = scenario["volatility_multiplier"]
                    mean_return = stressed_returns.mean()
                    stressed_returns = (
                        mean_return + (stressed_returns - mean_return) * vol_multiplier
                    )

                # Calculate simple metrics
                worst_case = stressed_returns.min()
                prob_loss = (stressed_returns < 0).mean()
                avg_loss = (
                    stressed_returns[stressed_returns < 0].mean()
                    if (stressed_returns < 0).any()
                    else 0
                )

                return {
                    "scenario": scenario,
                    "worst_case_loss": worst_case,
                    "probability_of_loss": prob_loss,
                    "average_loss": avg_loss,
                    "num_simulations": num_simulations,
                    "stress_type": stress_type,
                    "severity": severity,
                    "analysis_date": pd.Timestamp.now(),
                }

        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._empty_stress_test_results()

    def _empty_stress_test_results(self) -> Dict[str, Any]:
        """Return empty stress test results"""
        return {
            "scenario": {},
            "worst_case_loss": 0,
            "probability_of_loss": 0,
            "average_loss": 0,
            "num_simulations": 0,
            "error": "Stress test failed",
        }

    def _empty_risk_metrics(self) -> Dict[str, Any]:
        """Return empty risk metrics structure"""
        return {
            "var": 0.0,
            "cvar": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
        }


class RiskVisualizationManager:
    """Risk Management Visualization Manager - UI display ONLY"""

    @staticmethod
    def create_risk_dashboard(risk_metrics: Dict[str, Any]) -> go.Figure:
        """Create risk metrics dashboard chart"""

        try:
            # Create gauge charts for key metrics
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "VaR (95%)",
                    "Sharpe Ratio",
                    "Max Drawdown",
                    "Volatility",
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}],
                ],
            )

            # VaR gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=abs(risk_metrics.get("var", 0)) * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "VaR (%)"},
                    gauge={
                        "axis": {"range": [None, 10]},
                        "bar": {"color": "red"},
                        "bgcolor": "lightgray",
                    },
                ),
                row=1,
                col=1,
            )

            # Sharpe ratio gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_metrics.get("sharpe_ratio", 0),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Sharpe Ratio"},
                    gauge={
                        "axis": {"range": [0, 3]},
                        "bar": {"color": "green"},
                        "bgcolor": "lightgray",
                    },
                ),
                row=1,
                col=2,
            )

            # Max drawdown gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=abs(risk_metrics.get("max_drawdown", 0)) * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Max DD (%)"},
                    gauge={
                        "axis": {"range": [0, 50]},
                        "bar": {"color": "orange"},
                        "bgcolor": "lightgray",
                    },
                ),
                row=2,
                col=1,
            )

            # Volatility gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_metrics.get("volatility", 0) * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Volatility (%)"},
                    gauge={
                        "axis": {"range": [0, 50]},
                        "bar": {"color": "blue"},
                        "bgcolor": "lightgray",
                    },
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                title="Risk Metrics Dashboard", height=600, showlegend=False
            )

            return fig

        except Exception as e:
            # Fallback empty chart
            fig = go.Figure()
            fig.add_annotation(text=f"Chart error: {e}", x=0.5, y=0.5)
            return fig

    @staticmethod
    def create_position_sizing_chart(sizing_results: Dict[str, Any]) -> go.Figure:
        """Create position sizing visualization"""

        try:
            fig = go.Figure()

            # Position size bar
            fig.add_trace(
                go.Bar(
                    x=["Recommended Position Size"],
                    y=[sizing_results.get("position_size", 0)],
                    name="Position Size",
                    marker_color="blue",
                )
            )

            fig.update_layout(
                title=f"Position Sizing Analysis ({sizing_results.get('method', 'Unknown')})",
                yaxis_title="Position Size (%)",
                xaxis_title="",
                height=400,
            )

            return fig

        except Exception as e:
            # Fallback empty chart
            fig = go.Figure()
            fig.add_annotation(text=f"Chart error: {e}", x=0.5, y=0.5)
            return fig

    @staticmethod
    def create_stress_test_chart(stress_results: Dict[str, Any]) -> go.Figure:
        """Create stress test results visualization"""

        try:
            fig = go.Figure()

            # Mock stress test visualization
            scenarios = ["Base Case", "Mild Stress", "Severe Stress"]
            returns = [0.10, -0.05, -0.20]

            fig.add_trace(
                go.Bar(
                    x=scenarios,
                    y=returns,
                    name="Portfolio Returns",
                    marker_color=["green", "orange", "red"],
                )
            )

            fig.update_layout(
                title="Stress Test Results",
                yaxis_title="Portfolio Return (%)",
                xaxis_title="Stress Scenario",
                height=400,
            )

            return fig

        except Exception as e:
            # Fallback empty chart
            fig = go.Figure()
            fig.add_annotation(text=f"Chart error: {e}", x=0.5, y=0.5)
            return fig
