"""
Risk Management UI Utilities
Week 14 UI Integration - Risk Management Support Functions
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from pathlib import Path
import sys
import logging

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.risk.position_sizing import PositionSizer, PortfolioSizer
    from src.risk.risk_metrics import RiskMetrics, PortfolioRiskAnalyzer
    from src.risk.portfolio_optimization import (
        PortfolioOptimizer,
        AFMLPortfolioOptimizer,
    )
    from src.risk.stress_testing import (
        ScenarioGenerator,
        MonteCarloEngine,
        SensitivityAnalyzer,
        TailRiskAnalyzer,
        StressTesting,
    )
except ImportError as e:
    st.error(f"Failed to import risk management modules: {e}")


class RiskManagementProcessor:
    """
    Risk Management UI Data Processor

    Handles data preparation, calculation, and result formatting
    for risk management UI components.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_metrics = RiskMetrics()
        self.position_sizer = PositionSizer()

    def extract_backtest_data(self, backtest_key: str) -> pd.Series:
        """Extract returns data from backtest cache for risk analysis."""
        try:
            import streamlit as st

            if (
                "backtest_cache" not in st.session_state
                or backtest_key not in st.session_state.backtest_cache
            ):
                print(f"❌ No backtest data found for key: {backtest_key}")
                return None

            backtest_result = st.session_state.backtest_cache[backtest_key]
            print(
                f"🔍 Extracting data from backtest result keys: {list(backtest_result.keys())}"
            )

            # Extract portfolio returns - check for different formats
            returns = None

            # Strategy 1: Check for direct returns
            if "returns" in backtest_result:
                returns = backtest_result["returns"]
                if isinstance(returns, pd.Series) and len(returns) > 0:
                    print(f"✅ Found returns data: {len(returns)} periods")
                else:
                    returns = None

            # Strategy 2: Calculate from portfolio values
            if returns is None and "portfolio_values" in backtest_result:
                portfolio_values = backtest_result["portfolio_values"]
                if isinstance(portfolio_values, list) and len(portfolio_values) > 1:
                    portfolio_series = pd.Series(portfolio_values)
                    returns = portfolio_series.pct_change().dropna()
                    print(
                        f"✅ Calculated returns from portfolio values: {len(returns)} periods"
                    )
                elif (
                    isinstance(portfolio_values, pd.Series)
                    and len(portfolio_values) > 1
                ):
                    returns = portfolio_values.pct_change().dropna()
                    print(
                        f"✅ Calculated returns from portfolio series: {len(returns)} periods"
                    )

            # Strategy 3: Legacy support for other formats
            if returns is None:
                if "portfolio_returns" in backtest_result:
                    returns = backtest_result["portfolio_returns"]
                elif "equity_curve" in backtest_result:
                    equity_curve = backtest_result["equity_curve"]
                    returns = equity_curve.pct_change().dropna()
                else:
                    # Generate from trades or positions if available
                    returns = self._calculate_returns_from_trades(backtest_result)

            # Ensure returns is a pandas Series
            if returns is not None and not isinstance(returns, pd.Series):
                returns = pd.Series(returns)

            if returns is None or (isinstance(returns, pd.Series) and returns.empty):
                print("❌ No valid return data found")
                return None

            print(f"✅ Successfully extracted returns: {len(returns)} periods")
            return returns

        except Exception as e:
            self.logger.error(f"Error extracting backtest data: {e}")
            print(f"❌ Error in extract_backtest_data: {e}")
            import traceback

            traceback.print_exc()
            return {
                "returns": pd.Series(),
                "positions": pd.DataFrame(),
                "trades": pd.DataFrame(),
                "price_data": pd.DataFrame(),
            }

    def _calculate_returns_from_trades(self, backtest_result: Dict) -> pd.Series:
        """Calculate returns from trade data."""
        try:
            trades = backtest_result.get("trades", [])
            if not trades:
                return pd.Series()

            # Convert trades to DataFrame if it's a list
            if isinstance(trades, list):
                trades_df = pd.DataFrame(trades)
            else:
                trades_df = trades

            if trades_df.empty:
                return pd.Series()

            # Calculate daily returns from trade PnL
            if "pnl" in trades_df.columns and "date" in trades_df.columns:
                daily_pnl = trades_df.groupby("date")["pnl"].sum()
                # Assume initial capital for return calculation
                initial_capital = 100000  # Default
                returns = daily_pnl / initial_capital
                return returns

            return pd.Series()

        except Exception as e:
            self.logger.error(f"Error calculating returns from trades: {e}")
            return pd.Series()

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        var_method: str = "Historical",
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        try:
            if returns.empty:
                return self._empty_risk_metrics()

            # Initialize risk metrics calculator
            risk_calc = RiskMetrics(confidence_level=confidence_level)

            # Basic risk metrics
            metrics = {}

            # Volatility (annualized)
            metrics["volatility"] = returns.std() * np.sqrt(252)

            # Value at Risk
            if var_method == "Historical":
                metrics["var"] = risk_calc.value_at_risk(returns, method="historical")
            elif var_method == "Parametric":
                metrics["var"] = risk_calc.value_at_risk(returns, method="parametric")
            else:  # Cornish-Fisher
                metrics["var"] = risk_calc.value_at_risk(
                    returns, method="cornish_fisher"
                )

            # Conditional VaR
            if var_method == "Historical":
                metrics["cvar"] = risk_calc.conditional_var(
                    returns, method="historical"
                )
            elif var_method == "Parametric":
                metrics["cvar"] = risk_calc.conditional_var(
                    returns, method="parametric"
                )
            else:
                metrics["cvar"] = risk_calc.conditional_var(
                    returns, method="parametric"
                )

            # Maximum Drawdown
            dd_metrics = risk_calc.maximum_drawdown(returns)
            metrics["max_drawdown"] = dd_metrics.get("max_drawdown", 0.0)

            # Risk-adjusted return metrics
            risk_adj_metrics = risk_calc.risk_adjusted_returns(returns)
            metrics["sharpe_ratio"] = risk_adj_metrics.get("sharpe_ratio", 0.0)
            metrics["sortino_ratio"] = risk_adj_metrics.get("sortino_ratio", 0.0)
            metrics["calmar_ratio"] = risk_adj_metrics.get(
                "calmar_ratio", 0.0
            )  # Beta (if benchmark available)
            # metrics["beta"] = self._calculate_beta(returns, benchmark)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return self._empty_risk_metrics()

    def _empty_risk_metrics(self) -> Dict[str, Any]:
        """Return empty risk metrics structure."""
        return {
            "volatility": 0.0,
            "var": 0.0,
            "cvar": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
        }

    def calculate_position_sizes(
        self, returns: pd.Series, method: str = "Kelly", **kwargs
    ) -> Dict[str, float]:
        """Calculate optimal position sizes using various methods."""
        try:
            if returns.empty:
                return {"position_size": 0.0}

            sizer = PositionSizer()

            if method == "Kelly":
                win_rate = kwargs.get("win_rate", (returns > 0).mean())
                avg_win = kwargs.get("avg_win", returns[returns > 0].mean())
                avg_loss = kwargs.get("avg_loss", abs(returns[returns < 0].mean()))

                # Calculate win/loss ratio
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

                position_size = sizer.kelly_criterion(
                    win_prob=win_rate, win_loss_ratio=win_loss_ratio
                )

            elif method == "Risk Parity":
                # For single asset, use volatility targeting
                target_vol = kwargs.get("target_volatility", 0.15)
                current_vol = returns.std() * np.sqrt(252)
                position_size = target_vol / current_vol if current_vol > 0 else 0

            elif method == "Volatility Targeting":
                target_vol = kwargs.get("target_volatility", 0.15)
                position_size = sizer.volatility_targeting(
                    returns=returns, target_volatility=target_vol
                )

            else:  # Fixed Fractional
                risk_per_trade = kwargs.get("risk_per_trade", 0.02)
                stop_loss_pct = kwargs.get("stop_loss_pct", 0.05)
                position_size = sizer.fixed_fractional(
                    risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct
                )

            return {
                "position_size": position_size,
                "method": method,
                "parameters": kwargs,
            }

        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {e}")
            return {"position_size": 0.0}

    def run_portfolio_optimization(
        self, returns_data: pd.DataFrame, method: str = "Mean Variance", **kwargs
    ) -> Dict[str, Any]:
        """Run portfolio optimization."""
        try:
            if returns_data.empty:
                return {"weights": [], "expected_return": 0.0, "volatility": 0.0}

            optimizer = PortfolioOptimizer()

            if method == "Mean Variance":
                result = optimizer.mean_variance_optimization(
                    returns=returns_data,
                    target_return=kwargs.get("target_return"),
                    risk_aversion=kwargs.get("risk_aversion", 1.0),
                )

            elif method == "Risk Parity":
                result = optimizer.risk_parity_optimization(returns=returns_data)

            elif method == "Minimum Variance":
                result = optimizer.minimum_variance_optimization(returns=returns_data)

            elif method == "Black-Litterman":
                # Requires views and confidence levels
                views = kwargs.get("views", {})
                result = optimizer.black_litterman_optimization(
                    returns=returns_data, views=views
                )

            else:  # AFML Ensemble
                afml_optimizer = AFMLPortfolioOptimizer()
                result = afml_optimizer.ensemble_optimization(returns=returns_data)

            return result

        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return {"weights": [], "expected_return": 0.0, "volatility": 0.0}

    def run_stress_test(
        self,
        returns: pd.Series,
        stress_type: str,
        scenario_params: Dict,
        num_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """Run stress testing scenarios."""
        try:
            if returns.empty:
                return {"stressed_returns": [], "statistics": {}}

            stress_tester = StressTesting()

            if stress_type == "Market Crash":
                scenarios = stress_tester.market_crash_scenario(
                    returns=returns,
                    crash_magnitude=scenario_params.get("market_drop", 0.2),
                    num_simulations=num_simulations,
                )

            elif stress_type == "Interest Rate Shock":
                scenarios = stress_tester.interest_rate_shock(
                    returns=returns,
                    rate_change=scenario_params.get("rate_change", 100),
                    num_simulations=num_simulations,
                )

            elif stress_type == "Volatility Spike":
                scenarios = stress_tester.volatility_shock(
                    returns=returns,
                    vol_multiplier=scenario_params.get("vol_multiplier", 2.0),
                    num_simulations=num_simulations,
                )

            else:  # Custom Scenario
                scenarios = stress_tester.custom_scenario(
                    returns=returns,
                    shock_magnitude=scenario_params.get("custom_shock", -0.1),
                    num_simulations=num_simulations,
                )

            return scenarios

        except Exception as e:
            self.logger.error(f"Error in stress testing: {e}")
            return {"stressed_returns": [], "statistics": {}}


class RiskVisualizationManager:
    """
    Risk Management Visualization Manager

    Creates professional charts and displays for risk analysis results.
    """

    @staticmethod
    def create_risk_dashboard(risk_metrics: Dict[str, Any]) -> go.Figure:
        """Create comprehensive risk metrics dashboard."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Risk Metrics Overview",
                "VaR vs CVaR",
                "Risk-Return Profile",
                "Performance Ratios",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # Risk Metrics Overview (Gauge)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_metrics.get("volatility", 0) * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Volatility (%)"},
                gauge={
                    "axis": {"range": [None, 30]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 10], "color": "lightgreen"},
                        {"range": [10, 20], "color": "yellow"},
                        {"range": [20, 30], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 25,
                    },
                },
            ),
            row=1,
            col=1,
        )

        # VaR vs CVaR Comparison
        var_cvar_data = ["VaR", "CVaR"]
        var_cvar_values = [
            abs(risk_metrics.get("var", 0)) * 100,
            abs(risk_metrics.get("cvar", 0)) * 100,
        ]

        fig.add_trace(
            go.Bar(
                x=var_cvar_data,
                y=var_cvar_values,
                name="Risk Measures",
                marker_color=["red", "darkred"],
            ),
            row=1,
            col=2,
        )

        # Risk-Return Profile (placeholder for single asset)
        fig.add_trace(
            go.Scatter(
                x=[risk_metrics.get("volatility", 0) * 100],
                y=[0],  # Would need expected return
                mode="markers",
                marker=dict(size=15, color="blue"),
                name="Current Portfolio",
            ),
            row=2,
            col=1,
        )

        # Performance Ratios
        ratio_names = ["Sharpe", "Sortino", "Calmar"]
        ratio_values = [
            risk_metrics.get("sharpe_ratio", 0),
            risk_metrics.get("sortino_ratio", 0),
            risk_metrics.get("calmar_ratio", 0),
        ]

        colors = ["green" if x > 0 else "red" for x in ratio_values]

        fig.add_trace(
            go.Bar(
                x=ratio_names,
                y=ratio_values,
                name="Performance Ratios",
                marker_color=colors,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=600, title_text="Risk Management Dashboard", showlegend=False
        )

        return fig

    @staticmethod
    def create_drawdown_chart(returns: pd.Series) -> go.Figure:
        """Create drawdown analysis chart."""
        if returns.empty:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Calculate cumulative returns and drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Cumulative Returns", "Drawdown"),
            vertical_spacing=0.1,
        )

        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                mode="lines",
                name="Cumulative Returns",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode="lines",
                name="Drawdown (%)",
                fill="tonexty",
                line=dict(color="red"),
                fillcolor="rgba(255,0,0,0.3)",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=500, title_text="Drawdown Analysis", showlegend=False)

        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        return fig

    @staticmethod
    def create_stress_test_results(stress_results: Dict[str, Any]) -> go.Figure:
        """Create stress test results visualization."""
        stressed_returns = stress_results.get("stressed_returns", [])

        if not stressed_returns:
            return go.Figure().add_annotation(
                text="No stress test data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Convert to numpy array for analysis
        stressed_array = np.array(stressed_returns)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Return Distribution", "Stress Statistics"),
            specs=[[{"type": "histogram"}, {"type": "bar"}]],
        )

        # Return distribution histogram
        fig.add_trace(
            go.Histogram(
                x=stressed_array * 100,
                nbinsx=50,
                name="Stressed Returns",
                marker_color="red",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Statistics
        stats_data = {
            "Mean": np.mean(stressed_array) * 100,
            "Std": np.std(stressed_array) * 100,
            "Min": np.min(stressed_array) * 100,
            "Max": np.max(stressed_array) * 100,
            "5th %ile": np.percentile(stressed_array, 5) * 100,
            "95th %ile": np.percentile(stressed_array, 95) * 100,
        }

        fig.add_trace(
            go.Bar(
                x=list(stats_data.keys()),
                y=list(stats_data.values()),
                name="Statistics",
                marker_color=["green" if x > 0 else "red" for x in stats_data.values()],
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=400, title_text="Stress Test Results", showlegend=False
        )

        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Value (%)", row=1, col=2)

        return fig

    @staticmethod
    def create_position_sizing_analysis(sizing_results: Dict[str, Any]) -> go.Figure:
        """Create position sizing analysis chart."""
        # This would show optimal position sizes across different methods
        # For now, create a simple comparison chart

        methods = ["Kelly", "Risk Parity", "Vol Targeting", "Fixed Fractional"]
        # Sample values - in practice these would come from calculations
        sizes = [0.25, 0.20, 0.15, 0.10]  # Example position sizes

        fig = go.Figure(
            data=[
                go.Bar(
                    x=methods,
                    y=sizes,
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                )
            ]
        )

        fig.update_layout(
            title="Position Sizing Comparison",
            xaxis_title="Method",
            yaxis_title="Position Size",
            height=400,
        )

        return fig
