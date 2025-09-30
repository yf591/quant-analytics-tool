"""
Backtest UI Widgets for Streamlit Application

This module provides specialized UI widgets for backtesting interface
including strategy configuration, backtest management, and results visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class StrategyConfigWidget:
    """Widget for strategy configuration and parameter setting"""

    def __init__(self):
        self.strategy_types = {
            "Buy & Hold": {
                "description": "Simple buy and hold strategy",
                "parameters": {},
            },
            "Momentum": {
                "description": "Trend following momentum strategy",
                "parameters": {
                    "short_window": {
                        "type": "slider",
                        "min": 5,
                        "max": 50,
                        "default": 10,
                    },
                    "long_window": {
                        "type": "slider",
                        "min": 20,
                        "max": 200,
                        "default": 50,
                    },
                    "signal_threshold": {
                        "type": "slider",
                        "min": 0.0,
                        "max": 0.1,
                        "default": 0.02,
                    },
                },
            },
            "Mean Reversion": {
                "description": "Mean reversion strategy using Bollinger Bands",
                "parameters": {
                    "window": {"type": "slider", "min": 10, "max": 100, "default": 20},
                    "num_std": {
                        "type": "slider",
                        "min": 1.0,
                        "max": 4.0,
                        "default": 2.0,
                    },
                    "entry_threshold": {
                        "type": "slider",
                        "min": 0.5,
                        "max": 2.0,
                        "default": 1.0,
                    },
                },
            },
            "Model-Based": {
                "description": "ML model-based trading strategy",
                "parameters": {
                    "confidence_threshold": {
                        "type": "slider",
                        "min": 0.5,
                        "max": 0.9,
                        "default": 0.7,
                    },
                    "rebalance_frequency": {
                        "type": "selectbox",
                        "options": ["Daily", "Weekly", "Monthly"],
                        "default": "Daily",
                    },
                    "position_sizing": {
                        "type": "selectbox",
                        "options": ["Fixed", "Kelly", "Risk Parity"],
                        "default": "Fixed",
                    },
                },
            },
            "Multi-Asset": {
                "description": "Multi-asset portfolio strategy",
                "parameters": {
                    "rebalance_frequency": {
                        "type": "selectbox",
                        "options": ["Daily", "Weekly", "Monthly"],
                        "default": "Monthly",
                    },
                    "risk_model": {
                        "type": "selectbox",
                        "options": ["Equal Weight", "Risk Parity", "Min Variance"],
                        "default": "Equal Weight",
                    },
                    "max_position": {
                        "type": "slider",
                        "min": 0.05,
                        "max": 0.5,
                        "default": 0.2,
                    },
                },
            },
        }

    def render_strategy_selection(self, key_prefix: str = "") -> Dict[str, Any]:
        """Render strategy selection interface"""

        st.subheader("📈 Strategy Configuration")

        # Strategy type selection
        strategy_type = st.selectbox(
            "Select Strategy Type",
            list(self.strategy_types.keys()),
            key=f"{key_prefix}_strategy_type",
            help="Choose the trading strategy type",
        )

        # Strategy description
        strategy_info = self.strategy_types[strategy_type]
        st.info(f"**Description:** {strategy_info['description']}")

        # Strategy parameters
        parameters = {}
        if strategy_info["parameters"]:
            st.write("**Strategy Parameters:**")

            for param_name, param_config in strategy_info["parameters"].items():
                param_key = f"{key_prefix}_{strategy_type}_{param_name}"

                if param_config["type"] == "slider":
                    if isinstance(param_config["default"], float):
                        step = (param_config["max"] - param_config["min"]) / 100
                        parameters[param_name] = st.slider(
                            param_name.replace("_", " ").title(),
                            param_config["min"],
                            param_config["max"],
                            param_config["default"],
                            step=step,
                            key=param_key,
                        )
                    else:
                        parameters[param_name] = st.slider(
                            param_name.replace("_", " ").title(),
                            param_config["min"],
                            param_config["max"],
                            param_config["default"],
                            key=param_key,
                        )
                elif param_config["type"] == "selectbox":
                    default_index = param_config["options"].index(
                        param_config["default"]
                    )
                    parameters[param_name] = st.selectbox(
                        param_name.replace("_", " ").title(),
                        param_config["options"],
                        index=default_index,
                        key=param_key,
                    )

        return {"strategy_type": strategy_type, "parameters": parameters}


class BacktestConfigWidget:
    """Widget for backtest configuration"""

    def __init__(self):
        pass

    def render_backtest_config(self, key_prefix: str = "") -> Dict[str, Any]:
        """Render backtest configuration interface"""

        st.subheader("⚙️ Backtest Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Portfolio Settings:**")
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000,
                key=f"{key_prefix}_initial_capital",
            )

            max_position_size = (
                st.slider(
                    "Max Position Size (%)",
                    1,
                    50,
                    20,
                    help="Maximum position size as percentage of portfolio",
                    key=f"{key_prefix}_max_position",
                )
                / 100
            )

            leverage = st.slider(
                "Leverage",
                1.0,
                5.0,
                1.0,
                0.1,
                help="Maximum leverage allowed",
                key=f"{key_prefix}_leverage",
            )

        with col2:
            st.write("**Transaction Costs:**")
            commission_rate = (
                st.slider(
                    "Commission Rate (%)",
                    0.0,
                    1.0,
                    0.1,
                    0.01,
                    help="Commission rate per trade",
                    key=f"{key_prefix}_commission",
                )
                / 100
            )

            slippage_rate = (
                st.slider(
                    "Slippage Rate (%)",
                    0.0,
                    1.0,
                    0.05,
                    0.01,
                    help="Market impact and slippage",
                    key=f"{key_prefix}_slippage",
                )
                / 100
            )

            min_commission = st.number_input(
                "Minimum Commission ($)",
                0.0,
                10.0,
                1.0,
                0.1,
                key=f"{key_prefix}_min_commission",
            )

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            # Date range
            col_date1, col_date2 = st.columns(2)

            with col_date1:
                start_date = st.date_input(
                    "Backtest Start Date",
                    value=datetime.now() - timedelta(days=365),
                    key=f"{key_prefix}_start_date",
                )

            with col_date2:
                end_date = st.date_input(
                    "Backtest End Date",
                    value=datetime.now(),
                    key=f"{key_prefix}_end_date",
                )

            # Risk management
            st.write("**Risk Management:**")
            stop_loss = (
                st.slider(
                    "Stop Loss (%)",
                    0.0,
                    20.0,
                    0.0,
                    0.5,
                    help="Stop loss percentage (0 = disabled)",
                    key=f"{key_prefix}_stop_loss",
                )
                / 100
            )

            take_profit = (
                st.slider(
                    "Take Profit (%)",
                    0.0,
                    50.0,
                    0.0,
                    0.5,
                    help="Take profit percentage (0 = disabled)",
                    key=f"{key_prefix}_take_profit",
                )
                / 100
            )

            # Benchmark
            benchmark_options = ["None", "SPY", "QQQ", "IWM", "Custom"]
            benchmark = st.selectbox(
                "Benchmark", benchmark_options, key=f"{key_prefix}_benchmark"
            )

        return {
            "initial_capital": initial_capital,
            "max_position_size": max_position_size,
            "leverage": leverage,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "min_commission": min_commission,
            "start_date": start_date,
            "end_date": end_date,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "benchmark": benchmark if benchmark != "None" else None,
        }


class BacktestResultsWidget:
    """Widget for displaying backtest results"""

    def __init__(self):
        pass

    def render_results_overview(self, results: Dict[str, Any]) -> None:
        """Render backtest results overview"""

        st.subheader("📊 Backtest Results Overview")

        # Key metrics
        metrics = results.get("metrics")
        if metrics:
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                total_return = getattr(metrics, "total_return", 0) * 100
                st.metric(
                    "Total Return",
                    f"{total_return:.2f}%",
                    delta=f"{total_return:.2f}%" if total_return > 0 else None,
                )

            with col2:
                sharpe_ratio = getattr(metrics, "sharpe_ratio", 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")

            with col3:
                max_drawdown = getattr(metrics, "max_drawdown", 0) * 100
                st.metric(
                    "Max Drawdown",
                    f"{max_drawdown:.2f}%",
                    delta=f"{max_drawdown:.2f}%" if max_drawdown < 0 else None,
                )

            with col4:
                win_rate = results.get("win_rate", 0) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")

            with col5:
                num_trades = len(results.get("trades", []))
                st.metric("Total Trades", num_trades)

    def render_performance_chart(self, results: Dict[str, Any]) -> None:
        """Render performance chart"""

        st.subheader("📈 Portfolio Performance")

        portfolio_values = results.get("portfolio_values", [])
        returns = results.get("returns", pd.Series())

        if len(portfolio_values) > 0:
            # Create performance chart
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Portfolio Value", "Daily Returns"),
                row_heights=[0.7, 0.3],
            )

            # Portfolio value line
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=len(portfolio_values)),
                periods=len(portfolio_values),
                freq="D",
            )

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="blue", width=2),
                    hovertemplate="<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Returns bar chart
            if len(returns) > 0:
                fig.add_trace(
                    go.Bar(
                        x=dates[-len(returns) :],
                        y=returns * 100,
                        name="Daily Returns (%)",
                        marker_color=np.where(returns >= 0, "green", "red"),
                        opacity=0.6,
                        hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

            fig.update_layout(
                title="Backtest Performance",
                height=600,
                template="plotly_white",
                hovermode="x unified",
            )

            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Returns (%)", row=2, col=1)

            st.plotly_chart(
                fig, use_container_width=True, key="backtest_performance_chart"
            )

    def render_trades_analysis(self, results: Dict[str, Any]) -> None:
        """Render trades analysis"""

        st.subheader("🔍 Trade Analysis")

        trades = results.get("trades", [])

        if trades:
            trades_df = pd.DataFrame(trades)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Trade Summary:**")

                # Calculate trade statistics
                if "pnl" in trades_df.columns:
                    winning_trades = trades_df[trades_df["pnl"] > 0]
                    losing_trades = trades_df[trades_df["pnl"] <= 0]

                    trade_stats = {
                        "Total Trades": len(trades_df),
                        "Winning Trades": len(winning_trades),
                        "Losing Trades": len(losing_trades),
                        "Win Rate": (
                            f"{(len(winning_trades) / len(trades_df) * 100):.1f}%"
                            if len(trades_df) > 0
                            else "0%"
                        ),
                        "Avg Win": (
                            f"${winning_trades['pnl'].mean():.2f}"
                            if len(winning_trades) > 0
                            else "$0.00"
                        ),
                        "Avg Loss": (
                            f"${losing_trades['pnl'].mean():.2f}"
                            if len(losing_trades) > 0
                            else "$0.00"
                        ),
                        "Profit Factor": (
                            f"{abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()):.2f}"
                            if len(losing_trades) > 0
                            and losing_trades["pnl"].sum() != 0
                            else "N/A"
                        ),
                    }

                    for stat, value in trade_stats.items():
                        st.write(f"**{stat}:** {value}")

            with col2:
                st.write("**Recent Trades:**")
                display_trades = (
                    trades_df.tail(10) if len(trades_df) > 10 else trades_df
                )

                # Convert complex objects to strings to avoid Arrow serialization issues
                display_trades_clean = display_trades.copy()
                for col in display_trades_clean.columns:
                    if display_trades_clean[col].dtype == "object":
                        display_trades_clean[col] = display_trades_clean[col].astype(
                            str
                        )

                st.dataframe(display_trades_clean, use_container_width=True)

            # Trade distribution
            if "pnl" in trades_df.columns:
                st.write("**Trade P&L Distribution:**")

                fig = go.Figure(
                    data=[
                        go.Histogram(
                            x=trades_df["pnl"],
                            nbinsx=30,
                            marker_color="lightblue",
                            opacity=0.7,
                        )
                    ]
                )

                fig.update_layout(
                    title="Trade P&L Distribution",
                    xaxis_title="P&L ($)",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=400,
                )

                st.plotly_chart(
                    fig, use_container_width=True, key="trades_distribution_chart"
                )
        else:
            st.info("No trades executed in this backtest")

    def render_risk_metrics(self, results: Dict[str, Any]) -> None:
        """Render risk analysis metrics"""

        st.subheader("⚠️ Risk Analysis")

        metrics = results.get("metrics")
        returns = results.get("returns", pd.Series())

        if metrics and len(returns) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Return Metrics:**")
                st.write(
                    f"**Annual Return:** {getattr(metrics, 'annualized_return', 0) * 100:.2f}%"
                )
                st.write(
                    f"**Annual Volatility:** {getattr(metrics, 'volatility', 0) * 100:.2f}%"
                )
                st.write(f"**Skewness:** {returns.skew():.3f}")
                st.write(f"**Kurtosis:** {returns.kurtosis():.3f}")

            with col2:
                st.write("**Risk Metrics:**")
                st.write(f"**Sharpe Ratio:** {getattr(metrics, 'sharpe_ratio', 0):.3f}")
                st.write(
                    f"**Sortino Ratio:** {getattr(metrics, 'sortino_ratio', 0):.3f}"
                )
                st.write(f"**Calmar Ratio:** {getattr(metrics, 'calmar_ratio', 0):.3f}")
                st.write(
                    f"**Max Drawdown:** {getattr(metrics, 'max_drawdown', 0) * 100:.2f}%"
                )

            with col3:
                st.write("**Risk Measures:**")
                var_95 = np.percentile(returns, 5) * 100
                var_99 = np.percentile(returns, 1) * 100
                st.write(f"**VaR (95%):** {var_95:.2f}%")
                st.write(f"**VaR (99%):** {var_99:.2f}%")

                # Conditional VaR (Expected Shortfall)
                cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
                st.write(f"**CVaR (95%):** {cvar_95:.2f}%")

    def render_drawdown_analysis(self, results: Dict[str, Any]) -> None:
        """Render drawdown analysis"""

        st.subheader("📉 Drawdown Analysis")

        portfolio_values = results.get("portfolio_values", [])

        if len(portfolio_values) > 1:
            # Calculate drawdown
            portfolio_series = pd.Series(portfolio_values)
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max

            # Create drawdown chart
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=len(portfolio_values)),
                periods=len(portfolio_values),
                freq="D",
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=drawdown * 100,
                    mode="lines",
                    name="Drawdown (%)",
                    line=dict(color="red", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
                )
            )

            fig.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_white",
                height=400,
            )

            st.plotly_chart(
                fig, use_container_width=True, key="drawdown_analysis_chart"
            )

            # Drawdown statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                max_dd = drawdown.min() * 100
                st.metric("Maximum Drawdown", f"{max_dd:.2f}%")

            with col2:
                # Calculate average drawdown duration
                dd_periods = (drawdown < 0).astype(int)
                if dd_periods.sum() > 0:
                    avg_duration = (
                        dd_periods.sum() / (dd_periods.diff() == 1).sum()
                        if (dd_periods.diff() == 1).sum() > 0
                        else 0
                    )
                    st.metric("Avg DD Duration", f"{avg_duration:.0f} days")
                else:
                    st.metric("Avg DD Duration", "0 days")

            with col3:
                current_dd = drawdown.iloc[-1] * 100
                st.metric("Current Drawdown", f"{current_dd:.2f}%")


class BacktestComparisonWidget:
    """Widget for comparing multiple backtest results"""

    def __init__(self):
        pass

    def render_comparison_table(self, backtest_results: Dict[str, Dict]) -> None:
        """Render comparison table for multiple backtests"""

        st.subheader("🔀 Backtest Comparison")

        if len(backtest_results) < 2:
            st.info("Run multiple backtests to enable comparison")
            return

        # Create comparison data
        comparison_data = []

        for backtest_name, results in backtest_results.items():
            metrics = results.get("metrics")
            trades = results.get("trades", [])

            if metrics:
                row = {
                    "Backtest": backtest_name,
                    "Total Return (%)": f"{getattr(metrics, 'total_return', 0) * 100:.2f}",
                    "Annual Return (%)": f"{getattr(metrics, 'annualized_return', 0) * 100:.2f}",
                    "Volatility (%)": f"{getattr(metrics, 'volatility', 0) * 100:.2f}",
                    "Sharpe Ratio": f"{getattr(metrics, 'sharpe_ratio', 0):.3f}",
                    "Max Drawdown (%)": f"{getattr(metrics, 'max_drawdown', 0) * 100:.2f}",
                    "Total Trades": len(trades),
                    "Win Rate (%)": f"{results.get('win_rate', 0) * 100:.1f}",
                }
                comparison_data.append(row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Performance comparison chart
            self._render_comparison_chart(backtest_results)

    def _render_comparison_chart(self, backtest_results: Dict[str, Dict]) -> None:
        """Render comparison chart"""

        fig = go.Figure()

        for backtest_name, results in backtest_results.items():
            portfolio_values = results.get("portfolio_values", [])

            if len(portfolio_values) > 0:
                # Normalize to start at 100
                normalized_values = [
                    (v / portfolio_values[0]) * 100 for v in portfolio_values
                ]

                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=len(portfolio_values)),
                    periods=len(portfolio_values),
                    freq="D",
                )

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=normalized_values,
                        mode="lines",
                        name=backtest_name,
                        line=dict(width=2),
                        hovertemplate=f"<b>{backtest_name}</b><br>%{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
                    )
                )

                fig.update_layout(
                    title="Normalized Performance Comparison (Base = 100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Value",
                    template="plotly_white",
                    height=500,
                )

        st.plotly_chart(fig, use_container_width=True, key="backtest_comparison_chart")
