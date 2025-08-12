"""
Streamlit Page: Backtesting
Week 14 UI Integration - Professional Backtesting Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Week 11: Backtesting Framework Integration
    from src.backtesting import (
        BacktestEngine,
        BuyAndHoldStrategy,
        MomentumStrategy,
        MeanReversionStrategy,
        PerformanceCalculator,
        Portfolio,
        RiskModel,
    )
    from src.config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Backtesting Interface"""

    st.title("⚡ Backtesting")
    st.markdown("**Professional Strategy Backtesting Platform**")

    # Initialize session state
    if "backtest_cache" not in st.session_state:
        st.session_state.backtest_cache = {}

    # Check for available models
    if "model_cache" not in st.session_state or not st.session_state.model_cache:
        st.warning("🤖 Please train models first from Model Training page")
        return

    # Professional UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        backtest_control_panel()

    with col2:
        backtest_display_panel()


def backtest_control_panel():
    """Backtesting Control Panel"""

    st.subheader("🎯 Backtest Configuration")

    # Data and model selection
    available_models = list(st.session_state.model_cache.keys())
    selected_model = st.selectbox("Select Model", available_models)

    if not selected_model:
        return

    # Strategy selection
    st.subheader("📈 Strategy Configuration")

    strategy_type = st.selectbox(
        "Strategy Type",
        ["Buy & Hold", "Momentum", "Mean Reversion", "Model-Based"],
    )

    # Strategy parameters
    if strategy_type == "Momentum":
        short_window = st.slider("Short Window", 5, 30, 20, key="momentum_short")
        long_window = st.slider("Long Window", 20, 100, 50, key="momentum_long")
        strategy_params = {"short_window": short_window, "long_window": long_window}
    elif strategy_type == "Mean Reversion":
        window = st.slider("Window", 10, 50, 20, key="mr_window")
        num_std = st.slider("Num Std", 1.0, 3.0, 2.0, key="mr_std")
        strategy_params = {"window": window, "num_std": num_std}
    else:
        strategy_params = {}

    # Backtest parameters
    st.subheader("⚙️ Backtest Parameters")

    initial_capital = st.number_input(
        "Initial Capital ($)", value=100000, min_value=1000, step=1000
    )
    commission_rate = st.slider("Commission Rate (%)", 0.0, 1.0, 0.1, step=0.01) / 100
    slippage_rate = st.slider("Slippage Rate (%)", 0.0, 1.0, 0.05, step=0.01) / 100

    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        run_backtest(
            selected_model,
            strategy_type,
            strategy_params,
            initial_capital,
            commission_rate,
            slippage_rate,
        )


def backtest_display_panel():
    """Backtest Display and Analysis Panel"""

    if not st.session_state.backtest_cache:
        st.info("⚡ Configure and run backtest to see results")
        return

    # Backtest selection
    selected_backtest = st.selectbox(
        "Select Backtest", list(st.session_state.backtest_cache.keys())
    )

    if selected_backtest:
        display_backtest_overview(selected_backtest)
        display_backtest_performance(selected_backtest)
        display_backtest_analytics(selected_backtest)


def run_backtest(
    model_key: str,
    strategy_type: str,
    strategy_params: dict,
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
):
    """Run backtest using Week 11 BacktestEngine"""

    try:
        # Get model and data
        cached_model = st.session_state.model_cache[model_key]

        # Get feature data for backtesting
        feature_key = model_key.split("_")[-1]  # Extract feature key from model key
        if feature_key in st.session_state.feature_cache:
            cached_features = st.session_state.feature_cache[feature_key]
            data = cached_features["data"]
        else:
            st.error("Cannot find corresponding feature data for backtesting")
            return

        with st.spinner(f"Running {strategy_type} backtest..."):
            # Initialize backtesting engine using existing Week 11 module
            engine = BacktestEngine(
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            )

            # Add data to engine
            symbol = "ASSET"  # Generic symbol for backtesting
            engine.add_data(symbol, data)

            # Create strategy using existing Week 11 modules
            if strategy_type == "Buy & Hold":
                strategy = BuyAndHoldStrategy(symbols=[symbol])
            elif strategy_type == "Momentum":
                strategy = MomentumStrategy(
                    symbols=[symbol],
                    short_window=strategy_params["short_window"],
                    long_window=strategy_params["long_window"],
                )
            elif strategy_type == "Mean Reversion":
                strategy = MeanReversionStrategy(
                    symbols=[symbol],
                    window=strategy_params["window"],
                    num_std=strategy_params["num_std"],
                )
            else:  # Model-Based
                # Create a simple model-based strategy
                strategy = create_model_based_strategy(cached_model, symbol)

            # Set strategy and run backtest
            engine.set_strategy(strategy)
            results = engine.run_backtest()

            # Extract results
            if "error" not in results:
                portfolio_values = (
                    [value[1] for value in engine.portfolio_values]
                    if engine.portfolio_values
                    else [initial_capital]
                )

                trades_summary = engine.get_trades_summary()
                positions_summary = engine.get_positions_summary()
            else:
                portfolio_values = [initial_capital]
                trades_summary = []
                positions_summary = {}

            # Calculate performance metrics using existing Week 11 module
            calculator = PerformanceCalculator()
            returns = pd.Series(portfolio_values).pct_change().dropna()

            # Calculate comprehensive metrics
            metrics = calculator.calculate_comprehensive_metrics(
                returns=returns,
                portfolio_values=pd.Series(portfolio_values),
                trades=trades_summary,
                benchmark_returns=None,  # No benchmark for simplicity
                initial_capital=initial_capital,
            )

            # Store results
            backtest_key = (
                f"{strategy_type}_{model_key}_{datetime.now().strftime('%H%M%S')}"
            )
            st.session_state.backtest_cache[backtest_key] = {
                "strategy_type": strategy_type,
                "strategy_params": strategy_params,
                "model_key": model_key,
                "engine_config": {
                    "initial_capital": initial_capital,
                    "commission_rate": commission_rate,
                    "slippage_rate": slippage_rate,
                },
                "portfolio_values": portfolio_values,
                "returns": returns,
                "metrics": metrics,
                "trades": trades_summary,
                "positions": positions_summary,
                "data": data,
                "run_at": datetime.now(),
            }

        st.success(f"✅ Backtest completed: {strategy_type} strategy")
        st.rerun()

    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")


def create_model_based_strategy(cached_model, symbol):
    """Create a simple model-based strategy"""

    class ModelBasedStrategy:
        def __init__(self, model, symbol):
            self.model = model["model"]
            self.symbol = symbol
            self.engine = None

        def set_engine(self, engine):
            self.engine = engine

        def on_data(self, current_time):
            if self.engine is None:
                return

            # Simple strategy: buy if model predicts positive return
            try:
                # Get current data for prediction (simplified)
                current_price = self.engine.get_current_price(self.symbol)
                if current_price is None:
                    return

                # Make a simple prediction based on current price
                # This is a simplified example - in practice, you'd use proper features
                if hasattr(self.model, "predict"):
                    # For classification models, buy if prediction is 1 (positive)
                    if cached_model["task_type"] == "Classification":
                        # Simple feature: price change (mock)
                        mock_features = [[current_price / 100]]  # Normalized price
                        prediction = self.model.predict(mock_features)[0]

                        current_position = self.engine.positions.get(self.symbol)
                        position_size = (
                            current_position.quantity if current_position else 0
                        )

                        if prediction == 1 and position_size <= 0:
                            # Buy signal
                            self.engine.place_order(self.symbol, "BUY", 100)
                        elif prediction == 0 and position_size > 0:
                            # Sell signal
                            self.engine.place_order(
                                self.symbol, "SELL", abs(position_size)
                            )

            except Exception as e:
                # Ignore prediction errors
                pass

    return ModelBasedStrategy(cached_model, symbol)


def display_backtest_overview(backtest_key: str):
    """Display backtest overview with key metrics"""

    cached_backtest = st.session_state.backtest_cache[backtest_key]
    strategy_type = cached_backtest["strategy_type"]
    metrics = cached_backtest["metrics"]
    engine_config = cached_backtest["engine_config"]

    st.subheader(f"⚡ Backtest Overview: {backtest_key}")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = (
            cached_backtest["portfolio_values"][-1] / engine_config["initial_capital"]
            - 1
        ) * 100
        st.metric("Total Return", f"{total_return:.2f}%")

    with col2:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}")

    with col3:
        st.metric("Max Drawdown", f"{metrics.max_drawdown:.3f}")

    with col4:
        st.metric("Num Trades", len(cached_backtest["trades"]))

    # Strategy details
    st.subheader("📊 Strategy Details")

    details_col1, details_col2 = st.columns(2)

    with details_col1:
        st.write(f"**Strategy Type:** {strategy_type}")
        st.write(f"**Initial Capital:** ${engine_config['initial_capital']:,.2f}")
        st.write(f"**Final Value:** ${cached_backtest['portfolio_values'][-1]:,.2f}")

    with details_col2:
        st.write(f"**Commission Rate:** {engine_config['commission_rate']:.3%}")
        st.write(f"**Slippage Rate:** {engine_config['slippage_rate']:.3%}")
        run_time = cached_backtest["run_at"]
        st.write(f"**Run Time:** {run_time.strftime('%H:%M:%S')}")

    # Performance metrics table
    st.subheader("📈 Performance Metrics")

    metrics_data = {
        "Metric": [
            "Total Return",
            "Annualized Return",
            "Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Maximum Drawdown",
            "Calmar Ratio",
        ],
        "Value": [
            f"{total_return:.2f}%",
            f"{getattr(metrics, 'annualized_return', 0) * 100:.2f}%",
            f"{getattr(metrics, 'volatility', 0) * 100:.2f}%",
            f"{metrics.sharpe_ratio:.3f}",
            f"{metrics.sortino_ratio:.3f}",
            f"{metrics.max_drawdown:.3f}",
            f"{metrics.calmar_ratio:.3f}",
        ],
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def display_backtest_performance(backtest_key: str):
    """Display backtest performance charts"""

    cached_backtest = st.session_state.backtest_cache[backtest_key]
    portfolio_values = cached_backtest["portfolio_values"]
    returns = cached_backtest["returns"]
    data = cached_backtest["data"]

    st.subheader("📈 Performance Analysis")

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Portfolio Value Over Time",
            "Returns Distribution",
            "Drawdown Analysis",
        ),
        row_heights=[0.5, 0.25, 0.25],
    )

    # Portfolio value chart
    dates = (
        data.index[: len(portfolio_values)]
        if len(data.index) >= len(portfolio_values)
        else data.index
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values[: len(dates)],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Returns histogram
    fig.add_trace(
        go.Histogram(
            x=returns,
            name="Returns Distribution",
            nbinsx=50,
            opacity=0.7,
            marker_color="green",
        ),
        row=2,
        col=1,
    )

    # Drawdown chart
    portfolio_series = pd.Series(portfolio_values[: len(dates)], index=dates)
    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - rolling_max) / rolling_max

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color="red", width=1),
            fill="tozeroy",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=f"Backtest Performance: {backtest_key}",
        height=800,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def display_backtest_analytics(backtest_key: str):
    """Display detailed backtest analytics"""

    cached_backtest = st.session_state.backtest_cache[backtest_key]
    trades = cached_backtest["trades"]
    positions = cached_backtest["positions"]

    st.subheader("🔍 Trade Analysis")

    if trades:
        # Trades summary
        trades_df = pd.DataFrame(trades)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Recent Trades:**")
            st.dataframe(trades_df.tail(10), use_container_width=True)

        with col2:
            st.write("**Trade Statistics:**")

            # Calculate trade statistics
            if len(trades_df) > 0:
                buy_trades = trades_df[trades_df["side"] == "BUY"]
                sell_trades = trades_df[trades_df["side"] == "SELL"]

                stats_data = {
                    "Statistic": [
                        "Total Trades",
                        "Buy Trades",
                        "Sell Trades",
                        "Avg Trade Size",
                        "Total Volume",
                    ],
                    "Value": [
                        len(trades_df),
                        len(buy_trades),
                        len(sell_trades),
                        f"{trades_df['quantity'].mean():.2f}",
                        f"{trades_df['quantity'].sum():.2f}",
                    ],
                }

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed in this backtest")

    # Positions summary
    st.subheader("💼 Position Analysis")

    if positions:
        positions_data = []
        for symbol, position in positions.items():
            positions_data.append(
                {
                    "Symbol": symbol,
                    "Quantity": getattr(position, "quantity", 0),
                    "Avg Price": getattr(position, "avg_price", 0),
                    "Current Price": getattr(position, "current_price", 0),
                    "Market Value": getattr(position, "market_value", 0),
                    "P&L": getattr(position, "unrealized_pnl", 0),
                }
            )

        if positions_data:
            positions_df = pd.DataFrame(positions_data)
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
    else:
        st.info("No positions held at end of backtest")

    # Download option
    st.subheader("📥 Export Results")

    # Create export data
    export_data = {
        "backtest_config": {
            "strategy_type": cached_backtest["strategy_type"],
            "strategy_params": cached_backtest["strategy_params"],
            "engine_config": cached_backtest["engine_config"],
        },
        "portfolio_values": cached_backtest["portfolio_values"],
        "returns": cached_backtest["returns"].tolist(),
        "trades": trades,
        "positions": {k: str(v) for k, v in positions.items()},
    }

    import json

    export_json = json.dumps(export_data, indent=2, default=str)

    st.download_button(
        label="📥 Download Backtest Results JSON",
        data=export_json,
        file_name=f"backtest_{backtest_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
