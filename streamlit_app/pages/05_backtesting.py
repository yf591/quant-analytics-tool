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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import warnings
import traceback
import json

warnings.filterwarnings("ignore")

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

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
        Order,
        OrderSide,
        OrderType,
    )
    from src.config import settings

    # Streamlit components
    from components.backtest_widgets import (
        StrategyConfigWidget,
        BacktestConfigWidget,
        BacktestResultsWidget,
        BacktestComparisonWidget,
    )
    from components.charts import (
        create_price_chart,
        create_feature_importance_chart,
    )
    from components.data_display import (
        display_computation_status,
        display_alert_message,
    )
    from utils.backtest_utils import (
        BacktestDataPreparer,
        StrategyBuilder,
        BacktestResultProcessor,
        get_available_symbols_from_cache,
        validate_backtest_config,
    )

except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are properly installed.")
    st.stop()


def main():
    """Professional Backtesting Interface"""

    st.set_page_config(
        page_title="⚡ Backtesting",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("⚡ Professional Backtesting Platform")
    st.markdown(
        "### Comprehensive strategy backtesting with advanced analytics and risk management"
    )

    # Initialize session state
    initialize_session_state()

    # Check for available data
    data_available = check_data_availability()

    if not data_available:
        display_data_requirements()
        return

    # Professional workflow tabs
    tabs = st.tabs(
        [
            "🎯 Strategy Backtest",
            "📊 Results Analysis",
            "🔀 Comparison",
            "📈 Advanced Analytics",
            "💾 Management",
        ]
    )

    with tabs[0]:
        strategy_backtest_workflow()

    with tabs[1]:
        results_analysis_workflow()

    with tabs[2]:
        comparison_workflow()

    with tabs[3]:
        advanced_analytics_workflow()

    with tabs[4]:
        backtest_management_workflow()


def initialize_session_state():
    """Initialize session state for backtesting"""

    if "backtest_cache" not in st.session_state:
        st.session_state.backtest_cache = {}

    if "backtest_running" not in st.session_state:
        st.session_state.backtest_running = False

    if "backtest_progress" not in st.session_state:
        st.session_state.backtest_progress = {}


def check_data_availability() -> bool:
    """Check if required data is available for backtesting"""

    # Check for feature data
    feature_data_available = (
        "feature_cache" in st.session_state and st.session_state.feature_cache
    )

    # Check for model data (optional for some strategies)
    model_data_available = (
        "model_cache" in st.session_state and st.session_state.model_cache
    )

    # Check for data acquisition data
    raw_data_available = (
        "data_cache" in st.session_state and st.session_state.data_cache
    )

    return feature_data_available or raw_data_available


def display_data_requirements():
    """Display data requirements for backtesting"""

    st.warning("📊 **Data Required for Backtesting**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📥 Data Acquisition")
        if "data_cache" in st.session_state and st.session_state.data_cache:
            st.success("✅ Raw data available")
        else:
            st.error("❌ No raw data found")

        if st.button("Go to Data Acquisition", key="goto_data_acq"):
            st.switch_page("pages/01_data_acquisition.py")

    with col2:
        st.subheader("🔧 Feature Engineering")
        if "feature_cache" in st.session_state and st.session_state.feature_cache:
            st.success("✅ Features available")
        else:
            st.error("❌ No features found")

        if st.button("Go to Feature Engineering", key="goto_features"):
            st.switch_page("pages/02_feature_engineering.py")

    with col3:
        st.subheader("🤖 Model Training")
        if "model_cache" in st.session_state and st.session_state.model_cache:
            st.success("✅ Models available")
        else:
            st.info("ℹ️ Models optional for basic strategies")

        if st.button("Go to Model Training", key="goto_models"):
            st.switch_page("pages/03_model_training.py")

    st.info(
        "💡 **Tip:** You need at least Feature Engineering data or Raw Data to run backtests. Models are required for model-based strategies."
    )


def strategy_backtest_workflow():
    """Main strategy backtesting workflow"""

    st.header("🎯 Strategy Backtesting Workflow")

    # Create two-column layout
    col_config, col_results = st.columns([1, 1.5])

    with col_config:
        st.subheader("⚙️ Configuration")

        # Data source selection
        data_source = select_data_source()
        if not data_source:
            return

        # Strategy configuration
        strategy_widget = StrategyConfigWidget()
        strategy_config = strategy_widget.render_strategy_selection("main")

        # Backtest configuration
        config_widget = BacktestConfigWidget()
        backtest_config = config_widget.render_backtest_config("main")

        # Validation
        is_valid, errors = validate_backtest_config(backtest_config)

        if errors:
            for error in errors:
                st.error(error)

        # Run backtest button
        if st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True,
            disabled=not is_valid,
        ):
            run_comprehensive_backtest(data_source, strategy_config, backtest_config)

    with col_results:
        st.subheader("📊 Live Results")

        if st.session_state.backtest_running:
            display_backtest_progress()
        elif st.session_state.backtest_cache:
            # Show latest backtest result
            latest_backtest = max(
                st.session_state.backtest_cache.keys(),
                key=lambda x: st.session_state.backtest_cache[x].get(
                    "timestamp", datetime.min
                ),
            )
            display_quick_results(latest_backtest)
        else:
            st.info("Configure and run a backtest to see results here")


def select_data_source() -> Optional[str]:
    """Select data source for backtesting"""

    st.subheader("📊 Data Source Selection")

    available_sources = []

    # Check feature data
    if "feature_cache" in st.session_state and st.session_state.feature_cache:
        for feature_key in st.session_state.feature_cache.keys():
            available_sources.append(f"Features: {feature_key}")

    # Check raw data
    if "data_cache" in st.session_state and st.session_state.data_cache:
        for data_key in st.session_state.data_cache.keys():
            available_sources.append(f"Raw Data: {data_key}")

    if not available_sources:
        st.error("No data sources available")
        return None

    selected_source = st.selectbox(
        "Select Data Source",
        available_sources,
        help="Choose the data source for backtesting",
        key="data_source_select",
    )

    return selected_source


def run_comprehensive_backtest(
    data_source: str, strategy_config: Dict, backtest_config: Dict
):
    """Run comprehensive backtest with professional features"""

    try:
        # Update running status
        st.session_state.backtest_running = True

        # Create progress tracking
        progress_container = st.container()

        with progress_container:
            # Initialize data preparer and strategy builder
            data_preparer = BacktestDataPreparer()
            strategy_builder = StrategyBuilder()
            result_processor = BacktestResultProcessor()

            # Step 1: Prepare data
            display_computation_status("📊 Preparing data...", 0.1)
            time.sleep(0.5)

            # Get data based on source
            if data_source.startswith("Features:"):
                feature_key = data_source.replace("Features: ", "")
                feature_data = st.session_state.feature_cache[feature_key]
                data = data_preparer.prepare_feature_data(feature_data)
                data_key = feature_key
            else:  # Raw data
                data_key = data_source.replace("Raw Data: ", "")
                raw_data = st.session_state.data_cache[data_key]["data"]
                data = data_preparer._validate_price_data(raw_data)

            symbols = [data_key]

            # Step 2: Initialize backtest engine
            display_computation_status("⚙️ Initializing backtest engine...", 0.2)
            time.sleep(0.5)

            engine = BacktestEngine(
                initial_capital=backtest_config["initial_capital"],
                commission_rate=backtest_config["commission_rate"],
                slippage_rate=backtest_config["slippage_rate"],
                min_commission=backtest_config["min_commission"],
                max_position_size=backtest_config["max_position_size"],
            )

            # Add data to engine
            symbol = symbols[0] if symbols else "ASSET"
            engine.add_data(symbol, data)

            # Step 3: Build strategy
            display_computation_status("🎯 Building strategy...", 0.3)
            time.sleep(0.5)

            if strategy_config["strategy_type"] == "Model-Based":
                # Check if models are available
                if (
                    "model_cache" not in st.session_state
                    or not st.session_state.model_cache
                ):
                    st.error("No models available for model-based strategy")
                    return

                # Use first available model for demo
                model_key = list(st.session_state.model_cache.keys())[0]
                model_info = st.session_state.model_cache[model_key]
                strategy = create_model_strategy(
                    model_info, symbol, strategy_config["parameters"]
                )
            else:
                strategy = strategy_builder.build_strategy(strategy_config, [symbol])

            # Step 4: Run backtest
            display_computation_status("🚀 Running backtest...", 0.5)
            time.sleep(1.0)

            engine.set_strategy(strategy)

            # Set date range if specified
            start_date = backtest_config.get("start_date")
            end_date = backtest_config.get("end_date")

            raw_results = engine.run_backtest(start_date=start_date, end_date=end_date)

            # Step 5: Process results
            display_computation_status("📊 Processing results...", 0.8)
            time.sleep(0.5)

            results = result_processor.process_results(engine, backtest_config)

            # Step 6: Store results
            display_computation_status("💾 Storing results...", 0.9)
            time.sleep(0.3)

            backtest_id = f"{strategy_config['strategy_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            st.session_state.backtest_cache[backtest_id] = {
                **results,
                "strategy_config": strategy_config,
                "backtest_config": backtest_config,
                "data_source": data_source,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "id": backtest_id,
            }

            display_computation_status("✅ Backtest completed successfully!", 1.0)

            st.success(f"🎉 Backtest '{backtest_id}' completed successfully!")
            st.balloons()

            # Display quick summary
            display_quick_results(backtest_id)

    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        st.error(traceback.format_exc())

    finally:
        # Reset running status
        st.session_state.backtest_running = False


def create_model_strategy(model_info: Dict, symbol: str, parameters: Dict):
    """Create model-based strategy instance"""

    class ModelStrategy:
        def __init__(self, model_info, symbol, parameters):
            self.model = model_info["model"]
            self.symbol = symbol
            self.confidence_threshold = parameters.get("confidence_threshold", 0.7)
            self.position_sizing = parameters.get("position_sizing", "Fixed")
            self.engine = None

        def set_backtest_engine(self, engine):
            self.engine = engine

        def on_start(self):
            pass

        def on_data(self, current_time: datetime) -> List:
            signals = []

            try:
                # Simplified prediction logic
                current_price = self.engine.get_current_price(self.symbol)
                if current_price is None:
                    return signals

                # Mock prediction based on price trend
                historical_data = self.engine.data.get(self.symbol)
                if historical_data is not None and len(historical_data) > 5:
                    recent_prices = historical_data["Close"].tail(5)
                    price_trend = (
                        recent_prices.iloc[-1] - recent_prices.iloc[0]
                    ) / recent_prices.iloc[0]

                    # Simple signal generation
                    if price_trend > 0.01:  # 1% uptrend
                        confidence = min(abs(price_trend) * 10, 1.0)
                        if confidence > self.confidence_threshold:
                            position_size = self._calculate_position_size(confidence)

                            signals.append(
                                {
                                    "symbol": self.symbol,
                                    "side": OrderSide.BUY,
                                    "quantity": position_size,
                                    "type": OrderType.MARKET,
                                }
                            )

                    elif price_trend < -0.01:  # 1% downtrend
                        # Sell existing positions
                        position = self.engine.positions.get(self.symbol)
                        if position and position.quantity > 0:
                            signals.append(
                                {
                                    "symbol": self.symbol,
                                    "side": OrderSide.SELL,
                                    "quantity": position.quantity,
                                    "type": OrderType.MARKET,
                                }
                            )

            except Exception:
                pass

            return signals

        def _calculate_position_size(self, confidence: float) -> int:
            portfolio_value = self.engine.get_portfolio_value()
            current_price = self.engine.get_current_price(self.symbol)

            if self.position_sizing == "Fixed":
                position_value = portfolio_value * 0.1
            else:
                position_value = portfolio_value * confidence * 0.2

            return (
                max(1, int(position_value / current_price)) if current_price > 0 else 1
            )

        def on_finish(self):
            pass

    return ModelStrategy(model_info, symbol, parameters)


def display_backtest_progress():
    """Display backtest progress"""

    st.subheader("🔄 Backtest in Progress")

    # Show progress bar
    progress_bar = st.progress(0.5)
    st.write("Running backtest... Please wait.")

    # Show status
    status_placeholder = st.empty()
    status_placeholder.info("⚙️ Processing backtest...")


def display_quick_results(backtest_id: str):
    """Display quick backtest results"""

    if backtest_id not in st.session_state.backtest_cache:
        return

    result = st.session_state.backtest_cache[backtest_id]

    st.subheader(f"📊 Quick Results: {backtest_id}")

    # Key metrics
    metrics = result.get("metrics")
    if metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = getattr(metrics, "total_return", 0) * 100
            st.metric("Total Return", f"{total_return:.2f}%")

        with col2:
            sharpe_ratio = getattr(metrics, "sharpe_ratio", 0)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")

        with col3:
            max_drawdown = getattr(metrics, "max_drawdown", 0) * 100
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

        with col4:
            num_trades = len(result.get("trades", []))
            st.metric("Total Trades", num_trades)

    # Quick chart
    portfolio_values = result.get("portfolio_values", [])
    if len(portfolio_values) > 1:
        st.subheader("📈 Portfolio Performance")

        fig = go.Figure()

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
            )
        )

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True)


def results_analysis_workflow():
    """Results analysis workflow"""

    st.header("📊 Comprehensive Results Analysis")

    if not st.session_state.backtest_cache:
        st.info("No backtest results available. Run a backtest first.")
        return

    # Backtest selection
    backtest_ids = list(st.session_state.backtest_cache.keys())
    selected_backtest = st.selectbox(
        "Select Backtest for Analysis", backtest_ids, key="analysis_backtest_select"
    )

    if selected_backtest:
        result = st.session_state.backtest_cache[selected_backtest]

        # Create results widget and display comprehensive analysis
        results_widget = BacktestResultsWidget()

        # Results overview
        results_widget.render_results_overview(result)

        # Performance chart
        results_widget.render_performance_chart(result)

        # Trade analysis
        results_widget.render_trades_analysis(result)

        # Risk metrics
        results_widget.render_risk_metrics(result)

        # Drawdown analysis
        results_widget.render_drawdown_analysis(result)


def comparison_workflow():
    """Backtest comparison workflow"""

    st.header("🔀 Backtest Comparison")

    if len(st.session_state.backtest_cache) < 2:
        st.info("Run at least 2 backtests to enable comparison.")
        return

    # Create comparison widget
    comparison_widget = BacktestComparisonWidget()

    # Render comparison
    comparison_widget.render_comparison_table(st.session_state.backtest_cache)


def advanced_analytics_workflow():
    """Advanced analytics workflow"""

    st.header("📈 Advanced Analytics")

    if not st.session_state.backtest_cache:
        st.info("No backtest results available for advanced analytics.")
        return

    # Advanced analytics options
    analytics_type = st.selectbox(
        "Select Analytics Type",
        [
            "Monte Carlo Simulation",
            "Walk-Forward Analysis",
            "Sensitivity Analysis",
            "Risk Attribution",
        ],
    )

    if analytics_type == "Monte Carlo Simulation":
        monte_carlo_analysis()
    elif analytics_type == "Walk-Forward Analysis":
        walk_forward_analysis()
    elif analytics_type == "Sensitivity Analysis":
        sensitivity_analysis()
    elif analytics_type == "Risk Attribution":
        risk_attribution_analysis()


def monte_carlo_analysis():
    """Monte Carlo simulation analysis"""

    st.subheader("🎲 Monte Carlo Simulation")

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)

    with col2:
        time_horizon = st.slider("Time Horizon (days)", 30, 365, 252)

    if st.button("🚀 Run Monte Carlo Simulation"):
        run_monte_carlo_simulation(num_simulations, confidence_level, time_horizon)


def run_monte_carlo_simulation(
    num_simulations: int, confidence_level: float, time_horizon: int
):
    """Run Monte Carlo simulation"""

    # Get latest backtest for simulation
    if not st.session_state.backtest_cache:
        st.error("No backtest results available")
        return

    latest_backtest = max(
        st.session_state.backtest_cache.keys(),
        key=lambda x: st.session_state.backtest_cache[x].get("timestamp", datetime.min),
    )
    result = st.session_state.backtest_cache[latest_backtest]

    returns = result.get("returns", pd.Series())

    if len(returns) < 10:
        st.error("Insufficient return data for simulation")
        return

    # Calculate return statistics
    mean_return = returns.mean()
    std_return = returns.std()

    # Run simulations
    with st.spinner("Running Monte Carlo simulation..."):
        simulations = []

        for _ in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, time_horizon)

            # Calculate cumulative return
            cumulative_return = (1 + random_returns).prod() - 1
            simulations.append(cumulative_return)

        simulations = np.array(simulations)

    # Display results
    st.subheader("📊 Monte Carlo Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        mean_sim_return = np.mean(simulations) * 100
        st.metric("Expected Return", f"{mean_sim_return:.2f}%")

    with col2:
        var_value = np.percentile(simulations, (1 - confidence_level) * 100) * 100
        st.metric(f"VaR ({confidence_level:.0%})", f"{var_value:.2f}%")

    with col3:
        worst_case = np.min(simulations) * 100
        st.metric("Worst Case", f"{worst_case:.2f}%")

    # Distribution chart
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=simulations * 100, nbinsx=50, name="Return Distribution", opacity=0.7
        )
    )

    # Add VaR line
    fig.add_vline(
        x=var_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR ({confidence_level:.0%})",
    )

    fig.update_layout(
        title="Monte Carlo Return Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def walk_forward_analysis():
    """Walk-forward analysis placeholder"""
    st.subheader("🚶 Walk-Forward Analysis")
    st.info("Walk-forward analysis coming soon...")


def sensitivity_analysis():
    """Sensitivity analysis placeholder"""
    st.subheader("🔍 Sensitivity Analysis")
    st.info("Sensitivity analysis coming soon...")


def risk_attribution_analysis():
    """Risk attribution analysis placeholder"""
    st.subheader("⚠️ Risk Attribution Analysis")
    st.info("Risk attribution analysis coming soon...")


def backtest_management_workflow():
    """Backtest management workflow"""

    st.header("💾 Backtest Management")

    if not st.session_state.backtest_cache:
        st.info("No backtests to manage.")
        return

    # Backtest list
    st.subheader("📋 Backtest Results")

    # Create management table
    management_data = []
    for backtest_id, result in st.session_state.backtest_cache.items():
        strategy_config = result.get("strategy_config", {})
        metrics = result.get("metrics")

        management_data.append(
            {
                "ID": backtest_id,
                "Strategy": strategy_config.get("strategy_type", "Unknown"),
                "Total Return (%)": (
                    f"{getattr(metrics, 'total_return', 0) * 100:.2f}"
                    if metrics
                    else "N/A"
                ),
                "Sharpe Ratio": (
                    f"{getattr(metrics, 'sharpe_ratio', 0):.3f}" if metrics else "N/A"
                ),
                "Max Drawdown (%)": (
                    f"{getattr(metrics, 'max_drawdown', 0) * 100:.2f}"
                    if metrics
                    else "N/A"
                ),
                "Created": result.get("timestamp", datetime.now()).strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }
        )

    if management_data:
        management_df = pd.DataFrame(management_data)

        # Convert complex objects to strings to avoid Arrow serialization issues
        for col in management_df.columns:
            if management_df[col].dtype == "object":
                management_df[col] = management_df[col].astype(str)

        st.dataframe(management_df, use_container_width=True)

        # Management actions
        st.subheader("🔧 Management Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Export All Results"):
                export_all_results()

        with col2:
            if st.button("🗑️ Clear All Results"):
                if st.button("⚠️ Confirm Clear All", type="secondary"):
                    st.session_state.backtest_cache = {}
                    st.success("All backtest results cleared")
                    st.rerun()

        with col3:
            selected_for_deletion = st.selectbox(
                "Select for Deletion", list(st.session_state.backtest_cache.keys())
            )

            if st.button("🗑️ Delete Selected"):
                if selected_for_deletion in st.session_state.backtest_cache:
                    del st.session_state.backtest_cache[selected_for_deletion]
                    st.success(f"Deleted backtest: {selected_for_deletion}")
                    st.rerun()


def export_all_results():
    """Export all backtest results"""

    if not st.session_state.backtest_cache:
        st.warning("No results to export")
        return

    # Create export data
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_backtests": len(st.session_state.backtest_cache),
        "backtests": {},
    }

    for backtest_id, result in st.session_state.backtest_cache.items():
        # Create exportable version (remove non-serializable objects)
        export_result = {
            "id": backtest_id,
            "strategy_config": result.get("strategy_config", {}),
            "backtest_config": result.get("backtest_config", {}),
            "portfolio_values": result.get("portfolio_values", []),
            "returns": result.get("returns", pd.Series()).tolist(),
            "trades": result.get("trades", []),
            "summary": result.get("summary", {}),
            "timestamp": result.get("timestamp", datetime.now()).isoformat(),
        }

        export_data["backtests"][backtest_id] = export_result

    # Convert to JSON
    export_json = json.dumps(export_data, indent=2, default=str)

    # Download button
    st.download_button(
        label="📥 Download All Results (JSON)",
        data=export_json,
        file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
