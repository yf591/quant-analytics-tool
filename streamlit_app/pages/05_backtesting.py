"""
Streamlit Page: Backtesting (Clean Architecture Version)
Week 14 UI Integration - Clean Frontend with Backend Integration Only
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

try:
    # Backend imports only - NO local implementations
    from src.backtesting import (
        BacktestEngine,
        BuyAndHoldStrategy,
        MomentumStrategy,
        MeanReversionStrategy,
        ModelBasedStrategy,
        MultiAssetStrategy,
        PerformanceCalculator,
    )

    # UI utilities for data preparation and display only
    from utils.backtest_utils import (
        BacktestDataPreparer,
        ConfigurationHelper,
        BacktestResultProcessor,
        get_available_symbols_from_cache,
        validate_backtest_config,
    )

    # UI components
    from components.backtest_widgets import (
        StrategyConfigWidget,
        BacktestConfigWidget,
        BacktestResultsWidget,
    )
    from components.data_display import (
        display_computation_status,
        display_alert_message,
    )

except ImportError as e:
    st.error(f"Backend import error: {e}")
    st.error("Please ensure backend modules are properly configured.")
    st.stop()


def main():
    """Clean Backtesting Interface - UI Only"""

    st.set_page_config(
        page_title="Backtesting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        initialize_session_state()

        st.title("📈 Advanced Backtesting Engine")
        st.markdown("**Professional backtesting powered by AFML-compliant backend**")

        # Check data availability
        if not check_data_availability():
            display_data_requirements()
            return

        # Main workflow tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🚀 Strategy Backtest",
                "📊 Results Analysis",
                "🔄 Comparison",
                "📋 Management",
            ]
        )

        with tab1:
            display_backtest_interface()

        with tab2:
            results_analysis_workflow()

        with tab3:
            comparison_workflow()

        with tab4:
            backtest_management_workflow()

    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please check backend configuration and try again.")


def initialize_session_state():
    """Initialize session state for backtesting"""
    if "backtest_cache" not in st.session_state:
        st.session_state.backtest_cache = {}
    if "backtest_running" not in st.session_state:
        st.session_state.backtest_running = False


def check_data_availability() -> bool:
    """Check if required data is available for backtesting"""
    try:
        symbols = get_available_symbols_from_cache()
        return len(symbols) > 0
    except Exception:
        return False


def display_data_requirements():
    """Display data requirements for backtesting"""
    st.warning("📊 **Data Required for Backtesting**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
        **Data Source 1**
        - Select your data source
        - Configure parameters
        - Import from backend
        """
        )

    with col2:
        st.info(
            """
        **Raw Market Data**
        - Price history (OHLCV)
        - Volume data
        - Multiple symbols
        """
        )

    with col3:
        st.info(
            """
        **Model Data (Optional)**
        - Trained ML models
        - Prediction features
        - Model parameters
        """
        )


def display_backtest_interface():
    """Backtest interface display - UI only"""

    st.subheader("🎯 Strategy Configuration")

    # Data source selection
    data_source = select_data_source()
    if not data_source:
        return

    # Two-column layout for configuration
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strategy Settings**")
        # Strategy configuration using widget
        strategy_widget = StrategyConfigWidget()
        strategy_config = strategy_widget.render_strategy_selection("main")

    with col2:
        st.markdown("**Backtest Settings**")
        # Backtest configuration using widget
        backtest_widget = BacktestConfigWidget()
        backtest_config = backtest_widget.render_backtest_config("main")

    # Validation and execution section (full width)
    st.markdown("---")
    st.subheader("🚀 Execute Backtest")

    # Validation
    is_valid, errors = validate_backtest_config(backtest_config)
    if errors:
        for error in errors:
            st.error(error)

    # Run button (full width, centered styling)
    if st.button(
        "🎯 Run Backtest",
        type="primary",
        disabled=not is_valid,
        use_container_width=True,
    ):
        display_backtest_results(data_source, strategy_config, backtest_config)


def select_data_source() -> Optional[str]:
    """Select data source for backtesting - UI only"""

    try:
        # Get available data sources
        available_sources = []

        # Feature cache sources
        if hasattr(st.session_state, "feature_cache"):
            for key, data in st.session_state.feature_cache.items():
                available_sources.append(f"Features: {key}")

        # Raw data sources (would be implemented based on data management)

        if not available_sources:
            st.warning("No data sources available")
            return None

        return st.selectbox(
            "📊 Select Data Source",
            available_sources,
            help="Choose data source for backtesting",
        )

    except Exception as e:
        st.error(f"Error loading data sources: {e}")
        return None


def display_backtest_results(
    data_source: str, strategy_config: Dict, backtest_config: Dict
):
    """Run backtest using backend only - NO business logic here"""

    progress_container = st.container()

    with progress_container:
        try:
            # Update running status
            st.session_state.backtest_running = True

            display_computation_status("📊 Preparing data...", 0.2)

            # Data preparation (UI utility only)
            data_preparer = BacktestDataPreparer()
            prepared_data = data_preparer.find_metadata_for_features(
                data_source.split(": ")[1] if ": " in data_source else data_source,
                st.session_state,
            )

            if prepared_data is None:
                st.error("Failed to prepare data")
                return

            display_computation_status("🎯 Building strategy...", 0.4)

            # Strategy creation (uses backend classes directly)
            config_helper = ConfigurationHelper()
            strategy = config_helper.create_strategy_instance(
                strategy_config, ["AAPL"]  # Would extract from prepared_data
            )

            if strategy is None:
                st.error("Failed to create strategy")
                return

            display_computation_status("🚀 Running backtest...", 0.6)

            # Backend execution - ALL logic in backend
            engine = BacktestEngine(
                initial_capital=backtest_config["initial_capital"],
                commission_rate=backtest_config["commission_rate"],
                slippage_rate=backtest_config["slippage_rate"],
            )

            # Add data to engine
            try:
                engine.add_data("AAPL", prepared_data)
                engine.set_strategy(strategy)

                # Run backtest (backend handles everything)
                results = engine.run_backtest()

                if results is None:
                    st.error("Backtest execution returned no results")
                    return

            except Exception as e:
                st.error(f"Backtest execution failed: {e}")
                return

            display_computation_status("📈 Processing results...", 0.8)

            # Result processing (formatting only)
            try:
                result_processor = BacktestResultProcessor()
                processed_results = result_processor.process_results(
                    engine, backtest_config
                )

                if processed_results is None:
                    st.error("Failed to process backtest results")
                    return

                display_computation_status("✅ Complete!", 1.0)

                # Cache results
                backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.backtest_cache[backtest_id] = processed_results

                # Display success message and quick results
                st.success("🎉 Backtest completed successfully!")
                display_quick_results(backtest_id)

            except Exception as e:
                st.error(f"Result processing failed: {e}")
                st.error(
                    "The backtest executed but results could not be processed. Please check backend configuration."
                )
                return

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.error("Please check data and configuration")

        finally:
            st.session_state.backtest_running = False


def display_quick_results(backtest_id: str):
    """Display backtest results using widgets - UI only"""

    # Ensure full width display
    st.markdown("---")  # Visual separator

    try:
        results = st.session_state.backtest_cache[backtest_id]

        # Try to use results widget for display
        try:
            results_widget = BacktestResultsWidget()
            results_widget.render_results_overview(results)
            results_widget.render_performance_chart(results)
        except Exception as widget_error:
            # Fallback to basic display (full width)
            st.info("📊 Using enhanced basic results display")
            display_basic_results(results)

    except Exception as e:
        st.error(f"Error displaying results: {e}")


def display_basic_results(results: Dict[str, Any]):
    """Basic fallback results display - Full width layout"""

    # Use full container width for results
    with st.container():
        st.subheader("📊 Backtest Results")

    # Display basic metrics with full width
    if "summary" in results:
        summary = results["summary"]

        # Use 4 columns for better spacing
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", summary.get("total_return", "N/A"))
        with col2:
            st.metric("Sharpe Ratio", summary.get("sharpe_ratio", "N/A"))
        with col3:
            st.metric("Max Drawdown", summary.get("max_drawdown", "N/A"))
        with col4:
            # Additional metric if available
            st.metric("Status", "✅ Complete")

    # Display portfolio values with full width
    if "portfolio_values" in results and results["portfolio_values"]:
        import plotly.graph_objects as go
        import pandas as pd
        from datetime import datetime, timedelta

        values = results["portfolio_values"]

        # Create proper date range for x-axis
        # Try to get dates from backtest config or use default
        config = results.get("config", {})

        # Use sample data date range that matches the backend
        start_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, periods=len(values), freq="D")

        # If we have fewer than 30 days, show daily ticks, otherwise weekly
        tick_frequency = "D1" if len(values) <= 30 else "D7"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                name="Portfolio Value",
                line=dict(width=2, color="#1f77b4"),
                hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400,
            margin=dict(l=60, r=20, t=40, b=60),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                tickformat="%m/%d" if len(values) <= 30 else "%m/%d/%y",
                dtick=tick_frequency,
                tickangle=45 if len(values) > 10 else 0,
                showticklabels=True,
            ),
            yaxis=dict(
                showgrid=True, gridwidth=1, gridcolor="lightgray", tickformat="$,.0f"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display additional info
        st.markdown(
            f'        # Display additional detailed info in columns\n        info_col1, info_col2, info_col3 = st.columns(3)\n        with info_col1:\n            st.markdown(f"**Data Points:** {len(values)}")\n        with info_col2: \n            st.markdown(f"**Initial Value:** ${values[0]:,.2f}" if values else "N/A")\n        with info_col3:\n            st.markdown(f"**Final Value:** ${values[-1]:,.2f}" if values else "N/A")\n    else:\n        st.info("No portfolio value data available for charting")'
            if values
            else "No data"
        )
    else:
        st.info("No portfolio value data available for charting")


def results_analysis_workflow():
    """Results analysis interface - UI only"""

    if not st.session_state.backtest_cache:
        st.info("No backtest results available. Run a backtest first.")
        return

    # Select result to analyze
    result_keys = list(st.session_state.backtest_cache.keys())
    selected_key = st.selectbox("📊 Select Backtest Result", result_keys)

    if selected_key:
        results = st.session_state.backtest_cache[selected_key]

        # Display comprehensive analysis using widgets
        results_widget = BacktestResultsWidget()
        results_widget.render_results_overview(results)
        results_widget.render_performance_chart(results)
        results_widget.render_trades_analysis(results)
        results_widget.render_risk_metrics(results)


def comparison_workflow():
    """Backtest comparison interface - UI only"""

    if len(st.session_state.backtest_cache) < 2:
        st.info("Need at least 2 backtest results for comparison.")
        return

    # Multi-select for comparison
    result_keys = list(st.session_state.backtest_cache.keys())
    selected_keys = st.multiselect("📊 Select Results to Compare", result_keys)

    if len(selected_keys) >= 2:
        comparison_results = {
            key: st.session_state.backtest_cache[key] for key in selected_keys
        }

        # Display comparison table
        st.subheader("📊 Backtest Comparison")

        # Create comparison dataframe
        comparison_data = []
        for key, results in comparison_results.items():
            if "summary" in results:
                summary = results["summary"]
                comparison_data.append(
                    {
                        "Backtest": key,
                        "Total Return": summary.get("total_return", "N/A"),
                        "Sharpe Ratio": summary.get("sharpe_ratio", "N/A"),
                        "Max Drawdown": summary.get("max_drawdown", "N/A"),
                    }
                )

        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data))


def backtest_management_workflow():
    """Backtest management interface - UI only"""

    st.subheader("📋 Backtest Management")

    if not st.session_state.backtest_cache:
        st.info("No backtest results to manage.")
        return

    # Display cached results
    for backtest_id, results in st.session_state.backtest_cache.items():
        with st.expander(f"📈 {backtest_id}"):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Basic info from results
                if "summary" in results:
                    summary = results["summary"]
                    st.write(f"**Return:** {summary.get('total_return', 'N/A')}")
                    st.write(
                        f"**Sharpe:** {summary.get('sharpe_ratio', 'Calculating...')}"
                    )

            with col2:
                if st.button("View Details", key=f"view_{backtest_id}"):
                    st.session_state.selected_result = backtest_id

            with col3:
                if st.button("Delete", key=f"delete_{backtest_id}"):
                    del st.session_state.backtest_cache[backtest_id]
                    st.rerun()


if __name__ == "__main__":
    main()
