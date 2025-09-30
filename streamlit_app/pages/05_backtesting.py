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

# Add src and components directory to path
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

try:
    # Week 14: Streamlit utils integration - use utility managers
    from utils.backtest_utils import (
        BacktestDataPreparer,
        StrategyBuilder,
        BacktestResultProcessor,
        get_available_symbols_from_cache,
        validate_backtest_config,
    )

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

    # Import config safely
    try:
        from src.config import settings
    except ImportError:
        settings = None

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

except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are properly installed.")
    st.error(
        "Check that the virtual environment is activated and dependencies are installed."
    )
    st.stop()


def main():
    """Professional Backtesting Interface"""

    try:
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
            try:
                strategy_backtest_workflow()
            except Exception as e:
                st.error(f"Error in strategy backtest workflow: {e}")
                if st.checkbox("Show strategy backtest error details"):
                    st.error(traceback.format_exc())

        with tabs[1]:
            try:
                results_analysis_workflow()
            except Exception as e:
                st.error(f"Error in results analysis workflow: {e}")
                if st.checkbox("Show analysis error details"):
                    st.error(traceback.format_exc())

        with tabs[2]:
            try:
                comparison_workflow()
            except Exception as e:
                st.error(f"Error in comparison workflow: {e}")
                if st.checkbox("Show comparison error details"):
                    st.error(traceback.format_exc())

        with tabs[3]:
            try:
                advanced_analytics_workflow()
            except Exception as e:
                st.error(f"Error in advanced analytics workflow: {e}")
                if st.checkbox("Show analytics error details"):
                    st.error(traceback.format_exc())

        with tabs[4]:
            try:
                backtest_management_workflow()
            except Exception as e:
                st.error(f"Error in management workflow: {e}")
                if st.checkbox("Show management error details"):
                    st.error(traceback.format_exc())

    except Exception as e:
        st.error(f"Critical error in backtesting interface: {e}")
        st.error("Please check your environment setup and dependencies.")
        if st.checkbox("Show critical error details"):
            st.error(traceback.format_exc())


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

    try:
        # Check for feature data
        feature_data_available = (
            "feature_cache" in st.session_state
            and st.session_state.feature_cache
            and len(st.session_state.feature_cache) > 0
        )

        # Check for model data (optional for some strategies)
        model_data_available = (
            "model_cache" in st.session_state
            and st.session_state.model_cache
            and len(st.session_state.model_cache) > 0
        )

        # Check for data acquisition data
        raw_data_available = (
            "data_cache" in st.session_state
            and st.session_state.data_cache
            and len(st.session_state.data_cache) > 0
        )

        return feature_data_available or raw_data_available

    except Exception as e:
        st.error(f"Error checking data availability: {e}")
        return False


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

    # Configuration section
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

    # Results section - displayed below the button
    st.markdown("---")  # Separator line
    st.subheader("📊 Backtest Results")

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
    source_details = {}

    try:
        # Check feature data
        if "feature_cache" in st.session_state and st.session_state.feature_cache:
            for feature_key, feature_data in st.session_state.feature_cache.items():
                # Get basic info about the feature data
                data_info = ""
                price_indicator = ""

                # Determine if this contains price data
                contains_price_data = False

                if isinstance(feature_data, dict):
                    # Check if metadata contains original price data
                    if "original_data" in feature_data:
                        original_data = feature_data["original_data"]
                        if isinstance(original_data, pd.DataFrame):
                            columns_lower = [
                                col.lower() for col in original_data.columns
                            ]
                            price_indicators = [
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                            contains_price_data = any(
                                any(indicator in col for indicator in price_indicators)
                                for col in columns_lower
                            )

                    if "features" in feature_data:
                        features_df = feature_data["features"]
                        if hasattr(features_df, "shape"):
                            data_info = f" ({features_df.shape[0]} samples, {features_df.shape[1]} features)"
                    elif "data" in feature_data:
                        data_df = feature_data["data"]
                        if hasattr(data_df, "shape"):
                            data_info = f" ({data_df.shape[0]} samples, {data_df.shape[1]} columns)"
                elif hasattr(feature_data, "shape"):
                    # Check direct DataFrame for price data
                    columns_lower = [col.lower() for col in feature_data.columns]
                    price_indicators_check = [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "price",
                        "adj",
                    ]
                    contains_price_data = any(
                        any(indicator in col for indicator in price_indicators_check)
                        for col in columns_lower
                    )
                    data_info = f" ({feature_data.shape[0]} samples, {feature_data.shape[1]} columns)"

                # Add price data indicator
                if contains_price_data:
                    price_indicator = " 💰"  # Contains price data
                elif feature_key.endswith("_metadata"):
                    price_indicator = " 📊"  # Metadata with price data
                else:
                    # Check if corresponding metadata exists
                    metadata_key = f"{feature_key}_metadata"
                    if metadata_key in st.session_state.feature_cache:
                        price_indicator = (
                            " 🔧"  # Technical indicators with available price data
                        )
                    else:
                        price_indicator = " ⚠️"  # No price data available

                source_label = f"Features: {feature_key}{data_info}{price_indicator}"
                available_sources.append(source_label)
                source_details[source_label] = {"type": "features", "key": feature_key}

        # Check raw data
        if "data_cache" in st.session_state and st.session_state.data_cache:
            for data_key, data_item in st.session_state.data_cache.items():
                # Get basic info about the raw data
                data_info = ""
                contains_price_data = False

                if isinstance(data_item, dict) and "data" in data_item:
                    data_df = data_item["data"]
                    if hasattr(data_df, "shape"):
                        data_info = (
                            f" ({data_df.shape[0]} samples, {data_df.shape[1]} columns)"
                        )
                        # Check for price data in raw data
                        if hasattr(data_df, "columns"):
                            columns_lower = [col.lower() for col in data_df.columns]
                            price_indicators = [
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "price",
                            ]
                            contains_price_data = any(
                                any(indicator in col for indicator in price_indicators)
                                for col in columns_lower
                            )
                elif hasattr(data_item, "shape"):
                    data_info = (
                        f" ({data_item.shape[0]} samples, {data_item.shape[1]} columns)"
                    )
                    # Check for price data in direct DataFrame
                    if hasattr(data_item, "columns"):
                        columns_lower = [col.lower() for col in data_item.columns]
                        price_indicators = [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "price",
                        ]
                        contains_price_data = any(
                            any(indicator in col for indicator in price_indicators)
                            for col in columns_lower
                        )

                # Add price data indicator
                price_indicator = " 💰" if contains_price_data else " ⚠️"

                source_label = f"Raw Data: {data_key}{data_info}{price_indicator}"
                available_sources.append(source_label)
                source_details[source_label] = {"type": "raw", "key": data_key}

        if not available_sources:
            st.error("No data sources available")
            return None

        # Display available sources with details
        st.info(f"📈 Found {len(available_sources)} data source(s)")

        # Icon legend
        st.caption(
            "**Icons**: 💰 Contains price data | 📊 Metadata with price data | "
            "🔧 Technical indicators (price data available) | ⚠️ No price data"
        )

        # Debug: Show data structure in expander
        with st.expander("🔍 Debug: Data Structure", expanded=False):
            if "feature_cache" in st.session_state:
                st.write(
                    "Feature Cache Keys:", list(st.session_state.feature_cache.keys())
                )
                for key, data in st.session_state.feature_cache.items():
                    st.write(f"**{key}:**")
                    if isinstance(data, dict):
                        st.write(f"  - Dict keys: {list(data.keys())}")
                        for sub_key, sub_data in data.items():
                            if hasattr(sub_data, "shape"):
                                st.write(
                                    f"  - {sub_key}: {type(sub_data).__name__} {sub_data.shape}"
                                )
                                if hasattr(sub_data, "columns"):
                                    st.write(
                                        f"    Columns: {list(sub_data.columns)[:10]}..."
                                    )
                    elif hasattr(data, "shape"):
                        st.write(f"  - Direct DataFrame: {data.shape}")
                        if hasattr(data, "columns"):
                            st.write(f"    Columns: {list(data.columns)[:10]}...")

            if "data_cache" in st.session_state:
                st.write("Data Cache Keys:", list(st.session_state.data_cache.keys()))

        selected_source = st.selectbox(
            "Select Data Source",
            available_sources,
            help="Choose the data source for backtesting",
            key="data_source_select",
        )

        # Provide helpful advice for data source selection
        if selected_source:
            source_info = source_details.get(selected_source, {})
            data_key = source_info.get("key", "")

            # Check if selected source is a technical indicator without price data
            if source_info.get("type") == "features" and not data_key.endswith(
                "_metadata"
            ):
                feature_data = st.session_state.feature_cache.get(data_key)
                if isinstance(feature_data, pd.DataFrame):
                    columns_lower = [col.lower() for col in feature_data.columns]
                    price_indicators = [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "price",
                        "adj",
                    ]
                    has_price_data = any(
                        any(indicator in col for indicator in price_indicators)
                        for col in columns_lower
                    )

                    if not has_price_data:
                        metadata_key = f"{data_key}_metadata"
                        if metadata_key in st.session_state.feature_cache:
                            st.warning(
                                f"⚠️ **Note**: '{data_key}' contains technical indicators but no price data. "
                                f"The system will automatically use price data from '{metadata_key}' for backtesting."
                            )
                        else:
                            st.error(
                                f"❌ **Error**: '{data_key}' contains technical indicators but no price data, "
                                f"and no corresponding metadata was found. Please select a data source with price data (OHLCV)."
                            )

        # Store source details in session state for later use
        if "backtest_source_details" not in st.session_state:
            st.session_state.backtest_source_details = {}
        st.session_state.backtest_source_details = source_details

        return selected_source

    except Exception as e:
        st.error(f"Error selecting data source: {e}")
        return None


def run_comprehensive_backtest(
    data_source: str, strategy_config: Dict, backtest_config: Dict
):
    """Run comprehensive backtest with professional features"""

    progress_container = st.container()

    try:
        # Update running status
        st.session_state.backtest_running = True

        with progress_container:
            # Initialize utility classes
            try:
                data_preparer = BacktestDataPreparer()
                strategy_builder = StrategyBuilder()
                result_processor = BacktestResultProcessor()
            except Exception as e:
                st.error(f"Failed to initialize backtest utilities: {e}")
                return

            # Step 1: Prepare data
            display_computation_status("📊 Preparing data...", 0.1)
            time.sleep(0.3)

            # Get source details
            source_details = st.session_state.get("backtest_source_details", {})
            source_info = source_details.get(data_source, {})

            # Extract actual key from the data source
            if source_info:
                data_key = source_info["key"]
                source_type = source_info["type"]
            else:
                # Fallback parsing
                if data_source.startswith("Features:"):
                    data_key = data_source.split("Features: ")[1].split(" (")[0]
                    source_type = "features"
                else:
                    data_key = data_source.split("Raw Data: ")[1].split(" (")[0]
                    source_type = "raw"

            # Get and prepare data based on source type using BacktestDataPreparer
            try:
                if source_type == "features":
                    if data_key not in st.session_state.feature_cache:
                        st.error(f"Feature data '{data_key}' not found in cache")
                        return

                    feature_data = st.session_state.feature_cache[data_key]

                    # Debug: Show feature data structure
                    st.write(f"Debug: Processing feature data for key '{data_key}'")
                    st.write(f"Debug: Feature data type: {type(feature_data)}")

                    # First try to prepare feature data directly
                    data = data_preparer.prepare_feature_data(feature_data)

                    # If data preparation returns None (technical indicators without price data)
                    if data is None:
                        st.write(
                            f"Debug: Feature data lacks price information, checking for metadata..."
                        )

                        # Try to find metadata with price data
                        metadata_data = data_preparer.find_metadata_for_features(
                            data_key, st.session_state
                        )

                        if metadata_data is not None:
                            data = data_preparer._validate_price_data(metadata_data)
                            st.success(
                                f"✅ Successfully extracted price data from metadata: {data.shape}"
                            )
                            st.write(f"Debug: Price data columns: {list(data.columns)}")
                        else:
                            st.error(f"No price data found in the dataset")
                            st.write(
                                f"Data key: {data_key}, Source type: {source_type}"
                            )
                            st.info(
                                "💡 **Solution**: Use data that includes price information (OHLCV) or ensure technical indicator data has metadata with original price data."
                            )
                            return
                    elif data.empty:
                        st.error(f"Prepared data is empty for key '{data_key}'")
                        return
                    else:
                        st.success(
                            f"✅ Successfully prepared feature data: {data.shape}"
                        )
                        st.write(f"Debug: Price data columns: {list(data.columns)}")

                else:  # raw data
                    raw_data_item = st.session_state.data_cache[data_key]
                    if isinstance(raw_data_item, dict) and "data" in raw_data_item:
                        raw_data = raw_data_item["data"]
                    else:
                        raw_data = raw_data_item
                    data = data_preparer._validate_price_data(raw_data)

            except Exception as e:
                st.error(f"Failed to prepare data: {e}")
                st.error(f"Data key: {data_key}, Source type: {source_type}")
                if st.checkbox("Show detailed error", key="show_data_prep_error"):
                    st.error(traceback.format_exc())
                return

            if data is None or data.empty:
                st.error("No valid data available for backtesting")
                return

            symbols = [data_key]

            # Step 2: Initialize backtest engine
            display_computation_status("⚙️ Initializing backtest engine...", 0.2)
            time.sleep(0.3)

            try:
                engine = BacktestEngine(
                    initial_capital=backtest_config.get("initial_capital", 100000.0),
                    commission_rate=backtest_config.get("commission_rate", 0.001),
                    slippage_rate=backtest_config.get("slippage_rate", 0.0005),
                    min_commission=backtest_config.get("min_commission", 1.0),
                    max_position_size=backtest_config.get("max_position_size", 0.1),
                )
            except Exception as e:
                st.error(f"Failed to initialize backtest engine: {e}")
                return

            # Add data to engine
            symbol = symbols[0] if symbols else "ASSET"
            try:
                engine.add_data(symbol, data)
            except Exception as e:
                st.error(f"Failed to add data to engine: {e}")
                return

            # Step 3: Build strategy
            display_computation_status("🎯 Building strategy...", 0.4)
            time.sleep(0.3)

            try:
                if strategy_config.get("strategy_type") == "Model-Based":
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

                    # For now, use a simple strategy since model-based strategies need more implementation
                    st.info(
                        "Model-based strategies are currently being implemented. Using simple buy-and-hold strategy."
                    )
                    strategy_config["strategy_type"] = "buy_and_hold"
                    strategy = strategy_builder.build_strategy(
                        strategy_config, [symbol]
                    )
                else:
                    strategy = strategy_builder.build_strategy(
                        strategy_config, [symbol]
                    )
            except Exception as e:
                st.error(f"Failed to build strategy: {e}")
                return

            # Step 4: Run backtest
            display_computation_status("🚀 Running backtest...", 0.6)
            time.sleep(0.5)

            try:
                engine.set_strategy(strategy)

                # Set date range if specified and convert to proper format
                start_date = backtest_config.get("start_date")
                end_date = backtest_config.get("end_date")

                # Convert date objects to datetime if needed and ensure timezone-naive
                if start_date is not None:
                    import datetime as dt

                    if isinstance(start_date, dt.date) and not isinstance(
                        start_date, dt.datetime
                    ):
                        # Convert date to datetime at midnight
                        start_date = pd.Timestamp.combine(start_date, dt.time.min)
                    elif hasattr(start_date, "date") and not isinstance(
                        start_date, pd.Timestamp
                    ):
                        start_date = pd.Timestamp(start_date)
                    elif isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    else:
                        start_date = pd.Timestamp(start_date)

                    # Ensure timezone-naive for compatibility with backtest engine
                    if hasattr(start_date, "tz") and start_date.tz is not None:
                        start_date = start_date.tz_localize(None)

                if end_date is not None:
                    import datetime as dt

                    if isinstance(end_date, dt.date) and not isinstance(
                        end_date, dt.datetime
                    ):
                        # Convert date to datetime at end of day
                        end_date = pd.Timestamp.combine(end_date, dt.time.max)
                    elif hasattr(end_date, "date") and not isinstance(
                        end_date, pd.Timestamp
                    ):
                        end_date = pd.Timestamp(end_date)
                    elif isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    else:
                        end_date = pd.Timestamp(end_date)

                    # Ensure timezone-naive for compatibility with backtest engine
                    if hasattr(end_date, "tz") and end_date.tz is not None:
                        end_date = end_date.tz_localize(None)

                st.write(f"Debug: Running backtest from {start_date} to {end_date}")

                raw_results = engine.run_backtest(
                    start_date=start_date, end_date=end_date
                )
            except Exception as e:
                st.error(f"Failed to run backtest: {e}")
                if st.checkbox(
                    "Show detailed backtest error", key="show_backtest_error"
                ):
                    st.error(traceback.format_exc())
                return

            # Step 5: Process results
            display_computation_status("📊 Processing results...", 0.8)
            time.sleep(0.3)

            try:
                results = result_processor.process_results(engine, backtest_config)
            except Exception as e:
                st.error(f"Failed to process results: {e}")
                # Create minimal results for basic functionality
                results = {
                    "portfolio_values": (
                        engine.portfolio_values
                        if hasattr(engine, "portfolio_values")
                        else []
                    ),
                    "returns": pd.Series(),
                    "trades": [],
                    "summary": {"total_return": 0.0},
                    "metrics": None,
                }

            # Step 6: Store results
            display_computation_status("💾 Storing results...", 0.9)
            time.sleep(0.2)

            backtest_id = f"{strategy_config.get('strategy_type', 'Unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
        if st.checkbox("Show detailed error information"):
            st.error(traceback.format_exc())

    finally:
        # Reset running status
        st.session_state.backtest_running = False


# Model strategy creation is now handled by StrategyBuilder in utils/backtest_utils.py
# This eliminates duplication with src/backtesting backend


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
        st.error(f"Backtest {backtest_id} not found in cache")
        return

    try:
        result = st.session_state.backtest_cache[backtest_id]

        st.subheader(f"📊 Quick Results: {backtest_id}")

        # Key metrics with error handling
        metrics = result.get("metrics")
        summary = result.get("summary", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if metrics and hasattr(metrics, "total_return"):
                total_return = getattr(metrics, "total_return", 0) * 100
            else:
                total_return = summary.get("total_return", 0)
                if isinstance(total_return, str):
                    try:
                        total_return = float(total_return.replace("%", ""))
                    except:
                        total_return = 0.0
            st.metric("Total Return", f"{total_return:.2f}%")

        with col2:
            if metrics and hasattr(metrics, "sharpe_ratio"):
                sharpe_ratio = getattr(metrics, "sharpe_ratio", 0)
            else:
                sharpe_ratio = summary.get("sharpe_ratio", 0.0)
                if isinstance(sharpe_ratio, str):
                    try:
                        sharpe_ratio = float(sharpe_ratio)
                    except:
                        sharpe_ratio = 0.0
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")

        with col3:
            if metrics and hasattr(metrics, "max_drawdown"):
                max_drawdown = getattr(metrics, "max_drawdown", 0) * 100
            else:
                max_drawdown = summary.get("max_drawdown", 0.0)
                if isinstance(max_drawdown, str):
                    try:
                        max_drawdown = float(max_drawdown.replace("%", ""))
                    except:
                        max_drawdown = 0.0
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

        with col4:
            num_trades = len(result.get("trades", []))
            st.metric("Total Trades", num_trades)

        # Quick chart with error handling
        portfolio_values = result.get("portfolio_values", [])
        if len(portfolio_values) > 1:
            st.subheader("📈 Portfolio Performance")

            try:
                fig = go.Figure()

                # Create dates - try to use actual dates from data if available
                if "returns" in result and hasattr(result["returns"], "index"):
                    dates = result["returns"].index[: len(portfolio_values)]
                else:
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

            except Exception as e:
                st.warning(f"Could not create performance chart: {e}")
        else:
            st.info("Insufficient portfolio data for chart display")

    except Exception as e:
        st.error(f"Error displaying quick results: {e}")
        if st.checkbox("Show error details", key=f"error_details_{backtest_id}"):
            st.error(traceback.format_exc())


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

    try:
        # Create export data
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_backtests": len(st.session_state.backtest_cache),
            "backtests": {},
        }

        for backtest_id, result in st.session_state.backtest_cache.items():
            try:
                # Create exportable version (remove non-serializable objects)
                export_result = {
                    "id": backtest_id,
                    "strategy_config": result.get("strategy_config", {}),
                    "backtest_config": result.get("backtest_config", {}),
                    "portfolio_values": result.get("portfolio_values", []),
                    "summary": result.get("summary", {}),
                    "data_source": result.get("data_source", ""),
                    "symbol": result.get("symbol", ""),
                    "timestamp": result.get("timestamp", datetime.now()).isoformat(),
                }

                # Handle returns data safely
                returns_data = result.get("returns", pd.Series())
                if hasattr(returns_data, "tolist"):
                    export_result["returns"] = returns_data.tolist()
                elif isinstance(returns_data, list):
                    export_result["returns"] = returns_data
                else:
                    export_result["returns"] = []

                # Handle trades data safely
                trades_data = result.get("trades", [])
                if isinstance(trades_data, list):
                    export_result["trades"] = trades_data
                else:
                    export_result["trades"] = []

                # Handle metrics safely
                metrics = result.get("metrics")
                if metrics:
                    export_result["metrics"] = {
                        "total_return": getattr(metrics, "total_return", 0),
                        "sharpe_ratio": getattr(metrics, "sharpe_ratio", 0),
                        "max_drawdown": getattr(metrics, "max_drawdown", 0),
                        "volatility": getattr(metrics, "volatility", 0),
                        "win_rate": getattr(metrics, "win_rate", 0),
                    }

                export_data["backtests"][backtest_id] = export_result

            except Exception as e:
                st.warning(f"Failed to export backtest {backtest_id}: {e}")
                continue

        # Convert to JSON with error handling
        try:
            export_json = json.dumps(export_data, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to convert results to JSON: {e}")
            return

        # Create downloadable data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qat_backtest_results_{timestamp}.json"

        # Download button
        st.download_button(
            label="📥 Download All Results (JSON)",
            data=export_json,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
            help="Download all backtest results in JSON format",
        )

        # Success message
        st.success(
            f"✅ Export prepared: {len(export_data['backtests'])} backtest(s) ready for download"
        )

    except Exception as e:
        st.error(f"Export failed: {e}")
        if st.checkbox("Show export error details"):
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
