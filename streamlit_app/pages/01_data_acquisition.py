"""
Streamlit Page: Data Acquisition
Week 14 UI Integration - Professional Data Collection Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Add src and components directory to path
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

try:
    # Week 14: Streamlit utils integration - use utility managers
    from utils.data_utils import DataAcquisitionManager
    from utils.analysis_utils import AnalysisManager

    # Streamlit components
    from components.charts import create_price_chart, create_correlation_heatmap
    from components.data_display import (
        display_data_metrics,
        display_alert_message,
    )
    from components.forms import create_data_selection_form, create_date_range_form
    from components.data_management import (
        create_data_source_selection_form,
        display_collection_status,
        display_validation_results,
        display_storage_management,
        create_data_quality_dashboard,
        display_data_comparison,
        create_batch_operation_interface,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Data Acquisition Interface"""

    # Initialize data acquisition manager
    data_manager = DataAcquisitionManager()
    analysis_manager = AnalysisManager()

    # Initialize session state using the managers
    data_manager.initialize_session_state(st.session_state)
    analysis_manager.initialize_session_state(st.session_state)

    st.set_page_config(
        page_title="📊 Data Acquisition",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("📊 Data Acquisition & Management")
    st.markdown(
        "### Professional financial data collection with validation and storage"
    )

    # Store the managers in session state for access in other functions
    st.session_state.data_manager = data_manager
    st.session_state.analysis_manager = analysis_manager

    # Professional workflow tabs
    tabs = st.tabs(
        [
            "📥 Collection",
            "🔍 Validation",
            "💾 Storage",
            "📊 Analysis",
            "🔄 Batch Operations",
        ]
    )

    with tabs[0]:
        data_collection_workflow()

    with tabs[1]:
        data_validation_workflow()

    with tabs[2]:
        storage_management_workflow()

    with tabs[3]:
        data_analysis_workflow()

    with tabs[4]:
        batch_operations_workflow()


def data_collection_workflow():
    """Professional data collection workflow"""

    st.header("🎯 Data Collection Workflow")

    # Data source configuration
    source_config = create_data_source_selection_form()

    if not source_config:
        st.warning("Please configure data source parameters")
        return

    # Collection actions
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "📥 Start Collection",
            type="primary",
            use_container_width=True,
            key="start_collection_btn",
        ):
            with st.spinner("Starting data collection..."):
                if source_config["data_source"] == "Yahoo Finance":
                    start_yahoo_finance_collection(source_config)
                elif source_config["data_source"] == "Custom Upload":
                    start_custom_upload_collection(source_config)

    with col2:
        if st.button(
            "⏹️ Stop Collection", use_container_width=True, key="stop_collection_btn"
        ):
            st.session_state.collection_status = {}
            st.success("Collection stopped")

    # Display collection status if active
    if st.session_state.collection_status:
        display_collection_status(st.session_state.collection_status)

    # Display collected data overview
    if st.session_state.data_cache:
        st.subheader("📊 Collected Data Overview")

        # Create summary data
        summary_data = []
        for cache_key, cache_data in st.session_state.data_cache.items():
            data = cache_data["data"]

            # Extract symbol from cache data
            symbol = cache_data.get("symbol", cache_key.split("_")[0])

            summary_data.append(
                {
                    "Symbol": symbol,
                    "Records": len(data),
                    "Columns": len(data.columns),
                    "Start Date": cache_data.get("date_range", {}).get(
                        "start",
                        (
                            data.index.min().strftime("%Y-%m-%d")
                            if isinstance(data.index, pd.DatetimeIndex)
                            else "N/A"
                        ),
                    ),
                    "End Date": cache_data.get("date_range", {}).get(
                        "end",
                        (
                            data.index.max().strftime("%Y-%m-%d")
                            if isinstance(data.index, pd.DatetimeIndex)
                            else "N/A"
                        ),
                    ),
                    "Source": cache_data.get("source", "Unknown"),
                    "Collected": cache_data.get(
                        "collected_at", datetime.now()
                    ).strftime("%H:%M:%S"),
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Display simple metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📊 Total Symbols", len(st.session_state.data_cache))

        with col2:
            total_records = sum(
                len(cache_data["data"])
                for cache_data in st.session_state.data_cache.values()
            )
            st.metric("📈 Total Records", f"{total_records:,}")

        with col3:
            avg_records = (
                total_records / len(st.session_state.data_cache)
                if st.session_state.data_cache
                else 0
            )
            st.metric("📊 Avg Records/Symbol", f"{avg_records:.0f}")

        with col4:
            latest_collection = max(
                cache_data.get("collected_at", datetime.min)
                for cache_data in st.session_state.data_cache.values()
            )
            st.metric("🕒 Latest Collection", latest_collection.strftime("%H:%M:%S"))


def data_validation_workflow():
    """Data validation workflow"""

    st.header("🔍 Data Validation Workflow")

    if not st.session_state.data_cache:
        st.info("📊 Collect data first to enable validation")
        return

    # Symbol selection for validation
    selected_symbols = st.multiselect(
        "Select Symbols to Validate",
        options=list(st.session_state.data_cache.keys()),
        default=list(st.session_state.data_cache.keys()),
        help="Choose symbols to validate",
    )

    # Validation options
    col1, col2 = st.columns(2)

    with col1:
        validation_level = st.selectbox(
            "Validation Level",
            options=["Basic", "Standard", "Strict"],
            index=1,
            help="Level of validation strictness",
        )

    with col2:
        generate_report = st.checkbox(
            "Generate Report", value=True, help="Generate detailed validation report"
        )

    # Validation actions
    col_val1, col_val2 = st.columns(2)

    with col_val1:
        if st.button(
            "🔍 Validate Selected",
            type="primary",
            use_container_width=True,
            key="validate_selected_btn",
        ):
            validate_selected_data(selected_symbols, validation_level, generate_report)

    with col_val2:
        if st.button(
            "🔍 Validate All", use_container_width=True, key="validate_all_btn"
        ):
            validate_all_data(validation_level, generate_report)

    # Display validation results
    if (
        hasattr(st.session_state, "validation_results")
        and st.session_state.validation_results
    ):
        for symbol, results in st.session_state.validation_results.items():
            with st.expander(f"📊 Validation Results: {symbol}"):
                display_validation_results(results)


def storage_management_workflow():
    """Storage management workflow"""

    st.header("💾 Storage Management Workflow")

    # Display storage interface
    display_storage_management()

    # Auto-save options
    st.subheader("⚙️ Auto-Save Configuration")

    col1, col2 = st.columns(2)

    with col1:
        auto_save = st.checkbox("Enable Auto-Save", value=False)
        if auto_save:
            save_interval = st.slider("Save Interval (minutes)", 1, 60, 5)

    with col2:
        backup_enabled = st.checkbox("Enable Backup", value=True)
        if backup_enabled:
            backup_location = st.text_input("Backup Location", value="backups/")


def data_analysis_workflow():
    """Data analysis and visualization workflow"""

    st.header("📊 Data Analysis Workflow")

    if not st.session_state.data_cache:
        st.info("📊 Collect data first to enable analysis")
        return

    # Analysis type selection
    analysis_type = st.selectbox(
        "Analysis Type",
        options=[
            "Single Symbol",
            "Multi-Symbol Comparison",
            "Data Quality Dashboard",
            "Statistical Analysis",
        ],
        help="Choose the type of analysis to perform",
    )

    if analysis_type == "Single Symbol":
        single_symbol_analysis()
    elif analysis_type == "Multi-Symbol Comparison":
        multi_symbol_comparison()
    elif analysis_type == "Data Quality Dashboard":
        data_quality_analysis()
    elif analysis_type == "Statistical Analysis":
        statistical_analysis_for_data_acquisition()


def batch_operations_workflow():
    """Batch operations workflow"""

    st.header("🔄 Batch Operations Workflow")

    # Create batch operation interface
    batch_config = create_batch_operation_interface()

    if batch_config.get("collect_batch", False):
        execute_batch_collection(batch_config)

    if batch_config.get("validate_batch", False):
        execute_batch_validation()

    if batch_config.get("save_batch", False):
        execute_batch_save()


def start_yahoo_finance_collection(config: Dict[str, Any]):
    """Start Yahoo Finance data collection using DataAcquisitionManager"""

    try:
        # Get the data manager from session state
        data_manager = st.session_state.data_manager

        # Transform config to match the manager's expected format
        symbols_list = config.get("symbols", [])
        if isinstance(symbols_list, list):
            symbols_str = ",".join(symbols_list)
        else:
            symbols_str = str(symbols_list)

        manager_config = {
            "symbols": symbols_str,
            "period": config.get("period", "1y"),
            "interval": config.get("interval", "1d"),
        }

        # Use the manager to start collection
        success, message = data_manager.start_yahoo_finance_collection(
            manager_config, st.session_state
        )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    except Exception as e:
        error_msg = f"Collection failed: {str(e)}"
        st.error(error_msg)


def start_custom_upload_collection(config: Dict[str, Any]):
    """Start custom file upload collection using DataAcquisitionManager"""

    try:
        # Get the data manager from session state
        data_manager = st.session_state.data_manager

        uploaded_file = config.get("uploaded_file")
        if not uploaded_file:
            st.warning("Please upload a file")
            return

        # Use the manager to start upload collection
        with st.spinner("Processing uploaded file..."):
            success, message = data_manager.start_custom_upload_collection(
                [uploaded_file], config, st.session_state
            )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        st.error(error_msg)


def validate_selected_data(
    symbols: List[str], validation_level: str, generate_report: bool
):
    """Validate selected symbols using DataAcquisitionManager"""

    try:
        # Get the data manager from session state
        data_manager = st.session_state.data_manager

        # Use the manager to validate selected data
        with st.spinner("Validating selected data..."):
            success, message = data_manager.validate_selected_data(
                symbols, validation_level, generate_report, st.session_state
            )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        st.error(error_msg)


def validate_all_data(validation_level: str, generate_report: bool):
    """Validate all cached data using DataAcquisitionManager"""

    # Get the data manager from session state
    data_manager = st.session_state.data_manager

    # Use the manager to validate all data
    with st.spinner("Validating all data..."):
        success, message = data_manager.validate_all_data(
            validation_level, generate_report, st.session_state
        )

    if success:
        st.success(message)
        st.rerun()
    else:
        st.error(message)


def single_symbol_analysis():
    """Single symbol analysis"""

    # Create a mapping of display names to cache keys
    symbol_options = {}
    for cache_key, cache_data in st.session_state.data_cache.items():
        display_name = cache_data.get("symbol", cache_key.split("_")[0])
        symbol_options[display_name] = cache_key

    selected_symbol = st.selectbox("Select Symbol", options=list(symbol_options.keys()))

    if selected_symbol:
        cache_key = symbol_options[selected_symbol]
        data = st.session_state.data_cache[cache_key]["data"]

        # Create data quality dashboard
        create_data_quality_dashboard(data, selected_symbol)

        # Price chart
        if "Close" in data.columns:
            st.subheader("📈 Price Chart")
            chart = create_price_chart(data, height=500)
            st.plotly_chart(chart, use_container_width=True)


def multi_symbol_comparison():
    """Multi-symbol comparison analysis"""

    # Create a mapping with display names
    data_dict = {}
    for cache_key, cache_data in st.session_state.data_cache.items():
        symbol = cache_data.get("symbol", cache_key.split("_")[0])
        data_dict[symbol] = cache_data["data"]

    display_data_comparison(data_dict)


def data_quality_analysis():
    """Data quality analysis for all symbols"""

    st.subheader("📊 Overall Data Quality Analysis")

    quality_data = []
    for cache_key, cache_data in st.session_state.data_cache.items():
        data = cache_data["data"]
        symbol = cache_data.get("symbol", cache_key.split("_")[0])

        missing_pct = (
            data.isnull().sum().sum() / (len(data) * len(data.columns))
        ) * 100

        quality_data.append(
            {
                "Symbol": symbol,
                "Records": len(data),
                "Columns": len(data.columns),
                "Missing %": f"{missing_pct:.1f}%",
                "Date Range": f"{(data.index.max() - data.index.min()).days} days",
                "Latest Date": data.index.max().strftime("%Y-%m-%d"),
            }
        )

    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True)


def execute_batch_collection(config: Dict[str, Any]):
    """Execute batch collection using DataAcquisitionManager"""

    try:
        # Get the data manager from session state
        data_manager = st.session_state.data_manager

        # Use the manager to execute batch collection
        success, message = data_manager.execute_batch_collection(
            config, st.session_state
        )

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Batch collection failed: {str(e)}")


def execute_batch_validation():
    """Execute batch validation using DataAcquisitionManager"""

    try:
        # Get the data manager from session state
        data_manager = st.session_state.data_manager

        if not st.session_state.data_cache:
            st.warning("No data to validate")
            return

        # Use the manager to execute batch validation
        success, message = data_manager.execute_batch_validation(st.session_state)

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Batch validation failed: {str(e)}")


def execute_batch_save():
    """Execute batch save operation using DataAcquisitionManager"""

    try:
        # Get the data manager from session state
        data_manager = st.session_state.data_manager

        if not st.session_state.data_cache:
            st.warning("No data to save")
            return

        # Use the manager to execute batch save
        with st.spinner("Saving all data..."):
            success, message = data_manager.execute_batch_save(st.session_state)

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        error_msg = f"Batch save failed: {str(e)}"
        st.error(error_msg)


# Session state is now initialized in main() using DataAcquisitionManager
# This function is kept for backward compatibility but delegates to the manager
def initialize_session_state():
    """Initialize session state for data acquisition (deprecated - use DataAcquisitionManager)"""

    # This function is now handled by DataAcquisitionManager in main()
    # Kept for backward compatibility
    pass


def statistical_analysis_for_data_acquisition():
    """Statistical Analysis for collected data - Week 14 Integration"""

    st.subheader("📈 Statistical Analysis of Collected Data")
    st.markdown(
        "Perform statistical analysis on collected financial data to understand "
        "its characteristics before proceeding to feature engineering."
    )

    # Get available data
    available_data = list(st.session_state.data_cache.keys())

    if not available_data:
        st.warning("No data available for analysis")
        return

    # Data selection
    selected_ticker = st.selectbox(
        "Select Dataset for Analysis",
        options=available_data,
        help="Choose a dataset to analyze",
    )

    if not selected_ticker:
        return

    # Get data
    cache_data = st.session_state.data_cache[selected_ticker]
    data = cache_data["data"]
    symbol = cache_data.get("symbol", selected_ticker.split("_")[0])

    # Analysis options
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Price Data Analysis:**")

        # Select price column
        price_columns = [
            col
            for col in data.columns
            if "close" in col.lower() or "price" in col.lower()
        ]
        if not price_columns:
            price_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if price_columns:
            selected_price_col = st.selectbox(
                "Select Price Column", options=price_columns
            )

            if st.button("📊 Analyze Price Statistics", use_container_width=True):
                analyze_data_statistics(symbol, data[selected_price_col], "price")

            if st.button("📈 Analyze Returns", use_container_width=True):
                analyze_data_returns(symbol, data[selected_price_col])
        else:
            st.warning("No suitable price columns found")

    with col2:
        st.write("**Data Quality Analysis:**")

        if st.button("🔍 Data Quality Check", use_container_width=True):
            analyze_data_quality(symbol, data)

        if st.button("📊 Distribution Analysis", use_container_width=True):
            analyze_data_distribution(symbol, data)

    # Display analysis results
    display_data_analysis_results(symbol)


def analyze_data_statistics(ticker: str, data: pd.Series, data_type: str):
    """Analyze basic statistics for collected data"""

    try:
        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_basic_statistics(
            data, ticker, st.session_state, data_type
        )

        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")

    except Exception as e:
        st.error(f"Statistical analysis failed: {str(e)}")


def analyze_data_returns(ticker: str, price_data: pd.Series):
    """Analyze returns for collected data"""

    try:
        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_returns(
            price_data, ticker, st.session_state, "simple"
        )

        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")

    except Exception as e:
        st.error(f"Return analysis failed: {str(e)}")


def analyze_data_quality(ticker: str, data: pd.DataFrame):
    """Analyze data quality for collected data"""

    try:
        # Basic data quality metrics
        st.subheader("📊 Data Quality Report")

        # Missing values analysis
        missing_data = data.isnull().sum()
        missing_pct = (missing_data / len(data)) * 100

        quality_df = pd.DataFrame(
            {
                "Column": data.columns,
                "Missing Count": missing_data,
                "Missing %": missing_pct,
                "Data Type": [str(dtype) for dtype in data.dtypes],
                "Unique Values": [data[col].nunique() for col in data.columns],
            }
        )

        st.dataframe(quality_df, use_container_width=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(data))

        with col2:
            st.metric("Total Columns", len(data.columns))

        with col3:
            overall_missing = (
                data.isnull().sum().sum() / (len(data) * len(data.columns))
            ) * 100
            st.metric("Overall Missing %", f"{overall_missing:.2f}%")

        with col4:
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)

        st.success("✅ Data quality analysis completed")

    except Exception as e:
        st.error(f"Data quality analysis failed: {str(e)}")


def analyze_data_distribution(ticker: str, data: pd.DataFrame):
    """Analyze distribution for collected data"""

    try:
        # Select numeric columns for distribution analysis
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            st.warning("No numeric data available for distribution analysis")
            return

        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_distribution(
            numeric_data, ticker, st.session_state, "price"
        )

        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")

    except Exception as e:
        st.error(f"Distribution analysis failed: {str(e)}")


def display_data_analysis_results(ticker: str):
    """Display analysis results for data acquisition"""

    try:
        if "analysis_cache" not in st.session_state:
            return

        analysis_cache = st.session_state.analysis_cache

        # Find analyses for this ticker
        ticker_analyses = [key for key in analysis_cache.keys() if ticker in key]

        if not ticker_analyses:
            return

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        for analysis_key in ticker_analyses:
            results = analysis_cache[analysis_key]
            analysis_type = results.get("type", "unknown")

            with st.expander(
                f"📈 {analysis_type.replace('_', ' ').title()}", expanded=False
            ):

                if analysis_type == "basic_statistics":
                    display_basic_stats_results(results)
                elif analysis_type == "distribution_analysis":
                    display_distribution_analysis_results(results)
                elif analysis_type == "returns_analysis":
                    display_returns_analysis_results(results)

    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")


def display_basic_stats_results(results: Dict[str, Any]):
    """Display basic statistics results"""

    try:
        stats_results = results.get("results", {})
        analysis_manager = st.session_state.analysis_manager

        for name, basic_stats in stats_results.items():
            st.write(f"**{name} Statistics:**")
            formatted_stats = analysis_manager.format_statistics_for_display(
                basic_stats
            )

            # Display as table
            stats_df = pd.DataFrame([formatted_stats]).T.rename(columns={0: "Value"})
            st.table(stats_df)

    except Exception as e:
        st.error(f"Error displaying basic statistics: {str(e)}")


def display_distribution_analysis_results(results: Dict[str, Any]):
    """Display distribution analysis results"""

    try:
        dist_results = results.get("results", {})
        analysis_manager = st.session_state.analysis_manager

        for name, dist_analysis in dist_results.items():
            st.write(f"**{name} Distribution Analysis:**")
            formatted_dist = analysis_manager.format_distribution_for_display(
                dist_analysis
            )

            # Display as table
            dist_df = pd.DataFrame([formatted_dist]).T.rename(columns={0: "Value"})
            st.table(dist_df)

    except Exception as e:
        st.error(f"Error displaying distribution analysis: {str(e)}")


def display_returns_analysis_results(results: Dict[str, Any]):
    """Display return analysis results"""

    try:
        return_stats = results.get("return_statistics")
        if return_stats:
            st.write("**Return Analysis:**")

            # Create metrics display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean Return", f"{return_stats.mean:.6f}")
                st.metric("Std Deviation", f"{return_stats.std:.6f}")

            with col2:
                st.metric("Skewness", f"{return_stats.skewness:.4f}")
                st.metric("Kurtosis", f"{return_stats.kurtosis:.4f}")

            with col3:
                st.metric("Sharpe Ratio", f"{return_stats.sharpe_ratio:.4f}")
                st.metric("Max Drawdown", f"{return_stats.max_drawdown:.4f}")

    except Exception as e:
        st.error(f"Error displaying return analysis: {str(e)}")


if __name__ == "__main__":
    main()
