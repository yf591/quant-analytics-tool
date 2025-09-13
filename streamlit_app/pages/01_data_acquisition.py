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

    # Streamlit components
    from components.charts import create_price_chart, create_correlation_heatmap
    from components.data_display import (
        display_data_metrics,
        display_computation_status,
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

    # Initialize session state using the manager
    data_manager.initialize_session_state(st.session_state)

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

    # Store the manager in session state for access in other functions
    st.session_state.data_manager = data_manager

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
        st.info("� Collect data first to enable analysis")
        return

    # Analysis type selection
    analysis_type = st.selectbox(
        "Analysis Type",
        options=["Single Symbol", "Multi-Symbol Comparison", "Data Quality Dashboard"],
        help="Choose the type of analysis to perform",
    )

    if analysis_type == "Single Symbol":
        single_symbol_analysis()
    elif analysis_type == "Multi-Symbol Comparison":
        multi_symbol_comparison()
    elif analysis_type == "Data Quality Dashboard":
        data_quality_analysis()


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
            symbols_str = symbols_list

        manager_config = {
            "symbols": symbols_str,
            "period": "1y",  # Default period
            "interval": config.get("interval", "1d"),
        }

        # Use the manager to start collection
        success, message = data_manager.start_yahoo_finance_collection(
            manager_config, st.session_state
        )

        if success:
            display_computation_status(f"✅ {message}", 1.0)
            st.success(message)
        else:
            display_computation_status(f"❌ {message}")
            st.error(message)

        st.rerun()

    except Exception as e:
        error_msg = f"Collection failed: {str(e)}"
        display_computation_status(f"❌ {error_msg}")
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
        success, message = data_manager.start_custom_upload_collection(
            [uploaded_file], config, st.session_state
        )

        if success:
            display_computation_status(f"✅ {message}", 1.0)
            st.success(message)
        else:
            display_computation_status(f"❌ {message}")
            st.error(message)

        st.rerun()

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        display_computation_status(f"❌ {error_msg}")
        st.error(error_msg)


def validate_selected_data(
    symbols: List[str], validation_level: str, generate_report: bool
):
    """Validate selected symbols using DataAcquisitionManager"""

    try:
        # Get the data manager from session state
        data_manager = st.session_state.data_manager

        # Use the manager to validate selected data
        success, message = data_manager.validate_selected_data(
            symbols, validation_level, generate_report, st.session_state
        )

        if success:
            display_computation_status(f"✅ {message}", 1.0)
            st.success(message)
        else:
            display_computation_status(f"❌ {message}")
            st.error(message)

        st.rerun()

    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        display_computation_status(f"❌ {error_msg}")
        st.error(error_msg)


def validate_all_data(validation_level: str, generate_report: bool):
    """Validate all cached data using DataAcquisitionManager"""

    # Get the data manager from session state
    data_manager = st.session_state.data_manager

    # Use the manager to validate all data
    success, message = data_manager.validate_all_data(
        validation_level, generate_report, st.session_state
    )

    if success:
        display_computation_status(f"✅ {message}", 1.0)
        st.success(message)
    else:
        display_computation_status(f"❌ {message}")
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

        display_computation_status("💾 Saving all data...", 0.5)

        # Use the manager to execute batch save
        success, message = data_manager.execute_batch_save(st.session_state)

        if success:
            display_computation_status(f"✅ {message}", 1.0)
            st.success(message)
        else:
            display_computation_status(f"❌ {message}")
            st.error(message)

    except Exception as e:
        error_msg = f"Batch save failed: {str(e)}"
        display_computation_status(f"❌ {error_msg}")
        st.error(error_msg)


# Session state is now initialized in main() using DataAcquisitionManager
# This function is kept for backward compatibility but delegates to the manager
def initialize_session_state():
    """Initialize session state for data acquisition (deprecated - use DataAcquisitionManager)"""

    # This function is now handled by DataAcquisitionManager in main()
    # Kept for backward compatibility
    pass


if __name__ == "__main__":
    main()
