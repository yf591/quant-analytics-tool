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
    # Week 2: Data Collection Framework Integration
    from src.data.collectors import YFinanceCollector, DataRequest
    from src.data.validators import DataValidator
    from src.data.storage import SQLiteStorage
    from src.config import settings
    
    # Streamlit components
    from components.charts import (
        create_price_chart,
        create_correlation_heatmap
    )
    from components.data_display import (
        display_data_metrics,
        display_computation_status,
        display_alert_message
    )
    from components.forms import (
        create_data_selection_form,
        create_date_range_form
    )
    from components.data_management import (
        create_data_source_selection_form,
        display_collection_status,
        display_validation_results,
        display_storage_management,
        create_data_quality_dashboard,
        display_data_comparison,
        create_batch_operation_interface
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Data Acquisition Interface"""
    
    # Initialize session state
    initialize_session_state()
    
    st.set_page_config(
        page_title="📊 Data Acquisition",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("📊 Data Acquisition & Management")
    st.markdown("### Professional financial data collection with validation and storage")
    
    # Professional workflow tabs
    tabs = st.tabs([
        "📥 Collection",
        "🔍 Validation", 
        "💾 Storage",
        "📊 Analysis",
        "🔄 Batch Operations"
    ])
    
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
        if st.button("📥 Start Collection", type="primary", use_container_width=True):
            if source_config["data_source"] == "Yahoo Finance":
                start_yahoo_finance_collection(source_config)
            elif source_config["data_source"] == "Custom Upload":
                start_custom_upload_collection(source_config)
    
    with col2:
        if st.button("⏹️ Stop Collection", use_container_width=True):
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
        for symbol, cache_data in st.session_state.data_cache.items():
            data = cache_data["data"]
            metadata = cache_data["metadata"]
            
            summary_data.append({
                "Symbol": symbol,
                "Records": len(data),
                "Columns": len(data.columns),
                "Start Date": data.index.min().strftime("%Y-%m-%d") if isinstance(data.index, pd.DatetimeIndex) else "N/A",
                "End Date": data.index.max().strftime("%Y-%m-%d") if isinstance(data.index, pd.DatetimeIndex) else "N/A",
                "Source": metadata.get("source", "Unknown"),
                "Collected": metadata.get("collected_at", datetime.now()).strftime("%H:%M:%S")
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Display simple metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Symbols", len(st.session_state.data_cache))
        
        with col2:
            total_records = sum(len(cache_data["data"]) for cache_data in st.session_state.data_cache.values())
            st.metric("📈 Total Records", f"{total_records:,}")
        
        with col3:
            avg_records = total_records / len(st.session_state.data_cache) if st.session_state.data_cache else 0
            st.metric("📊 Avg Records/Symbol", f"{avg_records:.0f}")
        
        with col4:
            latest_collection = max(
                cache_data["metadata"].get("collected_at", datetime.min) 
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
        help="Choose symbols to validate"
    )
    
    # Validation options
    col1, col2 = st.columns(2)
    
    with col1:
        validation_level = st.selectbox(
            "Validation Level",
            options=["Basic", "Standard", "Strict"],
            index=1,
            help="Level of validation strictness"
        )
    
    with col2:
        generate_report = st.checkbox(
            "Generate Report",
            value=True,
            help="Generate detailed validation report"
        )
    
    # Validation actions
    col_val1, col_val2 = st.columns(2)
    
    with col_val1:
        if st.button("🔍 Validate Selected", type="primary", use_container_width=True):
            validate_selected_data(selected_symbols, validation_level, generate_report)
    
    with col_val2:
        if st.button("🔍 Validate All", use_container_width=True):
            validate_all_data(validation_level, generate_report)
    
    # Display validation results
    if st.session_state.validation_cache:
        for symbol, results in st.session_state.validation_cache.items():
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
        help="Choose the type of analysis to perform"
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
    """Start Yahoo Finance data collection"""
    
    try:
        symbols = config.get("symbols", [])
        if not symbols:
            st.error("No symbols specified")
            return
        
        display_computation_status("🔄 Initializing data collection...", 0.1)
        
        # Initialize collection status
        st.session_state.collection_status = {
            "total_symbols": len(symbols),
            "collected_symbols": 0,
            "failed_symbols": 0,
            "symbol_status": []
        }
        
        # Initialize collector
        collector = YFinanceCollector()
        
        progress_container = st.container()
        
        for i, symbol in enumerate(symbols):
            progress = (i + 1) / len(symbols)
            display_computation_status(f"🔄 Collecting data for {symbol}...", progress)
            
            try:
                # Create DataRequest
                request = DataRequest(
                    symbol=symbol,
                    start_date=config["start_date"].strftime("%Y-%m-%d"),
                    end_date=config["end_date"].strftime("%Y-%m-%d"),
                    interval=config["interval"],
                    auto_adjust=config.get("auto_adjust", True),
                    prepost=config.get("prepost", False)
                )
                
                data = collector.fetch_data(request)
                
                if data is not None and not data.empty:
                    # Store in session
                    st.session_state.data_cache[symbol] = {
                        "data": data,
                        "metadata": {
                            "ticker": symbol,
                            "start_date": config["start_date"],
                            "end_date": config["end_date"],
                            "interval": config["interval"],
                            "collected_at": datetime.now(),
                            "source": "yahoo_finance"
                        }
                    }
                    
                    st.session_state.collection_status["collected_symbols"] += 1
                    st.session_state.collection_status["symbol_status"].append({
                        "Symbol": symbol,
                        "Status": "✅ Success",
                        "Records": len(data),
                        "Date Range": f"{data.index[0].date()} to {data.index[-1].date()}"
                    })
                    
                else:
                    raise ValueError("No data returned")
                
            except Exception as e:
                st.session_state.collection_status["failed_symbols"] += 1
                st.session_state.collection_status["symbol_status"].append({
                    "Symbol": symbol,
                    "Status": f"❌ Failed: {str(e)[:50]}",
                    "Records": 0,
                    "Date Range": "N/A"
                })
        
        # Final status
        collected = st.session_state.collection_status["collected_symbols"]
        total = st.session_state.collection_status["total_symbols"]
        
        display_computation_status(
            f"✅ Collection completed! {collected}/{total} symbols successful",
            1.0
        )
        
        st.rerun()
        
    except Exception as e:
        display_computation_status(f"❌ Collection failed: {str(e)}")


def start_custom_upload_collection(config: Dict[str, Any]):
    """Start custom file upload collection"""
    
    try:
        uploaded_file = config.get("uploaded_file")
        if not uploaded_file:
            st.warning("Please upload a file")
            return
        
        display_computation_status("🔄 Processing uploaded file...", 0.5)
        
        # Read uploaded file
        data = pd.read_csv(
            uploaded_file,
            index_col=config.get("index_col", 0),
            parse_dates=config.get("parse_dates", True),
            header=config.get("header_row", 0)
        )
        
        # Extract symbol from filename or use default
        symbol = uploaded_file.name.split('.')[0].upper()
        
        # Store in session
        st.session_state.data_cache[symbol] = {
            "data": data,
            "metadata": {
                "ticker": symbol,
                "start_date": data.index[0] if len(data) > 0 else None,
                "end_date": data.index[-1] if len(data) > 0 else None,
                "interval": "Unknown",
                "collected_at": datetime.now(),
                "source": "custom_upload",
                "filename": uploaded_file.name
            }
        }
        
        display_computation_status(
            f"✅ File uploaded successfully! {len(data)} records for {symbol}",
            1.0
        )
        
        st.rerun()
        
    except Exception as e:
        display_computation_status(f"❌ Upload failed: {str(e)}")


def validate_selected_data(symbols: List[str], validation_level: str, generate_report: bool):
    """Validate selected symbols"""
    
    try:
        from src.data.validators import ValidationLevel
        
        # Map validation level
        level_map = {
            "Basic": ValidationLevel.BASIC,
            "Standard": ValidationLevel.STANDARD,
            "Strict": ValidationLevel.STRICT
        }
        
        validator = DataValidator(validation_level=level_map[validation_level])
        
        for symbol in symbols:
            display_computation_status(f"🔍 Validating {symbol}...", 0.5)
            
            data = st.session_state.data_cache[symbol]["data"]
            validation_result = validator.validate_ohlcv_data(data)
            
            # Store validation results
            st.session_state.validation_cache[symbol] = {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "statistics": validation_result.statistics,
                "quality_score": 1.0 - (len(validation_result.errors) / max(1, len(validation_result.errors) + len(validation_result.warnings) + 10))
            }
        
        display_computation_status(f"✅ Validation completed for {len(symbols)} symbols", 1.0)
        st.rerun()
        
    except Exception as e:
        display_computation_status(f"❌ Validation failed: {str(e)}")


def validate_all_data(validation_level: str, generate_report: bool):
    """Validate all cached data"""
    
    all_symbols = list(st.session_state.data_cache.keys())
    validate_selected_data(all_symbols, validation_level, generate_report)


def single_symbol_analysis():
    """Single symbol analysis"""
    
    symbol = st.selectbox(
        "Select Symbol",
        options=list(st.session_state.data_cache.keys())
    )
    
    if symbol:
        data = st.session_state.data_cache[symbol]["data"]
        
        # Create data quality dashboard
        create_data_quality_dashboard(data, symbol)
        
        # Price chart
        if "Close" in data.columns:
            st.subheader("📈 Price Chart")
            chart = create_price_chart(data, height=500)
            st.plotly_chart(chart, use_container_width=True)


def multi_symbol_comparison():
    """Multi-symbol comparison analysis"""
    
    data_dict = {
        symbol: cache_data["data"] 
        for symbol, cache_data in st.session_state.data_cache.items()
    }
    
    display_data_comparison(data_dict)


def data_quality_analysis():
    """Data quality analysis for all symbols"""
    
    st.subheader("📊 Overall Data Quality Analysis")
    
    quality_data = []
    for symbol, cache_data in st.session_state.data_cache.items():
        data = cache_data["data"]
        
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        
        quality_data.append({
            "Symbol": symbol,
            "Records": len(data),
            "Columns": len(data.columns),
            "Missing %": f"{missing_pct:.1f}%",
            "Date Range": f"{(data.index.max() - data.index.min()).days} days",
            "Latest Date": data.index.max().strftime("%Y-%m-%d")
        })
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True)


def execute_batch_collection(config: Dict[str, Any]):
    """Execute batch collection"""
    
    symbols = config.get("symbol_list", [])
    if not symbols:
        st.error("No symbols specified for batch collection")
        return
    
    # Use Yahoo Finance configuration with batch symbols
    yahoo_config = {
        "symbols": symbols,
        "start_date": datetime.now() - timedelta(days=365),
        "end_date": datetime.now(),
        "interval": config.get("interval", "1d"),
        "auto_adjust": True,
        "prepost": False
    }
    
    start_yahoo_finance_collection(yahoo_config)


def execute_batch_validation():
    """Execute batch validation"""
    
    if not st.session_state.data_cache:
        st.warning("No data to validate")
        return
    
    validate_all_data("Standard", True)


def execute_batch_save():
    """Execute batch save operation"""
    
    try:
        if not st.session_state.data_cache:
            st.warning("No data to save")
            return
        
        display_computation_status("💾 Saving all data...", 0.5)
        
        storage = SQLiteStorage()
        
        saved_count = 0
        for symbol, cache_data in st.session_state.data_cache.items():
            try:
                success = storage.store_data(
                    symbol=symbol,
                    data=cache_data["data"],
                    data_source="yahoo",
                    overwrite=True
                )
                if success:
                    saved_count += 1
            except Exception as e:
                st.error(f"Failed to save {symbol}: {str(e)}")
        
        display_computation_status(
            f"✅ Saved {saved_count}/{len(st.session_state.data_cache)} symbols",
            1.0
        )
        
    except Exception as e:
        display_computation_status(f"❌ Batch save failed: {str(e)}")


# Initialize session state
def initialize_session_state():
    """Initialize session state for data acquisition"""
    
    if "data_cache" not in st.session_state:
        st.session_state.data_cache = {}
    
    if "collection_status" not in st.session_state:
        st.session_state.collection_status = {}
    
    if "validation_cache" not in st.session_state:
        st.session_state.validation_cache = {}
    
    if "storage_status" not in st.session_state:
        st.session_state.storage_status = {}


if __name__ == "__main__":
    main()
