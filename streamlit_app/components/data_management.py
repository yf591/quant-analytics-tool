"""
Data Management Component Module

This module provides specialized components for data acquisition workflow.
Includes data source selection, collection status, validation results, and storage management.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_data_source_selection_form() -> Dict[str, Any]:
    """
    Create a comprehensive data source selection form.

    Returns:
        Dictionary of data source parameters
    """
    try:
        st.subheader("🎯 Data Source Configuration")

        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            options=["Yahoo Finance", "Alpha Vantage", "Custom Upload"],
            index=0,
            help="Select the data source for financial data collection",
        )

        config = {"data_source": data_source}

        if data_source == "Yahoo Finance":
            config.update(_create_yahoo_finance_form())
        elif data_source == "Alpha Vantage":
            config.update(_create_alpha_vantage_form())
        elif data_source == "Custom Upload":
            config.update(_create_custom_upload_form())

        return config

    except Exception as e:
        st.error(f"Error creating data source selection form: {str(e)}")
        return {}


def _create_yahoo_finance_form() -> Dict[str, Any]:
    """Create Yahoo Finance specific configuration form"""

    st.write("**Yahoo Finance Parameters:**")

    col1, col2 = st.columns(2)

    with col1:
        symbols = st.text_area(
            "Symbols (one per line)",
            value="AAPL\nMSFT\nGOOGL",
            height=100,
            help="Enter stock symbols, one per line",
        )

        interval = st.selectbox(
            "Data Interval",
            options=["1m", "2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo"],
            index=6,  # Default to 1d
            help="Data frequency interval",
        )

    with col2:
        # Date range selection
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Data collection start date",
            )
        with col_date2:
            end_date = st.date_input(
                "End Date", value=datetime.now(), help="Data collection end date"
            )

        # Advanced options
        with st.expander("Advanced Options"):
            auto_adjust = st.checkbox(
                "Auto Adjust", value=True, help="Adjust for splits and dividends"
            )
            prepost = st.checkbox(
                "Pre/Post Market", value=False, help="Include pre and post market data"
            )
            actions = st.checkbox(
                "Include Actions",
                value=False,
                help="Include dividends and stock splits",
            )

    # Parse symbols
    symbol_list = [s.strip().upper() for s in symbols.split("\n") if s.strip()]

    return {
        "symbols": symbol_list,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "prepost": prepost,
        "actions": actions,
    }


def _create_alpha_vantage_form() -> Dict[str, Any]:
    """Create Alpha Vantage specific configuration form"""

    st.write("**Alpha Vantage Parameters:**")
    st.info(
        "🔑 Alpha Vantage integration coming soon. Please use Yahoo Finance for now."
    )

    return {"enabled": False}


def _create_custom_upload_form() -> Dict[str, Any]:
    """Create custom file upload form"""

    st.write("**Custom Data Upload:**")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload a CSV file with financial data (OHLCV format)",
    )

    if uploaded_file:
        # File format options
        col1, col2 = st.columns(2)

        with col1:
            date_column = st.text_input("Date Column", value="Date")
            parse_dates = st.checkbox("Parse Dates", value=True)

        with col2:
            index_col = st.text_input("Index Column", value="Date")
            header_row = st.number_input("Header Row", value=0, min_value=0)

        return {
            "uploaded_file": uploaded_file,
            "date_column": date_column,
            "parse_dates": parse_dates,
            "index_col": index_col,
            "header_row": header_row,
        }

    return {"uploaded_file": None}


def display_collection_status(status_data: Dict[str, Any]) -> None:
    """
    Display data collection status and progress.

    Args:
        status_data: Dictionary containing status information
    """
    try:
        st.subheader("📊 Collection Status")

        # Overall status metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_symbols = status_data.get("total_symbols", 0)
            st.metric("Total Symbols", total_symbols)

        with col2:
            collected_symbols = status_data.get("collected_symbols", 0)
            st.metric("Collected", collected_symbols)

        with col3:
            failed_symbols = status_data.get("failed_symbols", 0)
            st.metric(
                "Failed", failed_symbols, delta="❌" if failed_symbols > 0 else "✅"
            )

        with col4:
            success_rate = (
                (collected_symbols / total_symbols * 100) if total_symbols > 0 else 0
            )
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # Progress bar
        if total_symbols > 0:
            progress = collected_symbols / total_symbols
            st.progress(progress)
            st.write(f"Progress: {collected_symbols}/{total_symbols} symbols completed")

        # Detailed status
        if "symbol_status" in status_data:
            st.write("**Symbol Status:**")
            status_df = pd.DataFrame(status_data["symbol_status"])
            st.dataframe(status_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying collection status: {str(e)}")


def display_validation_results(validation_results: Dict[str, Any]) -> None:
    """
    Display comprehensive data validation results.

    Args:
        validation_results: Dictionary containing validation results
    """
    try:
        st.subheader("🔍 Data Validation Results")

        # Overall validation status
        is_valid = validation_results.get("is_valid", False)

        if is_valid:
            st.success("✅ All data passed validation")
        else:
            st.error("❌ Data validation issues detected")

        # Validation metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            error_count = len(validation_results.get("errors", []))
            st.metric("Errors", error_count, delta="❌" if error_count > 0 else "✅")

        with col2:
            warning_count = len(validation_results.get("warnings", []))
            st.metric(
                "Warnings", warning_count, delta="⚠️" if warning_count > 0 else "✅"
            )

        with col3:
            data_quality = validation_results.get("quality_score", 0.0)
            st.metric("Quality Score", f"{data_quality:.2f}")

        # Detailed validation results
        errors = validation_results.get("errors", [])
        warnings = validation_results.get("warnings", [])
        statistics = validation_results.get("statistics", {})

        # Errors
        if errors:
            with st.expander("❌ Validation Errors", expanded=True):
                for error in errors:
                    st.error(f"• {error}")

        # Warnings
        if warnings:
            with st.expander("⚠️ Validation Warnings"):
                for warning in warnings:
                    st.warning(f"• {warning}")

        # Statistics
        if statistics:
            with st.expander("📊 Data Statistics"):
                stats_df = pd.DataFrame([statistics]).T
                stats_df.columns = ["Value"]
                st.dataframe(stats_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying validation results: {str(e)}")


def display_storage_management() -> None:
    """Display storage management interface"""

    try:
        st.subheader("💾 Storage Management")

        # Storage options
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Storage Options:**")
            storage_format = st.selectbox(
                "Format", options=["SQLite", "CSV", "Parquet", "HDF5"], index=0
            )

            overwrite_existing = st.checkbox(
                "Overwrite Existing",
                value=False,
                help="Overwrite existing data for the same symbol",
            )

        with col2:
            st.write("**Storage Location:**")

            if storage_format == "SQLite":
                db_path = st.text_input("Database Path", value="data/financial_data.db")
                st.info(f"Data will be stored in: {db_path}")
            else:
                data_dir = st.text_input("Data Directory", value="data/")
                st.info(f"Files will be stored in: {data_dir}")

        # Storage actions
        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            if st.button("💾 Save All Data", use_container_width=True):
                st.info("Saving all collected data...")

        with col_action2:
            if st.button("📂 Load Existing Data", use_container_width=True):
                st.info("Loading existing data...")

        with col_action3:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.session_state.data_cache = {}
                st.success("Cache cleared!")
                st.rerun()

        # Display existing data
        if "data_cache" in st.session_state and st.session_state.data_cache:
            st.write("**Cached Data:**")
            cache_info = []
            for symbol, data_info in st.session_state.data_cache.items():
                cache_info.append(
                    {
                        "Symbol": symbol,
                        "Records": len(data_info["data"]),
                        "Columns": len(data_info["data"].columns),
                        "Last Updated": data_info["metadata"]["collected_at"].strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    }
                )

            cache_df = pd.DataFrame(cache_info)
            st.dataframe(cache_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying storage management: {str(e)}")


def create_data_quality_dashboard(data: pd.DataFrame, symbol: str) -> None:
    """
    Create a comprehensive data quality dashboard.

    Args:
        data: DataFrame to analyze
        symbol: Symbol name for the data
    """
    try:
        st.subheader(f"📈 Data Quality Dashboard: {symbol}")

        # Data overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(data))

        with col2:
            missing_pct = (
                data.isnull().sum().sum() / (len(data) * len(data.columns))
            ) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")

        with col3:
            if "Close" in data.columns:
                price_range = data["Close"].max() - data["Close"].min()
                st.metric("Price Range", f"${price_range:.2f}")

        with col4:
            date_range = (data.index.max() - data.index.min()).days
            st.metric("Date Range", f"{date_range} days")

        # Data quality charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Missing data heatmap
            if data.isnull().sum().sum() > 0:
                st.write("**Missing Data Pattern:**")
                missing_data = data.isnull().sum()
                fig_missing = go.Figure(
                    data=[go.Bar(x=missing_data.index, y=missing_data.values)]
                )
                fig_missing.update_layout(
                    title="Missing Values by Column",
                    xaxis_title="Column",
                    yaxis_title="Missing Count",
                    height=300,
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing data detected")

        with col_chart2:
            # Data distribution
            if "Close" in data.columns:
                st.write("**Price Distribution:**")
                fig_dist = go.Figure(data=[go.Histogram(x=data["Close"], nbinsx=30)])
                fig_dist.update_layout(
                    title="Price Distribution",
                    xaxis_title="Price",
                    yaxis_title="Frequency",
                    height=300,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        # Data anomalies detection
        if "Close" in data.columns:
            st.write("**Data Anomalies:**")

            # Price gaps
            price_changes = data["Close"].pct_change().abs()
            large_changes = price_changes > 0.1  # 10% threshold

            col_anom1, col_anom2 = st.columns(2)

            with col_anom1:
                st.metric("Large Price Changes (>10%)", large_changes.sum())

            with col_anom2:
                if large_changes.any():
                    max_change = price_changes.max()
                    st.metric("Max Price Change", f"{max_change:.1%}")

        # Recent data preview
        st.write("**Recent Data Preview:**")
        st.dataframe(data.tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"Error creating data quality dashboard: {str(e)}")


def display_data_comparison(data_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Display comparison between multiple datasets.

    Args:
        data_dict: Dictionary of symbol -> DataFrame
    """
    try:
        if len(data_dict) < 2:
            st.info("Collect data for multiple symbols to enable comparison")
            return

        st.subheader("📊 Multi-Symbol Data Comparison")

        # Select symbols for comparison
        selected_symbols = st.multiselect(
            "Select Symbols to Compare",
            options=list(data_dict.keys()),
            default=list(data_dict.keys())[:3],  # Limit to first 3 by default
            help="Choose symbols to compare",
        )

        if len(selected_symbols) < 2:
            st.warning("Please select at least 2 symbols for comparison")
            return

        # Comparison metrics
        comparison_data = []
        for symbol in selected_symbols:
            data = data_dict[symbol]
            if "Close" in data.columns:
                comparison_data.append(
                    {
                        "Symbol": symbol,
                        "Records": len(data),
                        "Latest Price": f"${data['Close'].iloc[-1]:.2f}",
                        "Min Price": f"${data['Close'].min():.2f}",
                        "Max Price": f"${data['Close'].max():.2f}",
                        "Volatility": f"{data['Close'].std():.2f}",
                        "Date Range": f"{(data.index.max() - data.index.min()).days} days",
                    }
                )

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Price comparison chart
            st.write("**Price Comparison Chart:**")
            fig_comparison = go.Figure()

            for symbol in selected_symbols:
                data = data_dict[symbol]
                if "Close" in data.columns:
                    # Normalize prices to start at 100 for comparison
                    normalized_prices = (data["Close"] / data["Close"].iloc[0]) * 100

                    fig_comparison.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=normalized_prices,
                            mode="lines",
                            name=symbol,
                            line=dict(width=2),
                        )
                    )

            fig_comparison.update_layout(
                title="Normalized Price Comparison (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                height=400,
                template="plotly_white",
            )

            st.plotly_chart(fig_comparison, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying data comparison: {str(e)}")


def create_batch_operation_interface() -> Dict[str, Any]:
    """Create interface for batch data operations"""

    try:
        st.subheader("🔄 Batch Operations")

        # Batch collection setup
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Batch Collection:**")

            # Predefined symbol lists
            preset_lists = {
                "S&P 500 Top 10": [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "TSLA",
                    "META",
                    "NVDA",
                    "BRK-B",
                    "JPM",
                    "JNJ",
                ],
                "Tech Giants": [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "META",
                    "NVDA",
                    "NFLX",
                    "ADBE",
                ],
                "Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
                "Custom": [],
            }

            selected_preset = st.selectbox(
                "Symbol List", options=list(preset_lists.keys())
            )

            if selected_preset == "Custom":
                custom_symbols = st.text_area(
                    "Custom Symbols (one per line)",
                    height=100,
                    placeholder="AAPL\nMSFT\nGOOGL",
                )
                symbol_list = [
                    s.strip().upper() for s in custom_symbols.split("\n") if s.strip()
                ]
            else:
                symbol_list = preset_lists[selected_preset]
                st.info(
                    f"Selected {len(symbol_list)} symbols: {', '.join(symbol_list[:5])}{', ...' if len(symbol_list) > 5 else ''}"
                )

        with col2:
            st.write("**Batch Parameters:**")

            batch_interval = st.selectbox(
                "Data Interval", options=["1d", "1h", "30m", "15m"], index=0
            )

            delay_between_requests = st.slider(
                "Delay Between Requests (seconds)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Delay to respect API rate limits",
            )

            max_retries = st.number_input(
                "Max Retries per Symbol", min_value=1, max_value=5, value=3
            )

        # Operation buttons
        col_op1, col_op2, col_op3 = st.columns(3)

        with col_op1:
            collect_batch = st.button(
                "📥 Collect Batch", type="primary", use_container_width=True
            )

        with col_op2:
            validate_batch = st.button("🔍 Validate All", use_container_width=True)

        with col_op3:
            save_batch = st.button("💾 Save All", use_container_width=True)

        return {
            "symbol_list": symbol_list,
            "interval": batch_interval,
            "delay": delay_between_requests,
            "max_retries": max_retries,
            "collect_batch": collect_batch,
            "validate_batch": validate_batch,
            "save_batch": save_batch,
        }

    except Exception as e:
        st.error(f"Error creating batch operation interface: {str(e)}")
        return {}
