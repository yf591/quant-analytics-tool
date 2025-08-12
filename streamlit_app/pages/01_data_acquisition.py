"""
Streamlit Page: Data Acquisition
Week 14 UI Integration - Professional Data Collection Interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Week 2: Data Collection Framework Integration
    from src.data.collectors import YFinanceCollector, DataRequest
    from src.data.validators import DataValidator
    from src.data.storage import SQLiteStorage
    from src.config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Data Acquisition Interface"""

    st.title("📈 Data Acquisition")
    st.markdown("**Professional Financial Data Collection Platform**")

    # Initialize session state
    if "data_cache" not in st.session_state:
        st.session_state.data_cache = {}

    # Professional UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        data_collection_panel()

    with col2:
        data_display_panel()


def data_collection_panel():
    """Data Collection Control Panel"""

    st.subheader("🎯 Collection Parameters")

    # Input parameters
    ticker = st.text_input("Symbol", value="AAPL", help="Stock ticker symbol")

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("Start", value=datetime.now() - timedelta(days=365))
    with col_date2:
        end_date = st.date_input("End", value=datetime.now())

    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

    # Collection Actions
    st.subheader("🚀 Actions")

    if st.button("📥 Collect Data", type="primary", use_container_width=True):
        collect_data(ticker, start_date, end_date, interval)

    if st.button("🔍 Validate Data", use_container_width=True):
        if ticker in st.session_state.data_cache:
            validate_data(ticker)
        else:
            st.warning("No data to validate")

    if st.button("💾 Save Data", use_container_width=True):
        if ticker in st.session_state.data_cache:
            save_data(ticker)
        else:
            st.warning("No data to save")


def data_display_panel():
    """Data Display and Visualization Panel"""

    if not st.session_state.data_cache:
        st.info("📊 Collect data to see results")
        return

    # Data selection
    selected_ticker = st.selectbox(
        "Select Dataset", list(st.session_state.data_cache.keys())
    )

    if selected_ticker:
        display_data_overview(selected_ticker)
        display_data_chart(selected_ticker)


def collect_data(ticker: str, start_date, end_date, interval: str):
    """Collect data using Week 2 YFinanceCollector"""

    try:
        with st.spinner(f"Collecting {ticker} data..."):
            # Use existing Week 2 module
            collector = YFinanceCollector()

            # Create DataRequest object
            request = DataRequest(
                symbol=ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                interval=interval,
            )

            data = collector.fetch_data(request)

            # Store in session
            st.session_state.data_cache[ticker] = {
                "data": data,
                "metadata": {
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "collected_at": datetime.now(),
                },
            }

        st.success(f"✅ Collected {len(data)} records for {ticker}")
        st.rerun()

    except Exception as e:
        st.error(f"Collection failed: {str(e)}")


def validate_data(ticker: str):
    """Validate data using Week 2 DataValidator"""

    try:
        data = st.session_state.data_cache[ticker]["data"]

        with st.spinner("Validating data..."):
            # Use existing Week 2 module
            validator = DataValidator()
            validation_result = validator.validate_ohlcv_data(data)

        # Display results
        st.subheader(f"🔍 Validation Results: {ticker}")

        if validation_result.is_valid:
            st.success("✅ Data validation passed")
        else:
            st.error("❌ Data validation failed")

        # Show validation details
        if validation_result.errors:
            st.error("**Errors:**")
            for error in validation_result.errors:
                st.write(f"❌ {error}")

        if validation_result.warnings:
            st.warning("**Warnings:**")
            for warning in validation_result.warnings:
                st.write(f"⚠️ {warning}")

        # Show statistics if available
        if validation_result.statistics:
            st.info("**Statistics:**")
            st.json(validation_result.statistics)

    except Exception as e:
        st.error(f"Validation failed: {str(e)}")


def save_data(ticker: str):
    """Save data using Week 2 SQLiteStorage"""

    try:
        data = st.session_state.data_cache[ticker]["data"]
        metadata = st.session_state.data_cache[ticker]["metadata"]

        with st.spinner(f"Saving {ticker} data..."):
            # Use existing Week 2 module
            storage = SQLiteStorage()
            success = storage.store_data(
                symbol=ticker, data=data, data_source="yahoo", overwrite=True
            )

            if success:
                st.success(f"✅ Data saved for {ticker}")
            else:
                st.error(f"❌ Failed to save data for {ticker}")

    except Exception as e:
        st.error(f"Save failed: {str(e)}")


def display_data_overview(ticker: str):
    """Display data overview with metrics"""

    cached_data = st.session_state.data_cache[ticker]
    data = cached_data["data"]
    metadata = cached_data["metadata"]

    st.subheader(f"📊 Data Overview: {ticker}")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", len(data))

    with col2:
        st.metric("Columns", len(data.columns))

    with col3:
        if "Close" in data.columns:
            latest_price = data["Close"].iloc[-1]
            st.metric("Latest Price", f"${latest_price:.2f}")

    with col4:
        collection_time = metadata["collected_at"]
        st.metric("Collected", collection_time.strftime("%H:%M"))

    # Data preview
    st.dataframe(data.tail(10), use_container_width=True, height=300)

    # Download option
    csv = data.to_csv()
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{ticker}_{metadata['start_date']}_{metadata['end_date']}.csv",
        mime="text/csv",
    )


def display_data_chart(ticker: str):
    """Display interactive price chart"""

    data = st.session_state.data_cache[ticker]["data"]

    if data.empty or "Close" not in data.columns:
        return

    st.subheader("📈 Price Chart")

    # Create candlestick chart if OHLC data available
    fig = go.Figure()

    if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name=ticker,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["Close"], mode="lines", name=ticker)
        )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
