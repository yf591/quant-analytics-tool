"""
Data Explorer utility page for Quant Analytics Tool
"""

import streamlit as st


def show_data_explorer_page():
    """Display the data explorer utility page"""

    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("🔍 Data Explorer")

    st.info("🔄 This utility page provides interactive data exploration capabilities.")

    # Check for available data
    if "data_cache" not in st.session_state or not st.session_state.data_cache:
        st.warning(
            "⚠️ No data available for exploration. Please acquire data first using the Data Acquisition page."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Data selection
    st.markdown("### 📊 Data Selection")

    available_symbols = list(st.session_state.data_cache.keys())
    selected_symbol = st.selectbox("Select Symbol:", available_symbols)

    if selected_symbol:
        data_info = st.session_state.data_cache[selected_symbol]

        # Display data information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Symbol", selected_symbol)
        with col2:
            if "metadata" in data_info:
                st.metric("Source", data_info["metadata"].get("source", "Unknown"))
        with col3:
            if "data" in data_info:
                st.metric("Records", len(data_info["data"]))

        # Data preview
        st.markdown("### 📈 Data Preview")

        if "data" in data_info:
            data = data_info["data"]

            # Basic statistics
            st.subheader("📊 Summary Statistics")
            st.dataframe(data.describe())

            # Data preview
            st.subheader("🔍 Data Sample")
            st.dataframe(data.head(20))

            # Basic chart
            if len(data.columns) > 1:
                st.subheader("📈 Price Chart")
                if "Close" in data.columns:
                    st.line_chart(data["Close"])
                elif "close" in data.columns:
                    st.line_chart(data["close"])
                else:
                    # Show first numeric column
                    numeric_cols = data.select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 0:
                        st.line_chart(data[numeric_cols[0]])
        else:
            st.warning("No data available for this symbol")

    st.markdown("</div>", unsafe_allow_html=True)
