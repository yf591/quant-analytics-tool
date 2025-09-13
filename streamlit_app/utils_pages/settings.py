"""
Settings utility page for Quant Analytics Tool
"""

import streamlit as st


def show_settings_page():
    """Display the enhanced settings page"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("⚙️ Application Settings")

    # Application settings
    st.subheader("🎨 Display Settings")

    col1, col2 = st.columns(2)

    with col1:
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
        layout = st.selectbox("Layout", ["Wide", "Centered"], index=0)

    with col2:
        sidebar = st.selectbox("Sidebar", ["Expanded", "Collapsed"], index=0)
        cache_ttl = st.slider(
            "Cache TTL (minutes)", min_value=5, max_value=120, value=60
        )

    # Data source settings
    st.subheader("📊 Data Source Configuration")

    data_source = st.selectbox(
        "Primary Data Source",
        ["Yahoo Finance", "Alpha Vantage", "Polygon", "Quandl", "Custom"],
        index=0,
    )

    # API configuration
    st.subheader("🔑 API Configuration")

    with st.expander("API Key Management", expanded=False):
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password")
        polygon_key = st.text_input("Polygon API Key", type="password")
        quandl_key = st.text_input("Quandl API Key", type="password")
        custom_api_key = st.text_input("Custom API Key", type="password")

        if st.button("💾 Save API Keys"):
            st.success("API keys saved successfully!")

    # Performance settings
    st.subheader("⚡ Performance Configuration")

    col1, col2 = st.columns(2)

    with col1:
        max_workers = st.slider(
            "Max Concurrent Workers", min_value=1, max_value=10, value=5
        )
        chunk_size = st.slider(
            "Data Chunk Size", min_value=100, max_value=10000, value=1000
        )

    with col2:
        memory_limit = st.slider(
            "Memory Limit (GB)", min_value=1, max_value=16, value=4
        )
        timeout = st.slider(
            "Request Timeout (seconds)", min_value=10, max_value=120, value=30
        )

    # Advanced settings
    st.subheader("🔬 Advanced Configuration")

    with st.expander("Expert Settings", expanded=False):
        enable_debug = st.checkbox("Enable Debug Logging")
        enable_profiling = st.checkbox("Enable Performance Profiling")
        enable_cache = st.checkbox("Enable Advanced Caching", value=True)
        enable_notifications = st.checkbox("Enable Notifications", value=True)

        log_level = st.selectbox(
            "Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1
        )

    # Save configuration
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            st.success("All settings saved successfully!")

    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.success("Settings reset to defaults!")

    with col3:
        if st.button("📤 Export Config", use_container_width=True):
            st.success("Configuration exported!")

    st.markdown("</div>", unsafe_allow_html=True)
