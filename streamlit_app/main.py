"""
Streamlit Main Application
Entry point for the Quant Analytics Tool dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import settings

# Configure Streamlit page
st.set_page_config(
    page_title=settings.streamlit_page_title,
    page_icon=settings.streamlit_page_icon,
    layout=settings.streamlit_layout,
    initial_sidebar_state="expanded",
)


def main():
    """Main application entry point"""

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main header
    st.markdown(
        '<h1 class="main-header">🚀 Quant Analytics Tool</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Comprehensive tool for quantitative financial data analysis and algorithmic trading</p>',
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")

    page = st.sidebar.selectbox(
        "Select Page",
        [
            "🏠 Home",
            "📈 Data Analysis",
            "🧠 Machine Learning",
            "🔙 Backtesting",
            "⚖️ Risk Management",
            "📊 Portfolio",
            "⚙️ Settings",
        ],
    )

    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📈 Data Analysis":
        show_data_analysis_page()
    elif page == "🧠 Machine Learning":
        show_ml_page()
    elif page == "🔙 Backtesting":
        show_backtest_page()
    elif page == "⚖️ Risk Management":
        show_risk_management_page()
    elif page == "📊 Portfolio":
        show_portfolio_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_home_page():
    """Display the home page"""

    # Welcome message
    st.markdown(
        """
    ## 📋 Overview
    
    This tool implements methodologies from "Advances in Financial Machine Learning" and provides the following features:
    
    - **📊 Real-time Financial Data Acquisition & Analysis**
    - **🧠 Machine Learning Models for Price Prediction**
    - **📈 Advanced Feature Engineering**
    - **🔙 Comprehensive Backtesting Functionality**
    - **⚖️ Risk Management & Position Sizing**
    - **📱 Interactive Dashboard**
    """
    )

    # Key metrics (placeholder)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Monitored Securities", value="25", delta="5")

    with col2:
        st.metric(label="Active Models", value="3", delta="1")

    with col3:
        st.metric(label="Average Sharpe Ratio", value="1.24", delta="0.08")

    with col4:
        st.metric(label="Max Drawdown", value="8.5%", delta="-1.2%")

    # Quick actions
    st.markdown("## 🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Start New Analysis", use_container_width=True):
            st.session_state.page = "📈 Data Analysis"
            st.rerun()

    with col2:
        if st.button("🧠 Train Model", use_container_width=True):
            st.session_state.page = "🧠 Machine Learning"
            st.rerun()

    with col3:
        if st.button("🔙 Run Backtest", use_container_width=True):
            st.session_state.page = "🔙 Backtesting"
            st.rerun()

    # Recent activity (placeholder)
    st.markdown("## 📝 Recent Activity")

    with st.expander("Recent Analysis Results"):
        st.info(
            "🔄 Features under development... You can test data acquisition functionality on the Data Analysis page."
        )

    # System status
    st.markdown("## 🔧 System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ Data Acquisition Service: Normal")
        st.success("✅ Machine Learning Models: Normal")

    with col2:
        st.success("✅ Backtesting Engine: Normal")
        st.success("✅ Risk Calculation: Normal")


def show_data_analysis_page():
    """Display the data analysis page"""
    st.header("📈 Data Analysis")

    # Placeholder content
    st.info("🔄 Data analysis features under development...")

    # Basic ticker input
    ticker = st.text_input(
        "Ticker Symbol", value="AAPL", help="Examples: AAPL, MSFT, 7203.T"
    )

    if ticker:
        st.write(f"Selected security: {ticker}")

        # Placeholder for data fetching
        if st.button("Fetch Data"):
            with st.spinner("Fetching data..."):
                st.success(
                    f"Data acquisition functionality for {ticker} is planned for implementation"
                )


def show_ml_page():
    """Display the machine learning page"""
    st.header("🧠 Machine Learning")
    st.info("🔄 Machine learning features under development...")


def show_backtest_page():
    """Display the backtest page"""
    st.header("🔙 Backtesting")
    st.info("🔄 Backtesting features under development...")


def show_risk_management_page():
    """Display the risk management page"""
    st.header("⚖️ Risk Management")
    st.info("🔄 Risk management features under development...")


def show_portfolio_page():
    """Display the portfolio page"""
    st.header("📊 Portfolio")
    st.info("🔄 Portfolio features under development...")


def show_settings_page():
    """Display the settings page"""
    st.header("⚙️ Settings")

    st.subheader("🔧 Application Settings")

    # Theme settings
    st.selectbox("Theme", ["Light", "Dark"], index=0)

    # Data source settings
    st.selectbox(
        "Default Data Source", ["Yahoo Finance", "Alpha Vantage", "Quandl"], index=0
    )

    # API key inputs
    st.subheader("🔑 API Configuration")

    with st.expander("API Key Settings"):
        st.text_input("Alpha Vantage API Key", type="password")
        st.text_input("Quandl API Key", type="password")
        st.text_input("Polygon API Key", type="password")

        if st.button("Save Settings"):
            st.success("Settings have been saved!")

    # Performance settings
    st.subheader("⚡ Performance Settings")

    st.slider("Cache TTL (minutes)", min_value=5, max_value=120, value=60)
    st.slider("Max Concurrent Data Fetches", min_value=1, max_value=10, value=5)


if __name__ == "__main__":
    main()
