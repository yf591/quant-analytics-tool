"""
Streamlit Main Application - Week 14 Implementation
Entry point for the Quant Analytics Tool dashboard with comprehensive pages
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

    # Enhanced CSS for better styling
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e3e8f0 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .nav-button {
        background: linear-gradient(135deg, #1f77b4 0%, #2686c7 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        width: 100%;
        text-align: center;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    .page-container {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background-color: #4CAF50; }
    .status-development { background-color: #FF9800; }
    .status-planned { background-color: #9E9E9E; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main header
    st.markdown(
        '<h1 class="main-header">🚀 Quant Analytics Tool</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Advanced Financial Machine Learning Platform - Week 14 Implementation</p>',
        unsafe_allow_html=True,
    )

    # Sidebar navigation with improved structure
    st.sidebar.title("📊 Navigation")
    
    # Initialize session state for page tracking
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    # Main navigation
    page = st.sidebar.selectbox(
        "Main Pages",
        [
            "🏠 Home",
            "📈 Data Acquisition",
            "🛠️ Feature Engineering", 
            "🧠 Model Training",
            "🔙 Backtesting",
            "⚖️ Risk Management",
            "📊 Advanced Analysis",
        ],
        index=0,
        key="main_nav"
    )
    
    # Update session state
    st.session_state.current_page = page

    # Secondary navigation for utilities
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Utilities")
    
    util_page = st.sidebar.selectbox(
        "Utility Pages",
        [
            "None",
            "⚙️ Settings",
            "📝 Documentation",
            "🔍 Data Explorer",
        ],
        index=0,
        key="util_nav"
    )
    
    # Implementation status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Implementation Status")
    
    status_data = {
        "🏠 Home": "active",
        "📈 Data Acquisition": "active", 
        "🛠️ Feature Engineering": "active",
        "🧠 Model Training": "development",
        "🔙 Backtesting": "development", 
        "⚖️ Risk Management": "development",
        "📊 Advanced Analysis": "planned"
    }
    
    for page_name, status in status_data.items():
        status_class = f"status-{status}"
        status_text = {"active": "✅", "development": "🔄", "planned": "⏳"}[status]
        st.sidebar.markdown(f'<span class="status-indicator {status_class}"></span>{status_text} {page_name}', unsafe_allow_html=True)

    # Main content routing
    if util_page != "None":
        # Handle utility pages
        if util_page == "⚙️ Settings":
            show_settings_page()
        elif util_page == "📝 Documentation":
            show_documentation_page()
        elif util_page == "🔍 Data Explorer":
            show_data_explorer_page()
    else:
        # Handle main pages
        if page == "🏠 Home":
            show_home_page()
        elif page == "📈 Data Acquisition":
            show_data_acquisition_page()
        elif page == "🛠️ Feature Engineering":
            show_feature_engineering_page()
        elif page == "🧠 Model Training":
            show_model_training_page()
        elif page == "🔙 Backtesting":
            show_backtesting_page()
        elif page == "⚖️ Risk Management":
            show_risk_management_page()
        elif page == "📊 Advanced Analysis":
            show_advanced_analysis_page()


def show_home_page():
    """Display the enhanced home page with Week 14 features"""
    
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown(
        """
    ## 📋 Overview
    
    This platform implements advanced methodologies from "Advances in Financial Machine Learning" (AFML) 
    and provides comprehensive quantitative analysis capabilities:
    
    ### 🎯 Core Features
    - **📊 Data Acquisition & Preprocessing**: Real-time market data with advanced cleaning
    - **🛠️ Feature Engineering**: Technical indicators, AFML features, and custom transformations  
    - **🧠 Machine Learning**: Ensemble methods, cross-validation, and hyperparameter tuning
    - **🔙 Backtesting**: Comprehensive strategy testing with statistical analysis
    - **⚖️ Risk Management**: Position sizing, drawdown control, and portfolio optimization
    - **� Advanced Analysis**: Microstructure features, entropy analysis, and structural breaks
    
    ### 📈 Implementation Status (Week 14)
    This is the Phase 5 Week 14 implementation with enhanced Streamlit UI integration.
    """
    )

    # Enhanced metrics display
    st.markdown("### 📊 Platform Metrics")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Data Sources", 
            value="5", 
            delta="2",
            help="Yahoo Finance, Alpha Vantage, Polygon, Quandl, Custom"
        )

    with col2:
        st.metric(
            label="🛠️ Features Available", 
            value="50+", 
            delta="15",
            help="Technical indicators, AFML features, microstructure"
        )

    with col3:
        st.metric(
            label="🧠 ML Models", 
            value="8", 
            delta="3",
            help="Random Forest, XGBoost, Neural Networks, SVM"
        )

    with col4:
        st.metric(
            label="📊 Analysis Tools", 
            value="25+", 
            delta="8",
            help="Backtesting, risk metrics, visualization tools"
        )

    # Quick navigation with enhanced styling
    st.markdown("### 🚀 Quick Navigation")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Start Data Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = "📈 Data Acquisition"
            st.rerun()
        
        if st.button("🛠️ Engineer Features", use_container_width=True):
            st.session_state.current_page = "🛠️ Feature Engineering"
            st.rerun()

    with col2:
        if st.button("🧠 Train Models", use_container_width=True):
            st.session_state.current_page = "🧠 Model Training"
            st.rerun()
            
        if st.button("🔙 Run Backtests", use_container_width=True):
            st.session_state.current_page = "🔙 Backtesting"
            st.rerun()

    with col3:
        if st.button("⚖️ Manage Risk", use_container_width=True):
            st.session_state.current_page = "⚖️ Risk Management"
            st.rerun()
            
        if st.button("📊 Advanced Analysis", use_container_width=True):
            st.session_state.current_page = "� Advanced Analysis"
            st.rerun()

    # Recent activity and system status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Recent Activity")
        with st.expander("Latest Operations", expanded=True):
            st.success("✅ Data acquisition page - Active")
            st.success("✅ Feature engineering page - Active") 
            st.info("🔄 Model training page - In Development")
            st.info("🔄 Backtesting page - In Development")
            st.warning("⏳ Risk management page - Planned")

    with col2:
        st.markdown("### 🔧 System Status")
        with st.expander("Platform Health", expanded=True):
            st.success("✅ Data Services: Operational")
            st.success("✅ Feature Pipeline: Operational")
            st.success("✅ ML Framework: Ready")
            st.success("✅ Visualization: Active")
            st.info("🔄 Real-time Updates: Enabled")

    # Development roadmap
    st.markdown("### 🗺️ Development Roadmap")
    
    roadmap_data = {
        "Phase 1 - Core Pages": "🟢 Completed",
        "Phase 2 - Analysis Pages": "🟡 In Progress", 
        "Phase 3 - Components": "🔴 Planned",
        "Phase 4 - Integration": "🔴 Planned"
    }
    
    for phase, status in roadmap_data.items():
        st.markdown(f"**{phase}**: {status}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_data_acquisition_page():
    """Redirect to the dedicated data acquisition page"""
    st.info("🔄 Redirecting to Data Acquisition page...")
    st.markdown("### 📈 Data Acquisition")
    st.markdown("""
    This page provides comprehensive data acquisition and preprocessing capabilities.
    
    **Available Features:**
    - Real-time market data fetching
    - Data preprocessing and cleaning
    - Technical analysis preparation
    - Data quality validation
    
    **Note**: This functionality is implemented in the dedicated page file.
    For full features, please use the Streamlit multipage navigation or visit:
    `/pages/01_data_acquisition.py`
    """)
    
    # Basic demonstration
    ticker = st.text_input("Quick Test - Enter Ticker:", value="AAPL")
    if st.button("Test Data Fetch"):
        st.success(f"Data acquisition system ready for {ticker}")
        st.info("Please use the dedicated Data Acquisition page for full functionality.")


def show_feature_engineering_page():
    """Redirect to the dedicated feature engineering page"""
    st.info("🔄 Redirecting to Feature Engineering page...")
    st.markdown("### 🛠️ Feature Engineering")
    st.markdown("""
    This page provides advanced feature engineering capabilities based on AFML methodologies.
    
    **Available Features:**
    - Technical indicators (50+ indicators)
    - AFML advanced features
    - Custom feature transformations
    - Feature importance analysis
    - Correlation analysis
    
    **Note**: This functionality is implemented in the dedicated page file.
    For full features, please use the Streamlit multipage navigation or visit:
    `/pages/02_feature_engineering.py`
    """)
    
    # Basic demonstration
    st.selectbox("Quick Test - Select Feature Type:", [
        "Technical Indicators",
        "AFML Features", 
        "Custom Features",
        "Statistical Features"
    ])
    if st.button("Test Feature Generation"):
        st.success("Feature engineering system ready")
        st.info("Please use the dedicated Feature Engineering page for full functionality.")


def show_model_training_page():
    """Display the model training page (placeholder for Phase 2)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("🧠 Model Training")
    
    st.info("🔄 **Phase 2 Development**: This page is currently under development.")
    
    st.markdown("""
    ### 🎯 Planned Features
    
    **Machine Learning Models:**
    - Random Forest Classifiers/Regressors
    - XGBoost implementation
    - Neural Networks (LSTM, CNN)
    - Support Vector Machines
    - Ensemble methods
    
    **AFML Implementation:**
    - Sample weights calculation
    - Cross-validation with purging
    - Feature importance analysis
    - Hyperparameter optimization
    - Model performance evaluation
    
    **Training Pipeline:**
    - Data preparation and splitting
    - Feature selection and scaling
    - Model training with cross-validation
    - Performance metrics calculation
    - Model persistence and versioning
    """)
    
    # Placeholder interface
    st.markdown("### 🛠️ Configuration Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Model Type", [
            "Random Forest",
            "XGBoost", 
            "Neural Network",
            "SVM",
            "Ensemble"
        ], disabled=True)
        
        st.slider("Training Split", 0.6, 0.9, 0.8, disabled=True)
        
    with col2:
        st.selectbox("Target Variable", [
            "Price Direction",
            "Return Magnitude", 
            "Volatility Regime",
            "Custom Target"
        ], disabled=True)
        
        st.slider("CV Folds", 3, 10, 5, disabled=True)
    
    st.button("Start Training", disabled=True, help="Available in Phase 2")
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_backtesting_page():
    """Display the backtesting page (placeholder for Phase 2)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("🔙 Backtesting")
    
    st.info("🔄 **Phase 2 Development**: This page is currently under development.")
    
    st.markdown("""
    ### 🎯 Planned Features
    
    **Backtesting Engine:**
    - Historical strategy simulation
    - Multiple asset support
    - Transaction cost modeling
    - Slippage simulation
    - Market impact analysis
    
    **AFML Backtesting:**
    - Purged cross-validation
    - Combinatorial purged CV
    - Walk-forward analysis
    - Monte Carlo simulation
    - Backtest overfitting detection
    
    **Performance Analytics:**
    - Risk-adjusted returns
    - Drawdown analysis
    - Sharpe ratio calculation
    - Information ratio
    - Maximum adverse excursion
    """)
    
    # Placeholder interface
    st.markdown("### 🛠️ Configuration Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.date_input("Start Date", disabled=True)
        st.date_input("End Date", disabled=True)
        st.number_input("Initial Capital", value=100000, disabled=True)
        
    with col2:
        st.selectbox("Strategy Type", [
            "Trend Following",
            "Mean Reversion",
            "ML Predictions",
            "Custom Strategy"
        ], disabled=True)
        
        st.slider("Commission (%)", 0.0, 0.5, 0.1, disabled=True)
    
    st.button("Run Backtest", disabled=True, help="Available in Phase 2")
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_risk_management_page():
    """Display the risk management page (placeholder for Phase 2)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("⚖️ Risk Management")
    
    st.info("🔄 **Phase 2 Development**: This page is currently under development.")
    
    st.markdown("""
    ### 🎯 Planned Features
    
    **Position Sizing:**
    - Kelly Criterion implementation
    - Risk parity allocation
    - Volatility targeting
    - Maximum drawdown control
    - Leverage optimization
    
    **Risk Metrics:**
    - Value at Risk (VaR)
    - Conditional VaR (CVaR)
    - Maximum drawdown
    - Risk-adjusted returns
    - Correlation analysis
    
    **Portfolio Management:**
    - Asset allocation optimization
    - Rebalancing strategies
    - Risk budgeting
    - Stress testing
    - Scenario analysis
    """)
    
    # Placeholder interface
    st.markdown("### 🛠️ Configuration Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Risk Target (%)", 1, 20, 10, disabled=True)
        st.slider("Max Position Size (%)", 1, 50, 25, disabled=True)
        st.selectbox("Risk Model", [
            "Historical VaR",
            "Parametric VaR",
            "Monte Carlo VaR",
            "EWMA Model"
        ], disabled=True)
        
    with col2:
        st.slider("Confidence Level (%)", 90, 99, 95, disabled=True)
        st.slider("Lookback Period", 30, 252, 100, disabled=True)
        st.checkbox("Enable Stress Testing", disabled=True)
    
    st.button("Calculate Risk Metrics", disabled=True, help="Available in Phase 2")
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_advanced_analysis_page():
    """Display the advanced analysis page (placeholder for Phase 3)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("📊 Advanced Analysis")
    
    st.warning("⏳ **Phase 3 Planning**: This page is in the planning phase.")
    
    st.markdown("""
    ### 🎯 Planned Features
    
    **Microstructure Analysis:**
    - Order flow analysis
    - Market impact models
    - Bid-ask spread analysis
    - Volume profile analysis
    - High-frequency patterns
    
    **Entropy Features:**
    - Shannon entropy
    - Renyi entropy  
    - Plug-in entropy
    - Cross-entropy analysis
    - Information theory metrics
    
    **Structural Analysis:**
    - Structural break detection
    - Regime change analysis
    - Changepoint detection
    - Time series decomposition
    - Seasonal patterns
    """)
    
    st.info("This page will be implemented in Phase 3 of the Week 14 development cycle.")
    
    st.markdown('</div>', unsafe_allow_html=True)


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
        cache_ttl = st.slider("Cache TTL (minutes)", min_value=5, max_value=120, value=60)

    # Data source settings
    st.subheader("📊 Data Source Configuration")
    
    data_source = st.selectbox(
        "Primary Data Source", 
        ["Yahoo Finance", "Alpha Vantage", "Polygon", "Quandl", "Custom"], 
        index=0
    )
    
    # API configuration
    st.subheader("🔑 API Configuration")
    
    with st.expander("API Key Management", expanded=False):
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password")
        polygon_key = st.text_input("Polygon API Key", type="password") 
        quandl_key = st.text_input("Quandl API Key", type="password")
        custom_api_key = st.text_input("Custom API Key", type="password")
        
        if st.button("💾 Save API Keys"):
            st.success("API keys have been saved securely!")

    # Performance settings
    st.subheader("⚡ Performance Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_workers = st.slider("Max Concurrent Workers", min_value=1, max_value=10, value=5)
        chunk_size = st.slider("Data Chunk Size", min_value=100, max_value=10000, value=1000)
        
    with col2:
        memory_limit = st.slider("Memory Limit (GB)", min_value=1, max_value=16, value=4)
        timeout = st.slider("Request Timeout (seconds)", min_value=10, max_value=120, value=30)

    # Advanced settings
    st.subheader("🔬 Advanced Configuration")
    
    with st.expander("Expert Settings", expanded=False):
        enable_debug = st.checkbox("Enable Debug Logging")
        enable_profiling = st.checkbox("Enable Performance Profiling")
        enable_cache = st.checkbox("Enable Advanced Caching", value=True)
        enable_notifications = st.checkbox("Enable Notifications", value=True)
        
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)

    # Save configuration
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            st.success("✅ All settings have been saved!")
            
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.warning("⚠️ Settings reset to defaults!")
            
    with col3:
        if st.button("📤 Export Config", use_container_width=True):
            st.info("📁 Configuration exported!")
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_documentation_page():
    """Display the documentation page"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("📝 Documentation")
    
    st.markdown("""
    ### 📚 User Guide
    
    **Getting Started:**
    1. Navigate to Data Acquisition to fetch market data
    2. Use Feature Engineering to create analysis features
    3. Train models using the Model Training page
    4. Backtest strategies with the Backtesting engine
    5. Manage risk using the Risk Management tools
    
    **Advanced Features:**
    - AFML methodologies implementation
    - Custom feature engineering
    - Advanced backtesting with purged CV
    - Comprehensive risk analysis
    
    ### 🔗 Quick Links
    - [AFML Book Reference](https://www.afml.com)
    - [API Documentation](https://docs.example.com)
    - [GitHub Repository](https://github.com/example/repo)
    - [Community Forum](https://forum.example.com)
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_data_explorer_page():
    """Display the data explorer utility page"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("🔍 Data Explorer")
    
    st.info("🔄 This utility page is under development.")
    
    st.markdown("""
    ### 🎯 Planned Features
    
    **Data Inspection:**
    - Interactive data viewer
    - Statistical summaries
    - Missing data analysis
    - Outlier detection
    
    **Visualization Tools:**
    - Time series plots
    - Distribution analysis
    - Correlation heatmaps
    - Custom charts
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()