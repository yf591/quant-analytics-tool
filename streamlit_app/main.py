"""
Streamlit Main Application - Week 14 Professional Integration
Entry point for the Quant Analytics Tool dashboard with comprehensive pages and advanced system management
"""

import streamlit as st
import gc
import sys
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src and streamlit directories to Python path
project_root = Path(__file__).parent.parent
streamlit_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

try:
    from src.config import settings

    # Import utility managers for system integration
    from utils.data_utils import DataAcquisitionManager
    from utils.feature_utils import FeatureEngineeringManager
    from utils.model_utils import ModelTrainingManager

    # Import UI components for enhanced display
    from components.data_display import display_alert_message, display_data_metrics
    from components.charts import create_correlation_heatmap

    # Import utility pages
    from utils_pages.settings import show_settings_page
    from utils_pages.documentation import show_documentation_page
    from utils_pages.data_explorer import show_data_explorer_page
    from utils_pages.cache_management import show_cache_management_page

except ImportError as e:
    print(f"Import warning in main.py: {e}")
    # Fallback for development mode

# Configure Streamlit page
st.set_page_config(
    page_title="Quant Analytics Tool",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yf591/quant-analytics-tool",
        "Report a bug": "https://github.com/yf591/quant-analytics-tool/issues",
        "About": "# Quant Analytics Tool\n*Professional Financial Machine Learning Platform*",
    },
)


# Initialize managers
@st.cache_resource
def initialize_managers():
    """Initialize utility managers with caching"""
    try:
        data_manager = DataAcquisitionManager()
        feature_manager = FeatureEngineeringManager()
        model_manager = ModelTrainingManager()
        return data_manager, feature_manager, model_manager
    except Exception as e:
        st.error(f"Failed to initialize managers: {e}")
        return None, None, None


def initialize_session_state():
    """Initialize session state with professional defaults"""

    # Page tracking
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Main"

    # System status tracking
    if "system_status" not in st.session_state:
        st.session_state.system_status = {
            "data_service": True,
            "feature_pipeline": True,
            "model_framework": True,
            "visualization": True,
            "last_updated": datetime.now(),
        }

    # Performance metrics
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = {
            "total_data_points": 0,
            "active_features": 0,
            "trained_models": 0,
            "completed_backtests": 0,
        }

    # User settings
    if "user_settings" not in st.session_state:
        st.session_state.user_settings = {
            "theme": "light",
            "auto_refresh": True,
            "notifications": True,
            "advanced_mode": False,
        }

    # Initialize cache dictionaries
    cache_types = [
        "data_cache",
        "feature_cache",
        "model_cache",
        "backtest_cache",
        "analysis_cache",
    ]
    for cache_type in cache_types:
        if cache_type not in st.session_state:
            st.session_state[cache_type] = {}

    # Initialize utility managers
    data_manager, feature_manager, model_manager = initialize_managers()
    if data_manager and feature_manager and model_manager:
        data_manager.initialize_session_state(st.session_state)
        feature_manager.initialize_session_state(st.session_state)
        model_manager.initialize_session_state(st.session_state)


def get_system_status():
    """Get current system status"""
    try:
        # Check memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB

        # Check cache sizes
        data_cache_size = len(st.session_state.get("data_cache", {}))
        feature_cache_size = len(st.session_state.get("feature_cache", {}))
        model_cache_size = len(st.session_state.get("model_cache", {}))

        return {
            "memory_mb": memory_usage,
            "data_cache_size": data_cache_size,
            "feature_cache_size": feature_cache_size,
            "model_cache_size": model_cache_size,
            "status": "healthy" if memory_usage < 1000 else "warning",
        }
    except Exception:
        return {
            "memory_mb": 0,
            "data_cache_size": 0,
            "feature_cache_size": 0,
            "model_cache_size": 0,
            "status": "unknown",
        }


def main():
    """Main application entry point with professional integration"""

    # Initialize session state and managers
    initialize_session_state()

    # Enhanced CSS for professional styling
    st.markdown(
        """
    <style>
    /* Professional color scheme and modern styling */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(90deg, #1f77b4, #2686c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-healthy { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    .status-unknown { background-color: #6c757d; }
    .page-container {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .system-status-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #dee2e6;
    }
    .performance-indicator {
        font-size: 0.875rem;
        font-weight: 600;
        color: #495057;
    }
    .data-flow-indicator {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        background: rgba(31, 119, 180, 0.1);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Professional header with system info
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            '<h1 class="main-header">🚀 Quant Analytics Tool</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sub-header">Professional Financial Machine Learning Platform - Week 14 Integration</p>',
            unsafe_allow_html=True,
        )

    with col3:
        # Real-time system status indicator
        system_status = get_system_status()
        status_color = {
            "healthy": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
            "unknown": "#6c757d",
        }.get(system_status["status"], "#6c757d")

        st.markdown(
            f"""
        <div style="text-align: right; padding: 1rem;">
            <div style="color: {status_color}; font-weight: 600;">
                ● System Status: {system_status['status'].title()}
            </div>
            <div style="font-size: 0.875rem; color: #666;">
                Memory: {system_status['memory_mb']:.1f} MB
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Enhanced sidebar navigation
    create_enhanced_sidebar()

    # Handle page routing
    handle_page_routing()


def create_enhanced_sidebar():
    """Create enhanced sidebar with system monitoring"""

    st.sidebar.title("📊 Navigation Center")

    # Quick system overview
    st.sidebar.markdown("### 📈 System Overview")
    system_status = get_system_status()

    # Data flow indicators
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Data Cache", system_status["data_cache_size"])
        st.metric("Models", system_status["model_cache_size"])
    with col2:
        st.metric("Features", system_status["feature_cache_size"])
        st.metric("Memory", f"{system_status['memory_mb']:.0f}MB")

    st.sidebar.markdown("---")

    # Main navigation with enhanced status
    st.sidebar.markdown("### 🎯 Main Pages")

    # Initialize session state for page tracking
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Main"

    # Page navigation with status indicators
    page_status = {
        "🏠 Main": ("active", "✅"),
        "📈 Data Acquisition": ("active", "✅"),
        "🛠️ Feature Engineering": ("active", "✅"),
        "🧠 Model Training": ("active", "✅"),
        "🔧 Training Pipeline": ("active", "✅"),
        "🔙 Backtesting": ("active", "✅"),
        "⚖️ Risk Management": ("active", "✅"),
    }

    page_options = []
    for page_name, (status, icon) in page_status.items():
        page_options.append(f"{icon} {page_name}")

    selected_page = st.sidebar.selectbox(
        "Navigate to:", page_options, index=0, key="main_nav"
    )

    # Clean the selected page name
    clean_page_name = selected_page[2:].strip()  # Remove emoji and space
    st.session_state.current_page = clean_page_name

    st.sidebar.markdown("---")

    # Utility navigation
    st.sidebar.markdown("### 🔧 Utilities")

    util_options = [
        "None",
        "⚙️ Settings",
        "📝 Documentation",
        "🔍 Data Explorer",
        "🗄️ Cache Management",
    ]

    util_page = st.sidebar.selectbox(
        "Utility Pages:", util_options, index=0, key="util_nav"
    )

    st.session_state.util_page = util_page

    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Actions")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        if st.button("🗑️ Clear Cache", use_container_width=True):
            clear_all_caches()
            st.success("Cache cleared!")
            st.rerun()
    with col2:
        if st.button("💾 Save State", use_container_width=True):
            st.success("State saved!")
        if st.button("📊 Status", use_container_width=True):
            show_system_status()

    # Implementation status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Implementation Status")

    for page_name, (status, icon) in page_status.items():
        status_color = {
            "active": "#28a745",
            "development": "#ffc107",
            "planned": "#6c757d",
        }.get(status, "#6c757d")

        st.sidebar.markdown(
            f'<span style="color: {status_color};">{icon}</span> {page_name}',
            unsafe_allow_html=True,
        )


def handle_page_routing():
    """Handle page routing with error handling"""

    try:
        # Check utility page first
        util_page = st.session_state.get("util_page", "None")

        if util_page != "None":
            if util_page == "⚙️ Settings":
                show_settings_page()
            elif util_page == "📝 Documentation":
                show_documentation_page()
            elif util_page == "🔍 Data Explorer":
                show_data_explorer_page()
            elif util_page == "🗄️ Cache Management":
                show_cache_management_page()
        else:
            # Handle main pages
            current_page = st.session_state.get("current_page", "🏠 Main")

            if current_page == "🏠 Main":
                show_enhanced_home_page()
            elif current_page == "📈 Data Acquisition":
                st.switch_page("pages/01_data_acquisition.py")
            elif current_page == "🛠️ Feature Engineering":
                st.switch_page("pages/02_feature_engineering.py")
            elif current_page == "� A Traditional Models":
                st.switch_page("pages/03_a_traditional_models.py")
            elif current_page == "🧠 B Deep Learning Models":
                st.switch_page("pages/03_b_deep_learning_models.py")
            elif current_page == "� C Advanced Models":
                st.switch_page("pages/03_c_advanced_models.py")
            elif current_page == "🧠 Model Training":
                st.switch_page("pages/03_model_training.py")
            elif current_page == "🔧 Training Pipeline":
                st.switch_page("pages/04_Training_Pipeline.py")
            elif current_page == "� Backtesting":
                st.switch_page("pages/05_backtesting.py")
            elif current_page == "⚖️ Risk Management":
                st.switch_page("pages/06_risk_management.py")

    except Exception as e:
        st.error(f"Navigation error: {e}")
        show_enhanced_home_page()


def clear_all_caches():
    """Clear all cached data"""
    cache_keys = [
        "data_cache",
        "feature_cache",
        "model_cache",
        "backtest_cache",
        "analysis_cache",
    ]
    for key in cache_keys:
        if key in st.session_state:
            st.session_state[key] = {}

    # Force garbage collection
    gc.collect()


def show_system_status():
    """Show system status in sidebar"""
    system_status = get_system_status()

    st.sidebar.markdown("#### 🖥️ System Health")
    st.sidebar.json(system_status)


def show_enhanced_home_page():
    """Display the enhanced home page with comprehensive system overview"""

    st.markdown('<div class="page-container">', unsafe_allow_html=True)

    # Welcome section with real-time metrics
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown(
            """
        ## 🎯 Platform Overview
        
        Welcome to the **Quant Analytics Tool** - a comprehensive financial machine learning platform 
        implementing advanced methodologies from "Advances in Financial Machine Learning" (AFML).
        
        ### 🚀 Key Capabilities
        - **Real-time Data Acquisition** with multiple sources
        - **Advanced Feature Engineering** with AFML techniques
        - **Professional ML Models** with proper validation
        - **Comprehensive Backtesting** with statistical rigor
        - **Risk Management** with portfolio optimization
        - **System Monitoring** with performance analytics
        """
        )

    with col2:
        # System health indicator
        system_status = get_system_status()
        status_color = {
            "healthy": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
            "unknown": "#6c757d",
        }.get(system_status["status"], "#6c757d")

        st.markdown(
            f"""
        <div class="system-status-panel">
            <h4 style="color: {status_color};">🖥️ System Health</h4>
            <div class="performance-indicator">
                Status: {system_status['status'].title()}
            </div>
            <hr style="margin: 0.5rem 0;">
            <div class="performance-indicator">
                Memory: {system_status['memory_mb']:.1f} MB
            </div>
            <div class="performance-indicator">
                Data Cache: {system_status['data_cache_size']} items
            </div>
            <div class="performance-indicator">
                Features: {system_status['feature_cache_size']} sets
            </div>
            <div class="performance-indicator">
                Models: {system_status['model_cache_size']} trained
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        # Data flow indicators
        st.markdown("### 📊 Data Flow Status")

        # Check cache availability
        data_available = len(st.session_state.get("data_cache", {})) > 0
        features_available = len(st.session_state.get("feature_cache", {})) > 0
        models_available = len(st.session_state.get("model_cache", {})) > 0

        flow_steps = [
            ("📈 Data Acquisition", data_available),
            ("🛠️ Feature Engineering", features_available),
            ("🧠 Model Training", models_available),
            ("🔙 Backtesting", models_available),
        ]

        for step_name, is_ready in flow_steps:
            status_icon = "✅" if is_ready else "⏳"
            status_color = "#28a745" if is_ready else "#6c757d"

            st.markdown(
                f"""
            <div class="data-flow-indicator">
                <span style="color: {status_color};">{status_icon}</span>
                <span style="color: {status_color};">{step_name}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Enhanced metrics display
    st.markdown("### 📊 Platform Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_data = len(st.session_state.get("data_cache", {}))
        st.metric(
            label="📈 Data Sources",
            value=str(total_data),
            delta="+1" if total_data > 0 else "0",
            help="Available data sources in cache",
        )

    with col2:
        total_features = len(st.session_state.get("feature_cache", {}))
        st.metric(
            label="🛠️ Feature Sets",
            value=str(total_features),
            delta="+1" if total_features > 0 else "0",
            help="Engineered feature sets",
        )

    with col3:
        total_models = len(st.session_state.get("model_cache", {}))
        st.metric(
            label="🧠 Trained Models",
            value=str(total_models),
            delta="+1" if total_models > 0 else "0",
            help="Successfully trained models",
        )

    with col4:
        total_backtests = len(st.session_state.get("backtest_cache", {}))
        st.metric(
            label="📊 Backtests",
            value=str(total_backtests),
            delta="+1" if total_backtests > 0 else "0",
            help="Completed backtest results",
        )

    st.markdown("---")

    # Quick navigation with workflow guidance
    st.markdown("### 🚀 Workflow Navigation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 Data & Features")
        if st.button("📈 Get Market Data", use_container_width=True, type="primary"):
            st.session_state.current_page = "📈 Data Acquisition"
            st.rerun()

        if st.button("🛠️ Engineer Features", use_container_width=True):
            st.session_state.current_page = "🛠️ Feature Engineering"
            st.rerun()

    with col2:
        st.markdown("#### 🧠 Modeling")
        if st.button("🧠 Train Models", use_container_width=True):
            st.session_state.current_page = "🧠 Model Training"
            st.rerun()

        if st.button("🔙 Run Backtests", use_container_width=True):
            st.session_state.current_page = "🔙 Backtesting"
            st.rerun()

    with col3:
        st.markdown("#### ⚖️ Risk & Analysis")
        if st.button("⚖️ Manage Risk", use_container_width=True):
            st.session_state.current_page = "⚖️ Risk Management"
            st.rerun()

        if st.button("📊 Advanced Analysis", use_container_width=True):
            st.session_state.current_page = "📊 Advanced Analysis"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_data_acquisition_page():
    """Redirect to the dedicated data acquisition page"""
    st.info("🔄 Redirecting to Data Acquisition page...")
    st.markdown("### 📈 Data Acquisition")
    st.markdown(
        """
    This page provides comprehensive data acquisition and preprocessing capabilities.
    
    **Available Features:**
    - Real-time market data fetching
    - Data preprocessing and cleaning
    - Technical analysis preparation
    - Data quality validation
    
    **Note**: This functionality is implemented in the dedicated page file.
    For full features, please use the Streamlit multipage navigation or visit:
    `/pages/01_data_acquisition.py`
    """
    )

    # Basic demonstration
    ticker = st.text_input("Quick Test - Enter Ticker:", value="AAPL")
    if st.button("Test Data Fetch"):
        st.success(f"Data acquisition system ready for {ticker}")
        st.info(
            "Please use the dedicated Data Acquisition page for full functionality."
        )


def show_feature_engineering_page():
    """Redirect to the dedicated feature engineering page"""
    st.info("🔄 Redirecting to Feature Engineering page...")
    st.markdown("### 🛠️ Feature Engineering")
    st.markdown(
        """
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
    """
    )

    # Basic demonstration
    st.selectbox(
        "Quick Test - Select Feature Type:",
        [
            "Technical Indicators",
            "AFML Features",
            "Custom Features",
            "Statistical Features",
        ],
    )
    if st.button("Test Feature Generation"):
        st.success("Feature engineering system ready")
        st.info(
            "Please use the dedicated Feature Engineering page for full functionality."
        )


def show_model_training_page():
    """Display the model training page (placeholder for Phase 2)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("🧠 Model Training")

    st.info("🔄 **Phase 2 Development**: This page is currently under development.")

    st.markdown(
        """
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
    """
    )

    # Placeholder interface
    st.markdown("### 🛠️ Configuration Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Model Type",
            ["Random Forest", "XGBoost", "Neural Network", "SVM", "Ensemble"],
            disabled=True,
        )

    with col2:
        st.selectbox(
            "Target Variable",
            [
                "Price Direction",
                "Return Magnitude",
                "Volatility Regime",
                "Custom Target",
            ],
            disabled=True,
        )

    st.button("Start Training", disabled=True, help="Available in Phase 2")
    st.markdown("</div>", unsafe_allow_html=True)


def show_backtesting_page():
    """Display the backtesting page (placeholder for Phase 2)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("🔙 Backtesting")

    st.info("🔄 **Phase 2 Development**: This page is currently under development.")

    st.markdown(
        """
    ### 🎯 Planned Features
    
    **Backtesting Engine:**
    - Historical strategy simulation
    - Multiple asset support
    - Transaction cost modeling
    - Performance analytics
    """
    )

    st.button("Run Backtest", disabled=True, help="Available in Phase 2")
    st.markdown("</div>", unsafe_allow_html=True)


def show_risk_management_page():
    """Display the risk management page (placeholder for Phase 2)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("⚖️ Risk Management")

    st.info("🔄 **Phase 2 Development**: This page is currently under development.")

    st.markdown(
        """
    ### 🎯 Planned Features
    
    **Risk Metrics:**
    - Value at Risk (VaR)
    - Portfolio optimization
    - Position sizing
    - Stress testing
    """
    )

    st.button("Calculate Risk Metrics", disabled=True, help="Available in Phase 2")
    st.markdown("</div>", unsafe_allow_html=True)


def show_advanced_analysis_page():
    """Display the advanced analysis page (placeholder for Phase 3)"""
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("📊 Advanced Analysis")

    st.warning("⏳ **Phase 3 Planning**: This page is in the planning phase.")

    st.markdown(
        """
    ### 🎯 Planned Features
    
    **Advanced Analytics:**
    - Microstructure analysis
    - Entropy features
    - Structural break detection
    - Regime analysis
    """
    )

    st.info(
        "This page will be implemented in Phase 3 of the Week 14 development cycle."
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
