"""
Streamlit Page: Feature Engineering
Week 14 UI Integration - Professional Feature Engineering Interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Week 4-5: Feature Engineering Framework Integration
    from src.features.technical import TechnicalIndicators
    from src.features.advanced import AdvancedFeatures
    from src.features.pipeline import FeaturePipeline
    from src.config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Feature Engineering Interface"""

    st.title("🛠️ Feature Engineering")
    st.markdown("**Professional Financial Feature Generation Platform**")

    # Initialize session state
    if "feature_cache" not in st.session_state:
        st.session_state.feature_cache = {}

    # Check for available data
    if "data_cache" not in st.session_state or not st.session_state.data_cache:
        st.warning("📊 Please collect data first from Data Acquisition page")
        return

    # Professional UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        feature_control_panel()

    with col2:
        feature_display_panel()


def feature_control_panel():
    """Feature Engineering Control Panel"""

    st.subheader("🎯 Feature Configuration")

    # Data selection
    available_data = list(st.session_state.data_cache.keys())
    selected_ticker = st.selectbox("Select Data", available_data)

    if not selected_ticker:
        return

    # Feature type selection
    st.subheader("📊 Feature Types")

    tab1, tab2 = st.tabs(["Technical", "Advanced"])

    with tab1:
        technical_indicators_config(selected_ticker)

    with tab2:
        advanced_features_config(selected_ticker)


def technical_indicators_config(ticker: str):
    """Technical Indicators Configuration"""

    st.markdown("**Technical Indicators**")

    # Indicator selection
    enable_sma = st.checkbox("Simple Moving Average", value=True)
    sma_period = st.slider("SMA Period", 5, 50, 20) if enable_sma else 20

    enable_rsi = st.checkbox("RSI", value=True)
    rsi_period = st.slider("RSI Period", 5, 30, 14) if enable_rsi else 14

    enable_macd = st.checkbox("MACD", value=True)

    # Calculate button
    if st.button(
        "🔄 Calculate Technical Indicators", type="primary", use_container_width=True
    ):
        calculate_technical_indicators(
            ticker,
            {
                "sma": {"enabled": enable_sma, "period": sma_period},
                "rsi": {"enabled": enable_rsi, "period": rsi_period},
                "macd": {"enabled": enable_macd},
            },
        )


def advanced_features_config(ticker: str):
    """Advanced Features Configuration"""

    st.markdown("**Advanced Features (AFML)**")

    # Advanced feature selection
    enable_fractal = st.checkbox("Fractal Dimension", value=True)
    fractal_window = st.slider("Fractal Window", 20, 100, 50) if enable_fractal else 50

    enable_hurst = st.checkbox("Hurst Exponent", value=True)
    hurst_window = st.slider("Hurst Window", 50, 200, 100) if enable_hurst else 100

    # Calculate button
    if st.button(
        "🔬 Calculate Advanced Features", type="primary", use_container_width=True
    ):
        calculate_advanced_features(
            ticker,
            {
                "fractal": {"enabled": enable_fractal, "window": fractal_window},
                "hurst": {"enabled": enable_hurst, "window": hurst_window},
            },
        )


def feature_display_panel():
    """Feature Display and Visualization Panel"""

    if not st.session_state.feature_cache:
        st.info("🛠️ Configure and calculate features to see results")
        return

    # Feature selection
    selected_feature_set = st.selectbox(
        "Select Feature Set", list(st.session_state.feature_cache.keys())
    )

    if selected_feature_set:
        display_feature_overview(selected_feature_set)
        display_feature_chart(selected_feature_set)


def calculate_technical_indicators(ticker: str, config: dict):
    """Calculate technical indicators using Week 4 TechnicalIndicators"""

    try:
        data = st.session_state.data_cache[ticker]["data"]

        with st.spinner("Calculating technical indicators..."):
            # Use existing Week 4 module
            ti = TechnicalIndicators()

            # Build indicator list based on config
            indicators_to_calculate = []
            if config["sma"]["enabled"]:
                indicators_to_calculate.append("sma")
            if config["rsi"]["enabled"]:
                indicators_to_calculate.append("rsi")
            if config["macd"]["enabled"]:
                indicators_to_calculate.append("macd")

            # Calculate all enabled indicators
            all_results = ti.calculate_all_indicators(data, indicators_to_calculate)

            # Extract the values from TechnicalIndicatorResults objects
            results = {}
            for indicator_name, result_obj in all_results.items():
                if hasattr(result_obj, "values"):
                    results[indicator_name] = result_obj.values
                else:
                    results[indicator_name] = result_obj

            # Store results
            feature_key = f"{ticker}_technical"
            st.session_state.feature_cache[feature_key] = {
                "data": data,
                "features": results,
                "type": "technical",
                "calculated_at": datetime.now(),
            }

        st.success(f"✅ Calculated {len(results)} technical indicators for {ticker}")
        st.rerun()

    except Exception as e:
        st.error(f"Technical indicator calculation failed: {str(e)}")


def calculate_advanced_features(ticker: str, config: dict):
    """Calculate advanced features using Week 5 AdvancedFeatures"""

    try:
        data = st.session_state.data_cache[ticker]["data"]

        with st.spinner("Calculating advanced features..."):
            # Use existing Week 5 module
            af = AdvancedFeatures()
            results = {}

            # Use the correct column name (check both Close and close)
            price_col = None
            if "Close" in data.columns:
                price_col = "Close"
            elif "close" in data.columns:
                price_col = "close"

            if price_col is None:
                st.error("❌ No price column found (Close or close)")
                return

            # Calculate enabled features
            if config["fractal"]["enabled"]:
                results["fractal_dimension"] = af.calculate_fractal_dimension(
                    data[price_col], config["fractal"]["window"]
                )

            if config["hurst"]["enabled"]:
                results["hurst_exponent"] = af.calculate_hurst_exponent(
                    data[price_col], config["hurst"]["window"]
                )

            # Store results
            feature_key = f"{ticker}_advanced"
            st.session_state.feature_cache[feature_key] = {
                "data": data,
                "features": results,
                "type": "advanced",
                "calculated_at": datetime.now(),
            }

        st.success(f"✅ Calculated {len(results)} advanced features for {ticker}")
        st.rerun()

    except Exception as e:
        st.error(f"Advanced feature calculation failed: {str(e)}")


def display_feature_overview(feature_key: str):
    """Display feature overview with metrics"""

    cached_features = st.session_state.feature_cache[feature_key]
    features = cached_features["features"]
    feature_type = cached_features["type"]

    st.subheader(f"📊 Features Overview: {feature_key}")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Features", len(features))

    with col2:
        st.metric("Type", feature_type.title())

    with col3:
        data_points = len(cached_features["data"])
        st.metric("Data Points", data_points)

    with col4:
        calc_time = cached_features["calculated_at"]
        st.metric("Calculated", calc_time.strftime("%H:%M"))

    # Feature preview
    feature_df = pd.DataFrame()
    for name, values in features.items():
        if isinstance(values, pd.Series):
            feature_df[name] = values
        elif isinstance(values, dict):
            for sub_name, sub_values in values.items():
                if isinstance(sub_values, pd.Series):
                    feature_df[f"{name}_{sub_name}"] = sub_values

    if not feature_df.empty:
        st.dataframe(feature_df.tail(10), use_container_width=True, height=300)

        # Download option
        csv = feature_df.to_csv()
        st.download_button(
            label="📥 Download Features CSV",
            data=csv,
            file_name=f"{feature_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


def display_feature_chart(feature_key: str):
    """Display interactive feature chart"""

    cached_features = st.session_state.feature_cache[feature_key]
    data = cached_features["data"]
    features = cached_features["features"]

    if not features or "Close" not in data.columns:
        return

    st.subheader("📈 Feature Visualization")

    # Create subplots for different feature types
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Price with Indicators", "Technical Features"),
        row_heights=[0.7, 0.3],
    )

    # Price chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Add technical indicators to price chart
    for name, values in features.items():
        if isinstance(values, pd.Series) and name in ["SMA", "EMA"]:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=values, mode="lines", name=name, opacity=0.8
                ),
                row=1,
                col=1,
            )

    # Add other features to second subplot
    for name, values in features.items():
        if isinstance(values, pd.Series) and name not in ["SMA", "EMA"]:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=values, mode="lines", name=name, opacity=0.8
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        title=f"Feature Analysis: {feature_key}", height=600, template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
