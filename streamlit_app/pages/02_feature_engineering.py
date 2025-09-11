"""
Streamlit Page: Feature Engineering
Week 14 UI Integration - Professional Feature Engineering Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src and components directory to path
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

try:
    # Week 4-5: Feature Engineering Framework Integration
    from src.features.technical import TechnicalIndicators
    from src.features.advanced import AdvancedFeatures
    from src.features.pipeline import FeaturePipeline
    from src.features.importance import FeatureImportance
    from src.config import settings

    # Streamlit components
    from components.charts import (
        create_technical_indicators_chart,
        create_advanced_features_chart,
        create_feature_importance_chart,
        create_correlation_heatmap,
    )
    from components.data_display import (
        display_data_metrics,
        display_feature_table,
        display_feature_quality_metrics,
        display_computation_status,
    )
    from components.forms import (
        create_data_selection_form,
        create_technical_indicators_form,
        create_advanced_features_form,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Feature Engineering Interface"""

    st.title("🛠️ Feature Engineering")
    st.markdown("**Professional Financial Feature Generation Platform**")
    st.markdown("---")

    # Initialize session state
    if "feature_cache" not in st.session_state:
        st.session_state.feature_cache = {}
    if "feature_pipeline_cache" not in st.session_state:
        st.session_state.feature_pipeline_cache = {}

    # Check for available data
    if "data_cache" not in st.session_state or not st.session_state.data_cache:
        st.warning("📊 Please collect data first from Data Acquisition page")
        st.info(
            "💡 Go to the Data Acquisition page to collect financial data before proceeding with feature engineering."
        )
        return

    # Data selection
    selected_ticker = create_data_selection_form(
        available_data=list(st.session_state.data_cache.keys()),
        title="🎯 Select Dataset",
    )

    if not selected_ticker:
        st.warning("Please select a dataset to continue")
        return

    # Display data overview
    data = st.session_state.data_cache[selected_ticker]["data"]
    display_data_metrics(data, title="📊 Dataset Overview", show_detailed=True)

    st.markdown("---")

    # Feature Engineering Workflow
    st.header("🔧 Feature Engineering Workflow")

    # Create tabs for different feature engineering approaches
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔧 Technical Indicators",
            "🧠 Advanced Features",
            "⚡ Feature Pipeline",
            "📊 Feature Analysis",
        ]
    )

    with tab1:
        technical_indicators_workflow(selected_ticker, data)

    with tab2:
        advanced_features_workflow(selected_ticker, data)

    with tab3:
        feature_pipeline_workflow(selected_ticker, data)

    with tab4:
        feature_analysis_workflow(selected_ticker)


def technical_indicators_workflow(ticker: str, data: pd.DataFrame):
    """Technical Indicators Workflow"""

    st.subheader("🔧 Technical Indicators Configuration")
    st.markdown("Configure and calculate technical indicators for financial analysis.")

    # Use the sophisticated form component
    indicator_config = create_technical_indicators_form()

    if st.button(
        "🔄 Calculate Technical Indicators", type="primary", use_container_width=True
    ):
        calculate_technical_indicators(ticker, data, indicator_config)

    # Display existing results if available
    feature_key = f"{ticker}_technical"
    if feature_key in st.session_state.feature_cache:
        st.success("✅ Technical indicators calculated successfully!")
        display_technical_results(feature_key)


def advanced_features_workflow(ticker: str, data: pd.DataFrame):
    """Advanced Features Workflow"""

    st.subheader("🧠 Advanced Features Configuration")
    st.markdown("Configure advanced AFML-based features for sophisticated analysis.")

    # Use the sophisticated form component
    advanced_config = create_advanced_features_form()

    if st.button(
        "🔬 Calculate Advanced Features", type="primary", use_container_width=True
    ):
        calculate_advanced_features(ticker, data, advanced_config)

    # Display existing results if available
    feature_key = f"{ticker}_advanced"
    if feature_key in st.session_state.feature_cache:
        st.success("✅ Advanced features calculated successfully!")
        display_advanced_results(feature_key)


def feature_pipeline_workflow(ticker: str, data: pd.DataFrame):
    """Integrated Feature Pipeline Workflow"""

    st.subheader("⚡ Automated Feature Pipeline")
    st.markdown(
        "Execute comprehensive feature engineering pipeline with automated selection and validation."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Pipeline Configuration:**")
        include_technical = st.checkbox("Include Technical Indicators", value=True)
        include_advanced = st.checkbox("Include Advanced Features", value=True)
        feature_selection = st.checkbox("Enable Feature Selection", value=True)
        quality_validation = st.checkbox("Enable Quality Validation", value=True)

    with col2:
        st.write("**Pipeline Parameters:**")
        max_features = st.slider("Max Features", 10, 100, 50)
        correlation_threshold = st.slider("Correlation Threshold", 0.5, 0.95, 0.8)

    if st.button("⚡ Run Feature Pipeline", type="primary", use_container_width=True):
        run_feature_pipeline(
            ticker,
            data,
            {
                "include_technical": include_technical,
                "include_advanced": include_advanced,
                "feature_selection": feature_selection,
                "quality_validation": quality_validation,
                "max_features": max_features,
                "correlation_threshold": correlation_threshold,
            },
        )

    # Display pipeline results if available
    pipeline_key = f"{ticker}_pipeline"
    if pipeline_key in st.session_state.feature_pipeline_cache:
        st.success("✅ Feature pipeline executed successfully!")
        display_pipeline_results(pipeline_key)


def feature_analysis_workflow(ticker: str):
    """Feature Analysis and Visualization Workflow"""

    st.subheader("📊 Feature Analysis & Visualization")

    # Check for available feature sets
    available_features = [
        key for key in st.session_state.feature_cache.keys() if ticker in key
    ]
    available_pipelines = [
        key for key in st.session_state.feature_pipeline_cache.keys() if ticker in key
    ]

    if not available_features and not available_pipelines:
        st.info("🔧 Calculate features first to enable analysis")
        return

    # Feature set selection
    all_available = available_features + available_pipelines
    selected_feature_set = st.selectbox(
        "Select Feature Set for Analysis:",
        options=all_available,
        help="Choose a feature set to analyze and visualize",
    )

    if selected_feature_set:
        # Analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
            ["📈 Visualization", "📊 Statistics", "🔍 Quality Analysis"]
        )

        with analysis_tab1:
            display_feature_visualization(selected_feature_set)

        with analysis_tab2:
            display_feature_statistics(selected_feature_set)

        with analysis_tab3:
            display_feature_quality_analysis(selected_feature_set)


def calculate_technical_indicators(
    ticker: str, data: pd.DataFrame, config: Dict[str, Any]
):
    """Calculate technical indicators using Week 4 TechnicalIndicators"""

    try:
        display_computation_status("🔄 Calculating technical indicators...", 0.1)

        # Use existing Week 4 module
        ti = TechnicalIndicators()

        # Build indicator list based on config
        indicators_to_calculate = []

        if config.get("sma_enabled", False):
            indicators_to_calculate.append("sma")
        if config.get("ema_enabled", False):
            indicators_to_calculate.append("ema")
        if config.get("rsi_enabled", False):
            indicators_to_calculate.append("rsi")
        if config.get("macd_enabled", False):
            indicators_to_calculate.append("macd")
        if config.get("bb_enabled", False):
            indicators_to_calculate.append("bollinger_bands")
        if config.get("atr_enabled", False):
            indicators_to_calculate.append("atr")
        if config.get("stoch_enabled", False):
            indicators_to_calculate.append("stochastic")
        if config.get("williams_enabled", False):
            indicators_to_calculate.append("williams_r")

        if not indicators_to_calculate:
            st.warning(
                "⚠️ No indicators selected. Please enable at least one indicator."
            )
            return

        display_computation_status("🔄 Computing indicators...", 0.5)

        # Calculate all enabled indicators
        all_results = ti.calculate_all_indicators(data, indicators_to_calculate)

        display_computation_status("🔄 Processing results...", 0.8)

        # Extract the values from TechnicalIndicatorResults objects
        results = {}
        for indicator_name, result_obj in all_results.items():
            if hasattr(result_obj, "values"):
                if isinstance(result_obj.values, pd.DataFrame):
                    # Handle multi-column results like MACD
                    for col in result_obj.values.columns:
                        results[f"{indicator_name}_{col}"] = result_obj.values[col]
                else:
                    results[indicator_name] = result_obj.values
            else:
                results[indicator_name] = result_obj

        # Store results with additional metadata
        feature_key = f"{ticker}_technical"
        st.session_state.feature_cache[feature_key] = {
            "data": data,
            "features": results,
            "type": "technical",
            "config": config,
            "calculated_at": datetime.now(),
            "indicators_count": len(results),
        }

        display_computation_status(
            f"✅ Successfully calculated {len(results)} technical indicators for {ticker}",
            1.0,
        )
        st.rerun()

    except Exception as e:
        display_computation_status(
            f"❌ Technical indicator calculation failed: {str(e)}", details=str(e)
        )


def calculate_advanced_features(
    ticker: str, data: pd.DataFrame, config: Dict[str, Any]
):
    """Calculate advanced features using Week 5 AdvancedFeatures"""

    try:
        display_computation_status("🔄 Calculating advanced features...", 0.1)

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

        display_computation_status("🔄 Computing features...", 0.3)

        # Calculate enabled features
        if config.get("fractal_enabled", False):
            display_computation_status("🔄 Computing fractal dimension...", 0.4)
            results["fractal_dimension"] = af.calculate_fractal_dimension(
                data[price_col],
                config.get("fractal_window", 100),
                config.get("fractal_method", "higuchi"),
            )

        if config.get("hurst_enabled", False):
            display_computation_status("🔄 Computing Hurst exponent...", 0.6)
            results["hurst_exponent"] = af.calculate_hurst_exponent(
                data[price_col],
                config.get("hurst_window", 100),
                config.get("hurst_method", "rs"),
            )

        if config.get("info_bars_enabled", False):
            display_computation_status("🔄 Creating information bars...", 0.7)
            threshold = config.get("bar_threshold", None)
            if threshold == 0:
                threshold = None
            results["information_bars"] = af.create_information_bars(
                data, config.get("bar_type", "volume"), threshold, price_col
            )

        if config.get("frac_diff_enabled", False):
            display_computation_status(
                "🔄 Computing fractional differentiation...", 0.8
            )
            results["fractional_diff"] = af.fractional_differentiation(
                data[price_col],
                config.get("frac_diff_d", 0.4),
                config.get("frac_diff_threshold", 0.01),
            )

        if not results:
            st.warning(
                "⚠️ No advanced features selected. Please enable at least one feature."
            )
            return

        # Store results with additional metadata
        feature_key = f"{ticker}_advanced"
        st.session_state.feature_cache[feature_key] = {
            "data": data,
            "features": results,
            "type": "advanced",
            "config": config,
            "calculated_at": datetime.now(),
            "features_count": len(results),
        }

        display_computation_status(
            f"✅ Successfully calculated {len(results)} advanced features for {ticker}",
            1.0,
        )
        st.rerun()

    except Exception as e:
        display_computation_status(
            f"❌ Advanced feature calculation failed: {str(e)}", details=str(e)
        )


def run_feature_pipeline(ticker: str, data: pd.DataFrame, config: Dict[str, Any]):
    """Run comprehensive feature pipeline"""

    try:
        display_computation_status("🔄 Initializing feature pipeline...", 0.1)

        # Initialize feature pipeline
        fp = FeaturePipeline()

        display_computation_status("🔄 Generating features...", 0.3)

        # Generate features based on configuration
        pipeline_results = fp.generate_features(data)

        display_computation_status("🔄 Processing pipeline results...", 0.8)

        # Store pipeline results
        pipeline_key = f"{ticker}_pipeline"
        st.session_state.feature_pipeline_cache[pipeline_key] = {
            "data": data,
            "results": pipeline_results,
            "config": config,
            "calculated_at": datetime.now(),
        }

        display_computation_status(
            f"✅ Feature pipeline completed successfully for {ticker}", 1.0
        )
        st.rerun()

    except Exception as e:
        display_computation_status(
            f"❌ Feature pipeline failed: {str(e)}", details=str(e)
        )


def display_technical_results(feature_key: str):
    """Display technical indicators results"""

    cached_features = st.session_state.feature_cache[feature_key]
    features = cached_features["features"]
    data = cached_features["data"]

    # Display feature table
    display_feature_table(features, title="📊 Technical Indicators", show_stats=True)

    # Create and display chart
    if features:
        chart = create_technical_indicators_chart(
            data=data,
            indicators=features,
            height=600,
            title="Technical Indicators Analysis",
        )
        st.plotly_chart(chart, use_container_width=True)


def display_advanced_results(feature_key: str):
    """Display advanced features results"""

    cached_features = st.session_state.feature_cache[feature_key]
    features = cached_features["features"]
    data = cached_features["data"]

    # Display feature table
    display_feature_table(features, title="🧠 Advanced Features", show_stats=True)

    # Create and display chart
    if features:
        chart = create_advanced_features_chart(
            data=data, features=features, height=600, title="Advanced Features Analysis"
        )
        st.plotly_chart(chart, use_container_width=True)


def display_pipeline_results(pipeline_key: str):
    """Display feature pipeline results"""

    cached_pipeline = st.session_state.feature_pipeline_cache[pipeline_key]
    results = cached_pipeline["results"]

    if results.features is not None:
        # Display comprehensive results
        display_feature_table(
            results.features, title="⚡ Pipeline Features", show_stats=True
        )

        # Feature quality metrics
        if results.quality_metrics:
            st.subheader("📊 Feature Quality Metrics")
            quality_df = pd.DataFrame([results.quality_metrics]).T
            quality_df.columns = ["Score"]
            st.dataframe(quality_df, use_container_width=True)

        # Feature importance if available
        if results.feature_importance is not None:
            chart = create_feature_importance_chart(
                results.feature_importance, title="Feature Importance Analysis"
            )
            st.plotly_chart(chart, use_container_width=True)


def display_feature_visualization(feature_set_key: str):
    """Display feature visualization"""

    if feature_set_key in st.session_state.feature_cache:
        cached_features = st.session_state.feature_cache[feature_set_key]
        features = cached_features["features"]
        data = cached_features["data"]
        feature_type = cached_features["type"]

        if feature_type == "technical":
            chart = create_technical_indicators_chart(data, features, height=700)
        else:
            chart = create_advanced_features_chart(data, features, height=700)

        st.plotly_chart(chart, use_container_width=True)

    elif feature_set_key in st.session_state.feature_pipeline_cache:
        cached_pipeline = st.session_state.feature_pipeline_cache[feature_set_key]
        results = cached_pipeline["results"]

        if results.features is not None:
            # Create correlation heatmap for pipeline features
            if len(results.features.columns) > 1:
                corr_matrix = results.features.corr()
                chart = create_correlation_heatmap(
                    corr_matrix, title="Feature Correlation Matrix"
                )
                st.plotly_chart(chart, use_container_width=True)


def display_feature_statistics(feature_set_key: str):
    """Display feature statistics"""

    if feature_set_key in st.session_state.feature_cache:
        cached_features = st.session_state.feature_cache[feature_set_key]
        features = cached_features["features"]

        # Convert dict to DataFrame for statistics
        if isinstance(features, dict):
            feature_df = pd.DataFrame()
            for name, values in features.items():
                if isinstance(values, pd.Series):
                    feature_df[name] = values
        else:
            feature_df = features

        if not feature_df.empty:
            st.subheader("📊 Statistical Summary")
            st.dataframe(feature_df.describe(), use_container_width=True)

            # Additional statistics
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Correlation Matrix**")
                if len(feature_df.columns) > 1:
                    corr_matrix = feature_df.corr()
                    st.dataframe(corr_matrix, use_container_width=True)

            with col2:
                st.write("**Missing Values**")
                missing_df = pd.DataFrame(
                    {
                        "Column": feature_df.columns,
                        "Missing Count": feature_df.isnull().sum(),
                        "Missing %": (feature_df.isnull().sum() / len(feature_df))
                        * 100,
                    }
                )
                st.dataframe(missing_df, use_container_width=True)

    elif feature_set_key in st.session_state.feature_pipeline_cache:
        cached_pipeline = st.session_state.feature_pipeline_cache[feature_set_key]
        results = cached_pipeline["results"]

        if results.features is not None:
            st.subheader("📊 Pipeline Statistics")
            st.dataframe(results.features.describe(), use_container_width=True)


def display_feature_quality_analysis(feature_set_key: str):
    """Display feature quality analysis"""

    if feature_set_key in st.session_state.feature_cache:
        cached_features = st.session_state.feature_cache[feature_set_key]
        features = cached_features["features"]

        # Convert dict to DataFrame for quality analysis
        if isinstance(features, dict):
            feature_df = pd.DataFrame()
            for name, values in features.items():
                if isinstance(values, pd.Series):
                    feature_df[name] = values
        else:
            feature_df = features

        if not feature_df.empty:
            display_feature_quality_metrics(
                feature_df, title="🔍 Feature Quality Analysis"
            )

    elif feature_set_key in st.session_state.feature_pipeline_cache:
        cached_pipeline = st.session_state.feature_pipeline_cache[feature_set_key]
        results = cached_pipeline["results"]

        if results.features is not None:
            display_feature_quality_metrics(
                results.features, title="🔍 Pipeline Quality Analysis"
            )

            # Additional pipeline-specific quality metrics
            if results.quality_metrics:
                st.subheader("📊 Pipeline Quality Metrics")
                for metric_name, metric_value in results.quality_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        st.metric(
                            metric_name.replace("_", " ").title(), f"{metric_value:.4f}"
                        )


def display_feature_overview(feature_key: str):
    """Display feature overview with metrics (legacy compatibility)"""

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
    """Display interactive feature chart (legacy compatibility)"""

    cached_features = st.session_state.feature_cache[feature_key]
    data = cached_features["data"]
    features = cached_features["features"]

    # Use correct column name
    price_col = "Close" if "Close" in data.columns else "close"
    if price_col not in data.columns:
        st.warning("No price column found for chart display")
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
            y=data[price_col],
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
