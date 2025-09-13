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
from typing import Dict, Any, Optional, List

# Add src and components directory to path
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

try:
    # Week 14: Streamlit utils integration - use utility managers
    from utils.feature_utils import FeatureEngineeringManager
    from utils.analysis_utils import AnalysisManager

    # Streamlit components
    from components.charts import (
        create_technical_indicators_chart,
        create_advanced_features_chart,
        create_information_bars_chart,
        create_feature_importance_chart,
        create_correlation_heatmap,
    )
    from components.data_display import (
        display_data_metrics,
        display_feature_table,
        display_feature_quality_metrics,
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

    # Initialize managers
    feature_manager = FeatureEngineeringManager()
    analysis_manager = AnalysisManager()

    # Initialize session state using the managers
    feature_manager.initialize_session_state(st.session_state)
    analysis_manager.initialize_session_state(st.session_state)

    st.title("🛠️ Feature Engineering")
    st.markdown("**Professional Financial Feature Generation Platform**")
    st.markdown("---")

    # Store managers in session state for access in other functions
    st.session_state.feature_manager = feature_manager
    st.session_state.analysis_manager = analysis_manager

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🔧 Technical Indicators",
            "🧠 Advanced Features",
            "⚡ Feature Pipeline",
            "📊 Feature Analysis",
            "📈 Statistical Analysis",
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

    with tab5:
        statistical_analysis_workflow(selected_ticker, data)


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

    # Check for available feature sets (exclude metadata)
    available_features = [
        key
        for key in st.session_state.feature_cache.keys()
        if ticker in key and not key.endswith("_metadata")
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
    """Calculate technical indicators using FeatureEngineeringManager"""

    try:
        # Get the feature manager from session state
        feature_manager = st.session_state.feature_manager

        # Use the manager to calculate technical indicators
        with st.spinner("Calculating technical indicators..."):
            success, message = feature_manager.calculate_technical_indicators(
                ticker, data, config, st.session_state
            )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    except Exception as e:
        error_msg = f"Technical indicator calculation failed: {str(e)}"
        st.error(error_msg)


def calculate_advanced_features(
    ticker: str, data: pd.DataFrame, config: Dict[str, Any]
):
    """Calculate advanced features using FeatureEngineeringManager"""

    try:
        # Get the feature manager from session state
        feature_manager = st.session_state.feature_manager

        # Use the manager to calculate advanced features
        with st.spinner("Calculating advanced features..."):
            success, message = feature_manager.calculate_advanced_features(
                ticker, data, config, st.session_state
            )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    except Exception as e:
        error_msg = f"Advanced feature calculation failed: {str(e)}"
        st.error(error_msg)


def run_feature_pipeline(ticker: str, data: pd.DataFrame, config: Dict[str, Any]):
    """Run comprehensive feature pipeline using FeatureEngineeringManager"""

    try:
        # Get the feature manager from session state
        feature_manager = st.session_state.feature_manager

        # Use the manager to run feature pipeline
        with st.spinner("Running feature pipeline..."):
            success, message = feature_manager.run_feature_pipeline(
                ticker, data, config, st.session_state
            )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    except Exception as e:
        error_msg = f"Feature pipeline failed: {str(e)}"
        st.error(error_msg)


def statistical_analysis_workflow(ticker: str, data: pd.DataFrame):
    """Statistical Analysis Workflow - NEW: Week 14 Integration"""

    st.subheader("📈 Statistical Analysis & Quality Assessment")
    st.markdown(
        "Perform comprehensive statistical analysis on data and features using AFML methodologies."
    )

    # Analysis type selection
    analysis_type = st.selectbox(
        "Analysis Type",
        options=[
            "Raw Data Analysis",
            "Feature Quality Analysis",
            "Return Analysis",
            "Volatility Analysis",
        ],
        help="Choose the type of statistical analysis to perform",
    )

    if analysis_type == "Raw Data Analysis":
        raw_data_analysis_workflow(ticker, data)
    elif analysis_type == "Feature Quality Analysis":
        feature_quality_analysis_workflow(ticker)
    elif analysis_type == "Return Analysis":
        return_analysis_workflow(ticker, data)
    elif analysis_type == "Volatility Analysis":
        volatility_analysis_workflow(ticker, data)


def raw_data_analysis_workflow(ticker: str, data: pd.DataFrame):
    """Raw data statistical analysis"""

    st.subheader("📊 Raw Data Statistical Analysis")

    # Select column for analysis
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns found for analysis")
        return

    selected_column = st.selectbox(
        "Select Column for Analysis",
        options=numeric_columns,
        index=0 if "Close" in numeric_columns else 0,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Basic Statistics", type="primary", use_container_width=True):
            analyze_basic_statistics(ticker, data[selected_column], "price")

    with col2:
        if st.button("📈 Distribution Analysis", use_container_width=True):
            analyze_distribution(ticker, data[selected_column], "price")

    # Display results if available
    display_analysis_results(ticker, "price", ["statistics", "distribution"])


def feature_quality_analysis_workflow(ticker: str):
    """Feature quality analysis workflow"""

    st.subheader("🔍 Feature Quality Analysis")

    # Get available features
    available_features = [
        key
        for key in st.session_state.feature_cache.keys()
        if ticker in key and not key.endswith("_metadata")
    ]

    if not available_features:
        st.info("🔧 Calculate features first to enable quality analysis")
        return

    selected_feature_set = st.selectbox(
        "Select Feature Set", options=available_features
    )

    if st.button(
        "🔍 Analyze Feature Quality", type="primary", use_container_width=True
    ):
        feature_data = st.session_state.feature_cache[selected_feature_set]
        if isinstance(feature_data, pd.DataFrame):
            analyze_feature_quality(ticker, feature_data)

    # Display results
    display_analysis_results(
        ticker, "features", ["statistics", "distribution", "correlation", "quality"]
    )


def return_analysis_workflow(ticker: str, data: pd.DataFrame):
    """Return analysis workflow"""

    st.subheader("📈 Return Analysis")

    # Select price column
    price_columns = [
        col for col in data.columns if "close" in col.lower() or "price" in col.lower()
    ]
    if not price_columns:
        price_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if not price_columns:
        st.warning("No suitable price columns found")
        return

    selected_price_col = st.selectbox("Select Price Column", options=price_columns)
    return_type = st.selectbox("Return Type", options=["simple", "log"], index=0)

    if st.button("📈 Analyze Returns", type="primary", use_container_width=True):
        analyze_returns(ticker, data[selected_price_col], return_type)

    # Display results
    display_analysis_results(ticker, "returns", ["returns_analysis"])


def volatility_analysis_workflow(ticker: str, data: pd.DataFrame):
    """Volatility analysis workflow"""

    st.subheader("📊 Volatility Analysis")

    # Select price column
    price_columns = [
        col for col in data.columns if "close" in col.lower() or "price" in col.lower()
    ]
    if not price_columns:
        price_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if not price_columns:
        st.warning("No suitable price columns found")
        return

    selected_price_col = st.selectbox("Select Price Column", options=price_columns)

    # Check for OHLC data
    has_ohlc = all(col in data.columns for col in ["Open", "High", "Low", "Close"])
    include_ohlc = False
    if has_ohlc:
        include_ohlc = st.checkbox("Include OHLC-based volatility measures", value=True)

    if st.button("📊 Analyze Volatility", type="primary", use_container_width=True):
        ohlc_data = data[["Open", "High", "Low", "Close"]] if include_ohlc else None
        analyze_volatility(ticker, data[selected_price_col], ohlc_data)

    # Display results
    display_analysis_results(ticker, "volatility", ["volatility_analysis"])


def analyze_basic_statistics(ticker: str, data: pd.Series, data_type: str):
    """Analyze basic statistics"""

    try:
        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_basic_statistics(
            data, ticker, st.session_state, data_type
        )

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Statistical analysis failed: {str(e)}")


def analyze_distribution(ticker: str, data: pd.Series, data_type: str):
    """Analyze distribution"""

    try:
        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_distribution(
            data, ticker, st.session_state, data_type
        )

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Distribution analysis failed: {str(e)}")


def analyze_returns(ticker: str, price_data: pd.Series, return_type: str):
    """Analyze returns"""

    try:
        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_returns(
            price_data, ticker, st.session_state, return_type
        )

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Return analysis failed: {str(e)}")


def analyze_volatility(
    ticker: str, price_data: pd.Series, ohlc_data: Optional[pd.DataFrame]
):
    """Analyze volatility"""

    try:
        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_volatility(
            price_data, ticker, st.session_state, ohlc_data
        )

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Volatility analysis failed: {str(e)}")


def analyze_feature_quality(ticker: str, feature_data: pd.DataFrame):
    """Analyze feature quality"""

    try:
        analysis_manager = st.session_state.analysis_manager
        success, message = analysis_manager.analyze_feature_quality(
            feature_data, ticker, st.session_state
        )

        if success:
            st.success(message)
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Feature quality analysis failed: {str(e)}")


def display_analysis_results(ticker: str, data_type: str, analysis_types: List[str]):
    """Display analysis results"""

    try:
        analysis_manager = st.session_state.analysis_manager

        for analysis_type in analysis_types:
            analysis_key = f"{ticker}_{data_type}_{analysis_type}"
            results = analysis_manager.get_analysis_results(
                analysis_key, st.session_state
            )

            if results:
                with st.expander(
                    f"📊 {analysis_type.replace('_', ' ').title()} Results",
                    expanded=True,
                ):

                    if analysis_type == "statistics":
                        display_statistics_results(results)
                    elif analysis_type == "distribution":
                        display_distribution_results(results)
                    elif analysis_type == "returns_analysis":
                        display_returns_results(results)
                    elif analysis_type == "volatility_analysis":
                        display_volatility_results(results)
                    elif analysis_type == "correlation":
                        display_correlation_results(results)
                    elif analysis_type == "quality":
                        display_quality_results(results)

    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")


def display_statistics_results(results: Dict[str, Any]):
    """Display basic statistics results"""

    stats_results = results.get("results", {})
    analysis_manager = st.session_state.analysis_manager

    for name, basic_stats in stats_results.items():
        st.write(f"**{name} Statistics:**")
        formatted_stats = analysis_manager.format_statistics_for_display(basic_stats)

        # Display as metrics
        cols = st.columns(5)
        metric_items = list(formatted_stats.items())

        for i, (metric_name, metric_value) in enumerate(metric_items[:10]):
            with cols[i % 5]:
                st.metric(metric_name, metric_value)

        # Additional stats as table
        if len(formatted_stats) > 10:
            remaining_stats = {k: v for k, v in list(formatted_stats.items())[10:]}
            st.table(pd.DataFrame([remaining_stats]).T.rename(columns={0: "Value"}))


def display_distribution_results(results: Dict[str, Any]):
    """Display distribution analysis results"""

    dist_results = results.get("results", {})
    analysis_manager = st.session_state.analysis_manager

    for name, dist_analysis in dist_results.items():
        st.write(f"**{name} Distribution Analysis:**")
        formatted_dist = analysis_manager.format_distribution_for_display(dist_analysis)

        # Display as table
        dist_df = pd.DataFrame([formatted_dist]).T.rename(columns={0: "Value"})
        st.table(dist_df)


def display_returns_results(results: Dict[str, Any]):
    """Display return analysis results"""

    return_stats = results.get("return_statistics")
    if return_stats:
        st.write("**Return Statistics:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Return", f"{return_stats.mean:.6f}")
            st.metric("Std Deviation", f"{return_stats.std:.6f}")
            st.metric("Skewness", f"{return_stats.skewness:.4f}")

        with col2:
            st.metric("Kurtosis", f"{return_stats.kurtosis:.4f}")
            st.metric("Sharpe Ratio", f"{return_stats.sharpe_ratio:.4f}")
            st.metric("Max Drawdown", f"{return_stats.max_drawdown:.4f}")

        with col3:
            st.metric("Total Return", f"{return_stats.total_return:.4f}")
            st.metric("Annualized Return", f"{return_stats.annualized_return:.4f}")
            st.metric(
                "Annualized Volatility", f"{return_stats.annualized_volatility:.4f}"
            )


def display_volatility_results(results: Dict[str, Any]):
    """Display volatility analysis results"""

    vol_stats = results.get("volatility_statistics")
    if vol_stats:
        st.write("**Volatility Statistics:**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Volatility", f"{vol_stats.current_volatility:.4f}")
            st.metric("Average Volatility", f"{vol_stats.average_volatility:.4f}")
            st.metric("Volatility Std", f"{vol_stats.volatility_std:.4f}")

        with col2:
            st.metric("Volatility Skewness", f"{vol_stats.volatility_skewness:.4f}")
            st.metric("Volatility Kurtosis", f"{vol_stats.volatility_kurtosis:.4f}")
            if vol_stats.garch_persistence:
                st.metric("GARCH Persistence", f"{vol_stats.garch_persistence:.4f}")


def display_correlation_results(results: Dict[str, Any]):
    """Display correlation analysis results"""

    corr_matrix = results.get("correlation_matrix")
    if corr_matrix is not None:
        st.write("**Correlation Matrix:**")
        st.dataframe(corr_matrix, use_container_width=True)

        # Create correlation heatmap
        chart = create_correlation_heatmap(
            corr_matrix, title="Feature Correlation Matrix"
        )
        st.plotly_chart(chart, use_container_width=True)


def display_quality_results(results: Dict[str, Any]):
    """Display feature quality results"""

    st.write("**Feature Quality Summary:**")

    quality_metrics = [
        ("Basic Statistics", results.get("basic_statistics", False)),
        ("Distribution Analysis", results.get("distribution_analysis", False)),
        ("Correlation Analysis", results.get("correlation_analysis", False)),
    ]

    for metric_name, is_successful in quality_metrics:
        status = "✅ Complete" if is_successful else "❌ Failed"
        st.write(f"- **{metric_name}:** {status}")

    st.metric("Feature Count", results.get("feature_count", 0))
    st.metric("Sample Count", results.get("sample_count", 0))


def display_technical_results(feature_key: str):
    """Display technical indicators results"""

    # Check if we have the new DataFrame format or old dict format
    if feature_key in st.session_state.feature_cache:
        cached_data = st.session_state.feature_cache[feature_key]

        if isinstance(cached_data, pd.DataFrame):
            # New format: Direct DataFrame
            feature_df = cached_data
            metadata_key = f"{feature_key}_metadata"

            if metadata_key in st.session_state.feature_cache:
                metadata = st.session_state.feature_cache[metadata_key]
                original_data = metadata["original_data"]
                # ★★★ ここでチャート用の辞書を取得 ★★★
                features_dict_for_chart = metadata.get("features_dict_for_chart", {})
            else:
                # フォールバック
                original_data = feature_df
                features_dict_for_chart = {
                    col: feature_df[col] for col in feature_df.columns
                }
        else:
            # Old format: Dict with data and features
            cached_features = cached_data
            features_dict = cached_features["features"]
            original_data = cached_features["data"]

            # Convert to DataFrame for consistent handling
            feature_df = pd.DataFrame(index=original_data.index)
            for name, values in features_dict.items():
                if isinstance(values, pd.Series):
                    feature_df[name] = values
            features_dict_for_chart = features_dict

        # Debug information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write(f"**Number of calculated indicators:** {len(feature_df.columns)}")
            st.write(f"**Indicator names:** {list(feature_df.columns)}")
            st.write(f"**Chart indicators:** {list(features_dict_for_chart.keys())}")
            for col in feature_df.columns:
                valid_count = feature_df[col].dropna().shape[0]
                st.write(
                    f"**{col}:** {valid_count} valid values out of {len(feature_df)}"
                )

        # Display feature table - pass DataFrame directly
        display_feature_table(
            feature_df, title="📊 Technical Indicators", show_stats=True
        )

        # Create and display chart with proper data format
        if features_dict_for_chart:
            try:
                chart = create_technical_indicators_chart(
                    data=original_data,
                    indicators=features_dict_for_chart,  # ★ここが重要★
                    height=800,  # 高さを調整
                    title="Technical Indicators Analysis",
                )
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {e}")

                # Fallback: Show simple line charts
                st.subheader("📈 Individual Indicator Charts")
                for col in feature_df.columns:
                    values = feature_df[col].dropna()
                    if not values.empty:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(x=values.index, y=values, mode="lines", name=col)
                        )
                        fig.update_layout(title=f"{col} Indicator", height=300)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No indicators to display")
    else:
        st.error(f"Feature key {feature_key} not found in cache")


def display_advanced_results(feature_key: str):
    """Display advanced features results"""

    # Check if we have the new DataFrame format or old dict format
    if feature_key in st.session_state.feature_cache:
        cached_data = st.session_state.feature_cache[feature_key]

        if isinstance(cached_data, pd.DataFrame):
            # New format: Direct DataFrame
            feature_df = cached_data
            metadata_key = f"{feature_key}_metadata"

            if metadata_key in st.session_state.feature_cache:
                metadata = st.session_state.feature_cache[metadata_key]
                original_data = metadata["original_data"]
                features_dict = metadata[
                    "features_dict"
                ].copy()  # Make a copy to avoid modifying original
            else:
                original_data = feature_df  # Fallback
                features_dict = {col: feature_df[col] for col in feature_df.columns}
        else:
            # Old format: Dict with data and features
            cached_features = cached_data
            features_dict = cached_features["features"].copy()  # Make a copy
            original_data = cached_features["data"]

            # Convert to DataFrame for consistent handling
            feature_df = pd.DataFrame(index=original_data.index)
            for name, values in features_dict.items():
                if isinstance(values, pd.Series):
                    feature_df[name] = values

        # Separate information_bars from other features
        info_bars_data = features_dict.pop("information_bars", None)

        # Display Series-based features
        if features_dict:
            # Create DataFrame with only Series features
            series_features_df = pd.DataFrame(index=original_data.index)
            for name, values in features_dict.items():
                if isinstance(values, pd.Series):
                    series_features_df[name] = values

            if len(series_features_df.columns) > 0:
                display_feature_table(
                    series_features_df,
                    title="🧠 Advanced Features (Time Series)",
                    show_stats=True,
                )

                # Create and display chart for Series features only
                try:
                    chart = create_advanced_features_chart(
                        data=original_data,
                        features=features_dict,  # This now contains only Series
                        height=600,
                        title="Advanced Features Analysis",
                    )
                    st.plotly_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating advanced features chart: {e}")

        # Display Information Bars separately
        if (
            info_bars_data is not None
            and isinstance(info_bars_data, pd.DataFrame)
            and not info_bars_data.empty
        ):
            st.markdown("---")
            st.subheader("📊 Information-Driven Bars Analysis")
            st.markdown(
                "Data sampled by information content (e.g., volume) rather than time. "
                "Each bar represents a fixed amount of trading activity."
            )

            # Display basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bars", len(info_bars_data))
            with col2:
                if "volume" in info_bars_data.columns:
                    avg_volume = info_bars_data["volume"].mean()
                    st.metric("Avg Volume per Bar", f"{avg_volume:,.0f}")
            with col3:
                if "count" in info_bars_data.columns:
                    avg_ticks = info_bars_data["count"].mean()
                    st.metric("Avg Ticks per Bar", f"{avg_ticks:.1f}")

            # Display data table
            display_feature_table(
                info_bars_data.head(20),
                title="Information Bars Data (First 20 bars)",
                show_stats=True,
            )

            # Create and display specialized chart
            info_bars_chart = create_information_bars_chart(
                info_bars_data,
                title=f"Information-Driven Bars ({len(info_bars_data)} bars)",
            )
            st.plotly_chart(info_bars_chart, use_container_width=True)

            # Additional analysis
            if len(info_bars_data) > 1:
                st.subheader("📈 Information Bars Statistics")

                # Time between bars analysis
                time_diffs = info_bars_data.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    avg_time_diff = time_diffs.mean()
                    st.write(f"**Average time between bars:** {avg_time_diff}")

                    # Show distribution of time intervals
                    fig_hist = go.Figure(
                        data=[
                            go.Histogram(
                                x=time_diffs.dt.total_seconds()
                                / 60,  # Convert to minutes
                                nbinsx=20,
                                name="Time Intervals",
                            )
                        ]
                    )
                    fig_hist.update_layout(
                        title="Distribution of Time Intervals Between Bars (minutes)",
                        xaxis_title="Minutes",
                        yaxis_title="Frequency",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.error(f"Feature key {feature_key} not found in cache")


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

            # Safely display quality metrics with proper type conversion
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total Features", results.quality_metrics.get("total_features", 0)
                )
                if "completeness" in results.quality_metrics:
                    completeness = results.quality_metrics["completeness"]
                    st.metric("Data Completeness", f"{completeness:.2%}")

            with col2:
                if "correlation_threshold" in results.quality_metrics:
                    threshold = results.quality_metrics["correlation_threshold"]
                    st.metric("Correlation Threshold", f"{threshold:.2f}")

            # Display feature metadata separately
            if "feature_metadata" in results.quality_metrics:
                st.subheader("🔍 Feature Categories")
                metadata = results.quality_metrics["feature_metadata"]

                for category, features in metadata.items():
                    if isinstance(features, list) and features:
                        st.write(f"**{category.replace('_', ' ').title()}:**")
                        feature_text = ", ".join(features)
                        st.write(feature_text)

            # Show other metrics as JSON if they exist
            other_metrics = {
                k: v
                for k, v in results.quality_metrics.items()
                if k
                not in [
                    "total_features",
                    "completeness",
                    "correlation_threshold",
                    "feature_metadata",
                ]
            }
            if other_metrics:
                st.subheader("📋 Additional Metrics")
                st.json(other_metrics)

        # Feature importance if available
        if (
            hasattr(results, "feature_importance")
            and results.feature_importance is not None
        ):
            chart = create_feature_importance_chart(
                results.feature_importance, title="Feature Importance Analysis"
            )
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info(
                "Feature importance analysis requires target variable. This will be available when training models."
            )


def display_feature_visualization(feature_set_key: str):
    """Display feature visualization"""

    if feature_set_key in st.session_state.feature_cache:
        cached_data = st.session_state.feature_cache[feature_set_key]

        # Check if we have the new DataFrame format or old dict format
        if isinstance(cached_data, pd.DataFrame):
            # New format: Direct DataFrame
            feature_df = cached_data
            metadata_key = f"{feature_set_key}_metadata"

            if metadata_key in st.session_state.feature_cache:
                metadata = st.session_state.feature_cache[metadata_key]
                original_data = metadata["original_data"]
                # Handle different metadata formats
                if "features_dict_for_chart" in metadata:
                    features_dict = metadata[
                        "features_dict_for_chart"
                    ]  # Technical indicators
                elif "features_dict" in metadata:
                    features_dict = metadata["features_dict"]  # Advanced features
                else:
                    features_dict = {col: feature_df[col] for col in feature_df.columns}
                feature_type = metadata["type"]
            else:
                # Fallback if no metadata
                original_data = feature_df
                features_dict = {col: feature_df[col] for col in feature_df.columns}
                feature_type = "unknown"
        else:
            # Old format: Dict with data and features
            cached_features = cached_data

            # Check if this is actually old format with 'features' key
            if isinstance(cached_features, dict) and "features" in cached_features:
                features_dict = cached_features["features"]
                original_data = cached_features["data"]
                feature_type = cached_features["type"]
            else:
                # This might be a metadata dict that shouldn't be here
                st.error(
                    f"❌ Invalid feature data format for {feature_set_key}. Please regenerate features."
                )
                st.write("Debug info - cached_data type:", type(cached_data))
                if isinstance(cached_data, dict):
                    st.write("Available keys:", list(cached_data.keys()))
                return

        # Ensure data is a DataFrame
        if not isinstance(original_data, pd.DataFrame):
            st.error("❌ Data format error: Expected DataFrame for visualization")
            return

        # Create chart based on feature type
        try:
            if feature_type == "technical":
                chart = create_technical_indicators_chart(
                    data=original_data,
                    indicators=features_dict,
                    height=700,
                    title="Technical Indicators Analysis",
                )
            else:
                chart = create_advanced_features_chart(
                    data=original_data,
                    features=features_dict,
                    height=700,
                    title="Advanced Features Analysis",
                )
            st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {e}")

            # Fallback: Show simple visualization
            st.subheader("📈 Feature Visualization (Fallback)")
            if isinstance(cached_data, pd.DataFrame):
                # Show correlation heatmap for DataFrame
                if len(cached_data.columns) > 1:
                    corr_matrix = cached_data.corr()
                    chart = create_correlation_heatmap(
                        corr_matrix, title="Feature Correlation Matrix"
                    )
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
            else:
                st.info("📊 Need at least 2 features to display correlation matrix")
    else:
        st.error(f"Feature set {feature_set_key} not found in cache")


def display_feature_statistics(feature_set_key: str):
    """Display feature statistics"""

    if feature_set_key in st.session_state.feature_cache:
        cached_data = st.session_state.feature_cache[feature_set_key]

        # Check if we have the new DataFrame format or old dict format
        if isinstance(cached_data, pd.DataFrame):
            # New format: Direct DataFrame
            feature_df = cached_data
        else:
            # Old format: Dict with data and features
            cached_features = cached_data

            # Check if this is actually old format with 'features' key
            if isinstance(cached_features, dict) and "features" in cached_features:
                features = cached_features["features"]

                # Convert dict to DataFrame for statistics
                if isinstance(features, dict):
                    feature_df = pd.DataFrame()
                    for name, values in features.items():
                        if isinstance(values, pd.Series):
                            feature_df[name] = values
                else:
                    feature_df = features
            else:
                st.error(
                    f"❌ Invalid feature data format for {feature_set_key}. Please regenerate features."
                )
                return

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
        else:
            st.warning("No feature data available for statistics")

    elif feature_set_key in st.session_state.feature_pipeline_cache:
        cached_pipeline = st.session_state.feature_pipeline_cache[feature_set_key]
        results = cached_pipeline["results"]

        if results.features is not None:
            st.subheader("📊 Pipeline Statistics")
            st.dataframe(results.features.describe(), use_container_width=True)
    else:
        st.error(f"Feature set {feature_set_key} not found in cache")


def display_feature_quality_analysis(feature_set_key: str):
    """Display feature quality analysis"""

    if feature_set_key in st.session_state.feature_cache:
        cached_data = st.session_state.feature_cache[feature_set_key]

        # Check if we have the new DataFrame format or old dict format
        if isinstance(cached_data, pd.DataFrame):
            # New format: Direct DataFrame
            feature_df = cached_data
        else:
            # Old format: Dict with data and features
            try:
                cached_features = cached_data
                if "features" not in cached_features:
                    st.error(
                        f"Invalid data format for {feature_set_key}: missing 'features' key"
                    )
                    st.debug(
                        f"Available keys: {list(cached_features.keys()) if hasattr(cached_features, 'keys') else 'Not a dict'}"
                    )
                    return

                features = cached_features["features"]

                # Convert dict to DataFrame for quality analysis
                if isinstance(features, dict):
                    feature_df = pd.DataFrame()
                    for name, values in features.items():
                        if isinstance(values, pd.Series):
                            feature_df[name] = values
                else:
                    feature_df = features
            except Exception as e:
                st.error(
                    f"Error processing feature data for {feature_set_key}: {str(e)}"
                )
                st.debug(f"Data type: {type(cached_data)}")
                return

        if not feature_df.empty:
            display_feature_quality_metrics(
                feature_df, title="🔍 Feature Quality Analysis"
            )
        else:
            st.warning("No feature data available for quality analysis")

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
    else:
        st.error(f"Feature set {feature_set_key} not found in cache")


if __name__ == "__main__":
    main()
