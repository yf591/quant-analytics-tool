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
        create_information_bars_chart,
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
    """Calculate technical indicators using Week 4 TechnicalIndicators"""

    try:
        display_computation_status("🔄 Calculating technical indicators...", 0.1)

        # Validate and normalize data columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = []

        # Check for standard column names and their lowercase variants
        column_mapping = {}
        for col in required_columns:
            if col in data.columns:
                column_mapping[col] = col
            elif col.lower() in data.columns:
                column_mapping[col] = col.lower()
            else:
                missing_columns.append(col)

        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info(f"Available columns: {list(data.columns)}")
            return

        # Create normalized data with standard column names
        normalized_data = pd.DataFrame(index=data.index)
        for standard_col, actual_col in column_mapping.items():
            normalized_data[standard_col] = data[actual_col]

        # Remove any rows with NaN values in required columns
        before_clean = len(normalized_data)
        normalized_data = normalized_data.dropna()
        after_clean = len(normalized_data)

        if after_clean < before_clean:
            st.info(f"📊 Removed {before_clean - after_clean} rows with missing values")

        if len(normalized_data) < 50:
            st.warning(
                "⚠️ Insufficient data for technical analysis (need at least 50 data points)"
            )
            return

        display_computation_status("🔄 Initializing technical indicators...", 0.2)

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
        if config.get("momentum_enabled", False):
            indicators_to_calculate.append("momentum")

        if not indicators_to_calculate:
            st.warning(
                "⚠️ No indicators selected. Please enable at least one indicator."
            )
            return

        display_computation_status("🔄 Computing indicators...", 0.5)

        # Calculate all enabled indicators
        # Note: TechnicalIndicators expects lowercase column names
        lowercase_data = normalized_data.copy()
        lowercase_data.columns = [col.lower() for col in lowercase_data.columns]

        all_results = ti.calculate_all_indicators(
            lowercase_data, indicators_to_calculate
        )

        display_computation_status("🔄 Processing results...", 0.8)

        # ★★★ 重要な修正点：データ形式を分離 ★★★

        # 1. チャート関数に渡す用の辞書を作成（元の構造を維持）
        features_for_chart = {}
        for name, result_obj in all_results.items():
            if hasattr(result_obj, "values"):
                features_for_chart[name] = result_obj.values

        # 2. テーブル表示と後続処理用のDataFrameを作成（全カラムを展開）
        feature_df_for_table = pd.DataFrame(index=normalized_data.index)
        for name, values in features_for_chart.items():
            if isinstance(values, pd.DataFrame):
                # 複数カラムを持つ指標（MACD、Bollinger Bands、Stochastic等）
                for col in values.columns:
                    # 分かりやすい名前にする
                    if name == "bollinger_bands":
                        new_name = f"BB_{col}"
                    elif name == "stochastic":
                        new_name = f"Stoch_{col}"
                    else:
                        new_name = f"{name}_{col}"
                    feature_df_for_table[new_name] = values[col]
            elif isinstance(values, pd.Series):
                # 単一カラムの指標（SMA、EMA、RSI等）
                feature_df_for_table[name] = values

        if feature_df_for_table.empty:
            st.warning("⚠️ No technical indicators were calculated successfully.")
            st.info(f"📊 Requested indicators: {indicators_to_calculate}")
            return

        # 3. セッションステートに保存
        feature_key = f"{ticker}_technical"

        # テーブル表示やモデル学習用には、展開されたDataFrameを保存
        st.session_state.feature_cache[feature_key] = feature_df_for_table

        # メタデータには、チャート描画用の元の構造を保った辞書を保存
        st.session_state.feature_cache[f"{feature_key}_metadata"] = {
            "original_data": normalized_data,
            "features_dict_for_chart": features_for_chart,  # ★新しいキーで保存
            "type": "technical",
            "config": config,
            "calculated_at": datetime.now(),
            "indicators_count": len(features_for_chart),
        }

        display_computation_status(
            f"✅ Successfully calculated {len(features_for_chart)} technical indicators for {ticker}",
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

        # Validate and normalize data columns
        price_col = None
        volume_col = None

        # Check for price column (Close or close)
        if "Close" in data.columns:
            price_col = "Close"
        elif "close" in data.columns:
            price_col = "close"
        else:
            st.error("❌ No price column found (Close or close)")
            st.info(f"Available columns: {list(data.columns)}")
            return

        # Check for volume column (Volume or volume)
        if "Volume" in data.columns:
            volume_col = "Volume"
        elif "volume" in data.columns:
            volume_col = "volume"

        # Clean data for analysis
        clean_data = data.dropna(subset=[price_col])
        if len(clean_data) < len(data):
            st.info(
                f"📊 Removed {len(data) - len(clean_data)} rows with missing price data"
            )

        if len(clean_data) < 100:
            st.warning(
                "⚠️ Insufficient data for advanced features (need at least 100 data points)"
            )
            return

        # Use existing Week 5 module
        af = AdvancedFeatures()
        results = {}

        display_computation_status("🔄 Computing features...", 0.3)

        # Calculate enabled features
        if config.get("fractal_enabled", False):
            display_computation_status("🔄 Computing fractal dimension...", 0.4)
            try:
                results["fractal_dimension"] = af.calculate_fractal_dimension(
                    clean_data[price_col],
                    config.get("fractal_window", 100),
                    config.get("fractal_method", "higuchi"),
                )
            except Exception as e:
                st.warning(f"⚠️ Fractal dimension calculation failed: {str(e)}")

        if config.get("hurst_enabled", False):
            display_computation_status("🔄 Computing Hurst exponent...", 0.6)
            try:
                results["hurst_exponent"] = af.calculate_hurst_exponent(
                    clean_data[price_col],
                    config.get("hurst_window", 100),
                    config.get("hurst_method", "rs"),
                )
            except Exception as e:
                st.warning(f"⚠️ Hurst exponent calculation failed: {str(e)}")

        if config.get("info_bars_enabled", False) and volume_col:
            display_computation_status("🔄 Creating information bars...", 0.7)
            try:
                threshold = config.get("bar_threshold", None)
                if threshold == 0:
                    threshold = None
                results["information_bars"] = af.create_information_bars(
                    clean_data,
                    config.get("bar_type", "volume"),
                    threshold,
                    price_col,
                    volume_col,
                )
            except Exception as e:
                st.warning(f"⚠️ Information bars calculation failed: {str(e)}")

        if config.get("frac_diff_enabled", False):
            display_computation_status(
                "🔄 Computing fractional differentiation...", 0.8
            )
            try:
                results["fractional_diff"] = af.fractional_differentiation(
                    clean_data[price_col],
                    config.get("frac_diff_d", 0.4),
                    config.get("frac_diff_threshold", 0.01),
                )
            except Exception as e:
                st.warning(f"⚠️ Fractional differentiation calculation failed: {str(e)}")

        if not results:
            st.warning(
                "⚠️ No advanced features calculated successfully. Please check your data and try again."
            )
            return

        # Store results with additional metadata
        feature_key = f"{ticker}_advanced"

        # Convert features dict to DataFrame for Model Training compatibility
        feature_df = pd.DataFrame(index=clean_data.index)
        for name, values in results.items():
            if isinstance(values, pd.Series):
                feature_df[name] = values
            else:
                # Convert other types to Series if possible
                try:
                    feature_df[name] = pd.Series(values, index=clean_data.index)
                except:
                    st.warning(f"Could not convert {name} to Series, skipping")

        # Store DataFrame instead of dict for Model Training compatibility
        st.session_state.feature_cache[feature_key] = feature_df

        # Also store metadata separately if needed
        st.session_state.feature_cache[f"{feature_key}_metadata"] = {
            "original_data": clean_data,
            "features_dict": results,
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

        # Validate and normalize data
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        # Create column mapping for case-insensitive matching
        column_mapping = {}
        for col in required_columns:
            if col in data.columns:
                column_mapping[col] = col
            elif col.lower() in data.columns:
                column_mapping[col] = col.lower()

        # Create normalized data with standard column names
        normalized_data = pd.DataFrame(index=data.index)
        for standard_col, actual_col in column_mapping.items():
            if actual_col in data.columns:
                # Ensure numeric data types
                try:
                    normalized_data[standard_col] = pd.to_numeric(
                        data[actual_col], errors="coerce"
                    )
                except Exception:
                    normalized_data[standard_col] = data[actual_col]

        # Check if we have minimum required data
        if "Close" not in normalized_data.columns:
            st.error("❌ Price data (Close) is required for feature pipeline")
            return

        display_computation_status("🔄 Generating features...", 0.3)

        # Initialize feature pipeline with simplified configuration
        # Note: Since FeaturePipeline may have complex configuration requirements,
        # we'll create features manually based on the config

        all_features = pd.DataFrame(index=normalized_data.index)
        feature_metadata = {}

        # Add basic price features
        try:
            all_features["Price"] = normalized_data["Close"]
            all_features["Returns"] = normalized_data["Close"].pct_change()
            all_features["Log_Returns"] = np.log(
                normalized_data["Close"] / normalized_data["Close"].shift(1)
            )
            feature_metadata["basic_features"] = ["Price", "Returns", "Log_Returns"]
        except Exception as e:
            st.warning(f"⚠️ Basic features calculation failed: {str(e)}")

        # Add technical indicators if enabled
        if config.get("include_technical", True):
            display_computation_status("🔄 Adding technical indicators...", 0.5)
            try:
                from src.features.technical import (
                    calculate_sma,
                    calculate_ema,
                    calculate_rsi,
                )

                # Simple moving averages
                all_features["SMA_10"] = calculate_sma(normalized_data["Close"], 10)
                all_features["SMA_20"] = calculate_sma(normalized_data["Close"], 20)

                # Exponential moving averages
                all_features["EMA_10"] = calculate_ema(normalized_data["Close"], 10)
                all_features["EMA_20"] = calculate_ema(normalized_data["Close"], 20)

                # RSI
                all_features["RSI"] = calculate_rsi(normalized_data["Close"], 14)

                feature_metadata["technical_indicators"] = [
                    "SMA_10",
                    "SMA_20",
                    "EMA_10",
                    "EMA_20",
                    "RSI",
                ]

            except Exception as e:
                st.warning(f"⚠️ Technical indicators calculation failed: {str(e)}")

        # Add advanced features if enabled
        if config.get("include_advanced", True):
            display_computation_status("🔄 Adding advanced features...", 0.7)
            try:
                # Rolling volatility
                all_features["Volatility_20"] = (
                    normalized_data["Close"].rolling(20).std()
                )

                # Price momentum
                all_features["Momentum_5"] = normalized_data["Close"] - normalized_data[
                    "Close"
                ].shift(5)
                all_features["Momentum_10"] = normalized_data[
                    "Close"
                ] - normalized_data["Close"].shift(10)

                feature_metadata["advanced_features"] = [
                    "Volatility_20",
                    "Momentum_5",
                    "Momentum_10",
                ]

            except Exception as e:
                st.warning(f"⚠️ Advanced features calculation failed: {str(e)}")

        display_computation_status("🔄 Processing pipeline results...", 0.8)

        # Remove features with all NaN values
        all_features = all_features.dropna(axis=1, how="all")

        # Feature selection if enabled
        if config.get("feature_selection", True) and len(
            all_features.columns
        ) > config.get("max_features", 50):
            display_computation_status("🔄 Performing feature selection...", 0.9)

            # Simple correlation-based feature selection
            correlation_threshold = config.get("correlation_threshold", 0.8)
            correlation_matrix = all_features.corr().abs()

            # Find highly correlated features
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )

            # Select features to drop
            to_drop = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > correlation_threshold)
            ]

            if to_drop:
                all_features = all_features.drop(columns=to_drop)
                st.info(f"🔄 Removed {len(to_drop)} highly correlated features")

        # Create simplified pipeline results
        from dataclasses import dataclass

        @dataclass
        class SimplePipelineResults:
            features: pd.DataFrame = None
            feature_names: List[str] = None
            quality_metrics: Dict[str, Any] = None
            feature_importance: Optional[pd.Series] = None

        # Calculate quality metrics
        quality_metrics = {
            "total_features": len(all_features.columns),
            "completeness": (1 - all_features.isnull().mean().mean()),
            "correlation_threshold": config.get("correlation_threshold", 0.8),
            "feature_metadata": feature_metadata,
        }

        pipeline_results = SimplePipelineResults(
            features=all_features,
            feature_names=list(all_features.columns),
            quality_metrics=quality_metrics,
            feature_importance=None,  # Initialize as None for now
        )

        # Store pipeline results
        pipeline_key = f"{ticker}_pipeline"

        # Store the features DataFrame directly in feature_cache for Model Training compatibility
        st.session_state.feature_cache[pipeline_key] = all_features

        # Store metadata separately
        st.session_state.feature_cache[f"{pipeline_key}_metadata"] = {
            "original_data": normalized_data,
            "type": "pipeline",
            "config": config,
            "calculated_at": datetime.now(),
            "feature_metadata": feature_metadata,
        }

        # Also store in feature_pipeline_cache for pipeline-specific functionality
        st.session_state.feature_pipeline_cache[pipeline_key] = {
            "data": normalized_data,
            "results": pipeline_results,
            "config": config,
            "calculated_at": datetime.now(),
        }

        display_computation_status(
            f"✅ Feature pipeline completed successfully for {ticker} ({len(all_features.columns)} features)",
            1.0,
        )
        st.rerun()

    except Exception as e:
        display_computation_status(
            f"❌ Feature pipeline failed: {str(e)}", details=str(e)
        )


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

            # Safely display quality metrics, excluding complex objects
            display_metrics = {}
            for key, value in results.quality_metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    display_metrics[key] = value
                elif isinstance(value, dict):
                    # Display dict contents as string representation
                    display_metrics[f"{key}_info"] = str(value)
                else:
                    display_metrics[f"{key}_type"] = str(type(value).__name__)

            if display_metrics:
                quality_df = pd.DataFrame([display_metrics]).T
                quality_df.columns = ["Value"]
                st.dataframe(quality_df, use_container_width=True)
            else:
                st.write(
                    "Quality metrics available but cannot be displayed in table format"
                )
                st.json(results.quality_metrics)

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
