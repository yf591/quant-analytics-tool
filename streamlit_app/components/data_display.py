"""
Data Display Component Module

This module provides data display components for the Streamlit application.
Includes metrics display, data tables, progress bars, and alert messages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime


def display_data_metrics(
    data: pd.DataFrame, title: str = "Data Overview", show_detailed: bool = False
) -> None:
    """
    Display key metrics about the dataset.

    Args:
        data: DataFrame to analyze
        title: Title for the metrics section
        show_detailed: Whether to show detailed statistics
    """
    try:
        st.subheader(title)

        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="📊 Total Rows", value=f"{len(data):,}")

        with col2:
            st.metric(label="📈 Columns", value=len(data.columns))

        with col3:
            missing_count = data.isnull().sum().sum()
            missing_pct = (missing_count / (len(data) * len(data.columns))) * 100
            st.metric(
                label="❌ Missing Values",
                value=f"{missing_count:,}",
                delta=f"{missing_pct:.1f}%",
            )

        with col4:
            if isinstance(data.index, pd.DatetimeIndex):
                try:
                    date_range = (data.index.max() - data.index.min()).days
                    st.metric(label="📅 Date Range", value=f"{date_range} days")
                except Exception:
                    st.metric(label="📅 Data Points", value=len(data))
            else:
                st.metric(label="📅 Data Points", value=len(data))

        # Detailed statistics if requested
        if show_detailed:
            st.subheader("📊 Detailed Statistics")

            # Numeric columns statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Numeric Columns Summary:**")
                st.dataframe(data[numeric_cols].describe())

            # Missing values breakdown
            if missing_count > 0:
                st.write("**Missing Values by Column:**")
                missing_df = pd.DataFrame(
                    {
                        "Column": data.columns,
                        "Missing Count": data.isnull().sum(),
                        "Missing %": (data.isnull().sum() / len(data)) * 100,
                    }
                ).sort_values("Missing Count", ascending=False)
                st.dataframe(missing_df[missing_df["Missing Count"] > 0])

    except Exception as e:
        st.error(f"Error displaying data metrics: {str(e)}")


def display_feature_table(
    features: Union[pd.DataFrame, Dict[str, pd.Series]],
    title: str = "Feature Overview",
    max_rows: int = 100,
    show_stats: bool = True,
) -> None:
    """
    Display feature data in an interactive table.

    Args:
        features: Feature data as DataFrame or dict of Series
        title: Title for the table section
        max_rows: Maximum number of rows to display
        show_stats: Whether to show feature statistics
    """
    try:
        st.subheader(title)

        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            # Align all series to common index
            common_index = None
            for series in features.values():
                if isinstance(series, pd.Series):
                    if common_index is None:
                        common_index = series.index
                    else:
                        common_index = common_index.intersection(series.index)

            if common_index is not None and len(common_index) > 0:
                feature_df = pd.DataFrame(
                    {
                        name: series.reindex(common_index)
                        for name, series in features.items()
                        if isinstance(series, pd.Series)
                    }
                )
            else:
                st.warning("No common index found in feature data")
                return
        else:
            feature_df = features.copy()

        if len(feature_df) == 0:
            st.warning("No feature data to display")
            return

        # Feature statistics
        if show_stats:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="Features Count", value=len(feature_df.columns))

            with col2:
                non_null_pct = (
                    (feature_df.count().sum())
                    / (len(feature_df) * len(feature_df.columns))
                ) * 100
                st.metric(label="Data Completeness", value=f"{non_null_pct:.1f}%")

            with col3:
                if len(feature_df.columns) > 1:
                    avg_correlation = (
                        feature_df.corr()
                        .abs()
                        .values[np.triu_indices_from(feature_df.corr().values, k=1)]
                        .mean()
                    )
                    st.metric(label="Avg Correlation", value=f"{avg_correlation:.3f}")

        # Display table
        st.write(f"**Feature Data (showing up to {max_rows} rows):**")
        display_df = feature_df.head(max_rows)

        # Ensure all columns are properly formatted for Arrow serialization
        try:
            # Convert any object columns to strings if they contain non-numeric data
            for col in display_df.columns:
                if display_df[col].dtype == "object":
                    # Try to convert to numeric first
                    try:
                        display_df[col] = pd.to_numeric(
                            display_df[col], errors="coerce"
                        )
                    except:
                        # If conversion fails, convert to string
                        display_df[col] = display_df[col].astype(str)
        except Exception as convert_error:
            st.warning(f"Data type conversion warning: {convert_error}")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=min(400, len(display_df) * 35 + 50),
        )

        # Download button
        if len(feature_df) > 0:
            csv_data = feature_df.to_csv()
            # Create unique key based on title and timestamp
            unique_key = f"download_{title.replace(' ', '_').replace('📊', '').replace('🧠', '').replace('⚡', '').strip()}_{int(datetime.now().timestamp())}"
            st.download_button(
                label="📥 Download Features as CSV",
                data=csv_data,
                file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=unique_key,
            )

    except Exception as e:
        st.error(f"Error displaying feature table: {str(e)}")


def display_progress_bar(
    current: int, total: int, title: str = "Progress", description: str = ""
) -> None:
    """
    Display a progress bar with current status.

    Args:
        current: Current progress value
        total: Total progress value
        title: Progress bar title
        description: Additional description
    """
    try:
        progress = min(current / total, 1.0) if total > 0 else 0

        st.write(f"**{title}**")
        if description:
            st.write(description)

        progress_bar = st.progress(progress)
        st.write(f"{current}/{total} ({progress*100:.1f}%)")

        return progress_bar

    except Exception as e:
        st.error(f"Error displaying progress bar: {str(e)}")
        return None


def display_alert_message(
    message: str,
    alert_type: str = "info",
    title: Optional[str] = None,
    expandable: bool = False,
) -> None:
    """
    Display styled alert messages.

    Args:
        message: Alert message content
        alert_type: Type of alert ('info', 'success', 'warning', 'error')
        title: Optional title for the alert
        expandable: Whether to make the alert expandable
    """
    try:
        display_message = f"**{title}**\n\n{message}" if title else message

        if expandable and title:
            with st.expander(title):
                if alert_type == "info":
                    st.info(message)
                elif alert_type == "success":
                    st.success(message)
                elif alert_type == "warning":
                    st.warning(message)
                elif alert_type == "error":
                    st.error(message)
                else:
                    st.write(message)
        else:
            if alert_type == "info":
                st.info(display_message)
            elif alert_type == "success":
                st.success(display_message)
            elif alert_type == "warning":
                st.warning(display_message)
            elif alert_type == "error":
                st.error(display_message)
            else:
                st.write(display_message)

    except Exception as e:
        st.error(f"Error displaying alert message: {str(e)}")


def display_feature_quality_metrics(
    features: pd.DataFrame, title: str = "Feature Quality Analysis"
) -> None:
    """
    Display feature quality metrics and analysis.

    Args:
        features: Feature DataFrame
        title: Title for the analysis section
    """
    try:
        st.subheader(title)

        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)

        # Completeness
        with col1:
            completeness = (features.count() / len(features)).mean()
            st.metric(
                label="📊 Avg Completeness",
                value=f"{completeness:.2%}",
                delta=f"{'Good' if completeness > 0.9 else 'Fair' if completeness > 0.7 else 'Poor'}",
            )

        # Variability
        with col2:
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 0:
                avg_cv = (numeric_features.std() / numeric_features.mean().abs()).mean()
                st.metric(
                    label="📈 Avg Variability",
                    value=f"{avg_cv:.3f}",
                    delta=f"{'Good' if 0.1 < avg_cv < 2.0 else 'Check'}",
                )

        # Correlation
        with col3:
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr().abs()
                avg_corr = corr_matrix.values[
                    np.triu_indices_from(corr_matrix.values, k=1)
                ].mean()
                st.metric(
                    label="🔗 Avg Correlation",
                    value=f"{avg_corr:.3f}",
                    delta=f"{'Good' if avg_corr < 0.7 else 'High'}",
                )

        # Outliers
        with col4:
            if len(numeric_features.columns) > 0:
                # Simple outlier detection using IQR
                outlier_counts = []
                for col in numeric_features.columns:
                    Q1 = numeric_features[col].quantile(0.25)
                    Q3 = numeric_features[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = (
                        (numeric_features[col] < (Q1 - 1.5 * IQR))
                        | (numeric_features[col] > (Q3 + 1.5 * IQR))
                    ).sum()
                    outlier_counts.append(outliers)

                avg_outliers = np.mean(outlier_counts) if outlier_counts else 0
                outlier_pct = (avg_outliers / len(features)) * 100
                st.metric(
                    label="⚠️ Avg Outliers",
                    value=f"{outlier_pct:.1f}%",
                    delta=f"{'Good' if outlier_pct < 5 else 'Check'}",
                )

        # Feature ranking by quality
        if len(numeric_features.columns) > 0:
            st.write("**Feature Quality Ranking:**")

            quality_scores = []
            for col in numeric_features.columns:
                completeness_score = features[col].count() / len(features)

                # Variability score (CV in reasonable range)
                mean_val = numeric_features[col].mean()
                std_val = numeric_features[col].std()
                if abs(mean_val) > 1e-10:
                    cv = std_val / abs(mean_val)
                    variability_score = (
                        1.0 if 0.1 <= cv <= 2.0 else max(0, 1 - abs(cv - 1) / 2)
                    )
                else:
                    variability_score = 0.0

                # Simple quality score
                quality_score = (completeness_score + variability_score) / 2
                quality_scores.append(
                    {
                        "Feature": col,
                        "Quality Score": quality_score,
                        "Completeness": completeness_score,
                        "Variability (CV)": cv if abs(mean_val) > 1e-10 else np.nan,
                    }
                )

            quality_df = pd.DataFrame(quality_scores).sort_values(
                "Quality Score", ascending=False
            )
            st.dataframe(quality_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying feature quality metrics: {str(e)}")


def display_computation_status(
    status: str, progress: Optional[float] = None, details: Optional[str] = None
) -> None:
    """
    Display computation status with optional progress.

    Args:
        status: Current status message
        progress: Progress value (0.0 to 1.0)
        details: Additional details
    """
    try:
        status_container = st.container()

        with status_container:
            if progress is not None:
                st.progress(progress)

            if status.lower().startswith("error"):
                st.error(status)
            elif status.lower().startswith("warning"):
                st.warning(status)
            elif status.lower().startswith("success"):
                st.success(status)
            else:
                st.info(status)

            if details:
                with st.expander("Show Details"):
                    st.write(details)

    except Exception as e:
        st.error(f"Error displaying computation status: {str(e)}")


def display_model_metrics(
    evaluation_data: Dict[str, Any], title: str = "Model Performance"
) -> None:
    """
    Display model performance metrics.

    Args:
        evaluation_data: Dictionary containing evaluation metrics
        title: Title for the metrics section
    """
    try:
        st.subheader(title)

        if "classification_metrics" in evaluation_data:
            metrics = evaluation_data["classification_metrics"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                accuracy = metrics.get("accuracy", 0)
                st.metric("Accuracy", f"{accuracy:.4f}")

            with col2:
                precision = metrics.get("precision", 0)
                st.metric("Precision", f"{precision:.4f}")

            with col3:
                recall = metrics.get("recall", 0)
                st.metric("Recall", f"{recall:.4f}")

            with col4:
                f1_score = metrics.get("f1_score", 0)
                st.metric("F1 Score", f"{f1_score:.4f}")

        elif "regression_metrics" in evaluation_data:
            metrics = evaluation_data["regression_metrics"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                r2 = metrics.get("r2_score", 0)
                st.metric("R² Score", f"{r2:.4f}")

            with col2:
                mse = metrics.get("mse", 0)
                st.metric("MSE", f"{mse:.4f}")

            with col3:
                rmse = metrics.get("rmse", 0)
                st.metric("RMSE", f"{rmse:.4f}")

            with col4:
                mae = metrics.get("mae", 0)
                st.metric("MAE", f"{mae:.4f}")

        else:
            # Simple metrics display
            train_score = evaluation_data.get("train_score", 0)
            test_score = evaluation_data.get("test_score", 0)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Training Score", f"{train_score:.4f}")

            with col2:
                st.metric("Test Score", f"{test_score:.4f}")

    except Exception as e:
        st.error(f"Error displaying model metrics: {str(e)}")


def display_training_progress(
    progress_data: Dict[str, Any] = None, title: str = "Training Progress"
) -> None:
    """
    Display training progress information.

    Args:
        progress_data: Dictionary containing progress information
        title: Title for the progress section
    """
    try:
        st.subheader(title)

        if progress_data is None:
            st.info("No active training session.")
            return

        # Progress bar
        progress = progress_data.get("progress", 0)
        st.progress(progress)

        # Status information
        status = progress_data.get("status", "Unknown")
        st.write(f"**Status:** {status}")

        # Time information
        if "start_time" in progress_data:
            start_time = progress_data["start_time"]
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)

            elapsed = datetime.now() - start_time
            st.write(f"**Elapsed Time:** {elapsed}")

        # Model information
        if "model_config" in progress_data:
            config = progress_data["model_config"]
            st.write(f"**Model Type:** {config.get('model_type', 'Unknown')}")
            st.write(f"**Task Type:** {config.get('task_type', 'Unknown')}")

        # Current metrics
        if "metrics" in progress_data:
            metrics = progress_data["metrics"]
            with st.expander("Current Metrics", expanded=True):
                for metric, value in metrics.items():
                    st.metric(metric.replace("_", " ").title(), f"{value:.4f}")

    except Exception as e:
        st.error(f"Error displaying training progress: {str(e)}")


def display_model_comparison(
    comparison_data: List[Dict[str, Any]], title: str = "Model Comparison"
) -> None:
    """
    Display model comparison table.

    Args:
        comparison_data: List of dictionaries containing model data
        title: Title for the comparison section
    """
    try:
        st.subheader(title)

        if not comparison_data:
            st.info("No models available for comparison.")
            return

        # Convert to DataFrame for better display
        df = pd.DataFrame(comparison_data)

        # Format numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(4)

        st.dataframe(df, use_container_width=True)

        # Highlight best performing model
        if len(df) > 1:
            # Try to find a performance metric to highlight
            performance_cols = [
                col
                for col in df.columns
                if any(
                    metric in col.lower()
                    for metric in ["accuracy", "f1", "r2", "score"]
                )
            ]

            if performance_cols:
                best_col = performance_cols[0]
                best_idx = df[best_col].idxmax()
                best_model = (
                    df.loc[best_idx, "Model"]
                    if "Model" in df.columns
                    else f"Model {best_idx}"
                )
                best_score = df.loc[best_idx, best_col]

                st.success(
                    f"🏆 **Best Model:** {best_model} ({best_col}: {best_score:.4f})"
                )

    except Exception as e:
        st.error(f"Error displaying model comparison: {str(e)}")
