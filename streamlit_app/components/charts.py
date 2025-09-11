"""
Charts Component Module

This module provides interactive chart components using Plotly for the Streamlit application.
Includes price charts, technical indicators, feature visualization, and correlation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Any
from datetime import datetime


def create_price_chart(
    data: pd.DataFrame,
    price_col: str = "Close",
    volume_col: Optional[str] = "Volume",
    height: int = 400,
    title: str = "Price Chart",
) -> go.Figure:
    """
    Create an interactive price chart with optional volume.

    Args:
        data: DataFrame with price data
        price_col: Name of price column
        volume_col: Name of volume column (optional)
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        # Create subplots if volume is available
        if volume_col and volume_col in data.columns:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(title, "Volume"),
                row_heights=[0.7, 0.3],
            )

            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode="lines",
                    name="Price",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[volume_col],
                    name="Volume",
                    marker_color="#ff7f0e",
                    opacity=0.7,
                    hovertemplate="Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        else:
            # Price only chart
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode="lines",
                    name="Price",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                )
            )
            fig.update_layout(title=title)

        # Update layout
        fig.update_layout(
            height=height,
            template="plotly_white",
            showlegend=True,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price ($)"),
            hovermode="x unified",
        )

        return fig

    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return go.Figure()


def create_technical_indicators_chart(
    data: pd.DataFrame,
    price_col: str = "Close",
    indicators: Dict[str, pd.Series] = None,
    height: int = 600,
    title: str = "Technical Indicators",
) -> go.Figure:
    """
    Create chart with price and technical indicators.

    Args:
        data: DataFrame with price data
        price_col: Name of price column
        indicators: Dictionary of indicator name to Series
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        if indicators is None:
            indicators = {}

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(
                "Price with Moving Averages",
                "Oscillators (RSI, Stochastic)",
                "MACD",
            ),
            row_heights=[0.5, 0.25, 0.25],
        )

        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[price_col],
                mode="lines",
                name="Price",
                line=dict(color="#1f77b4", width=2),
            ),
            row=1,
            col=1,
        )

        # Add moving averages to price chart
        colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        color_idx = 0

        for name, series in indicators.items():
            if isinstance(series, pd.Series) and len(series) > 0:
                if "SMA" in name or "EMA" in name or "MA" in name:
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=name,
                            line=dict(color=colors[color_idx % len(colors)], width=1),
                        ),
                        row=1,
                        col=1,
                    )
                    color_idx += 1

                elif "RSI" in name:
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=name,
                            line=dict(color="#ff7f0e"),
                        ),
                        row=2,
                        col=1,
                    )
                    # Add RSI levels
                    fig.add_hline(
                        y=70, line_dash="dash", line_color="red", row=2, col=1
                    )
                    fig.add_hline(
                        y=30, line_dash="dash", line_color="green", row=2, col=1
                    )

                elif "MACD" in name:
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=name,
                            line=dict(color="#2ca02c"),
                        ),
                        row=3,
                        col=1,
                    )

        # Update layout
        fig.update_layout(
            height=height,
            title=title,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    except Exception as e:
        st.error(f"Error creating technical indicators chart: {str(e)}")
        return go.Figure()


def create_feature_importance_chart(
    importance_data: pd.Series,
    top_n: int = 20,
    height: int = 400,
    title: str = "Feature Importance",
) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.

    Args:
        importance_data: Series with feature names as index and importance as values
        top_n: Number of top features to display
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        # Get top features
        top_features = importance_data.nlargest(top_n)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation="h",
                    marker_color="lightblue",
                    hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=height,
            template="plotly_white",
            yaxis=dict(autorange="reversed"),  # Top feature at top
        )

        return fig

    except Exception as e:
        st.error(f"Error creating feature importance chart: {str(e)}")
        return go.Figure()


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    height: int = 500,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """
    Create correlation heatmap.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale="RdBu",
                zmid=0,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            height=height,
            template="plotly_white",
            xaxis=dict(tickangle=45),
            yaxis=dict(autorange="reversed"),
        )

        return fig

    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure()


def create_advanced_features_chart(
    data: pd.DataFrame,
    features: Dict[str, pd.Series],
    height: int = 500,
    title: str = "Advanced Features Analysis",
) -> go.Figure:
    """
    Create chart for advanced features like Fractal Dimension and Hurst Exponent.

    Args:
        data: Original price data
        features: Dictionary of feature name to Series
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        fig = make_subplots(
            rows=len(features) + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["Price"] + list(features.keys()),
        )

        # Price chart
        if "Close" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name="Price",
                    line=dict(color="#1f77b4"),
                ),
                row=1,
                col=1,
            )

        # Feature charts
        colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, (name, series) in enumerate(features.items()):
            if isinstance(series, pd.Series) and len(series) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode="lines",
                        name=name,
                        line=dict(color=colors[idx % len(colors)]),
                    ),
                    row=idx + 2,
                    col=1,
                )

        fig.update_layout(
            height=height,
            title=title,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
        )

        return fig

    except Exception as e:
        st.error(f"Error creating advanced features chart: {str(e)}")
        return go.Figure()
