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
        # Flexible column name handling
        actual_price_col = None
        actual_volume_col = None

        # Find price column (case-insensitive)
        for col in data.columns:
            if col.lower() == price_col.lower():
                actual_price_col = col
                break

        if actual_price_col is None:
            # Try common alternatives
            for alt_col in ["Close", "close", "CLOSE", "Price", "price"]:
                if alt_col in data.columns:
                    actual_price_col = alt_col
                    break

        if actual_price_col is None:
            st.error(f"Price column '{price_col}' not found in data")
            return go.Figure()

        # Find volume column if specified
        if volume_col:
            for col in data.columns:
                if col.lower() == volume_col.lower():
                    actual_volume_col = col
                    break

        # Create subplots if volume is available
        if actual_volume_col and actual_volume_col in data.columns:
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
                    y=data[actual_price_col],
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
                    y=data[actual_volume_col],
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
                    y=data[actual_price_col],
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
    indicators: Dict[str, Union[pd.Series, pd.DataFrame]] = None,
    height: int = 800,
    title: str = "Technical Indicators",
) -> go.Figure:
    """
    Create chart with price and technical indicators.

    Args:
        data: DataFrame with price data
        price_col: Name of price column
        indicators: Dictionary of indicator name to Series/DataFrame
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        if indicators is None:
            indicators = {}

        # Validate data input
        if not isinstance(data, pd.DataFrame):
            st.error("Data must be a DataFrame")
            return go.Figure()

        # Flexible column name handling for price
        actual_price_col = None
        for col in data.columns:
            if col.lower() == price_col.lower():
                actual_price_col = col
                break

        if actual_price_col is None:
            # Try common price column names
            for col in data.columns:
                if col.lower() in ["close", "price", "adj close"]:
                    actual_price_col = col
                    break

        if actual_price_col is None:
            st.error(f"Price column '{price_col}' not found in data")
            return go.Figure()

        # --- サブプロットの動的生成 ---
        subplot_titles = ["Price, Moving Averages & Bollinger Bands"]
        has_oscillators = any(
            "rsi" in k.lower() or "stochastic" in k.lower() or "williams" in k.lower()
            for k in indicators.keys()
        )
        has_macd = any("macd" in k.lower() for k in indicators.keys())
        has_atr = any("atr" in k.lower() for k in indicators.keys())

        if has_oscillators:
            subplot_titles.append("Oscillators (RSI, Stochastic, Williams %R)")
        if has_macd:
            subplot_titles.append("MACD")
        if has_atr:
            subplot_titles.append("ATR (Volatility)")

        # サブプロットを作成
        fig = make_subplots(
            rows=len(subplot_titles),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
        )

        # --- 各プロットへの描画 ---
        # 1. Price Chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[actual_price_col],
                name="Price",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # 行番号を動的に計算
        current_row = 2
        oscillator_row = current_row if has_oscillators else None
        if has_oscillators:
            current_row += 1
        macd_row = current_row if has_macd else None
        if has_macd:
            current_row += 1
        atr_row = current_row if has_atr else None

        # 2. 各インジケーターをループで描画
        colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        color_idx = 0

        for name, values in indicators.items():
            name_lower = name.lower()

            # Moving Averages (row 1)
            if "sma" in name_lower or "ema" in name_lower:
                fig.add_trace(
                    go.Scatter(
                        x=values.index,
                        y=values,
                        name=name.upper(),
                        line=dict(color=colors[color_idx % len(colors)], dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                color_idx += 1

            # Bollinger Bands (row 1)
            elif "bollinger" in name_lower:
                if isinstance(values, pd.DataFrame):
                    # Upper Band
                    if "Upper" in values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=values.index,
                                y=values["Upper"],
                                name="BB Upper",
                                line=dict(color="rgba(255,165,0,0.8)", dash="dash"),
                            ),
                            row=1,
                            col=1,
                        )
                    # Lower Band with fill
                    if "Lower" in values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=values.index,
                                y=values["Lower"],
                                name="BB Lower",
                                line=dict(color="rgba(255,165,0,0.8)", dash="dash"),
                                fill="tonexty",
                                fillcolor="rgba(255,165,0,0.1)",
                            ),
                            row=1,
                            col=1,
                        )
                    # Middle Band
                    if "Middle" in values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=values.index,
                                y=values["Middle"],
                                name="BB Middle",
                                line=dict(color="rgba(255,165,0,1.0)"),
                            ),
                            row=1,
                            col=1,
                        )

            # Oscillators (oscillator_row)
            elif oscillator_row and (
                "rsi" in name_lower
                or "stochastic" in name_lower
                or "williams" in name_lower
            ):
                if isinstance(values, pd.DataFrame):
                    # Stochastic case
                    if "%K" in values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=values.index,
                                y=values["%K"],
                                name="Stoch %K",
                                line=dict(color=colors[color_idx % len(colors)]),
                            ),
                            row=oscillator_row,
                            col=1,
                        )
                        color_idx += 1
                    if "%D" in values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=values.index,
                                y=values["%D"],
                                name="Stoch %D",
                                line=dict(color=colors[color_idx % len(colors)]),
                            ),
                            row=oscillator_row,
                            col=1,
                        )
                        color_idx += 1

                    # Add reference lines for Stochastic
                    fig.add_hline(
                        y=80,
                        line_dash="dash",
                        line_color="red",
                        row=oscillator_row,
                        col=1,
                    )
                    fig.add_hline(
                        y=20,
                        line_dash="dash",
                        line_color="green",
                        row=oscillator_row,
                        col=1,
                    )
                else:
                    # RSI, Williams %R (Series)
                    fig.add_trace(
                        go.Scatter(
                            x=values.index,
                            y=values,
                            name=name.upper(),
                            line=dict(color=colors[color_idx % len(colors)]),
                        ),
                        row=oscillator_row,
                        col=1,
                    )
                    color_idx += 1

                    # Add reference lines for RSI
                    if "rsi" in name_lower:
                        fig.add_hline(
                            y=70,
                            line_dash="dash",
                            line_color="red",
                            row=oscillator_row,
                            col=1,
                        )
                        fig.add_hline(
                            y=30,
                            line_dash="dash",
                            line_color="green",
                            row=oscillator_row,
                            col=1,
                        )

            # MACD (macd_row)
            elif macd_row and "macd" in name_lower:
                if isinstance(values, pd.DataFrame):
                    if "MACD" in values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=values.index,
                                y=values["MACD"],
                                name="MACD Line",
                                line=dict(color="#2ca02c"),
                            ),
                            row=macd_row,
                            col=1,
                        )
                    if "Signal" in values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=values.index,
                                y=values["Signal"],
                                name="Signal Line",
                                line=dict(color="#ff7f0e"),
                            ),
                            row=macd_row,
                            col=1,
                        )
                    if "Histogram" in values.columns:
                        fig.add_trace(
                            go.Bar(
                                x=values.index,
                                y=values["Histogram"],
                                name="MACD Histogram",
                                marker_color="rgba(255, 0, 0, 0.7)",
                            ),
                            row=macd_row,
                            col=1,
                        )

                    # Add zero line for MACD
                    fig.add_hline(
                        y=0, line_dash="dot", line_color="gray", row=macd_row, col=1
                    )

            # ATR (atr_row)
            elif atr_row and "atr" in name_lower:
                fig.add_trace(
                    go.Scatter(
                        x=values.index,
                        y=values,
                        name="ATR",
                        line=dict(color=colors[color_idx % len(colors)]),
                    ),
                    row=atr_row,
                    col=1,
                )
                color_idx += 1

        # レイアウト更新
        fig.update_layout(
            height=height,
            title=title,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )

        # Y軸のタイトルを設定
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if oscillator_row:
            fig.update_yaxes(title_text="Oscillator Values", row=oscillator_row, col=1)
        if macd_row:
            fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
        if atr_row:
            fig.update_yaxes(title_text="ATR", row=atr_row, col=1)

        fig.update_xaxes(title_text="Date", row=len(subplot_titles), col=1)

        return fig

    except Exception as e:
        st.error(f"Error creating technical indicators chart: {str(e)}")
        return go.Figure()

        # Validate data input
        if not isinstance(data, pd.DataFrame):
            st.error(f"Error: Expected DataFrame, got {type(data)}")
            return go.Figure()

        # Flexible column name handling for price
        actual_price_col = None
        for col in data.columns:
            if col.lower() == price_col.lower():
                actual_price_col = col
                break

        if actual_price_col is None:
            # Try common alternatives
            for alt_col in ["Close", "close", "CLOSE", "Price", "price"]:
                if alt_col in data.columns:
                    actual_price_col = alt_col
                    break

        if actual_price_col is None:
            st.error(f"Price column '{price_col}' not found in data")
            return go.Figure()

        # Determine number of subplots needed based on available indicators
        subplot_count = 1  # Always have price chart
        has_oscillators = any(
            "RSI" in name or "Stochastic" in name or "Williams" in name
            for name in indicators.keys()
        )
        has_macd = any("MACD" in name for name in indicators.keys())
        has_volume_indicators = any(
            "ATR" in name or "Volume" in name for name in indicators.keys()
        )

        if has_oscillators:
            subplot_count += 1
        if has_macd:
            subplot_count += 1
        if has_volume_indicators:
            subplot_count += 1

        # Create dynamic subplot titles and heights
        subplot_titles = ["Price with Moving Averages & Bollinger Bands"]
        row_heights = [0.4]

        if has_oscillators:
            subplot_titles.append("Oscillators (RSI, Stochastic, Williams %R)")
            row_heights.append(0.2)
        if has_macd:
            subplot_titles.append("MACD")
            row_heights.append(0.2)
        if has_volume_indicators:
            subplot_titles.append("Volume Indicators (ATR)")
            row_heights.append(0.2)

        # Normalize row heights to sum to 1
        total_height = sum(row_heights)
        row_heights = [h / total_height for h in row_heights]

        # Create subplots
        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
        )

        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[actual_price_col],
                mode="lines",
                name="Price",
                line=dict(color="#1f77b4", width=2),
            ),
            row=1,
            col=1,
        )

        # Add moving averages and Bollinger Bands to price chart
        colors = [
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
        color_idx = 0

        # Track current row for dynamic subplot assignment
        current_row = 1
        oscillator_row = None
        macd_row = None
        volume_row = None

        # Assign row numbers based on what indicators are available
        if has_oscillators:
            current_row += 1
            oscillator_row = current_row
        if has_macd:
            current_row += 1
            macd_row = current_row
        if has_volume_indicators:
            current_row += 1
            volume_row = current_row

        for name, series in indicators.items():
            if isinstance(series, pd.Series) and len(series) > 0:
                # Moving Averages
                if "SMA" in name or "EMA" in name or "MA" in name:
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=name,
                            line=dict(color=colors[color_idx % len(colors)], width=1),
                            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    color_idx += 1

                # Bollinger Bands
                elif "Bollinger" in name or "BB" in name:
                    if "Upper" in name:
                        fig.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode="lines",
                                name=name,
                                line=dict(
                                    color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"
                                ),
                                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
                            ),
                            row=1,
                            col=1,
                        )
                    elif "Lower" in name:
                        fig.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode="lines",
                                name=name,
                                line=dict(
                                    color="rgba(0, 255, 0, 0.5)", width=1, dash="dash"
                                ),
                                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
                            ),
                            row=1,
                            col=1,
                        )
                    else:  # Middle band
                        fig.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode="lines",
                                name=name,
                                line=dict(color="rgba(0, 0, 255, 0.7)", width=1),
                                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
                            ),
                            row=1,
                            col=1,
                        )

                # Oscillators (RSI, Stochastic, Williams %R)
                elif oscillator_row and (
                    "RSI" in name
                    or "Stochastic" in name
                    or "Williams" in name
                    or "Stoch" in name
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=name,
                            line=dict(color=colors[color_idx % len(colors)]),
                            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
                        ),
                        row=oscillator_row,
                        col=1,
                    )
                    color_idx += 1

                    # Add reference lines for RSI
                    if "RSI" in name:
                        fig.add_hline(
                            y=70,
                            line_dash="dash",
                            line_color="red",
                            row=oscillator_row,
                            col=1,
                            annotation_text="Overbought (70)",
                        )
                        fig.add_hline(
                            y=30,
                            line_dash="dash",
                            line_color="green",
                            row=oscillator_row,
                            col=1,
                            annotation_text="Oversold (30)",
                        )

                    # Add reference lines for Stochastic
                    elif "Stochastic" in name or "Stoch" in name:
                        fig.add_hline(
                            y=80,
                            line_dash="dash",
                            line_color="red",
                            row=oscillator_row,
                            col=1,
                        )
                        fig.add_hline(
                            y=20,
                            line_dash="dash",
                            line_color="green",
                            row=oscillator_row,
                            col=1,
                        )

                # MACD
                elif macd_row and "MACD" in name:
                    line_style = dict(color="#2ca02c")
                    if "Signal" in name:
                        line_style = dict(color="#ff7f0e", dash="dash")
                    elif "Histogram" in name:
                        # Use bar chart for MACD histogram
                        fig.add_trace(
                            go.Bar(
                                x=series.index,
                                y=series.values,
                                name=name,
                                marker_color="rgba(255, 0, 0, 0.7)",
                                hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
                            ),
                            row=macd_row,
                            col=1,
                        )
                        continue

                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=name,
                            line=line_style,
                            hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
                        ),
                        row=macd_row,
                        col=1,
                    )

                # Volume Indicators (ATR, etc.)
                elif volume_row and ("ATR" in name or "Volume" in name):
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=name,
                            line=dict(color=colors[color_idx % len(colors)]),
                            hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
                        ),
                        row=volume_row,
                        col=1,
                    )
                    color_idx += 1

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
        # Validate data input
        if not isinstance(data, pd.DataFrame):
            st.error(f"Error: Expected DataFrame, got {type(data)}")
            return go.Figure()

        # Validate features input
        if not isinstance(features, dict):
            st.error(f"Error: Expected dict for features, got {type(features)}")
            return go.Figure()

        fig = make_subplots(
            rows=len(features) + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["Price"] + list(features.keys()),
        )

        # Price chart - use flexible column matching
        price_col = None
        for col in data.columns:
            if col.lower() in ["close", "price"]:
                price_col = col
                break

        if price_col is None:
            # Try common alternatives
            for alt_col in ["Close", "CLOSE", "Price", "PRICE"]:
                if alt_col in data.columns:
                    price_col = alt_col
                    break

        if price_col and price_col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_col],
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


def create_information_bars_chart(
    bars_data: pd.DataFrame,
    height: int = 500,
    title: str = "Information-Driven Bars",
) -> go.Figure:
    """
    Create an interactive OHLC chart for information-driven bars.

    Args:
        bars_data: DataFrame with OHLC data for bars
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        required_cols = ["open", "high", "low", "close"]
        if not all(col in bars_data.columns for col in required_cols):
            st.error(
                f"Required columns {required_cols} not found in data. Available: {list(bars_data.columns)}"
            )
            return go.Figure()

        if len(bars_data) == 0:
            st.warning("No data available for information bars chart")
            return go.Figure()

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=bars_data.index,
                    open=bars_data["open"],
                    high=bars_data["high"],
                    low=bars_data["low"],
                    close=bars_data["close"],
                    name="Information Bars",
                )
            ]
        )

        # Add volume bars if available
        if "volume" in bars_data.columns:
            fig.add_trace(
                go.Bar(
                    x=bars_data.index,
                    y=bars_data["volume"],
                    name="Volume",
                    yaxis="y2",
                    opacity=0.3,
                ),
            )

            fig.update_layout(yaxis2=dict(title="Volume", overlaying="y", side="right"))

        fig.update_layout(
            title=title,
            xaxis_title="Time (Event-driven)",
            yaxis_title="Price",
            height=height,
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    except Exception as e:
        st.error(f"Error creating information bars chart: {str(e)}")
        return go.Figure()


def create_model_performance_chart(
    evaluation_data: Dict[str, Any],
    height: int = 400,
    title: str = "Model Performance Metrics",
) -> go.Figure:
    """
    Create model performance visualization chart.

    Args:
        evaluation_data: Dictionary containing evaluation metrics
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        # Extract metrics for classification or regression
        if "classification_metrics" in evaluation_data:
            metrics = evaluation_data["classification_metrics"]
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
            metric_values = [
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0),
                metrics.get("roc_auc", 0),
            ]
        elif "regression_metrics" in evaluation_data:
            metrics = evaluation_data["regression_metrics"]
            metric_names = ["R² Score", "MSE", "RMSE", "MAE"]
            metric_values = [
                metrics.get("r2_score", 0),
                metrics.get("mse", 0),
                metrics.get("rmse", 0),
                metrics.get("mae", 0),
            ]
        else:
            # Fallback for simple metrics
            train_score = evaluation_data.get("train_score", 0)
            test_score = evaluation_data.get("test_score", 0)
            metric_names = ["Train Score", "Test Score"]
            metric_values = [train_score, test_score]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    marker_color="lightblue",
                    hovertemplate="Metric: %{x}<br>Value: %{y:.4f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Score",
            height=height,
            template="plotly_white",
            showlegend=False,
        )

        return fig

    except Exception as e:
        st.error(f"Error creating model performance chart: {str(e)}")
        return go.Figure()


def create_confusion_matrix_chart(
    confusion_matrix: List[List[int]],
    class_labels: List[str] = None,
    height: int = 400,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """
    Create confusion matrix heatmap.

    Args:
        confusion_matrix: 2D array/list representing confusion matrix
        class_labels: List of class labels
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        import numpy as np

        # Convert to numpy array if needed
        cm = np.array(confusion_matrix)

        if class_labels is None:
            class_labels = [f"Class {i}" for i in range(len(cm))]

        # Create text annotations
        text = []
        for i in range(len(cm)):
            row_text = []
            for j in range(len(cm[0])):
                row_text.append(f"{cm[i][j]}")
            text.append(row_text)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=class_labels,
                y=class_labels,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale="Blues",
                hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=height,
            template="plotly_white",
        )

        return fig

    except Exception as e:
        st.error(f"Error creating confusion matrix chart: {str(e)}")
        return go.Figure()


def create_learning_curve_chart(
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: List[int] = None,
    height: int = 400,
    title: str = "Learning Curve",
) -> go.Figure:
    """
    Create learning curve chart showing training and validation scores.

    Args:
        train_scores: List of training scores
        val_scores: List of validation scores
        train_sizes: List of training set sizes
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        if train_sizes is None:
            train_sizes = list(range(1, len(train_scores) + 1))

        fig = go.Figure()

        # Training scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_scores,
                mode="lines+markers",
                name="Training Score",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
                hovertemplate="Size: %{x}<br>Training Score: %{y:.4f}<extra></extra>",
            )
        )

        # Validation scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_scores,
                mode="lines+markers",
                name="Validation Score",
                line=dict(color="red", width=2),
                marker=dict(size=6),
                hovertemplate="Size: %{x}<br>Validation Score: %{y:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            height=height,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    except Exception as e:
        st.error(f"Error creating learning curve chart: {str(e)}")
        return go.Figure()


def create_model_comparison_chart(
    comparison_data: List[Dict[str, Any]],
    metric_column: str = "Accuracy",
    height: int = 400,
    title: str = "Model Comparison",
) -> go.Figure:
    """
    Create model comparison bar chart.

    Args:
        comparison_data: List of dictionaries containing model data
        metric_column: Column name for the metric to compare
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        if not comparison_data:
            st.warning("No model comparison data available")
            return go.Figure()

        model_names = [
            data.get("Model", f"Model {i}") for i, data in enumerate(comparison_data)
        ]
        metric_values = [data.get(metric_column, 0) for data in comparison_data]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=model_names,
                    y=metric_values,
                    marker_color="lightcoral",
                    hovertemplate="Model: %{x}<br>"
                    + metric_column
                    + ": %{y:.4f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title=metric_column,
            height=height,
            template="plotly_white",
            showlegend=False,
            xaxis_tickangle=45,
        )

        return fig

    except Exception as e:
        st.error(f"Error creating model comparison chart: {str(e)}")
        return go.Figure()
