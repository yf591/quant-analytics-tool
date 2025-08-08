"""
Feature Engineering Module

This module implements feature engineering capabilities following
"Advances in Financial Machine Learning" (AFML) methodologies.

Modules:
    - technical: Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
    - advanced: Advanced features (Fractal dimension, Hurst exponent, Information-driven bars)
    - labeling: Meta-labeling methods (Triple barrier method)
    - pipeline: Automated feature generation and selection
"""

from .technical import (
    TechnicalIndicators,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_stochastic,
    calculate_williams_r,
    calculate_cci,
    calculate_momentum,
)

__all__ = [
    # Technical Indicators
    "TechnicalIndicators",
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_stochastic",
    "calculate_williams_r",
    "calculate_cci",
    "calculate_momentum",
]

__version__ = "1.0.0"
__author__ = "Quant Analytics Tool"
