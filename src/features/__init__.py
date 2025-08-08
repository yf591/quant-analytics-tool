"""
Feature Engineering Module

This module provides comprehensive feature engineering capabilities for financial data analysis.
Includes technical indicators, advanced features, feature pipeline, importance analysis, and quality validation.
"""

from .technical import TechnicalIndicators, TechnicalIndicatorResults
from .advanced import AdvancedFeatures, AdvancedFeatureResults
from .pipeline import FeaturePipeline, FeaturePipelineConfig, FeaturePipelineResults
from .importance import FeatureImportance, FeatureImportanceResults
from .validation import FeatureQualityValidator, FeatureQualityResults

__all__ = [
    "TechnicalIndicators",
    "TechnicalIndicatorResults",
    "AdvancedFeatures",
    "AdvancedFeatureResults",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "FeaturePipelineResults",
    "FeatureImportance",
    "FeatureImportanceResults",
    "FeatureQualityValidator",
    "FeatureQualityResults",
]

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
