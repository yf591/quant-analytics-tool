"""
Streamlit Application Utilities

This module provides utility functions for the Streamlit application,
implementing clean separation of concerns between UI and business logic.

Available Utilities:
- data_utils: Data acquisition and management utilities
- feature_utils: Feature engineering utilities
- model_utils: Model training utilities
- backtest_utils: Backtesting utilities

Design Principles:
- Separation of Concerns: UI logic vs Business logic
- Testability: Pure Python functions without Streamlit dependencies
- Reusability: Functions can be used across different pages
- Maintainability: Easy to modify business logic without touching UI
"""

from .data_utils import DataAcquisitionManager
from .feature_utils import FeatureEngineeringManager
from .model_utils import ModelTrainingManager

__all__ = [
    "DataAcquisitionManager",
    "FeatureEngineeringManager",
    "ModelTrainingManager",
]

__version__ = "1.0.0"
