"""
Streamlit Components Module

This module provides reusable UI components for the Streamlit application.
Includes charts, forms, data display, and model widgets for professional UI integration.
"""

from .charts import (
    create_price_chart,
    create_technical_indicators_chart,
    create_feature_importance_chart,
    create_correlation_heatmap,
)

from .data_display import (
    display_data_metrics,
    display_feature_table,
    display_progress_bar,
    display_alert_message,
)

from .forms import (
    create_data_selection_form,
    create_parameter_form,
    create_file_upload_form,
)

__all__ = [
    "create_price_chart",
    "create_technical_indicators_chart",
    "create_feature_importance_chart",
    "create_correlation_heatmap",
    "display_data_metrics",
    "display_feature_table",
    "display_progress_bar",
    "display_alert_message",
    "create_data_selection_form",
    "create_parameter_form",
    "create_file_upload_form",
]

__version__ = "1.0.0"
__author__ = "Quant Analytics Tool"
