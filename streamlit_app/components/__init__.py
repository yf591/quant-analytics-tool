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

# Model widgets (Week 14 addition)
try:
    from .model_widgets import (
        ModelSelectionWidget,
        HyperparameterWidget,
        ModelComparisonWidget,
        ProgressWidget,
    )

    _model_widgets_available = True
except ImportError:
    _model_widgets_available = False

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

# Add model widgets to __all__ if available
if _model_widgets_available:
    __all__.extend(
        [
            "ModelSelectionWidget",
            "HyperparameterWidget",
            "ModelComparisonWidget",
            "ProgressWidget",
        ]
    )

__version__ = "1.0.0"
__author__ = "Quant Analytics Tool"
