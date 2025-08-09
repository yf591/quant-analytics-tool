"""
Advanced Models Package

This package contains advanced machine learning models and techniques
specifically designed for financial time series analysis.

Modules:
- transformer: Transformer architecture for financial time series
- attention: Attention mechanisms and visualization tools
- ensemble: Ensemble methods including bagging, boosting, and stacking
- meta_labeling: Meta-labeling techniques for position sizing
- interpretation: Model interpretation and explainability tools
"""

# Module availability info
MODULE_AVAILABILITY = {
    "transformer": True,
    "attention": True,
    "ensemble": True,
    "meta_labeling": True,
    "interpretation": True,
}

# Import only key classes without causing circular imports
__all__ = [
    "MODULE_AVAILABILITY",
]
