"""
Test suite for analysis module initialization.
"""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_analysis_module_imports():
    """Test that all analysis modules can be imported."""
    try:
        from analysis.returns import ReturnAnalyzer
        from analysis.volatility import VolatilityAnalyzer
        from analysis.statistics import StatisticsAnalyzer
        from analysis.correlation import CorrelationAnalyzer

        assert ReturnAnalyzer is not None
        assert VolatilityAnalyzer is not None
        assert StatisticsAnalyzer is not None
        assert CorrelationAnalyzer is not None

    except ImportError as e:
        pytest.fail(f"Failed to import analysis modules: {e}")


def test_analysis_init_imports():
    """Test that the main analysis module imports work."""
    try:
        import analysis

        # Test that main classes are available
        assert hasattr(analysis, "ReturnAnalyzer")
        assert hasattr(analysis, "VolatilityAnalyzer")
        assert hasattr(analysis, "StatisticsAnalyzer")
        assert hasattr(analysis, "CorrelationAnalyzer")

    except ImportError as e:
        pytest.fail(f"Failed to import analysis package: {e}")


if __name__ == "__main__":
    test_analysis_module_imports()
    test_analysis_init_imports()
    print("All analysis module import tests passed!")
