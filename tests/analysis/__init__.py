"""
Test Suite for Complete Analysis Package

This package contains comprehensive test suites for all analysis components:

Basic Analysis Components:
- Return analysis tests
- Volatility analysis tests
- Statistics analysis tests
- Correlation analysis tests

Advanced Analysis Components (Week 13):
- Walk-forward analysis tests
- Monte Carlo simulation tests
- Sensitivity analysis tests
- Stress testing tests
- Performance attribution tests

All tests follow pytest conventions and include unit tests, integration tests,
and edge case validation to ensure AFML compliance and production readiness.
"""

import pytest
import sys
import os

__version__ = "1.0.0"
__description__ = "Test suite for complete analysis package"

# Test configuration
TEST_DATA_SIZE = 252  # One year of daily data
TEST_RANDOM_STATE = 42
TEST_TOLERANCE = 1e-6


def get_test_config():
    """
    Get standard test configuration for all test modules.

    Returns:
        Dictionary containing test configuration
    """
    return {
        "data_size": TEST_DATA_SIZE,
        "random_state": TEST_RANDOM_STATE,
        "tolerance": TEST_TOLERANCE,
        "confidence_levels": [0.95, 0.99],
        "simulation_runs": 1000,  # Reduced for testing speed
    }


def test_basic_analysis_module_imports():
    """Test that basic analysis modules can be imported."""
    try:
        # Basic analysis modules
        from src.analysis.returns import ReturnAnalyzer
        from src.analysis.volatility import VolatilityAnalyzer
        from src.analysis.statistics import StatisticsAnalyzer
        from src.analysis.correlation import CorrelationAnalyzer

        assert ReturnAnalyzer is not None
        assert VolatilityAnalyzer is not None
        assert StatisticsAnalyzer is not None
        assert CorrelationAnalyzer is not None

    except ImportError as e:
        pytest.fail(f"Failed to import basic analysis modules: {e}")


def test_advanced_analysis_module_imports():
    """Test that advanced analysis modules (Week 13) can be imported."""
    try:
        # Advanced analysis modules (Week 13)
        from src.analysis.walk_forward import (
            WalkForwardAnalyzer,
            PurgedGroupTimeSeriesSplit,
        )
        from src.analysis.monte_carlo import MonteCarloAnalyzer
        from src.analysis.sensitivity import SensitivityAnalyzer
        from src.analysis.stress_testing import AdvancedStressTester
        from src.analysis.performance_attribution import PerformanceAttributionAnalyzer

        assert WalkForwardAnalyzer is not None
        assert PurgedGroupTimeSeriesSplit is not None
        assert MonteCarloAnalyzer is not None
        assert SensitivityAnalyzer is not None
        assert AdvancedStressTester is not None
        assert PerformanceAttributionAnalyzer is not None

    except ImportError as e:
        pytest.fail(f"Failed to import advanced analysis modules: {e}")


if __name__ == "__main__":
    test_basic_analysis_module_imports()
    test_advanced_analysis_module_imports()
    print("All analysis module import tests passed!")

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

__version__ = "1.0.0"
__description__ = "Test suite for complete analysis package"

# Test configuration
TEST_DATA_SIZE = 252  # One year of daily data
TEST_RANDOM_STATE = 42
TEST_TOLERANCE = 1e-6


def get_test_config():
    """
    Get standard test configuration for all test modules.

    Returns:
        Dictionary containing test configuration
    """
    return {
        "data_size": TEST_DATA_SIZE,
        "random_state": TEST_RANDOM_STATE,
        "tolerance": TEST_TOLERANCE,
        "confidence_levels": [0.95, 0.99],
        "simulation_runs": 1000,  # Reduced for testing speed
    }


def test_analysis_module_imports():
    """Test that all analysis modules can be imported."""
    try:
        # Basic analysis modules
        from analysis.returns import ReturnAnalyzer
        from analysis.volatility import VolatilityAnalyzer
        from analysis.statistics import StatisticsAnalyzer
        from analysis.correlation import CorrelationAnalyzer

        # Advanced analysis modules (Week 13)
        from analysis.walk_forward import (
            WalkForwardAnalyzer,
            PurgedGroupTimeSeriesSplit,
        )
        from analysis.monte_carlo import MonteCarloAnalyzer
        from analysis.sensitivity import SensitivityAnalyzer
        from analysis.stress_testing import AdvancedStressTester
        from analysis.performance_attribution import PerformanceAttributionAnalyzer

        assert ReturnAnalyzer is not None
        assert VolatilityAnalyzer is not None
        assert StatisticsAnalyzer is not None
        assert CorrelationAnalyzer is not None
        assert WalkForwardAnalyzer is not None
        assert PurgedGroupTimeSeriesSplit is not None
        assert MonteCarloAnalyzer is not None
        assert SensitivityAnalyzer is not None
        assert AdvancedStressTester is not None
        assert PerformanceAttributionAnalyzer is not None

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
        assert hasattr(analysis, "WalkForwardAnalyzer")
        assert hasattr(analysis, "MonteCarloAnalyzer")
        assert hasattr(analysis, "SensitivityAnalyzer")
        assert hasattr(analysis, "AdvancedStressTester")
        assert hasattr(analysis, "PerformanceAttributionAnalyzer")

    except ImportError as e:
        pytest.fail(f"Failed to import analysis package: {e}")


if __name__ == "__main__":
    test_analysis_module_imports()
    test_analysis_init_imports()
    print("All analysis module import tests passed!")
