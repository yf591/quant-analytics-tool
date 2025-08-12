"""
Test Suite for Advanced Analysis Package

This package contains comprehensive test suites for all advanced analysis components:
- Walk-forward analysis tests
- Monte Carlo simulation tests
- Sensitivity analysis tests
- Stress testing tests
- Performance attribution tests

All tests follow pytest conventions and include unit tests, integration tests,
and edge case validation to ensure AFML compliance and production readiness.
"""

__version__ = "1.0.0"
__description__ = "Test suite for advanced analysis components"

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
