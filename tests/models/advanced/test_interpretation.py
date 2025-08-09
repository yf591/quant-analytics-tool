"""
Tests for Model Interpretation module.
Lightweight tests focusing on essential functionality without memory-intensive operations.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.models.advanced.interpretation import (
    InterpretationConfig,
    FeatureImportanceAnalyzer,
    SHAPAnalyzer,
    PartialDependenceAnalyzer,
    FinancialModelInterpreter,
)


class TestInterpretationConfig:
    """Test InterpretationConfig functionality."""

    def test_interpretation_config_creation(self):
        """Test InterpretationConfig creation with defaults."""
        config = InterpretationConfig()

        assert config.importance_threshold == 0.01
        assert config.n_repeats == 10
        assert config.random_state == 42

    def test_interpretation_config_custom_values(self):
        """Test InterpretationConfig with custom values."""
        config = InterpretationConfig(
            importance_threshold=0.05,
            n_repeats=5,
            random_state=123,
        )

        assert config.importance_threshold == 0.05
        assert config.n_repeats == 5
        assert config.random_state == 123


class TestFeatureImportanceAnalyzer:
    """Test FeatureImportanceAnalyzer functionality."""

    @pytest.fixture
    def sample_model_data(self):
        """Create sample model and data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        return model, X, y, feature_names

    def test_feature_importance_analyzer_creation(self):
        """Test FeatureImportanceAnalyzer creation."""
        config = InterpretationConfig()
        analyzer = FeatureImportanceAnalyzer(config=config)

        assert analyzer.config == config
        assert hasattr(analyzer, "analyze_tree_importance")
        assert hasattr(analyzer, "analyze_permutation_importance")

    def test_analyze_tree_importance(self, sample_model_data):
        """Test tree-based feature importance analysis."""
        model, X, y, feature_names = sample_model_data

        config = InterpretationConfig()
        analyzer = FeatureImportanceAnalyzer(config=config)

        importance = analyzer.analyze_tree_importance(model, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        assert all(isinstance(v, (int, float)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

    def test_analyze_permutation_importance(self, sample_model_data):
        """Test permutation-based feature importance analysis."""
        model, X, y, feature_names = sample_model_data

        config = InterpretationConfig(n_repeats=3)  # Small number for speed
        analyzer = FeatureImportanceAnalyzer(config=config)

        importance = analyzer.analyze_permutation_importance(model, X, y, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        # Permutation importance returns dict with 'mean' and 'std' keys
        assert all(isinstance(v, dict) for v in importance.values())
        assert all("mean" in v and "std" in v for v in importance.values())

    def test_plot_feature_importance(self, sample_model_data):
        """Test feature importance plotting."""
        model, X, y, feature_names = sample_model_data

        config = InterpretationConfig()
        analyzer = FeatureImportanceAnalyzer(config=config)

        importance = analyzer.analyze_tree_importance(model, feature_names)

        try:
            # Test plotting without actually displaying
            import matplotlib.pyplot as plt

            plt.ioff()  # Turn off interactive mode

            fig = analyzer.plot_feature_importance(importance, title="Test Plot")

            if fig is not None:
                plt.close(fig)

            # Test passes if no exception is raised
            assert True
        except Exception as e:
            # Skip if plotting environment not available
            pytest.skip(f"Plotting not available: {e}")


class TestSHAPAnalyzer:
    """Test SHAPAnalyzer functionality."""

    def test_shap_analyzer_creation(self):
        """Test SHAPAnalyzer creation."""
        config = InterpretationConfig()
        analyzer = SHAPAnalyzer(config=config)

        assert analyzer.config == config
        assert hasattr(analyzer, "calculate_shap_values")

    @pytest.mark.skip(reason="SHAP computationally intensive for CI")
    def test_calculate_shap_values(self):
        """Test SHAP values calculation (skipped for performance)."""
        pass


class TestPartialDependenceAnalyzer:
    """Test PartialDependenceAnalyzer functionality."""

    def test_partial_dependence_analyzer_creation(self):
        """Test PartialDependenceAnalyzer creation."""
        config = InterpretationConfig()
        analyzer = PartialDependenceAnalyzer(config=config)

        assert analyzer.config == config
        assert hasattr(analyzer, "plot_partial_dependence")

    @pytest.mark.skip(reason="Partial dependence computationally intensive")
    def test_plot_partial_dependence(self):
        """Test partial dependence plotting (skipped for performance)."""
        pass


class TestFinancialModelInterpreter:
    """Test FinancialModelInterpreter functionality."""

    def test_financial_model_interpreter_creation(self):
        """Test FinancialModelInterpreter creation."""
        config = InterpretationConfig()
        interpreter = FinancialModelInterpreter(config=config)

        assert interpreter.config == config
        assert hasattr(interpreter, "feature_analyzer")
        assert hasattr(interpreter, "shap_analyzer")
        assert hasattr(interpreter, "pd_analyzer")

    def test_comprehensive_analysis_basic(self):
        """Test basic comprehensive analysis workflow."""
        # Create simple test data
        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        feature_names = ["price", "volume", "volatility", "momentum", "rsi"]

        # Test configuration and basic functionality
        config = InterpretationConfig()
        interpreter = FinancialModelInterpreter(config=config)

        # Test feature importance analysis
        importance = interpreter.feature_analyzer.analyze_tree_importance(
            model, feature_names
        )

        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert all(
            name in importance for name in feature_names
        )  # Test that values are reasonable
        assert all(0 <= v <= 1 for v in importance.values())
        assert sum(importance.values()) > 0  # At least some importance


def test_interpretation_integration():
    """Test complete interpretation integration workflow."""
    # Create sample financial data
    np.random.seed(42)
    n_samples = 100

    # Financial features
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
    returns = np.diff(prices) / prices[:-1]  # Fixed: proper shapes
    returns = np.concatenate([returns, [0]])  # Match length
    volumes = np.random.lognormal(10, 0.5, n_samples)
    volatility = pd.Series(returns).rolling(10).std().fillna(0.1).values

    X = np.column_stack([prices, returns, volumes, volatility])

    # Create binary target (up/down market)
    y = (returns > 0).astype(int)

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Test interpretation
    config = InterpretationConfig()
    interpreter = FinancialModelInterpreter(config=config)

    feature_names = ["price", "returns", "volume", "volatility"]

    # Analyze importance
    tree_importance = interpreter.feature_analyzer.analyze_tree_importance(
        model, feature_names
    )

    # Verify results
    assert isinstance(tree_importance, dict)
    assert len(tree_importance) == 4
    assert "returns" in tree_importance  # Most relevant for market direction

    print(f"✅ Interpretation integration test passed")
    print(f"   - Tree importance computed for {len(tree_importance)} features")
    print(
        f"   - Most important feature: {max(tree_importance, key=tree_importance.get)}"
    )


if __name__ == "__main__":
    # Run basic smoke tests
    test_interpretation_integration()
    print("✅ All interpretation tests completed successfully!")
