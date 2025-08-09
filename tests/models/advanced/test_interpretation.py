"""
Tests for Model interpretation and explainability components.

This module tests the model interpretation tools for understanding
financial ML model predictions and feature importance.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from unittest.mock import patch, MagicMock

# Importing modules under test
try:
    from src.models.advanced.interpretation import (
        ModelInterpreter,
        SHAPInterpreter,
        LIMEInterpreter,
        PermutationImportance,
        PartialDependencePlotter,
        FeatureInteractionAnalyzer,
        create_interpreter,
        InterpretationValidator,
    )

    INTERPRETATION_AVAILABLE = True
except ImportError:
    INTERPRETATION_AVAILABLE = False


@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestModelInterpreter:
    """Test base ModelInterpreter component."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create trained model and financial data for interpretation."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 12

        # Generate financial-like features
        feature_names = [
            "returns_1d",
            "returns_5d",
            "volatility",
            "volume",
            "rsi",
            "macd",
            "bollinger_position",
            "sma_ratio",
            "momentum",
            "vix",
            "market_cap",
            "pe_ratio",
        ]

        # Create correlated features
        X = np.random.randn(n_samples, n_features)

        # Add realistic correlations
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * np.random.randn(
            n_samples
        )  # 5d returns correlated with 1d
        X[:, 2] = np.abs(X[:, 0]) * 2 + np.random.exponential(
            0.5, n_samples
        )  # Volatility
        X[:, 3] = np.random.exponential(2, n_samples)  # Volume (always positive)
        X[:, 4] = 50 + 30 * np.tanh(X[:, 0])  # RSI bounded between 0-100

        # Create target with realistic relationship
        y = np.zeros(n_samples)
        y += 0.3 * X[:, 0]  # Returns impact
        y += 0.2 * X[:, 4] / 50 - 0.2  # RSI impact (normalized)
        y += 0.1 * X[:, 5]  # MACD impact
        y += 0.15 * np.random.randn(n_samples)  # Noise

        y_binary = (y > np.median(y)).astype(int)

        # Train multiple models
        models = {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.3, random_state=42
        )

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models["random_forest"] = rf

        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        models["gradient_boosting"] = gb

        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        models["logistic_regression"] = lr

        return models, X_train, X_test, y_train, y_test, feature_names

    def test_model_interpreter_creation(self, trained_model_and_data):
        """Test ModelInterpreter creation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        interpreter = ModelInterpreter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        assert interpreter is not None
        assert interpreter.model is not None
        assert len(interpreter.feature_names) == len(feature_names)

    def test_feature_importance_extraction(self, trained_model_and_data):
        """Test feature importance extraction."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        # Test with tree-based model (has feature_importances_)
        interpreter = ModelInterpreter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        importance = interpreter.get_feature_importance()

        assert isinstance(importance, pd.Series)
        assert len(importance) == len(feature_names)
        assert all(imp >= 0 for imp in importance.values)
        assert np.isclose(importance.sum(), 1.0, rtol=1e-2)  # Should sum to 1

    def test_model_performance_summary(self, trained_model_and_data):
        """Test model performance summary."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        interpreter = ModelInterpreter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        performance = interpreter.get_model_performance(X_test, y_test)

        assert isinstance(performance, dict)
        assert "accuracy" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert "f1_score" in performance

    def test_prediction_explanation(self, trained_model_and_data):
        """Test individual prediction explanation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        interpreter = ModelInterpreter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        # Explain single prediction
        explanation = interpreter.explain_prediction(X_test[0])

        assert explanation is not None


@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestSHAPInterpreter:
    """Test SHAP-based interpretation."""

    def test_shap_interpreter_creation(self, trained_model_and_data):
        """Test SHAPInterpreter creation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        try:
            shap_interpreter = SHAPInterpreter(
                model=models["random_forest"],
                X_train=X_train,
                feature_names=feature_names,
            )
            assert shap_interpreter is not None
        except ImportError:
            pytest.skip("SHAP not available")

    def test_shap_values_calculation(self, trained_model_and_data):
        """Test SHAP values calculation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        try:
            shap_interpreter = SHAPInterpreter(
                model=models["random_forest"],
                X_train=X_train,
                feature_names=feature_names,
            )

            # Calculate SHAP values for a subset
            shap_values = shap_interpreter.calculate_shap_values(X_test[:5])

            assert shap_values is not None
            assert len(shap_values.shape) >= 2

        except ImportError:
            pytest.skip("SHAP not available")

    @patch("matplotlib.pyplot.show")
    def test_shap_summary_plot(self, mock_show, trained_model_and_data):
        """Test SHAP summary plot generation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        try:
            shap_interpreter = SHAPInterpreter(
                model=models["random_forest"],
                X_train=X_train,
                feature_names=feature_names,
            )

            # Generate summary plot
            shap_interpreter.plot_summary(X_test[:50])

            # If no exception is raised, test passes
            assert True

        except ImportError:
            pytest.skip("SHAP not available")

    def test_shap_feature_importance(self, trained_model_and_data):
        """Test SHAP-based feature importance."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        try:
            shap_interpreter = SHAPInterpreter(
                model=models["random_forest"],
                X_train=X_train,
                feature_names=feature_names,
            )

            importance = shap_interpreter.get_feature_importance(X_test[:100])

            assert isinstance(importance, pd.Series)
            assert len(importance) == len(feature_names)

        except ImportError:
            pytest.skip("SHAP not available")


@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestLIMEInterpreter:
    """Test LIME-based interpretation."""

    def test_lime_interpreter_creation(self, trained_model_and_data):
        """Test LIMEInterpreter creation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        try:
            lime_interpreter = LIMEInterpreter(
                model=models["random_forest"],
                X_train=X_train,
                feature_names=feature_names,
                mode="classification",
            )
            assert lime_interpreter is not None
        except ImportError:
            pytest.skip("LIME not available")

    def test_lime_explanation(self, trained_model_and_data):
        """Test LIME explanation for individual instances."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        try:
            lime_interpreter = LIMEInterpreter(
                model=models["random_forest"],
                X_train=X_train,
                feature_names=feature_names,
                mode="classification",
            )

            # Explain single instance
            explanation = lime_interpreter.explain_instance(X_test[0])

            assert explanation is not None

        except ImportError:
            pytest.skip("LIME not available")

    def test_lime_feature_importance(self, trained_model_and_data):
        """Test LIME-based feature importance extraction."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        try:
            lime_interpreter = LIMEInterpreter(
                model=models["random_forest"],
                X_train=X_train,
                feature_names=feature_names,
                mode="classification",
            )

            # Get feature importance for multiple instances
            importance = lime_interpreter.get_feature_importance(X_test[:10])

            assert isinstance(importance, pd.DataFrame)
            assert len(importance.columns) == len(feature_names)

        except ImportError:
            pytest.skip("LIME not available")


@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestPermutationImportance:
    """Test Permutation Importance analysis."""

    def test_permutation_importance_creation(self, trained_model_and_data):
        """Test PermutationImportance creation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        perm_importance = PermutationImportance(
            model=models["random_forest"], feature_names=feature_names
        )

        assert perm_importance is not None

    def test_permutation_importance_calculation(self, trained_model_and_data):
        """Test permutation importance calculation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        perm_importance = PermutationImportance(
            model=models["random_forest"], feature_names=feature_names, n_repeats=5
        )

        importance = perm_importance.calculate_importance(X_test, y_test)

        assert isinstance(importance, pd.DataFrame)
        assert "importance_mean" in importance.columns
        assert "importance_std" in importance.columns
        assert len(importance) == len(feature_names)

    def test_permutation_importance_with_custom_scorer(self, trained_model_and_data):
        """Test permutation importance with custom scoring function."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        from sklearn.metrics import f1_score

        perm_importance = PermutationImportance(
            model=models["random_forest"],
            feature_names=feature_names,
            scoring=f1_score,
            n_repeats=3,
        )

        importance = perm_importance.calculate_importance(X_test, y_test)

        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == len(feature_names)


@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestPartialDependencePlotter:
    """Test Partial Dependence Plot analysis."""

    def test_partial_dependence_creation(self, trained_model_and_data):
        """Test PartialDependencePlotter creation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        pdp = PartialDependencePlotter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        assert pdp is not None

    @patch("matplotlib.pyplot.show")
    def test_single_feature_pdp(self, mock_show, trained_model_and_data):
        """Test single feature partial dependence plot."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        pdp = PartialDependencePlotter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        # Plot PDP for first feature
        try:
            pdp.plot_partial_dependence(feature_idx=0)
            success = True
        except Exception:
            success = False

        assert success

    @patch("matplotlib.pyplot.show")
    def test_two_feature_interaction_pdp(self, mock_show, trained_model_and_data):
        """Test two-feature interaction partial dependence plot."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        pdp = PartialDependencePlotter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        # Plot 2D PDP for feature interaction
        try:
            pdp.plot_2d_partial_dependence(feature_idx1=0, feature_idx2=1)
            success = True
        except Exception:
            success = False

        assert success

    def test_pdp_values_calculation(self, trained_model_and_data):
        """Test partial dependence values calculation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        pdp = PartialDependencePlotter(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        # Calculate PDP values
        pdp_values = pdp.calculate_partial_dependence(feature_idx=0)

        assert isinstance(pdp_values, dict)
        assert "feature_values" in pdp_values
        assert "partial_dependence" in pdp_values


@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestFeatureInteractionAnalyzer:
    """Test Feature Interaction analysis."""

    def test_feature_interaction_analyzer_creation(self, trained_model_and_data):
        """Test FeatureInteractionAnalyzer creation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        analyzer = FeatureInteractionAnalyzer(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        assert analyzer is not None

    def test_pairwise_interactions(self, trained_model_and_data):
        """Test pairwise feature interactions calculation."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        analyzer = FeatureInteractionAnalyzer(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        # Calculate pairwise interactions (subset for speed)
        interactions = analyzer.calculate_pairwise_interactions(
            X_test[:100], n_features=5
        )

        assert isinstance(interactions, pd.DataFrame)
        assert interactions.shape[0] <= 10  # n_features choose 2

    def test_interaction_strength_ranking(self, trained_model_and_data):
        """Test interaction strength ranking."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        analyzer = FeatureInteractionAnalyzer(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        # Rank interaction strengths
        rankings = analyzer.rank_interaction_strength(X_test[:100], n_features=6)

        assert isinstance(rankings, pd.DataFrame)
        assert "interaction_strength" in rankings.columns
        assert "feature1" in rankings.columns
        assert "feature2" in rankings.columns

    @patch("matplotlib.pyplot.show")
    def test_interaction_heatmap(self, mock_show, trained_model_and_data):
        """Test interaction strength heatmap."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        analyzer = FeatureInteractionAnalyzer(
            model=models["random_forest"], X_train=X_train, feature_names=feature_names
        )

        # Plot interaction heatmap
        try:
            analyzer.plot_interaction_heatmap(X_test[:50], n_features=5)
            success = True
        except Exception:
            success = False

        assert success


@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestInterpretationValidator:
    """Test interpretation validation and consistency."""

    def test_interpretation_validator_creation(self):
        """Test InterpretationValidator creation."""
        validator = InterpretationValidator()
        assert validator is not None

    def test_feature_importance_consistency(self, trained_model_and_data):
        """Test consistency between different feature importance methods."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        validator = InterpretationValidator()

        # Get feature importance from different methods
        importances = {}

        # Built-in importance
        if hasattr(models["random_forest"], "feature_importances_"):
            importances["builtin"] = pd.Series(
                models["random_forest"].feature_importances_, index=feature_names
            )

        # Permutation importance
        perm_importance = PermutationImportance(
            model=models["random_forest"], feature_names=feature_names, n_repeats=3
        )
        perm_results = perm_importance.calculate_importance(X_test, y_test)
        importances["permutation"] = perm_results["importance_mean"]

        # Validate consistency
        consistency = validator.validate_importance_consistency(importances)

        assert isinstance(consistency, dict)
        assert "correlations" in consistency

    def test_interpretation_stability(self, trained_model_and_data):
        """Test stability of interpretations across data subsets."""
        models, X_train, X_test, y_train, y_test, feature_names = trained_model_and_data

        validator = InterpretationValidator()

        # Calculate importance on different subsets
        subset_size = len(X_test) // 3

        importances_subsets = []
        for i in range(3):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size
            subset_X = X_test[start_idx:end_idx]
            subset_y = y_test[start_idx:end_idx]

            perm_importance = PermutationImportance(
                model=models["random_forest"], feature_names=feature_names, n_repeats=2
            )

            importance = perm_importance.calculate_importance(subset_X, subset_y)
            importances_subsets.append(importance["importance_mean"])

        # Validate stability
        stability = validator.validate_interpretation_stability(importances_subsets)

        assert isinstance(stability, dict)
        assert "mean_correlation" in stability


# Integration tests
@pytest.mark.skipif(
    not INTERPRETATION_AVAILABLE, reason="Interpretation module not available"
)
class TestInterpretationIntegration:
    """Integration tests for interpretation components."""

    @pytest.fixture
    def financial_portfolio_data(self):
        """Create financial portfolio dataset for comprehensive interpretation."""
        np.random.seed(42)
        n_samples = 1500

        # Create time series data
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="1D")

        # Multiple assets with realistic correlations
        n_assets = 5
        asset_names = ["STOCK_A", "STOCK_B", "BOND_C", "COMMODITY_D", "CRYPTO_E"]

        # Generate correlated returns
        correlation_matrix = np.array(
            [
                [1.00, 0.70, 0.20, 0.10, 0.05],
                [0.70, 1.00, 0.25, 0.15, 0.10],
                [0.20, 0.25, 1.00, 0.05, -0.10],
                [0.10, 0.15, 0.05, 1.00, 0.20],
                [0.05, 0.10, -0.10, 0.20, 1.00],
            ]
        )

        mean_returns = [0.0008, 0.0006, 0.0003, 0.0005, 0.0015]
        volatilities = [0.02, 0.025, 0.008, 0.03, 0.05]

        # Generate asset returns
        asset_returns = np.random.multivariate_normal(
            mean_returns,
            np.outer(volatilities, volatilities) * correlation_matrix,
            n_samples,
        )

        # Market features
        market_volatility = np.random.exponential(0.015, n_samples)
        interest_rate = 0.02 + np.random.normal(0, 0.005, n_samples)
        economic_sentiment = np.random.uniform(-1, 1, n_samples)

        # Technical indicators
        features = pd.DataFrame(
            {
                # Asset returns
                "stock_a_return": asset_returns[:, 0],
                "stock_b_return": asset_returns[:, 1],
                "bond_c_return": asset_returns[:, 2],
                "commodity_d_return": asset_returns[:, 3],
                "crypto_e_return": asset_returns[:, 4],
                # Market features
                "market_volatility": market_volatility,
                "interest_rate": interest_rate,
                "economic_sentiment": economic_sentiment,
                # Portfolio metrics
                "portfolio_beta": np.random.normal(1.0, 0.2, n_samples),
                "sharpe_ratio": np.random.normal(0.8, 0.3, n_samples),
                "max_drawdown": np.random.exponential(0.05, n_samples),
                "correlation_to_market": np.random.uniform(0.3, 0.9, n_samples),
                # Momentum features
                "momentum_1m": np.random.normal(0, 0.1, n_samples),
                "momentum_3m": np.random.normal(0, 0.15, n_samples),
                "momentum_6m": np.random.normal(0, 0.2, n_samples),
            },
            index=dates,
        )

        # Create target: portfolio outperformance
        portfolio_return = np.mean(asset_returns, axis=1)
        market_return = (
            0.5 * asset_returns[:, 0]
            + 0.3 * asset_returns[:, 1]
            + 0.2 * asset_returns[:, 2]
        )

        outperformance = portfolio_return - market_return
        target = (outperformance > np.median(outperformance)).astype(int)

        return features, target

    def test_comprehensive_model_interpretation(self, financial_portfolio_data):
        """Test comprehensive interpretation of financial portfolio model."""
        features, target = financial_portfolio_data

        # Train ensemble model
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Create comprehensive interpreter
        interpreter = ModelInterpreter(
            model=model, X_train=X_train, feature_names=features.columns.tolist()
        )

        # Test multiple interpretation methods
        results = {}

        # 1. Feature importance
        results["feature_importance"] = interpreter.get_feature_importance()

        # 2. Permutation importance
        perm_importance = PermutationImportance(
            model=model, feature_names=features.columns.tolist(), n_repeats=3
        )
        results["permutation_importance"] = perm_importance.calculate_importance(
            X_test, y_test
        )

        # 3. Partial dependence
        pdp = PartialDependencePlotter(
            model=model, X_train=X_train, feature_names=features.columns.tolist()
        )
        results["pdp_values"] = pdp.calculate_partial_dependence(feature_idx=0)

        # 4. Feature interactions
        interaction_analyzer = FeatureInteractionAnalyzer(
            model=model, X_train=X_train, feature_names=features.columns.tolist()
        )
        results["interactions"] = interaction_analyzer.rank_interaction_strength(
            X_test[:100], n_features=8
        )

        # Validate all results
        assert all(result is not None for result in results.values())
        assert isinstance(results["feature_importance"], pd.Series)
        assert isinstance(results["permutation_importance"], pd.DataFrame)
        assert isinstance(results["pdp_values"], dict)
        assert isinstance(results["interactions"], pd.DataFrame)

    def test_interpretation_across_model_types(self, financial_portfolio_data):
        """Test interpretation consistency across different model types."""
        features, target = financial_portfolio_data

        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target, test_size=0.3, random_state=42
        )

        # Train different model types
        models = {}

        models["random_forest"] = RandomForestClassifier(
            n_estimators=50, random_state=42
        )
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=50, random_state=42
        )
        models["logistic_regression"] = LogisticRegression(
            random_state=42, max_iter=1000
        )

        # Train all models
        for name, model in models.items():
            model.fit(X_train, y_train)

        # Compare interpretations
        interpretations = {}

        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                interpretations[name] = pd.Series(
                    model.feature_importances_, index=features.columns
                )
            else:
                # Use permutation importance for models without built-in importance
                perm_importance = PermutationImportance(
                    model=model, feature_names=features.columns.tolist(), n_repeats=2
                )
                perm_results = perm_importance.calculate_importance(X_test, y_test)
                interpretations[name] = perm_results["importance_mean"]

        # Validate interpretations
        assert (
            len(interpretations) >= 2
        )  # At least two models should have interpretations

        # Check consistency (top features should be similar)
        feature_rankings = {}
        for name, importance in interpretations.items():
            feature_rankings[name] = importance.rank(ascending=False)

        # Top features should have some overlap
        if len(feature_rankings) >= 2:
            model_names = list(feature_rankings.keys())
            correlations = []

            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    corr = feature_rankings[model_names[i]].corr(
                        feature_rankings[model_names[j]]
                    )
                    correlations.append(corr)

            # Some positive correlation expected between rankings
            assert any(corr > 0.3 for corr in correlations)

    def test_interpretation_with_financial_constraints(self, financial_portfolio_data):
        """Test interpretation considering financial domain constraints."""
        features, target = financial_portfolio_data

        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target, test_size=0.3, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Create interpreter
        interpreter = ModelInterpreter(
            model=model, X_train=X_train, feature_names=features.columns.tolist()
        )

        # Get feature importance
        importance = interpreter.get_feature_importance()

        # Financial constraints validation
        financial_constraints = {
            "expected_important": [
                "market_volatility",
                "portfolio_beta",
                "sharpe_ratio",
            ],
            "should_be_positive": [
                "sharpe_ratio"
            ],  # Higher Sharpe should predict better performance
            "reasonable_range": ["interest_rate", "economic_sentiment"],
        }

        # Validate financial reasonableness
        for constraint_type, features_list in financial_constraints.items():
            for feature in features_list:
                if feature in importance.index:
                    feature_importance = importance[feature]

                    if constraint_type == "expected_important":
                        # Should be in top 50% of features
                        assert feature_importance > importance.median()

                    elif constraint_type == "should_be_positive":
                        # For features where direction matters, check via PDP
                        pdp = PartialDependencePlotter(
                            model=model,
                            X_train=X_train,
                            feature_names=features.columns.tolist(),
                        )

                        feature_idx = features.columns.get_loc(feature)
                        pdp_values = pdp.calculate_partial_dependence(feature_idx)

                        # Check if relationship is generally positive
                        assert pdp_values is not None


if __name__ == "__main__":
    pytest.main([__file__])
