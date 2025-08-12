"""
Test suite for sensitivity analysis implementation.

Tests parameter sensitivity, feature importance analysis, robustness testing,
and scenario sensitivity functionality.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.analysis.sensitivity import SensitivityAnalyzer, create_default_scenarios


class TestSensitivityAnalyzer:
    """Test suite for SensitivityAnalyzer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        n_features = 5

        # Create features with some structure
        data = np.random.randn(len(dates), n_features)
        X = pd.DataFrame(
            data, index=dates, columns=[f"feature_{i}" for i in range(n_features)]
        )

        # Create target with some predictable pattern
        y = pd.Series(
            np.where(X["feature_0"] + X["feature_1"] * 0.5 > 0, 1, 0),
            index=dates,
            name="target",
        )

        return X, y

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.model_type = "classifier"
        model.clone.return_value = model

        # Mock parameters
        model.learning_rate = 0.01
        model.max_depth = 5
        model.n_estimators = 100

        def mock_predict(X):
            # Simple prediction based on first two features
            return (X.iloc[:, 0] + X.iloc[:, 1] * 0.5 > 0).astype(int).values

        def mock_score(X, y):
            pred = mock_predict(X)
            return np.mean(pred == y.values)

        def mock_set_params(**params):
            for key, value in params.items():
                setattr(model, key, value)
            return model

        model.predict = mock_predict
        model.score = mock_score
        model.set_params = Mock(side_effect=mock_set_params)
        model.fit = Mock()  # Add feature importances for testing
        model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

        return model

    def test_analyzer_initialization(self):
        """Test SensitivityAnalyzer initialization."""
        analyzer = SensitivityAnalyzer(
            perturbation_range=0.15,
            n_permutations=50,
            confidence_level=0.99,
            random_state=123,
        )

        assert analyzer.perturbation_range == 0.15
        assert analyzer.n_permutations == 50
        assert analyzer.confidence_level == 0.99
        assert analyzer.random_state == 123
        assert analyzer.parameter_sensitivity_ == {}
        assert analyzer.feature_importance_ == {}
        assert analyzer.robustness_results_ == {}

    def test_default_initialization(self):
        """Test default initialization parameters."""
        analyzer = SensitivityAnalyzer()

        assert analyzer.perturbation_range == 0.1
        assert analyzer.n_permutations == 100
        assert analyzer.confidence_level == 0.95
        assert analyzer.random_state is None

    def test_parameter_sensitivity_analysis(self, sample_data, mock_model):
        """Test parameter sensitivity analysis."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Define parameter ranges for testing
        parameter_ranges = {
            "learning_rate": [0.001, 0.01, 0.1],
            "max_depth": [3, 5, 10],
            "n_estimators": [50, 100, 200],
        }

        results = analyzer.parameter_sensitivity_analysis(
            mock_model, X, y, parameter_ranges=parameter_ranges
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "baseline_score" in results
        assert "parameter_effects" in results
        assert "parameter_rankings" in results
        assert "stability_metrics" in results

        # Check that all parameters were tested
        assert len(results["parameter_effects"]) == 3
        assert "learning_rate" in results["parameter_effects"]
        assert "max_depth" in results["parameter_effects"]
        assert "n_estimators" in results["parameter_effects"]

        # Check parameter effect structure
        for param_name, effects in results["parameter_effects"].items():
            assert "values" in effects
            assert "scores" in effects
            assert "sensitivity_metrics" in effects
            assert "score_range" in effects
            assert "relative_impact" in effects

        # Check that model was fitted multiple times
        assert mock_model.fit.called
        assert mock_model.set_params.called

    def test_get_default_parameter_ranges(self, mock_model):
        """Test default parameter range generation."""
        analyzer = SensitivityAnalyzer()

        default_ranges = analyzer._get_default_parameter_ranges(mock_model)

        # Should include parameters that exist on the model
        assert "learning_rate" in default_ranges
        assert "max_depth" in default_ranges
        assert "n_estimators" in default_ranges

        # Check that ranges are lists
        for param_name, param_range in default_ranges.items():
            assert isinstance(param_range, list)
            assert len(param_range) > 1

    def test_calculate_parameter_sensitivity(self):
        """Test parameter sensitivity calculation."""
        analyzer = SensitivityAnalyzer()

        param_values = [0.01, 0.05, 0.1, 0.2]
        scores = [0.8, 0.85, 0.9, 0.75]
        baseline_score = 0.82

        sensitivity = analyzer._calculate_parameter_sensitivity(
            param_values, scores, baseline_score
        )

        assert isinstance(sensitivity, dict)
        assert "sensitivity" in sensitivity
        assert "normalized_sensitivity" in sensitivity
        assert "score_range" in sensitivity
        assert "param_range" in sensitivity

        # Check calculations
        expected_score_range = max(scores) - min(scores)
        expected_param_range = max(param_values) - min(param_values)

        assert sensitivity["score_range"] == expected_score_range
        assert sensitivity["param_range"] == expected_param_range

    def test_feature_importance_sensitivity(self, sample_data, mock_model):
        """Test feature importance sensitivity analysis."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(n_permutations=10, random_state=42)

        # Mock permutation importance
        with patch("src.analysis.sensitivity.permutation_importance") as mock_perm_imp:
            mock_result = Mock()
            mock_result.importances_mean = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
            mock_perm_imp.return_value = mock_result

            results = analyzer.feature_importance_sensitivity(
                mock_model, X, y, importance_methods=["permutation", "builtin"]
            )

        # Check results structure
        assert isinstance(results, dict)
        assert "methods" in results
        assert "consistency_metrics" in results
        assert "stable_features" in results
        assert "unstable_features" in results
        assert "perturbation_sensitivity" in results

        # Check methods results
        assert "permutation" in results["methods"]
        assert "builtin" in results["methods"]

        for method, method_results in results["methods"].items():
            assert "scores" in method_results
            assert "ranking" in method_results
            assert "top_features" in method_results

        # Check that model was fitted
        assert mock_model.fit.called

    def test_calculate_feature_importance_methods(self, sample_data, mock_model):
        """Test different feature importance calculation methods."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer()

        # Test builtin method
        builtin_importance = analyzer._calculate_feature_importance(
            mock_model, X, y, "builtin"
        )
        assert len(builtin_importance) == X.shape[1]
        np.testing.assert_array_equal(
            builtin_importance, mock_model.feature_importances_
        )

        # Test coefficient method (when model doesn't have coef_)
        # Add mock coef_ attribute for testing
        mock_model.coef_ = Mock()
        mock_model.coef_.ndim = 1  # Mock ndim attribute
        mock_model.coef_.__abs__ = Mock(
            return_value=np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        )

        coef_importance = analyzer._calculate_feature_importance(
            mock_model, X, y, "coefficient"
        )
        assert len(coef_importance) == X.shape[1]

        # Test fallback method
        fallback_importance = analyzer._calculate_feature_importance(
            mock_model, X, y, "unknown_method"
        )
        expected_uniform = np.ones(X.shape[1]) / X.shape[1]
        np.testing.assert_array_almost_equal(fallback_importance, expected_uniform)

    def test_analyze_importance_consistency(self):
        """Test feature importance consistency analysis."""
        analyzer = SensitivityAnalyzer()

        # Create mock importance methods results
        importance_methods = {
            "method1": {
                "ranking": [0, 1, 2, 3, 4],
                "scores": np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
            },
            "method2": {
                "ranking": [0, 2, 1, 4, 3],  # Similar but slightly different
                "scores": np.array([0.28, 0.22, 0.24, 0.13, 0.13]),
            },
        }

        consistency = analyzer._analyze_importance_consistency(importance_methods)

        assert isinstance(consistency, dict)
        assert "consistency_score" in consistency
        assert "n_comparisons" in consistency
        assert "pairwise_correlations" in consistency

        # Should have one comparison (method1 vs method2)
        assert consistency["n_comparisons"] == 1
        assert len(consistency["pairwise_correlations"]) == 1

    def test_identify_stable_features(self):
        """Test stable feature identification."""
        analyzer = SensitivityAnalyzer()

        # Create mock importance methods with overlapping top features
        importance_methods = {
            "method1": {"top_features": [0, 1, 2, 3, 4]},
            "method2": {"top_features": [0, 1, 3, 5, 6]},
            "method3": {"top_features": [0, 1, 2, 4, 7]},
        }

        stable_features, unstable_features = analyzer._identify_stable_features(
            importance_methods
        )

        # Features 0 and 1 appear in all methods (stable)
        assert 0 in stable_features
        assert 1 in stable_features

        # Other features appear in fewer methods (unstable)
        assert len(unstable_features) > 0

    def test_robustness_testing(self, sample_data, mock_model):
        """Test robustness testing functionality."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Use smaller test parameters
        noise_levels = [0.0, 0.1, 0.2]
        sample_fractions = [1.0, 0.8, 0.6]

        results = analyzer.robustness_testing(
            mock_model,
            X,
            y,
            noise_levels=noise_levels,
            sample_fractions=sample_fractions,
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "baseline_score" in results
        assert "noise_sensitivity" in results
        assert "sample_size_sensitivity" in results
        assert "missing_data_sensitivity" in results
        assert "combined_stress_test" in results

        # Check noise sensitivity
        noise_results = results["noise_sensitivity"]
        assert "noise_levels" in noise_results
        assert "scores" in noise_results
        assert "degradation_curve" in noise_results
        assert len(noise_results["scores"]) == len(noise_levels)

        # Check sample size sensitivity
        sample_results = results["sample_size_sensitivity"]
        assert "sample_fractions" in sample_results
        assert "scores" in sample_results
        assert len(sample_results["scores"]) == len(sample_fractions)

        # Check missing data sensitivity
        missing_results = results["missing_data_sensitivity"]
        assert "missing_fractions" in missing_results
        assert "scores" in missing_results

        # Check combined stress test
        stress_results = results["combined_stress_test"]
        assert "mild_stress" in stress_results
        assert "moderate_stress" in stress_results
        assert "high_stress" in stress_results

        # Check that model was fitted multiple times
        assert mock_model.fit.call_count > 10  # Should be many model fits

    def test_noise_robustness(self, sample_data, mock_model):
        """Test noise robustness testing."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Test different noise levels
        score_no_noise = analyzer._test_noise_robustness(mock_model, X, y, 0.0)
        score_with_noise = analyzer._test_noise_robustness(mock_model, X, y, 0.1)

        # Both should return valid scores
        assert isinstance(score_no_noise, (int, float))
        assert isinstance(score_with_noise, (int, float))

        # Noise typically degrades performance, but not always with mock model
        assert 0 <= score_no_noise <= 1
        assert 0 <= score_with_noise <= 1

    def test_sample_size_robustness(self, sample_data, mock_model):
        """Test sample size robustness testing."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Test different sample fractions
        score_full = analyzer._test_sample_size_robustness(mock_model, X, y, 1.0)
        score_half = analyzer._test_sample_size_robustness(mock_model, X, y, 0.5)

        # Both should return valid scores
        assert isinstance(score_full, (int, float))
        assert isinstance(score_half, (int, float))
        assert 0 <= score_full <= 1
        assert 0 <= score_half <= 1

    def test_missing_data_robustness(self, sample_data, mock_model):
        """Test missing data robustness testing."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        results = analyzer._test_missing_data_robustness(mock_model, X, y)

        assert isinstance(results, dict)
        assert "missing_fractions" in results
        assert "scores" in results
        assert "missing_data_effects" in results

        # Check that we get results for different missing data levels
        assert len(results["scores"]) > 0
        assert len(results["missing_data_effects"]) > 0

    def test_scenario_sensitivity_analysis(self, sample_data, mock_model):
        """Test scenario sensitivity analysis."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Define test scenarios
        scenarios = {
            "bull_market": {
                "feature_scale": 1.2,
                "target_shift": 0.01,
                "volatility_multiplier": 0.8,
            },
            "bear_market": {
                "feature_scale": 0.8,
                "target_shift": -0.01,
                "volatility_multiplier": 1.5,
            },
        }

        results = analyzer.scenario_sensitivity_analysis(mock_model, X, y, scenarios)

        # Check results structure
        assert isinstance(results, dict)
        assert "baseline_score" in results
        assert "scenario_scores" in results
        assert "worst_case_scenario" in results
        assert "best_case_scenario" in results
        assert "scenario_rankings" in results

        # Check scenario scores
        assert "bull_market" in results["scenario_scores"]
        assert "bear_market" in results["scenario_scores"]

        for scenario_name, scenario_result in results["scenario_scores"].items():
            assert "score" in scenario_result
            assert "score_change" in scenario_result
            assert "relative_change" in scenario_result
            assert "parameters" in scenario_result

        # Check worst/best case identification
        if results["worst_case_scenario"]:
            assert "name" in results["worst_case_scenario"]
            assert "score" in results["worst_case_scenario"]
            assert "parameters" in results["worst_case_scenario"]

        if results["best_case_scenario"]:
            assert "name" in results["best_case_scenario"]
            assert "score" in results["best_case_scenario"]

    def test_apply_scenario_modifications(self, sample_data):
        """Test scenario data modifications."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        scenario_params = {
            "feature_scale": 1.5,
            "target_shift": 0.1,
            "volatility_multiplier": 2.0,
        }

        X_modified, y_modified = analyzer._apply_scenario_modifications(
            X, y, scenario_params
        )

        # Check that modifications were applied
        assert X_modified.shape == X.shape
        assert y_modified.shape == y.shape

        # Feature scaling should change values
        assert not np.allclose(X_modified.values, X.values)

        # Target shift should change values
        assert not np.allclose(y_modified.values, y.values)

    def test_degradation_curve_calculation(self):
        """Test degradation curve calculation."""
        analyzer = SensitivityAnalyzer()

        baseline_score = 0.8
        test_scores = [0.8, 0.75, 0.7, 0.6]

        degradation = analyzer._calculate_degradation_curve(baseline_score, test_scores)

        # Check calculation
        expected = [0.0, 0.0625, 0.125, 0.25]  # (0.8 - score) / 0.8

        assert len(degradation) == len(test_scores)
        np.testing.assert_array_almost_equal(degradation, expected)

    def test_comprehensive_summary(self, sample_data, mock_model):
        """Test comprehensive summary generation."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Run some analyses first
        parameter_ranges = {"learning_rate": [0.01, 0.1]}
        analyzer.parameter_sensitivity_analysis(mock_model, X, y, parameter_ranges)

        with patch("src.analysis.sensitivity.permutation_importance") as mock_perm_imp:
            mock_result = Mock()
            mock_result.importances_mean = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
            mock_perm_imp.return_value = mock_result

            analyzer.feature_importance_sensitivity(mock_model, X, y, ["permutation"])

        # Get comprehensive summary
        summary = analyzer.get_comprehensive_summary()

        assert isinstance(summary, dict)
        assert "parameter_sensitivity" in summary
        assert "feature_importance" in summary
        assert "robustness_results" in summary
        assert "overall_assessment" in summary

        # Check overall assessment
        assessment = summary["overall_assessment"]
        assert "model_stability" in assessment
        assert "key_risk_factors" in assessment
        assert "recommendations" in assessment

    def test_overall_assessment_generation(self, sample_data, mock_model):
        """Test overall assessment generation."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Run parameter sensitivity analysis
        parameter_ranges = {"learning_rate": [0.01, 0.1]}
        analyzer.parameter_sensitivity_analysis(mock_model, X, y, parameter_ranges)

        assessment = analyzer._generate_overall_assessment()

        assert isinstance(assessment, dict)
        assert "model_stability" in assessment
        assert assessment["model_stability"] in ["high", "moderate", "low", "unknown"]
        assert "key_risk_factors" in assessment
        assert "recommendations" in assessment

        assert isinstance(assessment["key_risk_factors"], list)
        assert isinstance(assessment["recommendations"], list)

    def test_plot_data_generation(self, sample_data, mock_model):
        """Test plot data generation."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer(random_state=42)

        # Run some analyses first
        parameter_ranges = {"learning_rate": [0.01, 0.1]}
        analyzer.parameter_sensitivity_analysis(mock_model, X, y, parameter_ranges)

        plot_data = analyzer.plot_sensitivity_results()

        assert isinstance(plot_data, dict)
        assert "parameter_sensitivity" in plot_data
        assert "feature_importance" in plot_data
        assert "robustness_curves" in plot_data

        # Check parameter sensitivity plot data
        param_data = plot_data["parameter_sensitivity"]
        assert "parameter_effects" in param_data
        assert "rankings" in param_data

    def test_error_handling(self, sample_data):
        """Test error handling in various scenarios."""
        X, y = sample_data

        analyzer = SensitivityAnalyzer()

        # Test summary without results
        with pytest.raises(
            ValueError, match="No sensitivity analysis results available"
        ):
            analyzer.get_comprehensive_summary()

        # Test plot without results
        with pytest.raises(ValueError, match="No results available"):
            analyzer.plot_sensitivity_results()

        # Test with model that doesn't have required methods
        basic_model = Mock()
        basic_model.clone.return_value = basic_model
        basic_model.fit = Mock()

        # Should handle gracefully
        try:
            results = analyzer.parameter_sensitivity_analysis(basic_model, X, y, {})
            # Should complete without crashing
            assert isinstance(results, dict)
        except Exception as e:
            # Should fail gracefully
            assert "score" in str(e).lower() or "method" in str(e).lower()


class TestDefaultScenarios:
    """Test suite for default scenario creation."""

    def test_create_default_scenarios(self):
        """Test default scenario creation."""
        scenarios = create_default_scenarios()

        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

        # Check that expected scenarios exist
        expected_scenarios = [
            "bull_market",
            "bear_market",
            "high_volatility",
            "low_volatility",
            "market_crash",
        ]

        for scenario in expected_scenarios:
            assert scenario in scenarios

            # Check scenario structure
            scenario_params = scenarios[scenario]
            assert isinstance(scenario_params, dict)

            # Check that each scenario has expected parameters
            for param_name, param_value in scenario_params.items():
                assert isinstance(param_value, (int, float))

    def test_scenario_parameter_ranges(self):
        """Test that scenario parameters are within reasonable ranges."""
        scenarios = create_default_scenarios()

        for scenario_name, scenario_params in scenarios.items():
            # Feature scale should be positive
            if "feature_scale" in scenario_params:
                assert scenario_params["feature_scale"] > 0

            # Volatility multiplier should be positive
            if "volatility_multiplier" in scenario_params:
                assert scenario_params["volatility_multiplier"] > 0

            # Target shift should be reasonable
            if "target_shift" in scenario_params:
                assert abs(scenario_params["target_shift"]) < 1.0  # Not too extreme


class TestIntegration:
    """Integration tests for sensitivity analysis."""

    def test_full_sensitivity_workflow(self):
        """Test complete sensitivity analysis workflow."""
        # Create realistic financial data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=80, freq="D")

        # Features: returns, volatility, momentum
        returns = np.random.normal(0.001, 0.02, len(dates))
        volatility = np.abs(np.random.normal(0.02, 0.01, len(dates)))
        momentum = pd.Series(returns).rolling(5).mean().fillna(0)

        X = pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "momentum": momentum,
                "volume": np.random.lognormal(10, 0.5, len(dates)),
            },
            index=dates,
        )

        # Binary target: up/down movement
        y = pd.Series(np.where(returns > 0, 1, 0), index=dates, name="direction")

        # Create realistic model mock
        model = Mock()
        model.model_type = "classifier"
        model.clone.return_value = model
        model.learning_rate = 0.01
        model.max_depth = 5

        def realistic_predict(X_input):
            # Strategy based on momentum and volatility
            signal = (X_input["momentum"] > 0) & (X_input["volatility"] < 0.025)
            return signal.astype(int).values

        def realistic_score(X_input, y_input):
            pred = realistic_predict(X_input)
            return np.mean(pred == y_input.values) if len(y_input) > 0 else 0.0

        def mock_set_params(**params):
            for key, value in params.items():
                setattr(model, key, value)
            return model

        model.predict = realistic_predict
        model.score = realistic_score
        model.set_params = Mock(side_effect=mock_set_params)
        model.fit = Mock()
        model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])

        # Initialize analyzer
        analyzer = SensitivityAnalyzer(
            perturbation_range=0.05, n_permutations=10, random_state=42
        )

        # Run comprehensive sensitivity analysis

        # 1. Parameter sensitivity
        parameter_ranges = {
            "learning_rate": [0.005, 0.01, 0.05],
            "max_depth": [3, 5, 10],
        }
        param_results = analyzer.parameter_sensitivity_analysis(
            model, X, y, parameter_ranges
        )

        # 2. Feature importance sensitivity
        with patch("src.analysis.sensitivity.permutation_importance") as mock_perm_imp:
            mock_result = Mock()
            mock_result.importances_mean = np.array([0.35, 0.35, 0.2, 0.1])
            mock_perm_imp.return_value = mock_result

            importance_results = analyzer.feature_importance_sensitivity(
                model, X, y, ["permutation", "builtin"]
            )

        # 3. Robustness testing
        robustness_results = analyzer.robustness_testing(
            model, X, y, noise_levels=[0.0, 0.05, 0.1], sample_fractions=[1.0, 0.8, 0.6]
        )

        # 4. Scenario sensitivity
        scenarios = create_default_scenarios()
        scenario_results = analyzer.scenario_sensitivity_analysis(
            model, X, y, scenarios
        )

        # Verify all analyses completed successfully
        assert param_results["baseline_score"] is not None
        assert len(param_results["parameter_effects"]) == 2

        assert len(importance_results["methods"]) == 2
        assert "consistency_metrics" in importance_results

        assert robustness_results["baseline_score"] is not None
        assert "noise_sensitivity" in robustness_results

        assert scenario_results["baseline_score"] is not None
        assert len(scenario_results["scenario_scores"]) > 0

        # Test comprehensive summary
        summary = analyzer.get_comprehensive_summary()
        assert "parameter_sensitivity" in summary
        assert "feature_importance" in summary
        assert "robustness_results" in summary
        assert "overall_assessment" in summary

        # Test plot data generation
        plot_data = analyzer.plot_sensitivity_results()
        assert "parameter_sensitivity" in plot_data
        assert "feature_importance" in plot_data
        assert "robustness_curves" in plot_data

        # Verify model was used appropriately
        assert model.fit.call_count > 20  # Many model fits across all analyses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
