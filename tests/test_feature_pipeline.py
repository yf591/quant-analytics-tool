"""
Test Feature Pipeline Module

Tests for the comprehensive feature engineering pipeline including
technical indicators, advanced features, feature selection, scaling, and validation.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.features import (
    FeaturePipeline,
    FeaturePipelineConfig,
    FeaturePipelineResults,
    FeatureImportance,
    FeatureImportanceResults,
    FeatureQualityValidator,
    FeatureQualityResults,
)


class TestFeaturePipeline:
    """Test cases for FeaturePipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        price = 100 * np.exp(np.cumsum(returns))

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": price * (1 + np.random.normal(0, 0.001, len(dates))),
                "high": price * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                "low": price * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                "close": price,
                "volume": np.random.exponential(1000000, len(dates)),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    @pytest.fixture
    def sample_target(self, sample_data):
        """Create sample target variable."""
        # Future returns as target
        target = sample_data["close"].pct_change(5).shift(-5)
        return target.dropna()

    def test_pipeline_initialization(self):
        """Test pipeline initialization with different configurations."""
        # Default configuration
        pipeline1 = FeaturePipeline()
        assert pipeline1.config is not None
        assert hasattr(pipeline1, "technical_indicators")
        assert hasattr(pipeline1, "advanced_features")

        # Custom configuration
        custom_config = {
            "technical_indicators": {"trend": {"sma": {"windows": [10, 20]}}},
            "advanced_features": {"fractal_dimension": {"window": 50}},
            "feature_selection": {"method": "variance"},
            "scaling": {"method": "standard"},
            "validation": {"check_stationarity": True},
            "caching": {"enabled": False},
            "parallel": {"enabled": False},
        }

        pipeline2 = FeaturePipeline(custom_config)
        assert pipeline2.config.technical_indicators["trend"]["sma"]["windows"] == [
            10,
            20,
        ]

    def test_generate_features_basic(self, sample_data):
        """Test basic feature generation without target."""
        pipeline = FeaturePipeline()

        # Generate features
        results = pipeline.generate_features(sample_data)

        assert isinstance(results, FeaturePipelineResults)
        assert results.features is not None
        assert len(results.features) > 0
        assert len(results.feature_names) > 0
        assert results.technical_results is not None
        assert results.advanced_results is not None
        assert results.quality_metrics is not None

    def test_generate_features_with_target(self, sample_data, sample_target):
        """Test feature generation with target variable for supervised selection."""
        pipeline = FeaturePipeline()

        # Generate features with target
        results = pipeline.generate_features(sample_data, target=sample_target)

        assert isinstance(results, FeaturePipelineResults)
        assert results.features is not None
        assert results.feature_importance is not None
        assert results.selection_mask is not None
        assert results.scaling_params is not None

    def test_feature_selection_methods(self, sample_data, sample_target):
        """Test different feature selection methods."""
        methods = ["variance", "mdi", "correlation"]

        for method in methods:
            config = {
                "technical_indicators": {
                    "trend": {"sma": {"windows": [5, 10]}, "ema": {"windows": [5, 10]}}
                },
                "advanced_features": {"fractal_dimension": {"window": 50}},
                "feature_selection": {"method": method, "threshold": 0.01},
                "scaling": {"method": "none"},
                "validation": {"check_stationarity": False},
                "caching": {"enabled": False},
                "parallel": {"enabled": False},
            }

            pipeline = FeaturePipeline(config)
            results = pipeline.generate_features(sample_data, target=sample_target)

            assert results.features is not None
            print(f"Method {method}: Generated {len(results.feature_names)} features")

    def test_scaling_methods(self, sample_data):
        """Test different scaling methods."""
        scaling_methods = ["standard", "minmax", "robust", "none"]

        for method in scaling_methods:
            config = {
                "technical_indicators": {"trend": {"sma": {"windows": [10]}}},
                "advanced_features": {"fractal_dimension": {"window": 50}},
                "feature_selection": {"method": "variance"},
                "scaling": {"method": method},
                "validation": {"check_stationarity": False},
                "caching": {"enabled": False},
                "parallel": {"enabled": False},
            }

            pipeline = FeaturePipeline(config)
            results = pipeline.generate_features(sample_data)

            assert results.features is not None
            print(f"Scaling method {method}: Features shape {results.features.shape}")

    def test_transform_new_data(self, sample_data):
        """Test transforming new data using fitted pipeline."""
        pipeline = FeaturePipeline()

        # Fit on first part of data
        train_data = sample_data.iloc[:800]
        results = pipeline.generate_features(train_data)

        # Transform new data
        test_data = sample_data.iloc[800:]
        transformed_features = pipeline.transform_new_data(test_data, results)

        assert isinstance(transformed_features, pd.DataFrame)
        assert len(transformed_features.columns) == len(results.feature_names)

    def test_feature_descriptions(self):
        """Test feature description functionality."""
        pipeline = FeaturePipeline()
        descriptions = pipeline.get_feature_description()

        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0
        assert "returns" in descriptions
        assert "sma" in descriptions


class TestFeatureImportance:
    """Test cases for FeatureImportance."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature matrix."""
        np.random.seed(42)
        n_samples, n_features = 500, 10

        # Create correlated features
        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        return pd.DataFrame(X, columns=feature_names)

    @pytest.fixture
    def sample_classification_target(self, sample_features):
        """Create sample classification target."""
        np.random.seed(42)
        # Create target based on first few features
        target = (
            sample_features.iloc[:, 0]
            + sample_features.iloc[:, 1]
            + np.random.randn(len(sample_features)) * 0.5
            > 0
        ).astype(int)
        return target

    @pytest.fixture
    def sample_regression_target(self, sample_features):
        """Create sample regression target."""
        np.random.seed(42)
        # Create target based on first few features
        target = (
            sample_features.iloc[:, 0] * 2
            + sample_features.iloc[:, 1] * 1.5
            + sample_features.iloc[:, 2] * 0.5
            + np.random.randn(len(sample_features)) * 0.5
        )
        return target

    def test_mdi_importance(self, sample_features, sample_classification_target):
        """Test MDI importance calculation."""
        importance_analyzer = FeatureImportance()

        mdi_scores = importance_analyzer.calculate_mdi_importance(
            sample_features, sample_classification_target
        )

        assert isinstance(mdi_scores, pd.Series)
        assert len(mdi_scores) == len(sample_features.columns)
        assert all(mdi_scores >= 0)
        assert abs(mdi_scores.sum() - 1.0) < 0.001  # Should sum to 1

    def test_mda_importance(self, sample_features, sample_classification_target):
        """Test MDA importance calculation."""
        importance_analyzer = FeatureImportance()

        mda_scores = importance_analyzer.calculate_mda_importance(
            sample_features.iloc[:100],
            sample_classification_target.iloc[:100],
            cv_folds=3,
        )

        assert isinstance(mda_scores, pd.Series)
        assert len(mda_scores) == len(sample_features.columns)

    def test_sfi_importance(self, sample_features, sample_classification_target):
        """Test SFI importance calculation."""
        importance_analyzer = FeatureImportance()

        sfi_scores = importance_analyzer.calculate_sfi_importance(
            sample_features.iloc[:100],
            sample_classification_target.iloc[:100],
            cv_folds=3,
        )

        assert isinstance(sfi_scores, pd.Series)
        assert len(sfi_scores) == len(sample_features.columns)
        assert all(sfi_scores >= 0)

    def test_all_importance_methods(self, sample_features, sample_regression_target):
        """Test all importance methods together."""
        importance_analyzer = FeatureImportance()

        results = importance_analyzer.calculate_all_importance(
            sample_features.iloc[:200], sample_regression_target.iloc[:200], cv_folds=3
        )

        assert isinstance(results, FeatureImportanceResults)
        assert results.mdi_importance is not None
        assert results.mda_importance is not None
        assert results.sfi_importance is not None
        assert results.feature_ranking is not None

    def test_feature_selection(self, sample_features, sample_classification_target):
        """Test feature selection based on importance."""
        importance_analyzer = FeatureImportance()

        results = importance_analyzer.calculate_all_importance(
            sample_features.iloc[:200],
            sample_classification_target.iloc[:200],
            cv_folds=3,
        )

        # Select top 5 features
        selected_features = importance_analyzer.select_features_by_importance(
            results, method="mdi", n_features=5
        )

        assert isinstance(selected_features, list)
        assert len(selected_features) <= 5

    def test_feature_stability(self, sample_features, sample_classification_target):
        """Test feature importance stability assessment."""
        importance_analyzer = FeatureImportance()

        stability_results = importance_analyzer.get_feature_stability(
            sample_features.iloc[:200],
            sample_classification_target.iloc[:200],
            n_trials=5,
            sample_fraction=0.8,
        )

        assert isinstance(stability_results, pd.DataFrame)
        assert "stability_score" in stability_results.columns
        assert all(stability_results["stability_score"] >= 0)
        assert all(stability_results["stability_score"] <= 1)


class TestFeatureQualityValidator:
    """Test cases for FeatureQualityValidator."""

    @pytest.fixture
    def sample_quality_features(self):
        """Create sample features with known quality issues."""
        np.random.seed(42)
        n_samples = 500

        # Create features with different quality characteristics
        features = pd.DataFrame(
            {
                # Good quality feature
                "good_feature": np.random.randn(n_samples),
                # Highly correlated features
                "corr_feature_1": np.random.randn(n_samples),
                "corr_feature_2": None,  # Will be set to correlated
                # Feature with missing data
                "missing_feature": np.random.randn(n_samples),
                # Non-stationary feature (trend)
                "trend_feature": np.cumsum(np.random.randn(n_samples))
                + np.arange(n_samples) * 0.1,
                # Feature with outliers
                "outlier_feature": np.random.randn(n_samples),
                # Constant feature
                "constant_feature": np.ones(n_samples) * 5.0,
            }
        )

        # Create correlation
        features["corr_feature_2"] = (
            features["corr_feature_1"] + np.random.randn(n_samples) * 0.01
        )

        # Add missing data
        missing_indices = np.random.choice(n_samples, size=50, replace=False)
        features.loc[missing_indices, "missing_feature"] = np.nan

        # Add outliers
        outlier_indices = np.random.choice(n_samples, size=10, replace=False)
        features.loc[outlier_indices, "outlier_feature"] = np.random.randn(10) * 10 + 50

        return features

    def test_validate_all_features(self, sample_quality_features):
        """Test comprehensive feature quality validation."""
        validator = FeatureQualityValidator()

        results = validator.validate_all_features(sample_quality_features)

        assert isinstance(results, FeatureQualityResults)
        assert results.stationarity_results is not None
        assert results.multicollinearity_results is not None
        assert results.distribution_results is not None
        assert results.completeness_results is not None
        assert results.stability_results is not None
        assert results.outlier_results is not None
        assert results.quality_score is not None
        assert results.recommendations is not None

    def test_stationarity_detection(self, sample_quality_features):
        """Test stationarity detection."""
        validator = FeatureQualityValidator()

        stationarity_results = validator.test_stationarity(sample_quality_features)

        assert "summary" in stationarity_results
        assert "adf_test" in stationarity_results

        # Check that trend feature is detected as non-stationary
        if "trend_feature" in stationarity_results["summary"]:
            assert stationarity_results["summary"]["trend_feature"] in [
                "non_stationary",
                "questionable",
            ]

    def test_multicollinearity_detection(self, sample_quality_features):
        """Test multicollinearity detection."""
        validator = FeatureQualityValidator()

        multicollinearity_results = validator.detect_multicollinearity(
            sample_quality_features
        )

        assert "correlation_matrix" in multicollinearity_results
        assert "high_correlations" in multicollinearity_results
        assert "vif_scores" in multicollinearity_results

        # Should detect high correlation between corr_feature_1 and corr_feature_2
        high_corr_features = {
            pair["feature1"] for pair in multicollinearity_results["high_correlations"]
        }
        high_corr_features.update(
            {
                pair["feature2"]
                for pair in multicollinearity_results["high_correlations"]
            }
        )

        assert (
            "corr_feature_1" in high_corr_features
            or "corr_feature_2" in high_corr_features
        )

    def test_completeness_check(self, sample_quality_features):
        """Test data completeness checking."""
        validator = FeatureQualityValidator()

        completeness_results = validator.check_completeness(sample_quality_features)

        assert "missing_percentages" in completeness_results
        assert "completeness_score" in completeness_results

        # Should detect missing data in missing_feature
        missing_pct = completeness_results["missing_percentages"]["missing_feature"]
        assert missing_pct > 0

    def test_outlier_detection(self, sample_quality_features):
        """Test outlier detection."""
        validator = FeatureQualityValidator()

        outlier_results = validator.detect_outliers(sample_quality_features)

        assert "iqr_outliers" in outlier_results
        assert "zscore_outliers" in outlier_results
        assert "outlier_summary" in outlier_results

        # Should detect outliers in outlier_feature
        outlier_pct = outlier_results["iqr_outliers"]["outlier_feature"]
        assert outlier_pct > 0

    def test_quality_score_computation(self, sample_quality_features):
        """Test overall quality score computation."""
        validator = FeatureQualityValidator()

        results = validator.validate_all_features(sample_quality_features)
        quality_scores = results.quality_score

        assert isinstance(quality_scores, pd.Series)
        assert all(quality_scores >= 0)
        assert all(quality_scores <= 1)

        # Good feature should have higher score than problematic ones
        good_score = quality_scores["good_feature"]
        constant_score = quality_scores["constant_feature"]
        assert good_score > constant_score

    def test_recommendations_generation(self, sample_quality_features):
        """Test recommendation generation."""
        validator = FeatureQualityValidator()

        results = validator.validate_all_features(sample_quality_features)
        recommendations = results.recommendations

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should have recommendations about correlations and missing data
        rec_text = " ".join(recommendations)
        assert "correlat" in rec_text.lower() or "missing" in rec_text.lower()


def run_comprehensive_test():
    """Run comprehensive test of the feature pipeline system."""
    print("=== Comprehensive Feature Pipeline Test ===")

    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")

    returns = np.random.normal(0.001, 0.02, len(dates))
    price = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "open": price * (1 + np.random.normal(0, 0.001, len(dates))),
            "high": price * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            "low": price * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            "close": price,
            "volume": np.random.exponential(1000000, len(dates)),
        },
        index=dates,
    )

    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    # Create target
    target = data["close"].pct_change(5).shift(-5).dropna()

    print(f"Test data shape: {data.shape}")
    print(f"Target shape: {target.shape}")

    # 1. Test feature pipeline
    print("\n1. Testing Feature Pipeline...")
    pipeline = FeaturePipeline()
    results = pipeline.generate_features(data, target=target)

    print(f"Generated features: {len(results.feature_names)}")
    print(f"Feature names: {results.feature_names[:10]}...")
    print(f"Features shape: {results.features.shape}")

    # 2. Test feature importance
    print("\n2. Testing Feature Importance...")
    importance_analyzer = FeatureImportance()

    # Use subset of data for faster computation
    subset_features = results.features.dropna().iloc[:200]
    subset_target = target.reindex(subset_features.index).dropna()

    # Align data
    common_index = subset_features.index.intersection(subset_target.index)
    subset_features = subset_features.loc[common_index]
    subset_target = subset_target.loc[common_index]

    if len(subset_features) > 50:
        importance_results = importance_analyzer.calculate_all_importance(
            subset_features, subset_target, cv_folds=3
        )

        print(f"MDI importance (top 5):")
        if importance_results.mdi_importance is not None:
            print(importance_results.mdi_importance.head())

        print(f"Feature ranking (top 5):")
        if importance_results.feature_ranking is not None:
            print(importance_results.feature_ranking.head())

    # 3. Test feature quality validation
    print("\n3. Testing Feature Quality Validation...")
    validator = FeatureQualityValidator()
    quality_results = validator.validate_all_features(results.features.dropna())

    print(f"Quality scores (top 5):")
    if quality_results.quality_score is not None:
        print(quality_results.quality_score.head())

    print(f"Recommendations:")
    for i, rec in enumerate(quality_results.recommendations[:3], 1):
        print(f"  {i}. {rec}")

    print("\n=== Test Completed Successfully ===")


if __name__ == "__main__":
    run_comprehensive_test()
