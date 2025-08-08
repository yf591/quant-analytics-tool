"""
Feature Pipeline Example

This example demonstrates the complete feature engineering pipeline
including technical indicators, advanced features, feature selection,
scaling, and quality validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features import (
    FeaturePipeline,
    FeaturePipelineConfig,
    FeatureImportance,
    FeatureQualityValidator,
)


def create_sample_data(n_periods: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create sample financial data for demonstration."""
    np.random.seed(seed)

    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")

    # Generate realistic price movements with some trends and volatility clustering
    returns = []
    volatility = 0.02

    for i in range(n_periods):
        # Volatility clustering
        if i > 0:
            volatility = 0.8 * volatility + 0.2 * 0.02 + 0.1 * abs(returns[-1])

        # Add some trend periods
        trend = 0.001 if i % 200 < 100 else -0.0005

        daily_return = np.random.normal(trend, volatility)
        returns.append(daily_return)

    # Convert to prices
    price = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "open": price * (1 + np.random.normal(0, 0.001, n_periods)),
            "high": price * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
            "low": price * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
            "close": price,
            "volume": np.random.exponential(1000000, n_periods),
        },
        index=dates,
    )

    # Ensure OHLC relationships
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    return data


def example_1_basic_pipeline():
    """Example 1: Basic feature pipeline usage."""
    print("=== Example 1: Basic Feature Pipeline ===")

    # Create sample data
    data = create_sample_data(800)
    print(f"Sample data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Initialize pipeline with default configuration
    pipeline = FeaturePipeline()

    # Generate features
    print("\nGenerating features...")
    results = pipeline.generate_features(data)

    print(f"Generated {len(results.feature_names)} features")
    print(f"Features shape: {results.features.shape}")
    print(f"Feature names: {results.feature_names[:10]}...")

    # Display quality metrics
    print(f"\nQuality metrics:")
    print(f"  - Number of features: {results.quality_metrics['n_features']}")
    print(f"  - Number of observations: {results.quality_metrics['n_observations']}")
    print(
        f"  - Average missing percentage: {results.quality_metrics['missing_percentage'].mean():.3f}"
    )

    return results


def example_2_supervised_pipeline():
    """Example 2: Supervised feature pipeline with target variable."""
    print("\n=== Example 2: Supervised Feature Pipeline ===")

    # Create sample data
    data = create_sample_data(800)

    # Create target variable (future returns)
    target = data["close"].pct_change(5).shift(-5)  # 5-day forward returns
    target = target.dropna()

    print(f"Target variable shape: {target.shape}")
    print(f"Target statistics: mean={target.mean():.4f}, std={target.std():.4f}")

    # Configure pipeline for supervised learning
    config = {
        "technical_indicators": {
            "trend": {
                "sma": {"windows": [5, 10, 20, 50]},
                "ema": {"windows": [5, 10, 20]},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
            },
            "momentum": {
                "rsi": {"window": 14},
                "stochastic": {"k_period": 14, "d_period": 3},
            },
            "volatility": {
                "bollinger_bands": {"window": 20, "std_dev": 2},
                "atr": {"window": 14},
            },
        },
        "advanced_features": {
            "fractal_dimension": {"window": 100},
            "hurst_exponent": {"window": 100},
        },
        "feature_selection": {"method": "mdi", "n_features": 20},
        "scaling": {"method": "standard"},
        "validation": {"check_stationarity": True, "check_multicollinearity": True},
        "caching": {"enabled": False},
        "parallel": {"enabled": False},
    }

    pipeline = FeaturePipeline(config)

    # Generate features with target
    print("\nGenerating features with supervised selection...")
    results = pipeline.generate_features(data, target=target)

    print(f"Selected {len(results.feature_names)} features from supervised pipeline")
    print(f"Final features shape: {results.features.shape}")

    # Display feature importance
    if results.feature_importance is not None:
        print(f"\nTop 10 most important features:")
        for i, (feature, importance) in enumerate(
            results.feature_importance.head(10).iterrows(), 1
        ):
            print(f"  {i:2d}. {feature}: {importance.iloc[0]:.4f}")

    return results


def example_3_feature_importance_analysis():
    """Example 3: Detailed feature importance analysis."""
    print("\n=== Example 3: Feature Importance Analysis ===")

    # Create sample data
    data = create_sample_data(500)

    # Generate features
    pipeline = FeaturePipeline()
    results = pipeline.generate_features(data)

    # Create target
    target = data["close"].pct_change(3).shift(-3).dropna()

    # Prepare features for importance analysis
    features = results.features.dropna()

    # Align features and target
    common_index = features.index.intersection(target.index)
    features_aligned = features.loc[common_index]
    target_aligned = target.loc[common_index]

    print(
        f"Aligned data shape: {features_aligned.shape}, target: {len(target_aligned)}"
    )

    # Initialize importance analyzer
    importance_analyzer = FeatureImportance(
        n_estimators=50
    )  # Fewer estimators for speed

    # Calculate all importance measures
    print("\nCalculating feature importance (this may take a moment)...")
    importance_results = importance_analyzer.calculate_all_importance(
        features_aligned, target_aligned, cv_folds=3
    )

    # Display results
    print(f"\nMDI Importance (top 10):")
    if importance_results.mdi_importance is not None:
        for i, (feature, score) in enumerate(
            importance_results.mdi_importance.head(10).items(), 1
        ):
            print(f"  {i:2d}. {feature}: {score:.4f}")

    print(f"\nMDA Importance (top 10):")
    if importance_results.mda_importance is not None:
        for i, (feature, score) in enumerate(
            importance_results.mda_importance.head(10).items(), 1
        ):
            print(f"  {i:2d}. {feature}: {score:.4f}")

    print(f"\nSFI Scores (top 10):")
    if importance_results.sfi_importance is not None:
        for i, (feature, score) in enumerate(
            importance_results.sfi_importance.head(10).items(), 1
        ):
            print(f"  {i:2d}. {feature}: {score:.4f}")

    # Feature selection based on importance
    selected_features = importance_analyzer.select_features_by_importance(
        importance_results, method="avg_rank", n_features=15
    )

    print(f"\nSelected top 15 features based on average ranking:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feature}")

    return importance_results


def example_4_quality_validation():
    """Example 4: Feature quality validation."""
    print("\n=== Example 4: Feature Quality Validation ===")

    # Create sample data with some quality issues
    data = create_sample_data(400)

    # Generate features
    pipeline = FeaturePipeline()
    results = pipeline.generate_features(data)

    # Add some problematic features for demonstration
    features = results.features.copy()

    # Add a highly correlated feature
    if "sma_10" in features.columns:
        features["sma_10_copy"] = features["sma_10"] + np.random.normal(
            0, 0.001, len(features)
        )

    # Add a feature with missing data
    missing_indices = np.random.choice(
        len(features), size=int(0.1 * len(features)), replace=False
    )
    features.loc[features.index[missing_indices], "problematic_feature"] = np.nan

    # Add a non-stationary feature (trend)
    features["trend_feature"] = (
        np.cumsum(np.random.randn(len(features))) + np.arange(len(features)) * 0.01
    )

    print(f"Features with potential quality issues: {features.shape}")

    # Initialize quality validator
    validator = FeatureQualityValidator()

    # Validate feature quality
    print("\nValidating feature quality...")
    quality_results = validator.validate_all_features(features)

    # Display stationarity results
    print(f"\nStationarity Test Results:")
    for feature, status in list(
        quality_results.stationarity_results["summary"].items()
    )[:10]:
        print(f"  {feature}: {status}")

    # Display multicollinearity results
    print(f"\nMulticollinearity Issues:")
    if quality_results.multicollinearity_results["high_correlations"]:
        for pair in quality_results.multicollinearity_results["high_correlations"][:5]:
            print(
                f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}"
            )
    else:
        print("  No high correlations detected")

    # Display completeness results
    print(f"\nData Completeness:")
    missing_features = {
        k: v
        for k, v in quality_results.completeness_results["missing_percentages"].items()
        if v > 0
    }
    if missing_features:
        for feature, missing_pct in list(missing_features.items())[:5]:
            print(f"  {feature}: {missing_pct:.3f} missing")
    else:
        print("  No missing data detected")

    # Display quality scores
    print(f"\nTop 10 Features by Quality Score:")
    if quality_results.quality_score is not None:
        for i, (feature, score) in enumerate(
            quality_results.quality_score.head(10).items(), 1
        ):
            print(f"  {i:2d}. {feature}: {score:.3f}")

    # Display recommendations
    print(f"\nRecommendations:")
    for i, rec in enumerate(quality_results.recommendations[:5], 1):
        print(f"  {i}. {rec}")

    return quality_results


def example_5_complete_workflow():
    """Example 5: Complete feature engineering workflow."""
    print("\n=== Example 5: Complete Workflow ===")

    # 1. Data preparation
    print("1. Preparing data...")
    data = create_sample_data(1000)
    target = data["close"].pct_change(5).shift(-5).dropna()

    # Split into train/test
    split_point = int(0.8 * len(data))
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    train_target = target.reindex(train_data.index).dropna()

    print(f"Train data: {train_data.shape}, Test data: {test_data.shape}")

    # 2. Feature generation and selection
    print("\n2. Feature generation and selection...")
    config = {
        "technical_indicators": {
            "trend": {
                "sma": {"windows": [5, 10, 20, 50]},
                "ema": {"windows": [5, 10, 20]},
            },
            "momentum": {"rsi": {"window": 14}, "momentum": {"window": 10}},
            "volatility": {"bollinger_bands": {"window": 20}, "atr": {"window": 14}},
        },
        "advanced_features": {"fractal_dimension": {"window": 100}},
        "feature_selection": {"method": "mdi", "n_features": 15},
        "scaling": {"method": "standard"},
        "validation": {"check_stationarity": True, "check_multicollinearity": True},
        "caching": {"enabled": False},
        "parallel": {"enabled": False},
    }

    pipeline = FeaturePipeline(config)
    train_results = pipeline.generate_features(train_data, target=train_target)

    print(f"Selected {len(train_results.feature_names)} features")

    # 3. Quality validation
    print("\n3. Quality validation...")
    validator = FeatureQualityValidator()
    quality_results = validator.validate_all_features(train_results.features)

    avg_quality = quality_results.quality_score.mean()
    print(f"Average feature quality score: {avg_quality:.3f}")

    # 4. Transform test data
    print("\n4. Transforming test data...")
    test_features = pipeline.transform_new_data(test_data, train_results)

    print(f"Test features shape: {test_features.shape}")
    print(
        f"Features aligned: {set(train_results.feature_names) == set(test_features.columns)}"
    )

    # 5. Summary
    print("\n5. Workflow Summary:")
    print(f"  - Original data features: {data.shape[1]}")
    print(f"  - Generated features: {len(train_results.feature_names)}")
    print(f"  - Average quality score: {avg_quality:.3f}")
    print(f"  - Train/test consistency: ✓")

    recommendations = quality_results.recommendations
    if len(recommendations) <= 2:
        print(f"  - Feature quality: Good ({len(recommendations)} recommendations)")
    else:
        print(
            f"  - Feature quality: Needs attention ({len(recommendations)} recommendations)"
        )

    return {
        "train_features": train_results.features,
        "test_features": test_features,
        "feature_names": train_results.feature_names,
        "quality_score": avg_quality,
    }


def main():
    """Run all examples."""
    print("Feature Pipeline Examples")
    print("=" * 50)

    try:
        # Run examples
        example_1_basic_pipeline()
        example_2_supervised_pipeline()
        example_3_feature_importance_analysis()
        example_4_quality_validation()
        workflow_results = example_5_complete_workflow()

        print(f"\n{'='*50}")
        print("All examples completed successfully!")
        print(
            f"Final workflow generated {len(workflow_results['feature_names'])} high-quality features"
        )
        print(f"Average quality score: {workflow_results['quality_score']:.3f}")

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
