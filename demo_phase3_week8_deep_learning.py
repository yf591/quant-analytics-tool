#!/usr/bin/env python3
"""
Week 8: Deep Learning Models - Comprehensive Demo

This demo showcases LSTM and GRU models for financial time series analysis.
Demonstrates both classification and regression tasks with comprehensive evaluation.

PURPOSE: Production-level testing and complete feature validation
TARGET: Full system validation, presentation, research use
SCOPE: Complete feature pipeline with financial metrics and advanced evaluation

For lightweight testing and development, use: demo_week8_simple.py

Requirements:
- TensorFlow 2.19.0+
- Comprehensive feature engineering from previous weeks
- Financial time series data
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from src.data import BaseDataCollector, DataRequest
from src.features import FeaturePipeline
from src.models.deep_learning import (
    QuantLSTMClassifier,
    QuantLSTMRegressor,
    QuantGRUClassifier,
    QuantGRURegressor,
    LSTMDataPreprocessor,
)
from src.models.deep_learning.utils import (
    DeepLearningEvaluator,
    FinancialMetrics,
    ModelComparison,
)

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def main():
    """
    Main demo function for Week 8: Deep Learning Models
    """
    print("=" * 60)
    print("WEEK 8: DEEP LEARNING MODELS DEMO")
    print("=" * 60)
    print("Implementation: LSTM & GRU Models for Financial Time Series")
    print("Features: Advanced RNN architectures with financial optimization")
    print()

    # 1. Data Collection and Feature Engineering
    print("1. DATA COLLECTION AND FEATURE ENGINEERING")
    print("-" * 50)

    try:
        # Use synthetic data for demonstration
        print("Creating synthetic financial data for demonstration...")
        data = create_synthetic_financial_data()
        print(f"✓ Successfully created {len(data)} synthetic records")

        print(f"Data shape: {data.shape}")
        print()

        # 2. Feature Engineering
        print("2. FEATURE ENGINEERING")
        print("-" * 50)

        # Initialize feature pipeline
        feature_pipeline = FeaturePipeline()

        # Generate comprehensive features
        features_result = feature_pipeline.generate_features(data)
        features_df = features_result.features

        print(f"Generated {len(features_df.columns)} features:")
        for i, feature in enumerate(features_df.columns):
            if i < 10:  # Show first 10 features
                print(f"  - {feature}")
            elif i == 10:
                print(f"  ... and {len(features_df.columns) - 10} more")
                break
        print()

        # 3. Prepare Targets for Different Tasks
        print("3. TARGET PREPARATION")
        print("-" * 50)

        # Classification target: Next day return direction
        features_df["next_return"] = data["close"].pct_change(1).shift(-1)
        features_df["return_direction"] = (features_df["next_return"] > 0).astype(int)

        # Regression target: Next day return magnitude
        features_df["return_magnitude"] = features_df["next_return"]

        # Clean data
        features_df = features_df.dropna()

        if len(features_df) < 100:
            print("Insufficient data after cleaning. Using synthetic data...")
            features_df = create_synthetic_feature_data()

        print(f"Final dataset shape: {features_df.shape}")
        print(f"Classification target distribution:")
        print(features_df["return_direction"].value_counts())
        print()

        # 4. LSTM Models Demonstration
        print("4. LSTM MODELS DEMONSTRATION")
        print("-" * 50)

        # Prepare features (exclude target columns)
        feature_cols = [
            col
            for col in features_df.columns
            if col not in ["next_return", "return_direction", "return_magnitude"]
        ]
        X = features_df[feature_cols].values
        y_class = features_df["return_direction"].values
        y_reg = features_df["return_magnitude"].values

        # Split data (temporal split for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_class_train, y_class_test = y_class[:split_idx], y_class[split_idx:]
        y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print()

        # 4.1 LSTM Classification
        print("4.1 LSTM Classification (Return Direction Prediction)")
        print("-" * 40)

        lstm_classifier = QuantLSTMClassifier(
            sequence_length=30,
            lstm_units=[64, 32],
            dense_units=[16],
            dropout_rate=0.3,
            bidirectional=True,
            batch_size=32,
            epochs=20,
            early_stopping_patience=5,
            validation_split=0.2,
            verbose=1,
        )

        print("Training LSTM Classifier...")
        lstm_classifier.fit(X_train, y_class_train)

        # Evaluate classifier
        evaluator_class = DeepLearningEvaluator(lstm_classifier, "LSTM Classifier")
        # Adjust test data length to match predictions
        y_class_test_aligned = y_class_test[lstm_classifier.sequence_length - 1 :]
        class_results = evaluator_class.evaluate_classifier(
            X_test, y_class_test_aligned
        )

        print(f"LSTM Classification Results:")
        print(f"  Accuracy: {class_results['accuracy']:.4f}")
        print(f"  Precision: {class_results['precision']:.4f}")
        print(f"  Recall: {class_results['recall']:.4f}")
        print(f"  F1-Score: {class_results['f1_score']:.4f}")
        print(f"  ROC AUC: {class_results['roc_auc']:.4f}")
        print()

        # 4.2 LSTM Regression
        print("4.2 LSTM Regression (Return Magnitude Prediction)")
        print("-" * 40)

        lstm_regressor = QuantLSTMRegressor(
            sequence_length=30,
            lstm_units=[64, 32],
            dense_units=[16],
            dropout_rate=0.3,
            bidirectional=True,
            batch_size=32,
            epochs=20,
            early_stopping_patience=5,
            validation_split=0.2,
            verbose=1,
        )

        print("Training LSTM Regressor...")
        lstm_regressor.fit(X_train, y_reg_train)

        # Evaluate regressor
        evaluator_reg = DeepLearningEvaluator(lstm_regressor, "LSTM Regressor")
        # Adjust test data length to match predictions
        y_reg_test_aligned = y_reg_test[lstm_regressor.sequence_length - 1 :]
        reg_results = evaluator_reg.evaluate_regressor(
            X_test, y_reg_test_aligned, uncertainty_estimation=True
        )

        print(f"LSTM Regression Results:")
        print(f"  R² Score: {reg_results['r2_score']:.4f}")
        print(f"  RMSE: {reg_results['rmse']:.4f}")
        print(f"  MAE: {reg_results['mae']:.4f}")
        print(f"  Directional Accuracy: {reg_results['directional_accuracy']:.4f}")
        print(f"  Hit Ratio: {reg_results['hit_ratio']:.4f}")
        print()

        # 5. GRU Models Demonstration
        print("5. GRU MODELS DEMONSTRATION")
        print("-" * 50)

        # 5.1 GRU Classification
        print("5.1 GRU Classification")
        print("-" * 25)

        gru_classifier = QuantGRUClassifier(
            sequence_length=30,
            gru_units=[64, 32],
            dense_units=[16],
            dropout_rate=0.3,
            bidirectional=True,
            batch_size=32,
            epochs=15,
            early_stopping_patience=5,
            validation_split=0.2,
            verbose=1,
        )

        print("Training GRU Classifier...")
        gru_classifier.fit(X_train, y_class_train)

        # 5.2 GRU Regression
        print("5.2 GRU Regression")
        print("-" * 20)

        gru_regressor = QuantGRURegressor(
            sequence_length=30,
            gru_units=[64, 32],
            dense_units=[16],
            dropout_rate=0.3,
            bidirectional=True,
            batch_size=32,
            epochs=15,
            early_stopping_patience=5,
            validation_split=0.2,
            verbose=1,
        )

        print("Training GRU Regressor...")
        gru_regressor.fit(X_train, y_reg_train)

        # 6. Model Comparison
        print("6. MODEL COMPARISON")
        print("-" * 50)

        # Compare classification models
        print("6.1 Classification Models Comparison")
        comparison_class = ModelComparison()
        comparison_class.add_model(
            "LSTM", lstm_classifier, X_test, y_class_test_aligned
        )
        comparison_class.add_model("GRU", gru_classifier, X_test, y_class_test_aligned)

        class_comparison_df = comparison_class.compare_metrics()
        comparison_class.print_summary()

        # Compare regression models
        print("\\n6.2 Regression Models Comparison")
        comparison_reg = ModelComparison()
        comparison_reg.add_model("LSTM", lstm_regressor, X_test, y_reg_test_aligned)
        comparison_reg.add_model("GRU", gru_regressor, X_test, y_reg_test_aligned)

        reg_comparison_df = comparison_reg.compare_metrics()
        comparison_reg.print_summary()

        # 7. Financial Metrics Analysis
        print("\\n7. FINANCIAL METRICS ANALYSIS")
        print("-" * 50)

        # Generate trading signals based on predictions
        lstm_predictions = lstm_regressor.predict(X_test)
        gru_predictions = gru_regressor.predict(X_test)

        # Calculate returns based on predictions
        actual_returns = y_reg_test_aligned[
            : len(lstm_predictions)
        ]  # Align with prediction length
        lstm_signals = np.sign(lstm_predictions)
        gru_signals = np.sign(gru_predictions)

        lstm_strategy_returns = lstm_signals * actual_returns
        gru_strategy_returns = gru_signals * actual_returns

        # Calculate financial metrics
        print("Financial Performance Metrics:")
        print("\\nLSTM Strategy:")
        print(
            f"  Sharpe Ratio: {FinancialMetrics.sharpe_ratio(lstm_strategy_returns):.4f}"
        )
        print(
            f"  Max Drawdown: {FinancialMetrics.max_drawdown(lstm_strategy_returns):.4f}"
        )
        print(
            f"  Calmar Ratio: {FinancialMetrics.calmar_ratio(lstm_strategy_returns):.4f}"
        )
        print(
            f"  Sortino Ratio: {FinancialMetrics.sortino_ratio(lstm_strategy_returns):.4f}"
        )

        print("\\nGRU Strategy:")
        print(
            f"  Sharpe Ratio: {FinancialMetrics.sharpe_ratio(gru_strategy_returns):.4f}"
        )
        print(
            f"  Max Drawdown: {FinancialMetrics.max_drawdown(gru_strategy_returns):.4f}"
        )
        print(
            f"  Calmar Ratio: {FinancialMetrics.calmar_ratio(gru_strategy_returns):.4f}"
        )
        print(
            f"  Sortino Ratio: {FinancialMetrics.sortino_ratio(gru_strategy_returns):.4f}"
        )

        # 8. Advanced Features Demonstration
        print("\\n8. ADVANCED FEATURES DEMONSTRATION")
        print("-" * 50)

        # 8.1 Uncertainty Estimation
        print("8.1 Uncertainty Estimation (Monte Carlo Dropout)")
        lstm_mean, lstm_std = lstm_regressor.predict_with_uncertainty(
            X_test[:50], n_samples=50
        )

        # Plot uncertainty
        plt.figure(figsize=(12, 6))
        x_axis = range(len(lstm_mean))
        plt.plot(
            x_axis, y_reg_test_aligned[: len(lstm_mean)], label="Actual", alpha=0.8
        )
        plt.plot(x_axis, lstm_mean, label="Predicted", alpha=0.8)
        plt.fill_between(
            x_axis,
            lstm_mean - 2 * lstm_std,
            lstm_mean + 2 * lstm_std,
            alpha=0.3,
            label="95% Confidence",
        )
        plt.title("LSTM Predictions with Uncertainty Estimation")
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.legend()
        plt.show()

        # 8.2 Model Architecture Visualization
        print("\\n8.2 Model Architecture Summary")
        print("LSTM Classifier Architecture:")
        print(lstm_classifier.get_model_summary())

        # 8.3 Training History Analysis
        print("\\n8.3 Training History Analysis")
        lstm_classifier.plot_training_history()
        lstm_regressor.plot_training_history()

        print("\\n" + "=" * 60)
        print("WEEK 8 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Key Achievements:")
        print("✓ LSTM Classification and Regression models implemented")
        print("✓ GRU Classification and Regression models implemented")
        print("✓ Bidirectional architectures with financial optimization")
        print("✓ Comprehensive evaluation with financial metrics")
        print("✓ Uncertainty estimation using Monte Carlo Dropout")
        print("✓ Model comparison and performance analysis")
        print("✓ Advanced features: early stopping, learning rate scheduling")
        print(
            "\\nNext Steps: Integration with trading strategies and portfolio optimization"
        )

    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback

        traceback.print_exc()


def create_synthetic_financial_data(
    n_samples: int = 1000, n_features: int = 5
) -> pd.DataFrame:
    """
    Create synthetic financial time series data for demonstration.

    Args:
        n_samples: Number of time periods
        n_features: Number of base features (OHLCV)

    Returns:
        DataFrame with synthetic financial data
    """
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    # Generate price data with realistic patterns
    price = 100  # Starting price
    prices = [price]

    for i in range(1, n_samples):
        # Add trend, volatility, and mean reversion
        trend = 0.0001  # Small upward trend
        volatility = 0.02
        mean_reversion = -0.1 * (price - 100) / 100

        change = trend + mean_reversion + np.random.normal(0, volatility)
        price = price * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = data["close"].shift(1) * (1 + np.random.normal(0, 0.001, n_samples))
    data["high"] = np.maximum(data["open"], data["close"]) * (
        1 + np.abs(np.random.normal(0, 0.005, n_samples))
    )
    data["low"] = np.minimum(data["open"], data["close"]) * (
        1 - np.abs(np.random.normal(0, 0.005, n_samples))
    )
    data["volume"] = np.random.lognormal(10, 1, n_samples)

    # Remove NaN values
    data = data.dropna()

    return data


def create_synthetic_feature_data(
    n_samples: int = 500, n_features: int = 20
) -> pd.DataFrame:
    """
    Create synthetic feature data with targets for demonstration.

    Args:
        n_samples: Number of samples
        n_features: Number of features

    Returns:
        DataFrame with features and targets
    """
    np.random.seed(42)

    # Generate feature names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

    # Generate correlated features
    data = np.random.randn(n_samples, n_features)

    # Add some structure to make it more realistic
    for i in range(1, n_features):
        data[:, i] = 0.3 * data[:, i - 1] + 0.7 * data[:, i]  # Add autocorrelation

    # Create DataFrame
    features_df = pd.DataFrame(data, columns=feature_names)

    # Create targets based on features
    features_df["next_return"] = (
        0.1 * features_df["feature_1"]
        + 0.05 * features_df["feature_2"]
        + np.random.normal(0, 0.02, n_samples)
    )
    features_df["return_direction"] = (features_df["next_return"] > 0).astype(int)
    features_df["return_magnitude"] = features_df["next_return"]

    return features_df


if __name__ == "__main__":
    main()
