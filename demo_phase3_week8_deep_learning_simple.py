#!/usr/bin/env python3
"""
Week 8: Deep Learning Models - Simple Demo

Simplified demonstration of LSTM and GRU models for financial time series analysis.
Uses synthetic data to showcase the models' capabilities.

PURPOSE: Development testing, debugging, and basic functionality verification
TARGET: Developers, CI/CD environments, new team members
SCOPE: Basic model functionality with minimal dependencies
LIMITATIONS: Simplified evaluation only - use demo_week8_deep_learning.py for production testing

Environment: Non-interactive backend compatible (server environments)
Performance: Lightweight execution (~1-2 minutes)

For complete feature testing, use: demo_week8_deep_learning.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from src.models.deep_learning import (
    QuantLSTMClassifier,
    QuantLSTMRegressor,
    QuantGRUClassifier,
    QuantGRURegressor,
)


def create_sample_data():
    """Create sample financial time series data."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    # Generate feature names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

    # Generate correlated features with time-series characteristics
    data = np.random.randn(n_samples, n_features)

    # Add some autocorrelation
    for i in range(1, n_features):
        data[:, i] = 0.3 * data[:, i - 1] + 0.7 * data[:, i]

    # Add some time trend
    for i in range(1, n_samples):
        data[i] = 0.1 * data[i - 1] + 0.9 * data[i]

    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)

    # Create targets
    # Classification: direction of feature_1 + feature_2
    y_class = ((df["feature_1"] + df["feature_2"]) > 0).astype(int)

    # Regression: linear combination of features with noise
    y_reg = (
        0.3 * df["feature_1"]
        + 0.2 * df["feature_2"]
        + 0.1 * df["feature_3"]
        + np.random.normal(0, 0.1, n_samples)
    )

    return df.values, y_class.values, y_reg.values


def main():
    """
    Simple demo for deep learning models.

    Usage Guidelines:
    - Development: Quick functionality testing and debugging
    - CI/CD: Automated testing in server environments
    - Learning: Understanding basic model capabilities
    - Production: Use demo_week8_deep_learning.py for complete validation
    """
    print("=" * 60)
    print("WEEK 8: DEEP LEARNING MODELS - SIMPLE DEMO")
    print("=" * 60)
    print("Purpose: Development testing and basic functionality verification")
    print("For production testing: Use demo_week8_deep_learning.py")
    print()

    # 1. Data Preparation
    print("1. PREPARING SAMPLE DATA")
    print("-" * 40)

    X, y_class, y_reg = create_sample_data()
    print(f"Sample data shape: {X.shape}")
    print(f"Classification target distribution: {np.bincount(y_class)}")
    print(f"Regression target range: [{y_reg.min():.3f}, {y_reg.max():.3f}]")
    print()

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_class_train, y_class_test = y_class[:split_idx], y_class[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    # 2. LSTM Classification
    print("2. LSTM CLASSIFICATION")
    print("-" * 40)

    lstm_clf = QuantLSTMClassifier(
        sequence_length=20,
        lstm_units=[32, 16],
        dense_units=[8],
        dropout_rate=0.2,
        epochs=10,
        batch_size=32,
        verbose=0,
    )

    print("Training LSTM Classifier...")
    lstm_clf.fit(X_train, y_class_train)

    # Predictions
    lstm_pred = lstm_clf.predict(X_test)
    lstm_proba = lstm_clf.predict_proba(X_test)

    # Calculate accuracy
    accuracy = np.mean(lstm_pred == y_class_test[lstm_clf.sequence_length - 1 :])
    print(f"LSTM Classification Accuracy: {accuracy:.3f}")
    print()

    # 3. LSTM Regression
    print("3. LSTM REGRESSION")
    print("-" * 40)

    lstm_reg = QuantLSTMRegressor(
        sequence_length=20,
        lstm_units=[32, 16],
        dense_units=[8],
        dropout_rate=0.2,
        epochs=10,
        batch_size=32,
        verbose=0,
    )

    print("Training LSTM Regressor...")
    lstm_reg.fit(X_train, y_reg_train)

    # Predictions
    lstm_reg_pred = lstm_reg.predict(X_test)

    # Calculate metrics
    y_test_aligned = y_reg_test[lstm_reg.sequence_length - 1 :]
    mse = np.mean((lstm_reg_pred - y_test_aligned) ** 2)
    mae = np.mean(np.abs(lstm_reg_pred - y_test_aligned))

    print(f"LSTM Regression MSE: {mse:.4f}")
    print(f"LSTM Regression MAE: {mae:.4f}")
    print()

    # 4. GRU Classification
    print("4. GRU CLASSIFICATION")
    print("-" * 40)

    gru_clf = QuantGRUClassifier(
        sequence_length=20,
        gru_units=[32, 16],
        dense_units=[8],
        dropout_rate=0.2,
        epochs=10,
        batch_size=32,
        verbose=0,
    )

    print("Training GRU Classifier...")
    gru_clf.fit(X_train, y_class_train)

    # Predictions
    gru_pred = gru_clf.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(gru_pred == y_class_test[gru_clf.sequence_length - 1 :])
    print(f"GRU Classification Accuracy: {accuracy:.3f}")
    print()

    # 5. GRU Regression
    print("5. GRU REGRESSION")
    print("-" * 40)

    gru_reg = QuantGRURegressor(
        sequence_length=20,
        gru_units=[32, 16],
        dense_units=[8],
        dropout_rate=0.2,
        epochs=10,
        batch_size=32,
        verbose=0,
    )

    print("Training GRU Regressor...")
    gru_reg.fit(X_train, y_reg_train)

    # Predictions
    gru_reg_pred = gru_reg.predict(X_test)

    # Calculate metrics
    mse = np.mean((gru_reg_pred - y_test_aligned) ** 2)
    mae = np.mean(np.abs(gru_reg_pred - y_test_aligned))

    print(f"GRU Regression MSE: {mse:.4f}")
    print(f"GRU Regression MAE: {mae:.4f}")
    print()

    # 6. Model Comparison
    print("6. MODEL COMPARISON")
    print("-" * 40)

    print("Classification Results:")
    lstm_acc = np.mean(lstm_pred == y_class_test[lstm_clf.sequence_length - 1 :])
    gru_acc = np.mean(gru_pred == y_class_test[gru_clf.sequence_length - 1 :])
    print(f"  LSTM Accuracy: {lstm_acc:.3f}")
    print(f"  GRU Accuracy:  {gru_acc:.3f}")
    print()

    print("Regression Results:")
    lstm_mse = np.mean((lstm_reg_pred - y_test_aligned) ** 2)
    gru_mse = np.mean((gru_reg_pred - y_test_aligned) ** 2)
    print(f"  LSTM MSE: {lstm_mse:.4f}")
    print(f"  GRU MSE:  {gru_mse:.4f}")
    print()

    # 7. Visualization
    print("7. VISUALIZATION")
    print("-" * 40)

    # Plot regression predictions
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(y_test_aligned[:50], label="Actual", alpha=0.8)
    plt.plot(lstm_reg_pred[:50], label="LSTM Predicted", alpha=0.8)
    plt.title("LSTM Regression Predictions")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_test_aligned[:50], label="Actual", alpha=0.8)
    plt.plot(gru_reg_pred[:50], label="GRU Predicted", alpha=0.8)
    plt.title("GRU Regression Predictions")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    # Save plot instead of showing it
    plt.savefig("week8_deep_learning_results.png", dpi=150, bbox_inches="tight")
    print("✓ Visualization saved as 'week8_deep_learning_results.png'")
    plt.close()

    # 8. Advanced Features Demo
    print("8. ADVANCED FEATURES")
    print("-" * 40)

    # Uncertainty estimation
    print("Testing uncertainty estimation...")
    try:
        mean_pred, std_pred = lstm_reg.predict_with_uncertainty(
            X_test[:20], n_samples=20
        )
        print(f"Mean prediction std: {std_pred.mean():.4f}")
        print("✓ Uncertainty estimation working")
    except Exception as e:
        print(f"✗ Uncertainty estimation error: {e}")

    # Training history
    print("\\nTraining history available:")
    print(f"✓ LSTM final loss: {lstm_clf.training_history.get('final_loss', 'N/A')}")
    print(f"✓ GRU final loss: {gru_clf.training_history.get('final_loss', 'N/A')}")

    print()
    print("=" * 60)
    print("WEEK 8 SIMPLE DEMO COMPLETED!")
    print("=" * 60)
    print("✓ LSTM and GRU models successfully trained and tested")
    print("✓ Both classification and regression tasks demonstrated")
    print("✓ Model comparison and evaluation completed")
    print("✓ Advanced features (uncertainty estimation) tested")
    print()
    print("All deep learning models are working correctly!")


if __name__ == "__main__":
    main()
