"""
Traditional ML Models Demo

This script demonstrates the usage of all traditional ML models
with financial data examples and comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.traditional import (
    QuantRandomForestClassifier,
    QuantRandomForestRegressor,
    QuantXGBoostClassifier,
    QuantXGBoostRegressor,
    QuantSVMClassifier,
    QuantSVMRegressor,
)
from src.models.evaluation import ModelEvaluator, CrossValidator


def generate_financial_data(n_samples=2000, task="classification"):
    """
    Generate synthetic financial data for demonstration.

    Args:
        n_samples: Number of samples to generate
        task: 'classification' or 'regression'

    Returns:
        Tuple of (X, y) data
    """
    np.random.seed(42)

    if task == "classification":
        # Generate binary classification data (e.g., buy/sell signals)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            random_state=42,
            class_sep=0.8,
        )

        feature_names = [
            "rsi",
            "macd",
            "bollinger_upper",
            "bollinger_lower",
            "sma_20",
            "ema_12",
            "ema_26",
            "volume_ratio",
            "price_change",
            "volatility",
            "momentum",
            "williams_r",
            "stoch_k",
            "stoch_d",
            "atr",
        ]

    else:
        # Generate regression data (e.g., return prediction)
        X, y = make_regression(
            n_samples=n_samples,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=42,
        )

        feature_names = [
            "return_1d",
            "return_5d",
            "return_20d",
            "vol_1d",
            "vol_5d",
            "price_momentum",
            "volume_momentum",
            "rsi_momentum",
            "macd_signal",
            "bb_position",
            "ma_distance",
            "correlation_spy",
            "beta",
            "alpha",
            "sharpe",
        ]

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


def demo_classification_models():
    """Demonstrate classification models with financial data."""
    print("=" * 80)
    print("TRADITIONAL ML MODELS - CLASSIFICATION DEMO")
    print("=" * 80)

    # Generate data
    X, y = generate_financial_data(n_samples=2000, task="classification")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X.columns)[:5]}...")
    print()

    # Models to test
    models = {
        "Random Forest": QuantRandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        ),
        "XGBoost": QuantXGBoostClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "SVM": QuantSVMClassifier(
            kernel="rbf", C=1.0, random_state=42, scale_features=True
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        if name == "XGBoost":
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Evaluate using ModelEvaluator
        evaluator = ModelEvaluator(problem_type="classification")
        metrics = evaluator.evaluate_model(model, X_test, y_test)

        results[name] = {
            "model": model,
            "predictions": y_pred,
            "probabilities": y_proba,
            "metrics": metrics,
        }

        # Print metrics
        print(f"Accuracy: {metrics.accuracy:.3f}")
        print(f"Precision: {metrics.precision:.3f}")
        print(f"Recall: {metrics.recall:.3f}")
        print(f"F1 Score: {metrics.f1_score:.3f}")
        print(f"AUC: {metrics.roc_auc:.3f}")

        # Print feature importance if available
        if hasattr(model, "get_feature_importance"):
            importance = model.get_feature_importance()
            print(f"Top 3 features: {importance.head(3).to_dict()}")

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison_df = pd.DataFrame(
        {
            name: {
                "Accuracy": results[name]["metrics"].accuracy,
                "Precision": results[name]["metrics"].precision,
                "Recall": results[name]["metrics"].recall,
                "F1": results[name]["metrics"].f1_score,
                "AUC": results[name]["metrics"].roc_auc,
            }
            for name in results.keys()
        }
    ).round(3)

    print(comparison_df)

    return results


def demo_regression_models():
    """Demonstrate regression models with financial data."""
    print("\n" + "=" * 80)
    print("TRADITIONAL ML MODELS - REGRESSION DEMO")
    print("=" * 80)

    # Generate data
    X, y = generate_financial_data(n_samples=2000, task="regression")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print()

    # Models to test
    models = {
        "Random Forest": QuantRandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42
        ),
        "XGBoost": QuantXGBoostRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "SVM": QuantSVMRegressor(kernel="rbf", C=1.0, epsilon=0.1, scale_features=True),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        if name == "XGBoost":
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate using ModelEvaluator
        evaluator = ModelEvaluator(problem_type="regression")
        metrics = evaluator.evaluate_model(model, X_test, y_test)

        results[name] = {"model": model, "predictions": y_pred, "metrics": metrics}

        # Print metrics
        print(f"R² Score: {metrics.r2:.3f}")
        print(f"MSE: {metrics.mse:.3f}")
        print(f"RMSE: {metrics.rmse:.3f}")
        print(f"MAE: {metrics.mae:.3f}")

        # Special features for certain models
        if name == "Random Forest":
            # Test quantile predictions
            quantiles = model.predict_quantiles(X_test[:10], quantiles=[0.1, 0.5, 0.9])
            print(f"Quantile predictions shape: {quantiles.shape}")

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison_df = pd.DataFrame(
        {
            name: {
                "R²": results[name]["metrics"].r2,
                "MSE": results[name]["metrics"].mse,
                "RMSE": results[name]["metrics"].rmse,
                "MAE": results[name]["metrics"].mae,
            }
            for name in results.keys()
        }
    ).round(3)

    print(comparison_df)

    return results


def demo_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING DEMO")
    print("=" * 80)

    # Generate small dataset for faster tuning
    X, y = generate_financial_data(n_samples=500, task="classification")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest tuning
    print("\nTuning Random Forest...")
    rf_model = QuantRandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 8, 12],
        "min_samples_split": [2, 5, 10],
    }

    tuning_results = rf_model.tune_hyperparameters(
        X_train, y_train, param_grid=param_grid, cv=3, method="random", n_iter=10
    )

    print(f"Best parameters: {tuning_results['best_params']}")
    print(f"Best CV score: {tuning_results['best_score']:.3f}")

    # Test tuned model
    y_pred = rf_model.predict(X_test)
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy with tuned params: {accuracy:.3f}")


def demo_cross_validation():
    """Demonstrate cross-validation evaluation."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION DEMO")
    print("=" * 80)

    # Generate data
    X, y = generate_financial_data(n_samples=1000, task="classification")

    # Models to cross-validate
    models = {
        "Random Forest": QuantRandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42
        ),
        "XGBoost": QuantXGBoostClassifier(
            n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42
        ),
    }

    cv_evaluator = CrossValidator(cv_folds=5, random_state=42)

    for name, model in models.items():
        print(f"\nCross-validating {name}...")

        cv_results = cv_evaluator.cross_validate(
            model, X, y, metrics=["accuracy", "precision", "recall", "f1", "auc"]
        )

        print(f"CV Results:")
        for metric, scores in cv_results.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {metric}: {mean_score:.3f} ± {std_score:.3f}")


def create_visualizations(classification_results, regression_results):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(20, 15))

    # 1. Classification metrics comparison
    plt.subplot(2, 3, 1)
    metrics_data = []
    for name, result in classification_results.items():
        metrics = result["metrics"]
        metrics_data.append(
            {
                "Model": name,
                "Accuracy": metrics.accuracy,
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "F1": metrics.f1_score,
                "AUC": metrics.roc_auc,
            }
        )

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Model").plot(kind="bar", ax=plt.gca())
    plt.title("Classification Metrics Comparison")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)

    # 2. Regression metrics comparison
    plt.subplot(2, 3, 2)
    reg_metrics_data = []
    for name, result in regression_results.items():
        metrics = result["metrics"]
        reg_metrics_data.append(
            {
                "Model": name,
                "R²": metrics.r2,
                "RMSE": metrics.rmse / 100,  # Scale for visualization
                "MAE": metrics.mae / 100,
            }
        )

    reg_metrics_df = pd.DataFrame(reg_metrics_data)
    reg_metrics_df.set_index("Model").plot(kind="bar", ax=plt.gca())
    plt.title("Regression Metrics Comparison")
    plt.ylabel("Score (RMSE/MAE scaled by 100)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)

    # 3. Feature importance for Random Forest
    plt.subplot(2, 3, 3)
    rf_model = classification_results["Random Forest"]["model"]
    importance = rf_model.get_feature_importance()
    importance.head(10).plot(kind="barh", ax=plt.gca())
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")

    # 4. Training time comparison (simulated)
    plt.subplot(2, 3, 4)
    training_times = {"Random Forest": 2.5, "XGBoost": 3.8, "SVM": 8.2}

    models = list(training_times.keys())
    times = list(training_times.values())
    bars = plt.bar(models, times, color=["skyblue", "lightgreen", "lightcoral"])
    plt.title("Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{time}s",
            ha="center",
            va="bottom",
        )

    # 5. Prediction distribution
    plt.subplot(2, 3, 5)
    rf_pred = classification_results["Random Forest"]["probabilities"][:, 1]
    xgb_pred = classification_results["XGBoost"]["probabilities"][:, 1]

    plt.hist(rf_pred, alpha=0.6, bins=30, label="Random Forest", density=True)
    plt.hist(xgb_pred, alpha=0.6, bins=30, label="XGBoost", density=True)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()

    # 6. Model complexity comparison
    plt.subplot(2, 3, 6)
    complexity_data = {
        "Random Forest": {"Interpretability": 3, "Training Speed": 4, "Accuracy": 4},
        "XGBoost": {"Interpretability": 2, "Training Speed": 3, "Accuracy": 5},
        "SVM": {"Interpretability": 1, "Training Speed": 2, "Accuracy": 3},
    }

    complexity_df = pd.DataFrame(complexity_data).T
    complexity_df.plot(kind="bar", ax=plt.gca())
    plt.title("Model Characteristics (1-5 scale)")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("traditional_ml_demo_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Visualizations saved as 'traditional_ml_demo_results.png'")


def main():
    """Run the complete demo."""
    print("🚀 Starting Traditional ML Models Demo")
    print("This demo showcases Random Forest, XGBoost, and SVM models")
    print("optimized for financial applications.\n")

    try:
        # Run classification demo
        classification_results = demo_classification_models()

        # Run regression demo
        regression_results = demo_regression_models()

        # Demonstrate hyperparameter tuning
        demo_hyperparameter_tuning()

        # Demonstrate cross-validation
        demo_cross_validation()

        # Create visualizations
        create_visualizations(classification_results, regression_results)

        print("\n" + "=" * 80)
        print("DEMO COMPLETE!")
        print("=" * 80)
        print("✅ All traditional ML models tested successfully")
        print("✅ Classification and regression tasks completed")
        print("✅ Hyperparameter tuning demonstrated")
        print("✅ Cross-validation evaluation performed")
        print("✅ Comprehensive visualizations created")
        print("\nAll models are ready for financial analysis tasks!")

    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
