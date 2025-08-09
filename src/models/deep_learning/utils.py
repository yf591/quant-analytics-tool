"""
Deep Learning Utilities for Financial Time Series

This module provides utility functions for deep learning models including
model evaluation, visualization, and financial-specific metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

# Scikit-learn imports
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


class DeepLearningEvaluator:
    """
    Comprehensive evaluator for deep learning models in financial applications.

    Provides evaluation metrics, visualization tools, and financial-specific
    performance measures for LSTM/GRU models.
    """

    def __init__(self, model, model_name: str = "Deep Learning Model"):
        """
        Initialize evaluator.

        Args:
            model: Fitted deep learning model
            model_name: Name of the model for display purposes
        """
        self.model = model
        self.model_name = model_name

    def evaluate_classifier(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        plot_results: bool = True,
        figsize: Tuple[int, int] = (15, 10),
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification models.

        Args:
            X_test: Test features
            y_test: Test targets
            plot_results: Whether to plot evaluation results
            figsize: Figure size for plots

        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "predictions": y_pred,
            "probabilities": y_proba,
        }

        if plot_results:
            self._plot_classification_results(results, y_test, figsize)

        return results

    def evaluate_regressor(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        plot_results: bool = True,
        figsize: Tuple[int, int] = (15, 10),
        uncertainty_estimation: bool = False,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for regression models.

        Args:
            X_test: Test features
            y_test: Test targets
            plot_results: Whether to plot evaluation results
            figsize: Figure size for plots
            uncertainty_estimation: Whether to include uncertainty estimation

        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate basic regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate financial metrics
        directional_accuracy = self._calculate_directional_accuracy(y_test, y_pred)
        hit_ratio = self._calculate_hit_ratio(y_test, y_pred)

        results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "directional_accuracy": directional_accuracy,
            "hit_ratio": hit_ratio,
            "predictions": y_pred,
            "actual": y_test,
        }

        # Add uncertainty estimation if available
        if uncertainty_estimation and hasattr(self.model, "predict_with_uncertainty"):
            y_pred_mean, y_pred_std = self.model.predict_with_uncertainty(X_test)
            results["uncertainty_mean"] = y_pred_mean
            results["uncertainty_std"] = y_pred_std

        if plot_results:
            self._plot_regression_results(results, figsize)

        return results

    def _calculate_directional_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate directional accuracy for financial predictions."""
        # Convert to returns if not already
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            return np.mean(true_direction == pred_direction)
        return 0.0

    def _calculate_hit_ratio(
        self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0
    ) -> float:
        """Calculate hit ratio (percentage of correct sign predictions)."""
        true_signs = np.sign(y_true - threshold)
        pred_signs = np.sign(y_pred - threshold)
        return np.mean(true_signs == pred_signs)

    def _plot_classification_results(
        self, results: Dict[str, Any], y_test: np.ndarray, figsize: Tuple[int, int]
    ):
        """Plot classification evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"{self.model_name} - Classification Results", fontsize=16)

        # Confusion Matrix
        sns.heatmap(
            results["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")

        # ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, results["probabilities"][:, 1])
            axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], "k--")
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("True Positive Rate")
            axes[0, 1].set_title("ROC Curve")
            axes[0, 1].legend()
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Multiclass\nROC Curve",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("ROC Curve (Multiclass)")

        # Metrics Bar Plot
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
        values = [
            results["accuracy"],
            results["precision"],
            results["recall"],
            results["f1_score"],
            results["roc_auc"],
        ]

        axes[1, 0].bar(metrics, values)
        axes[1, 0].set_title("Performance Metrics")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Prediction Distribution
        axes[1, 1].hist(
            (
                results["probabilities"][:, 1]
                if len(np.unique(y_test)) == 2
                else results["probabilities"].max(axis=1)
            ),
            bins=30,
            alpha=0.7,
        )
        axes[1, 1].set_title("Prediction Probability Distribution")
        axes[1, 1].set_xlabel("Probability")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def _plot_regression_results(
        self, results: Dict[str, Any], figsize: Tuple[int, int]
    ):
        """Plot regression evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"{self.model_name} - Regression Results", fontsize=16)

        y_true = results["actual"]
        y_pred = results["predictions"]

        # Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual Values")
        axes[0, 0].set_ylabel("Predicted Values")
        axes[0, 0].set_title("Actual vs Predicted")

        # Residuals Plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals Plot")

        # Metrics Bar Plot
        metrics = ["R²", "RMSE", "MAE", "Dir. Acc.", "Hit Ratio"]
        values = [
            results["r2_score"],
            results["rmse"],
            results["mae"],
            results["directional_accuracy"],
            results["hit_ratio"],
        ]

        axes[1, 0].bar(metrics, values)
        axes[1, 0].set_title("Performance Metrics")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Time Series Plot (if indices are available)
        if hasattr(y_true, "index"):
            axes[1, 1].plot(y_true.index, y_true, label="Actual", alpha=0.7)
            axes[1, 1].plot(y_true.index, y_pred, label="Predicted", alpha=0.7)
        else:
            axes[1, 1].plot(y_true, label="Actual", alpha=0.7)
            axes[1, 1].plot(y_pred, label="Predicted", alpha=0.7)

        axes[1, 1].set_title("Time Series Comparison")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Values")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def plot_model_architecture(self, save_path: Optional[str] = None):
        """
        Plot model architecture diagram.

        Args:
            save_path: Path to save the diagram (optional)
        """
        if hasattr(self.model, "model_"):
            plot_model(
                self.model.model_,
                to_file=save_path or f"{self.model_name.lower()}_architecture.png",
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
            )
            print(
                f"Model architecture saved as {save_path or f'{self.model_name.lower()}_architecture.png'}"
            )
        else:
            print("Model architecture not available")


class FinancialMetrics:
    """
    Financial-specific metrics for deep learning model evaluation.
    """

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio."""
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns)
        if tracking_error == 0:
            return 0.0
        return np.mean(active_returns) / tracking_error

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        annual_return = np.mean(returns) * 252  # Assuming daily returns
        max_dd = FinancialMetrics.max_drawdown(returns)
        if max_dd == 0:
            return 0.0
        return annual_return / abs(max_dd)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        return np.mean(excess_returns) / downside_deviation

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()
        if losses == 0:
            return float("inf") if gains > 0 else 1.0
        return gains / losses


class ModelComparison:
    """
    Compare multiple deep learning models.
    """

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name: str, model, X_test: np.ndarray, y_test: np.ndarray):
        """
        Add model for comparison.

        Args:
            name: Model name
            model: Fitted model
            X_test: Test features
            y_test: Test targets
        """
        self.models[name] = model

        # Evaluate model
        evaluator = DeepLearningEvaluator(model, name)

        if hasattr(model, "predict_proba"):
            # Classification
            results = evaluator.evaluate_classifier(X_test, y_test, plot_results=False)
        else:
            # Regression
            results = evaluator.evaluate_regressor(X_test, y_test, plot_results=False)

        self.results[name] = results

    def compare_metrics(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Compare metrics across models.

        Args:
            figsize: Figure size
        """
        if not self.results:
            print("No models to compare. Add models first.")
            return

        # Determine if classification or regression
        first_result = list(self.results.values())[0]
        is_classification = "accuracy" in first_result

        if is_classification:
            metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
        else:
            metrics = ["r2_score", "rmse", "mae", "directional_accuracy", "hit_ratio"]
            metric_names = ["R²", "RMSE", "MAE", "Directional Accuracy", "Hit Ratio"]

        # Create comparison DataFrame
        comparison_data = {}
        for model_name, results in self.results.items():
            comparison_data[model_name] = [results.get(metric, 0) for metric in metrics]

        df_comparison = pd.DataFrame(comparison_data, index=metric_names)

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar plot
        df_comparison.plot(kind="bar", ax=ax1)
        ax1.set_title("Model Comparison - Metrics")
        ax1.set_ylabel("Score")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.tick_params(axis="x", rotation=45)

        # Heatmap
        sns.heatmap(df_comparison.T, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax2)
        ax2.set_title("Model Comparison - Heatmap")

        plt.tight_layout()
        plt.show()

        return df_comparison

    def print_summary(self):
        """Print summary of model comparison."""
        if not self.results:
            print("No models to compare. Add models first.")
            return

        print("=" * 50)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 50)

        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print("-" * len(model_name))

            if "accuracy" in results:
                # Classification
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  Precision: {results['precision']:.4f}")
                print(f"  Recall: {results['recall']:.4f}")
                print(f"  F1-Score: {results['f1_score']:.4f}")
                print(f"  ROC AUC: {results['roc_auc']:.4f}")
            else:
                # Regression
                print(f"  R² Score: {results['r2_score']:.4f}")
                print(f"  RMSE: {results['rmse']:.4f}")
                print(f"  MAE: {results['mae']:.4f}")
                print(f"  Directional Accuracy: {results['directional_accuracy']:.4f}")
                print(f"  Hit Ratio: {results['hit_ratio']:.4f}")
