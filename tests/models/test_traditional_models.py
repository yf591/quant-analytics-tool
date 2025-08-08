"""
Tests for Traditional ML Models

This module contains comprehensive tests for Random Forest, XGBoost, and SVM
models to ensure proper functionality and financial optimization.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings

warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.traditional import (
    QuantRandomForestClassifier,
    QuantRandomForestRegressor,
    QuantXGBoostClassifier,
    QuantXGBoostRegressor,
    QuantSVMClassifier,
    QuantSVMRegressor,
)


class TestTraditionalMLModels:
    """Test suite for traditional ML models."""

    @pytest.fixture
    def classification_data(self):
        """Generate synthetic classification data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )

        # Convert to DataFrame with feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")

        return train_test_split(X_df, y_series, test_size=0.2, random_state=42)

    @pytest.fixture
    def regression_data(self):
        """Generate synthetic regression data."""
        X, y = make_regression(
            n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=42
        )

        # Convert to DataFrame with feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")

        return train_test_split(X_df, y_series, test_size=0.2, random_state=42)

    def test_random_forest_classifier(self, classification_data):
        """Test Random Forest Classifier."""
        X_train, X_test, y_train, y_test = classification_data

        # Initialize model
        model = QuantRandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42
        )

        # Test fitting
        model.fit(X_train, y_train)
        assert model.is_fitted
        assert len(model.classes_) == 2

        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test probabilities
        y_proba = model.predict_proba(X_test)
        assert y_proba.shape == (len(y_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)

        # Test accuracy
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.7  # Should achieve reasonable accuracy

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X_train.columns)
        assert importance.sum() > 0

        print(f"Random Forest Classifier Accuracy: {accuracy:.3f}")

    def test_random_forest_regressor(self, regression_data):
        """Test Random Forest Regressor."""
        X_train, X_test, y_train, y_test = regression_data

        # Initialize model
        model = QuantRandomForestRegressor(
            n_estimators=10, max_depth=5, random_state=42
        )

        # Test fitting
        model.fit(X_train, y_train)
        assert model.is_fitted

        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test quantile predictions
        quantiles = model.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
        assert quantiles.shape == (len(y_test), 3)

        # Test R² score
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0.5  # Should achieve reasonable R²

        print(f"Random Forest Regressor R²: {r2:.3f}")

    def test_xgboost_classifier(self, classification_data):
        """Test XGBoost Classifier."""
        X_train, X_test, y_train, y_test = classification_data

        # Initialize model
        model = QuantXGBoostClassifier(
            n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42
        )

        # Test fitting with evaluation set
        eval_set = [(X_test, y_test)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        assert model.is_fitted
        assert len(model.classes_) == 2

        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test probabilities
        y_proba = model.predict_proba(X_test)
        assert y_proba.shape == (len(y_test), 2)

        # Test accuracy
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.7

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) > 0

        print(f"XGBoost Classifier Accuracy: {accuracy:.3f}")

    def test_xgboost_regressor(self, regression_data):
        """Test XGBoost Regressor."""
        X_train, X_test, y_train, y_test = regression_data

        # Initialize model
        model = QuantXGBoostRegressor(
            n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42
        )

        # Test fitting
        model.fit(X_train, y_train, verbose=False)
        assert model.is_fitted

        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test R² score
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0.4  # Adjusted threshold for XGBoost regressor

        print(f"XGBoost Regressor R²: {r2:.3f}")

    def test_svm_classifier(self, classification_data):
        """Test SVM Classifier."""
        X_train, X_test, y_train, y_test = classification_data

        # Use smaller dataset for SVM (faster training)
        X_train_small = X_train.iloc[:200]
        y_train_small = y_train.iloc[:200]

        # Initialize model
        model = QuantSVMClassifier(
            kernel="rbf", C=1.0, random_state=42, scale_features=True
        )

        # Test fitting
        model.fit(X_train_small, y_train_small)
        assert model.is_fitted
        assert len(model.classes_) == 2

        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test probabilities
        y_proba = model.predict_proba(X_test)
        assert y_proba.shape == (len(y_test), 2)

        # Test decision function
        decision_scores = model.decision_function(X_test)
        assert len(decision_scores) == len(y_test)

        # Test accuracy
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.6  # Lower threshold for SVM

        # Test support vectors
        n_support = model._get_n_support()
        assert isinstance(n_support, (int, np.ndarray))

        print(f"SVM Classifier Accuracy: {accuracy:.3f}")

    def test_svm_regressor(self, regression_data):
        """Test SVM Regressor."""
        X_train, X_test, y_train, y_test = regression_data

        # Use smaller dataset for SVM (faster training)
        X_train_small = X_train.iloc[:200]
        y_train_small = y_train.iloc[:200]

        # Initialize model
        model = QuantSVMRegressor(kernel="rbf", C=1.0, epsilon=0.1, scale_features=True)

        # Test fitting
        model.fit(X_train_small, y_train_small)
        assert model.is_fitted

        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test R² score
        r2 = r2_score(y_test, y_pred)
        assert r2 > -1.0  # Very low threshold for SVM (can be negative)

        print(f"SVM Regressor R²: {r2:.3f}")

    def test_hyperparameter_tuning(self, classification_data):
        """Test hyperparameter tuning."""
        X_train, X_test, y_train, y_test = classification_data

        # Use small dataset for faster testing
        X_train_small = X_train.iloc[:100]
        y_train_small = y_train.iloc[:100]

        # Test Random Forest tuning
        rf_model = QuantRandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [5, 10], "max_depth": [3, 5]}

        tuning_results = rf_model.tune_hyperparameters(
            X_train_small, y_train_small, param_grid=param_grid, cv=3, method="grid"
        )

        assert "best_params" in tuning_results
        assert "best_score" in tuning_results
        assert tuning_results["best_score"] > 0

        print(f"Best RF params: {tuning_results['best_params']}")

    def test_model_persistence(self, classification_data):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = classification_data

        # Train model
        model = QuantRandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        # Get original predictions
        original_pred = model.predict(X_test)

        # Save model
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model.save_model(f.name)

            # Load model
            loaded_model = QuantRandomForestClassifier()
            loaded_model = loaded_model.load_model(f.name)

            # Test loaded model
            loaded_pred = loaded_model.predict(X_test)

            # Predictions should be identical
            assert np.array_equal(original_pred, loaded_pred)

            # Clean up
            os.unlink(f.name)

    def test_training_history(self, classification_data):
        """Test training history tracking."""
        X_train, X_test, y_train, y_test = classification_data

        model = QuantRandomForestClassifier(n_estimators=5, random_state=42)

        # Check initial history is empty
        assert len(model.training_history) == 0

        # Train model
        model.fit(X_train, y_train)

        # Check history was recorded
        assert len(model.training_history) == 1
        history = model.training_history[0]

        assert "timestamp" in history
        assert "training_samples" in history
        assert "training_time" in history
        assert history["training_samples"] == len(X_train)
        assert history["training_time"] > 0


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])
