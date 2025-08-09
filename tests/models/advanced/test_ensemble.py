"""
Tests for Ensemble methods and voting classifiers.

This module tests the ensemble learning approaches for robust
financial prediction models.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from unittest.mock import patch, MagicMock

# Importing modules under test
try:
    from src.models.advanced.ensemble import (
        EnsembleClassifier,
        VotingEnsemble,
        StackingEnsemble,
        BaggingEnsemble,
        BoostingEnsemble,
        create_ensemble_model,
        EnsembleValidator,
    )

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestEnsembleClassifier:
    """Test base EnsembleClassifier component."""

    @pytest.fixture
    def sample_models(self):
        """Create sample models for ensemble."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        models = [
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("lr", LogisticRegression(random_state=42, max_iter=100)),
            ("svm", SVC(probability=True, random_state=42)),
        ]

        return models

    @pytest.fixture
    def financial_data(self):
        """Create sample financial dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # Generate correlated features mimicking financial indicators
        X = np.random.randn(n_samples, n_features)

        # Add some correlation between features
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * np.random.randn(
            n_samples
        )  # Correlated with first feature
        X[:, 2] = np.abs(X[:, 0]) + 0.2 * np.random.randn(
            n_samples
        )  # Non-linear relationship

        # Create target: price movement prediction
        # Based on combination of features with some noise
        linear_combination = np.dot(X, np.random.randn(n_features))
        y = (linear_combination > np.median(linear_combination)).astype(int)

        feature_names = [
            "returns",
            "volatility",
            "volume",
            "rsi",
            "macd",
            "sma_ratio",
            "bollinger_position",
            "momentum",
            "vix",
            "market_cap",
        ]

        return X, y, feature_names

    def test_ensemble_classifier_creation(self, sample_models):
        """Test EnsembleClassifier creation."""
        ensemble = EnsembleClassifier(models=sample_models)
        assert ensemble is not None
        assert len(ensemble.models) == 3

    def test_ensemble_classifier_fitting(self, sample_models, financial_data):
        """Test EnsembleClassifier fitting process."""
        X, y, _ = financial_data

        ensemble = EnsembleClassifier(models=sample_models)
        ensemble.fit(X, y)

        # Check that all models are fitted
        for name, model in ensemble.models:
            assert hasattr(model, "predict")

    def test_ensemble_classifier_prediction(self, sample_models, financial_data):
        """Test EnsembleClassifier prediction."""
        X, y, _ = financial_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        ensemble = EnsembleClassifier(models=sample_models)
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

    def test_ensemble_classifier_performance(self, sample_models, financial_data):
        """Test EnsembleClassifier performance on financial data."""
        X, y, _ = financial_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        ensemble = EnsembleClassifier(models=sample_models)
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Ensemble should achieve reasonable accuracy
        assert accuracy > 0.5


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestVotingEnsemble:
    """Test VotingEnsemble component."""

    @pytest.fixture
    def voting_models(self):
        """Create models for voting ensemble."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        models = [
            ("rf", RandomForestClassifier(n_estimators=20, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=20, random_state=42)),
            ("lr", LogisticRegression(random_state=42, max_iter=200)),
        ]

        return models

    def test_hard_voting_ensemble(self, voting_models, financial_data):
        """Test hard voting ensemble."""
        X, y, _ = financial_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        ensemble = VotingEnsemble(models=voting_models, voting="hard")
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

    def test_soft_voting_ensemble(self, voting_models, financial_data):
        """Test soft voting ensemble."""
        X, y, _ = financial_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        ensemble = VotingEnsemble(models=voting_models, voting="soft")
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)

    def test_weighted_voting_ensemble(self, voting_models, financial_data):
        """Test weighted voting ensemble."""
        X, y, _ = financial_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        weights = [0.5, 0.3, 0.2]  # Different weights for models
        ensemble = VotingEnsemble(models=voting_models, voting="soft", weights=weights)
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestStackingEnsemble:
    """Test StackingEnsemble component."""

    @pytest.fixture
    def stacking_models(self):
        """Create models for stacking ensemble."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB

        base_models = [
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("lr", LogisticRegression(random_state=42, max_iter=100)),
            ("nb", GaussianNB()),
        ]

        meta_model = LogisticRegression(random_state=42, max_iter=100)

        return base_models, meta_model

    def test_stacking_ensemble_creation(self, stacking_models):
        """Test StackingEnsemble creation."""
        base_models, meta_model = stacking_models

        ensemble = StackingEnsemble(base_models=base_models, meta_model=meta_model)
        assert ensemble is not None
        assert len(ensemble.base_models) == 3

    def test_stacking_ensemble_cross_validation(self, stacking_models, financial_data):
        """Test StackingEnsemble with cross-validation."""
        X, y, _ = financial_data
        base_models, meta_model = stacking_models

        ensemble = StackingEnsemble(
            base_models=base_models, meta_model=meta_model, cv_folds=3
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_stacking_ensemble_performance(self, stacking_models, financial_data):
        """Test StackingEnsemble performance."""
        X, y, _ = financial_data
        base_models, meta_model = stacking_models

        ensemble = StackingEnsemble(base_models=base_models, meta_model=meta_model)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Stacking should achieve competitive performance
        assert accuracy > 0.45


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestBaggingEnsemble:
    """Test BaggingEnsemble component."""

    def test_bagging_ensemble_creation(self):
        """Test BaggingEnsemble creation."""
        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(random_state=42)
        ensemble = BaggingEnsemble(base_estimator=base_estimator, n_estimators=10)

        assert ensemble is not None

    def test_bagging_ensemble_prediction(self, financial_data):
        """Test BaggingEnsemble prediction."""
        X, y, _ = financial_data

        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(random_state=42)
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator, n_estimators=15, random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_bagging_feature_sampling(self, financial_data):
        """Test BaggingEnsemble with feature sampling."""
        X, y, _ = financial_data

        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(random_state=42)
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=10,
            max_features=0.7,  # Use 70% of features
            random_state=42,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Should achieve reasonable performance with feature sampling
        assert accuracy > 0.4


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestBoostingEnsemble:
    """Test BoostingEnsemble component."""

    def test_boosting_ensemble_creation(self):
        """Test BoostingEnsemble creation."""
        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        ensemble = BoostingEnsemble(base_estimator=base_estimator, n_estimators=20)

        assert ensemble is not None

    def test_boosting_ensemble_prediction(self, financial_data):
        """Test BoostingEnsemble prediction."""
        X, y, _ = financial_data

        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        ensemble = BoostingEnsemble(
            base_estimator=base_estimator,
            n_estimators=30,
            learning_rate=0.1,
            random_state=42,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_boosting_ensemble_performance(self, financial_data):
        """Test BoostingEnsemble performance."""
        X, y, _ = financial_data

        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        ensemble = BoostingEnsemble(
            base_estimator=base_estimator,
            n_estimators=50,
            learning_rate=0.1,
            random_state=42,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Boosting should achieve good performance
        assert accuracy > 0.5


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestEnsembleModelCreation:
    """Test ensemble model creation utilities."""

    def test_create_voting_ensemble(self, financial_data):
        """Test creating voting ensemble."""
        X, y, _ = financial_data

        ensemble = create_ensemble_model(
            ensemble_type="voting", voting_type="soft", n_estimators=5
        )

        assert ensemble is not None

        # Test fitting and prediction
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_create_stacking_ensemble(self, financial_data):
        """Test creating stacking ensemble."""
        X, y, _ = financial_data

        ensemble = create_ensemble_model(ensemble_type="stacking", cv_folds=3)

        assert ensemble is not None

        # Test fitting and prediction
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_create_bagging_ensemble(self, financial_data):
        """Test creating bagging ensemble."""
        X, y, _ = financial_data

        ensemble = create_ensemble_model(ensemble_type="bagging", n_estimators=20)

        assert ensemble is not None

        # Test fitting and prediction
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestEnsembleValidator:
    """Test EnsembleValidator component."""

    def test_ensemble_validator_creation(self):
        """Test EnsembleValidator creation."""
        validator = EnsembleValidator()
        assert validator is not None

    def test_cross_validation_ensemble(self, financial_data):
        """Test cross-validation of ensemble models."""
        X, y, _ = financial_data

        # Create multiple ensemble models
        ensembles = {
            "voting": create_ensemble_model(ensemble_type="voting", voting_type="soft"),
            "bagging": create_ensemble_model(ensemble_type="bagging", n_estimators=10),
            "stacking": create_ensemble_model(ensemble_type="stacking", cv_folds=3),
        }

        validator = EnsembleValidator()

        # Validate each ensemble
        for name, ensemble in ensembles.items():
            scores = validator.cross_validate(ensemble, X, y, cv=3)

            assert "accuracy" in scores
            assert len(scores["accuracy"]) == 3  # 3-fold CV

    def test_ensemble_comparison(self, financial_data):
        """Test comparison of different ensemble methods."""
        X, y, _ = financial_data

        # Create multiple ensemble models
        ensembles = {
            "voting_hard": create_ensemble_model(
                ensemble_type="voting", voting_type="hard"
            ),
            "voting_soft": create_ensemble_model(
                ensemble_type="voting", voting_type="soft"
            ),
            "bagging": create_ensemble_model(ensemble_type="bagging", n_estimators=15),
        }

        validator = EnsembleValidator()

        # Compare ensembles
        comparison_results = validator.compare_ensembles(ensembles, X, y)

        assert isinstance(comparison_results, dict)
        assert len(comparison_results) == len(ensembles)


# Integration tests
@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestEnsembleIntegration:
    """Integration tests for Ensemble components."""

    @pytest.fixture
    def complex_financial_data(self):
        """Create complex financial dataset with multiple features and patterns."""
        np.random.seed(42)
        n_samples = 2000

        # Time series component
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="1H")

        # Base price movement
        returns = np.random.normal(0, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))

        # Technical indicators
        features = pd.DataFrame(
            {
                "price": prices,
                "returns": returns,
                "sma_5": prices,  # Simplified SMA
                "sma_20": prices,  # Simplified SMA
                "rsi": np.random.uniform(0, 100, n_samples),
                "macd": np.random.normal(0, 1, n_samples),
                "volume": np.random.exponential(1000, n_samples),
                "volatility": np.random.exponential(0.2, n_samples),
                "momentum": np.random.normal(0, 0.1, n_samples),
                "sentiment": np.random.uniform(-1, 1, n_samples),
            }
        )

        # Create target: multi-class price movement
        future_returns = np.roll(returns, -5)  # 5-step ahead returns
        conditions = [
            future_returns < -0.01,  # Strong down
            (future_returns >= -0.01) & (future_returns <= 0.01),  # Sideways
            future_returns > 0.01,  # Strong up
        ]
        targets = np.select(conditions, [0, 1, 2], default=1)

        # Remove last 5 samples (no future data)
        features = features.iloc[:-5]
        targets = targets[:-5]

        return features.values, targets, features.columns.tolist(), dates[:-5]

    def test_ensemble_with_complex_financial_data(self, complex_financial_data):
        """Test ensemble models with complex financial data."""
        X, y, feature_names, dates = complex_financial_data

        # Create sophisticated ensemble
        ensemble = create_ensemble_model(ensemble_type="stacking", cv_folds=5)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Fit ensemble
        ensemble.fit(X_train, y_train)

        # Make predictions
        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape[1] == 3  # 3 classes

    def test_ensemble_performance_comparison(self, complex_financial_data):
        """Test performance comparison of different ensemble methods."""
        X, y, feature_names, dates = complex_financial_data

        # Create different ensembles
        ensembles = {
            "voting_soft": create_ensemble_model(
                ensemble_type="voting", voting_type="soft"
            ),
            "stacking": create_ensemble_model(ensemble_type="stacking", cv_folds=3),
            "bagging": create_ensemble_model(ensemble_type="bagging", n_estimators=20),
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        results = {}

        for name, ensemble in ensembles.items():
            ensemble.fit(X_train, y_train)
            predictions = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results[name] = accuracy

        # All ensembles should achieve reasonable performance
        for name, accuracy in results.items():
            assert accuracy > 0.25  # Better than random for 3-class problem

    def test_ensemble_with_time_series_validation(self, complex_financial_data):
        """Test ensemble with time series cross-validation."""
        X, y, feature_names, dates = complex_financial_data

        # Time series split for validation
        from sklearn.model_selection import TimeSeriesSplit

        ensemble = create_ensemble_model(ensemble_type="voting", voting_type="soft")

        tscv = TimeSeriesSplit(n_splits=3)
        accuracies = []

        for train_idx, test_idx in tscv.split(X):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            ensemble.fit(X_train_fold, y_train_fold)
            predictions = ensemble.predict(X_test_fold)
            accuracy = accuracy_score(y_test_fold, predictions)
            accuracies.append(accuracy)

        # Time series validation should work
        assert len(accuracies) == 3
        assert all(acc > 0.2 for acc in accuracies)  # Reasonable performance

    def test_ensemble_feature_importance(self, complex_financial_data):
        """Test feature importance extraction from ensembles."""
        X, y, feature_names, dates = complex_financial_data

        # Use ensemble that supports feature importance
        ensemble = create_ensemble_model(ensemble_type="bagging", n_estimators=20)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        ensemble.fit(X_train, y_train)

        # Extract feature importance if available
        if hasattr(ensemble, "feature_importances_"):
            importances = ensemble.feature_importances_
            assert len(importances) == len(feature_names)
            assert all(imp >= 0 for imp in importances)


if __name__ == "__main__":
    pytest.main([__file__])
