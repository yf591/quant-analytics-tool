"""
Simple tests for implemented Ensemble methods.

This module tests only the actually implemented ensemble classes:
- FinancialRandomForest
- EnsembleConfig
- VotingEnsemble
- StackingEnsemble
- TimeSeriesBagging
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing implemented modules
try:
    from src.models.advanced.ensemble import (
        EnsembleConfig,
        FinancialRandomForest,
        VotingEnsemble,
        StackingEnsemble,
        TimeSeriesBagging,
    )

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestEnsembleConfig:
    """Test EnsembleConfig class."""

    def test_ensemble_config_creation(self):
        """Test EnsembleConfig creation with default values."""
        config = EnsembleConfig()
        assert config is not None
        assert hasattr(config, "n_estimators")

    def test_ensemble_config_custom_values(self):
        """Test EnsembleConfig with custom values."""
        config = EnsembleConfig(n_estimators=50, random_state=123)
        assert config.n_estimators == 50
        assert config.random_state == 123


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestFinancialRandomForest:
    """Test FinancialRandomForest class."""

    @pytest.fixture
    def financial_data(self):
        """Create sample financial dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 8

        # Generate financial-like features
        X = np.random.randn(n_samples, n_features)
        # Binary classification target
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.3 > 0).astype(int)

        feature_names = [
            "returns",
            "volume",
            "volatility",
            "rsi",
            "macd",
            "momentum",
            "vix",
            "market_cap",
        ]

        return X, y, feature_names

    def test_financial_random_forest_creation(self):
        """Test FinancialRandomForest creation."""
        config = EnsembleConfig(n_estimators=10, random_state=42)
        ensemble = FinancialRandomForest(config=config)
        assert ensemble is not None
        assert ensemble.config.n_estimators == 10

    def test_financial_random_forest_fit_predict(self, financial_data):
        """Test FinancialRandomForest fit and predict."""
        X, y, _ = financial_data

        config = EnsembleConfig(n_estimators=10, random_state=42)
        ensemble = FinancialRandomForest(config=config)

        # Fit the model
        ensemble.fit(X, y)

        # Check that model is fitted
        assert hasattr(ensemble, "model")
        assert ensemble.model is not None

        # Test prediction
        predictions = ensemble.predict(X[:100])
        assert len(predictions) == 100
        assert all(pred in [0, 1] for pred in predictions)

    def test_financial_random_forest_performance(self, financial_data):
        """Test FinancialRandomForest performance."""
        X, y, _ = financial_data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        config = EnsembleConfig(n_estimators=20, random_state=42)
        ensemble = FinancialRandomForest(config=config)

        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.5  # Should be better than random


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestVotingEnsemble:
    """Test VotingEnsemble class."""

    @pytest.fixture
    def financial_data(self):
        """Create sample financial dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 6

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    def test_voting_ensemble_creation(self):
        """Test VotingEnsemble creation."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        estimators = [
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("lr", LogisticRegression(random_state=42, max_iter=100)),
        ]
        config = EnsembleConfig()

        ensemble = VotingEnsemble(estimators=estimators, config=config)
        assert ensemble is not None
        assert len(ensemble.estimators) == 2

    def test_voting_ensemble_fit_predict(self, financial_data):
        """Test VotingEnsemble fit and predict."""
        X, y = financial_data

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        estimators = [
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("lr", LogisticRegression(random_state=42, max_iter=100)),
        ]

        ensemble = VotingEnsemble(estimators=estimators, config=EnsembleConfig())
        ensemble.fit(X, y)

        predictions = ensemble.predict(X[:50])
        assert len(predictions) == 50
        assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestStackingEnsemble:
    """Test StackingEnsemble class."""

    @pytest.fixture
    def financial_data(self):
        """Create sample financial dataset."""
        np.random.seed(42)
        n_samples = 300
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        return X, y

    def test_stacking_ensemble_creation(self):
        """Test StackingEnsemble creation."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        base_estimators = [
            RandomForestClassifier(n_estimators=5, random_state=42),
            LogisticRegression(random_state=42, max_iter=100),
        ]

        config = EnsembleConfig(meta_model="logistic")

        ensemble = StackingEnsemble(base_estimators=base_estimators, config=config)
        assert ensemble is not None
        assert len(ensemble.base_estimators) == 2

    def test_stacking_ensemble_fit_predict(self, financial_data):
        """Test StackingEnsemble fit and predict."""
        X, y = financial_data

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        base_estimators = [
            RandomForestClassifier(n_estimators=5, random_state=42),
            LogisticRegression(random_state=42, max_iter=100),
        ]

        config = EnsembleConfig(meta_model="logistic")

        ensemble = StackingEnsemble(base_estimators=base_estimators, config=config)

        ensemble.fit(X, y)
        predictions = ensemble.predict(X[:30])

        assert len(predictions) == 30
        assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestTimeSeriesBagging:
    """Test TimeSeriesBagging class."""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series dataset."""
        np.random.seed(42)
        n_samples = 200
        n_features = 4

        # Create time series-like data
        X = np.random.randn(n_samples, n_features)
        # Add some temporal dependency
        for i in range(1, n_samples):
            X[i] = 0.7 * X[i - 1] + 0.3 * X[i]

        y = (X[:, 0] > np.mean(X[:, 0])).astype(int)

        return X, y

    def test_time_series_bagging_creation(self):
        """Test TimeSeriesBagging creation."""
        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(random_state=42)
        config = EnsembleConfig(n_estimators=5)
        ensemble = TimeSeriesBagging(base_estimator=base_estimator, config=config)

        assert ensemble is not None
        assert ensemble.config.n_estimators == 5

    def test_time_series_bagging_fit_predict(self, time_series_data):
        """Test TimeSeriesBagging fit and predict."""
        X, y = time_series_data

        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(random_state=42)
        config = EnsembleConfig(n_estimators=5)
        ensemble = TimeSeriesBagging(base_estimator=base_estimator, config=config)

        ensemble.fit(X, y)
        predictions = ensemble.predict(X[:20])

        assert len(predictions) == 20
        assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble module not available")
class TestEnsembleIntegration:
    """Test ensemble integration scenarios."""

    def test_ensemble_with_financial_features(self):
        """Test ensemble methods with financial-like features."""
        np.random.seed(42)
        n_samples = 300

        # Create financial-like dataset
        returns = np.random.normal(0, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.exponential(1000, n_samples)
        volatility = np.abs(returns)

        X = np.column_stack([returns[:-1], volumes[:-1], volatility[:-1]])
        y = (returns[1:] > 0).astype(int)  # Predict next day direction

        # Test FinancialRandomForest
        config = EnsembleConfig(n_estimators=10, random_state=42)
        rf_ensemble = FinancialRandomForest(config=config)
        rf_ensemble.fit(X, y)
        rf_predictions = rf_ensemble.predict(X[:50])

        assert len(rf_predictions) == 50

        # Test VotingEnsemble
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        estimators = [
            ("rf", RandomForestClassifier(n_estimators=5, random_state=42)),
            ("lr", LogisticRegression(random_state=42, max_iter=100)),
        ]

        voting_ensemble = VotingEnsemble(estimators=estimators, config=config)
        voting_ensemble.fit(X, y)
        voting_predictions = voting_ensemble.predict(X[:50])

        assert len(voting_predictions) == 50

    def test_ensemble_performance_comparison(self):
        """Test performance comparison between different ensemble methods."""
        np.random.seed(42)
        n_samples = 400
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.2 > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # FinancialRandomForest
        config = EnsembleConfig(n_estimators=15, random_state=42)
        rf_ensemble = FinancialRandomForest(config=config)
        rf_ensemble.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_test, rf_ensemble.predict(X_test))

        # All methods should achieve reasonable performance
        assert rf_accuracy > 0.5
