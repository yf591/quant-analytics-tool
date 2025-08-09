"""
Working tests for Meta-labeling module.
These tests are designed to work with the current API implementation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.models.advanced.meta_labeling import (
    MetaLabelingConfig,
    TripleBarrierLabeling,
    MetaLabelingModel,
)


class TestMetaLabelingConfigWorking:
    """Test MetaLabelingConfig functionality."""

    def test_meta_labeling_config_creation(self):
        """Test MetaLabelingConfig creation with defaults."""
        config = MetaLabelingConfig()

        assert config.profit_target == 0.02
        assert config.stop_loss == 0.01
        assert config.max_holding_period == 5
        assert config.meta_model_type == "random_forest"
        assert config.n_estimators == 100

    def test_meta_labeling_config_custom_values(self):
        """Test MetaLabelingConfig with custom values."""
        config = MetaLabelingConfig(
            profit_target=0.05,
            stop_loss=0.02,
            max_holding_period=15,
            meta_model_type="logistic",
            n_estimators=50,
        )

        assert config.profit_target == 0.05
        assert config.stop_loss == 0.02
        assert config.max_holding_period == 15
        assert config.meta_model_type == "logistic"
        assert config.n_estimators == 50


class TestTripleBarrierLabelingWorking:
    """Test TripleBarrierLabeling functionality."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    def test_triple_barrier_labeling_creation(self):
        """Test TripleBarrierLabeling creation."""
        config = MetaLabelingConfig()
        labeler = TripleBarrierLabeling(config=config)

        assert labeler.config == config
        assert hasattr(labeler, "apply_triple_barrier")

    def test_triple_barrier_labeling_apply_working(self, sample_price_data):
        """Test applying triple barrier labeling with corrected API."""
        config = MetaLabelingConfig(
            profit_target=0.02, stop_loss=0.01, max_holding_period=5
        )
        labeler = TripleBarrierLabeling(config=config)

        # Create proper events data (as DatetimeIndex, not Series)
        events = sample_price_data.index[:20]  # DatetimeIndex directly

        labels = labeler.apply_triple_barrier(prices=sample_price_data, events=events)

        assert isinstance(labels, pd.DataFrame)
        assert len(labels) > 0


class TestMetaLabelingModelWorking:
    """Test MetaLabelingModel functionality with working API."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for meta-labeling."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate price data
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
        returns = np.random.normal(0.001, 0.02, n_samples)
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

        # Generate events (subset of dates)
        events = pd.Series(index=dates[: n_samples // 2])

        return X, prices, events

    def test_meta_labeling_model_creation(self):
        """Test MetaLabelingModel creation."""
        from sklearn.ensemble import RandomForestClassifier

        primary_model = RandomForestClassifier(n_estimators=10, random_state=42)
        config = MetaLabelingConfig()
        model = MetaLabelingModel(primary_model=primary_model, config=config)

        assert model.primary_model == primary_model
        assert model.config == config
        assert model.meta_model is None  # Not fitted yet

    def test_meta_labeling_model_fit_working(self, sample_data):
        """Test MetaLabelingModel fit with correct API."""
        X, prices, events = sample_data

        from sklearn.ensemble import RandomForestClassifier

        primary_model = RandomForestClassifier(n_estimators=5, random_state=42)
        config = MetaLabelingConfig(meta_model_type="random_forest", n_estimators=10)
        model = MetaLabelingModel(primary_model=primary_model, config=config)

        # Fit model with correct API (X, prices, events)
        model.fit(X, prices, events)

        # Verify model was fitted
        assert model.meta_model is not None
        assert hasattr(model.meta_model, "predict")


def test_basic_integration_working():
    """Test basic meta-labeling integration that should work."""
    # 1. Create simple test data
    np.random.seed(42)
    n_periods = 50
    dates = pd.date_range(start="2023-01-01", periods=n_periods, freq="D")
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    # 2. Create features
    X = np.random.randn(n_periods, 3)

    # 3. Create events (subset)
    events = prices.index[:20]  # Use DatetimeIndex directly

    # 4. Test triple barrier labeling
    config = MetaLabelingConfig(
        profit_target=0.015, stop_loss=0.01, max_holding_period=5
    )
    labeler = TripleBarrierLabeling(config=config)

    labels = labeler.apply_triple_barrier(prices=prices, events=events)
    assert isinstance(labels, pd.DataFrame)

    # 5. Test meta-labeling model
    from sklearn.ensemble import RandomForestClassifier

    primary_model = RandomForestClassifier(n_estimators=5, random_state=42)
    meta_model = MetaLabelingModel(primary_model=primary_model, config=config)

    # Create proper events Series for model
    events_series = pd.Series(index=events)
    meta_model.fit(X[: len(events)], prices, events_series)

    assert meta_model.meta_model is not None
    print("✅ Basic meta-labeling integration working!")


if __name__ == "__main__":
    test_basic_integration_working()
