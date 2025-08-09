"""
Tests for Meta-labeling approaches and advanced labeling strategies.

This module tests the meta-labeling techniques from Advances in Financial
Machine Learning (AFML) for enhanced prediction accuracy.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from unittest.mock import patch, MagicMock

# Importing modules under test
try:
    from src.models.advanced.meta_labeling import (
        MetaLabeler,
        TripleBarrierLabeling,
        FixedTimeHorizonLabeling,
        VolatilityAdjustedLabeling,
        CumulativeReturnsLabeling,
        AFMLMetaLabeler,
        create_meta_labeler,
        MetaLabelingValidator,
    )

    META_LABELING_AVAILABLE = True
except ImportError:
    META_LABELING_AVAILABLE = False


@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestMetaLabeler:
    """Test base MetaLabeler component."""

    @pytest.fixture
    def sample_primary_model(self):
        """Create sample primary model for meta-labeling."""
        from sklearn.ensemble import RandomForestClassifier

        primary_model = RandomForestClassifier(n_estimators=20, random_state=42)
        return primary_model

    @pytest.fixture
    def financial_time_series_data(self):
        """Create realistic financial time series data."""
        np.random.seed(42)
        n_samples = 1500

        # Create time index
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="1H")

        # Generate realistic price series
        returns = np.random.normal(0, 0.015, n_samples)
        # Add some autocorrelation
        for i in range(1, n_samples):
            returns[i] += 0.1 * returns[i - 1]

        prices = 100 * np.exp(np.cumsum(returns))

        # Create features DataFrame
        features = pd.DataFrame(
            {
                "price": prices,
                "returns": returns,
                "volume": np.random.exponential(1000, n_samples),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
                "volatility": np.random.exponential(0.02, n_samples),
                "rsi": np.random.uniform(20, 80, n_samples),
                "macd": np.random.normal(0, 0.5, n_samples),
                "sma_ratio": 1 + np.random.normal(0, 0.05, n_samples),
                "momentum": np.random.normal(0, 0.1, n_samples),
            },
            index=dates,
        )

        return features

    def test_meta_labeler_creation(self, sample_primary_model):
        """Test MetaLabeler creation."""
        meta_labeler = MetaLabeler(primary_model=sample_primary_model)
        assert meta_labeler is not None
        assert meta_labeler.primary_model is not None

    def test_meta_labeler_primary_predictions(
        self, sample_primary_model, financial_time_series_data
    ):
        """Test primary model predictions in meta-labeling."""
        features = financial_time_series_data

        # Create simple binary target for primary model
        target = (features["returns"] > 0).astype(int)

        meta_labeler = MetaLabeler(primary_model=sample_primary_model)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target.values, test_size=0.3, random_state=42
        )

        # Fit primary model
        meta_labeler.fit_primary(X_train, y_train)

        # Get primary predictions
        primary_preds = meta_labeler.predict_primary(X_test)

        assert len(primary_preds) == len(X_test)
        assert all(pred in [0, 1] for pred in primary_preds)

    def test_meta_labeler_confidence_scores(
        self, sample_primary_model, financial_time_series_data
    ):
        """Test confidence score calculation."""
        features = financial_time_series_data
        target = (features["returns"] > 0).astype(int)

        meta_labeler = MetaLabeler(primary_model=sample_primary_model)

        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target.values, test_size=0.3, random_state=42
        )

        meta_labeler.fit_primary(X_train, y_train)

        # Get confidence scores
        confidence_scores = meta_labeler.get_confidence_scores(X_test)

        assert len(confidence_scores) == len(X_test)
        assert all(0 <= score <= 1 for score in confidence_scores)


@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestTripleBarrierLabeling:
    """Test Triple Barrier Labeling method."""

    def test_triple_barrier_labeling_creation(self):
        """Test TripleBarrierLabeling creation."""
        labeler = TripleBarrierLabeling(
            pt_sl=[0.02, 0.02],  # 2% profit taking and stop loss
            min_ret=0.005,  # 0.5% minimum return
            num_days=5,  # 5-day horizon
        )
        assert labeler is not None

    def test_triple_barrier_labels_generation(self, financial_time_series_data):
        """Test triple barrier labels generation."""
        features = financial_time_series_data

        labeler = TripleBarrierLabeling(
            pt_sl=[0.01, 0.01],  # 1% barriers
            min_ret=0.002,  # 0.2% minimum return
            num_days=3,  # 3-day horizon
        )

        # Generate labels
        labels = labeler.generate_labels(features)

        assert isinstance(labels, pd.Series)
        assert len(labels) <= len(features)  # May be shorter due to horizon
        assert all(label in [-1, 0, 1] for label in labels.dropna())

    def test_triple_barrier_with_volatility_adjustment(
        self, financial_time_series_data
    ):
        """Test triple barrier with volatility-adjusted barriers."""
        features = financial_time_series_data

        labeler = TripleBarrierLabeling(
            pt_sl=[0.01, 0.01], min_ret=0.002, num_days=3, volatility_adjusted=True
        )

        labels = labeler.generate_labels(features)

        assert isinstance(labels, pd.Series)
        assert not labels.empty

    def test_triple_barrier_side_prediction(self, financial_time_series_data):
        """Test triple barrier with side prediction input."""
        features = financial_time_series_data

        # Create side predictions (direction predictions)
        side_predictions = pd.Series(
            np.random.choice([-1, 1], size=len(features)), index=features.index
        )

        labeler = TripleBarrierLabeling(pt_sl=[0.015, 0.015], min_ret=0.003, num_days=4)

        labels = labeler.generate_labels(features, side_predictions=side_predictions)

        assert isinstance(labels, pd.Series)
        assert not labels.empty


@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestFixedTimeHorizonLabeling:
    """Test Fixed Time Horizon Labeling method."""

    def test_fixed_time_horizon_creation(self):
        """Test FixedTimeHorizonLabeling creation."""
        labeler = FixedTimeHorizonLabeling(horizon=5, threshold=0.01)
        assert labeler is not None

    def test_fixed_time_horizon_labels(self, financial_time_series_data):
        """Test fixed time horizon label generation."""
        features = financial_time_series_data

        labeler = FixedTimeHorizonLabeling(horizon=3, threshold=0.005)
        labels = labeler.generate_labels(features)

        assert isinstance(labels, pd.Series)
        assert len(labels) == len(features) - 3  # Horizon adjustment
        assert all(label in [-1, 0, 1] for label in labels)

    def test_fixed_time_horizon_multi_threshold(self, financial_time_series_data):
        """Test fixed time horizon with multiple thresholds."""
        features = financial_time_series_data

        labeler = FixedTimeHorizonLabeling(
            horizon=5, threshold=[0.003, 0.008]  # Different thresholds for buy/sell
        )

        labels = labeler.generate_labels(features)

        assert isinstance(labels, pd.Series)
        assert not labels.empty


@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestVolatilityAdjustedLabeling:
    """Test Volatility Adjusted Labeling method."""

    def test_volatility_adjusted_creation(self):
        """Test VolatilityAdjustedLabeling creation."""
        labeler = VolatilityAdjustedLabeling(
            base_threshold=0.01, volatility_window=20, volatility_multiplier=2.0
        )
        assert labeler is not None

    def test_volatility_adjusted_labels(self, financial_time_series_data):
        """Test volatility-adjusted label generation."""
        features = financial_time_series_data

        labeler = VolatilityAdjustedLabeling(
            base_threshold=0.005, volatility_window=10, volatility_multiplier=1.5
        )

        labels = labeler.generate_labels(features)

        assert isinstance(labels, pd.Series)
        assert not labels.empty

    def test_volatility_adjustment_calculation(self, financial_time_series_data):
        """Test volatility adjustment calculation."""
        features = financial_time_series_data

        labeler = VolatilityAdjustedLabeling(
            base_threshold=0.01, volatility_window=15, volatility_multiplier=2.5
        )

        # Test internal volatility calculation
        volatility = labeler.calculate_volatility(features["returns"])

        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(features)
        assert all(vol >= 0 for vol in volatility.dropna())


@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestCumulativeReturnsLabeling:
    """Test Cumulative Returns Labeling method."""

    def test_cumulative_returns_creation(self):
        """Test CumulativeReturnsLabeling creation."""
        labeler = CumulativeReturnsLabeling(lookback_window=10, threshold=0.02)
        assert labeler is not None

    def test_cumulative_returns_labels(self, financial_time_series_data):
        """Test cumulative returns label generation."""
        features = financial_time_series_data

        labeler = CumulativeReturnsLabeling(lookback_window=5, threshold=0.01)

        labels = labeler.generate_labels(features)

        assert isinstance(labels, pd.Series)
        assert not labels.empty

    def test_cumulative_returns_with_decay(self, financial_time_series_data):
        """Test cumulative returns with exponential decay."""
        features = financial_time_series_data

        labeler = CumulativeReturnsLabeling(
            lookback_window=8, threshold=0.015, decay_factor=0.95
        )

        labels = labeler.generate_labels(features)

        assert isinstance(labels, pd.Series)
        assert not labels.empty


@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestAFMLMetaLabeler:
    """Test AFML-style Meta Labeler implementation."""

    @pytest.fixture
    def primary_model_predictions(self, financial_time_series_data):
        """Create primary model predictions for meta-labeling."""
        features = financial_time_series_data

        # Simulate primary model predictions with some skill
        np.random.seed(42)
        primary_predictions = np.random.choice([0, 1], size=len(features), p=[0.6, 0.4])

        # Add some correlation with actual future returns
        future_returns = features["returns"].shift(-3).fillna(0)
        skill_mask = (future_returns * (primary_predictions * 2 - 1)) > 0

        # Enhance predictions where there's skill
        enhanced_predictions = primary_predictions.copy()
        enhanced_predictions[skill_mask] = (future_returns[skill_mask] > 0).astype(int)

        return pd.Series(enhanced_predictions, index=features.index)

    def test_afml_meta_labeler_creation(self):
        """Test AFMLMetaLabeler creation."""
        meta_labeler = AFMLMetaLabeler(
            labeling_method="triple_barrier", pt_sl=[0.02, 0.02], min_ret=0.005
        )
        assert meta_labeler is not None

    def test_afml_meta_labeling_process(
        self, financial_time_series_data, primary_model_predictions
    ):
        """Test AFML meta-labeling process."""
        features = financial_time_series_data
        primary_preds = primary_model_predictions

        meta_labeler = AFMLMetaLabeler(
            labeling_method="triple_barrier",
            pt_sl=[0.015, 0.015],
            min_ret=0.003,
            num_days=3,
        )

        # Generate meta labels
        meta_labels = meta_labeler.generate_meta_labels(features, primary_preds)

        assert isinstance(meta_labels, pd.DataFrame)
        assert "primary_pred" in meta_labels.columns
        assert "meta_label" in meta_labels.columns

    def test_afml_meta_labeler_with_side_information(self, financial_time_series_data):
        """Test AFML meta-labeler with side information."""
        features = financial_time_series_data

        # Create side information (e.g., structural breaks, regime changes)
        side_info = pd.Series(
            np.random.choice([-1, 1], size=len(features)), index=features.index
        )

        meta_labeler = AFMLMetaLabeler(
            labeling_method="triple_barrier",
            pt_sl=[0.01, 0.01],
            min_ret=0.002,
            use_side_information=True,
        )

        # Simulate primary predictions
        primary_preds = pd.Series(
            np.random.choice([0, 1], size=len(features)), index=features.index
        )

        meta_labels = meta_labeler.generate_meta_labels(
            features, primary_preds, side_information=side_info
        )

        assert isinstance(meta_labels, pd.DataFrame)
        assert not meta_labels.empty

    def test_afml_sample_weights_calculation(
        self, financial_time_series_data, primary_model_predictions
    ):
        """Test AFML sample weights calculation for meta-labeling."""
        features = financial_time_series_data
        primary_preds = primary_model_predictions

        meta_labeler = AFMLMetaLabeler(
            labeling_method="triple_barrier",
            pt_sl=[0.02, 0.02],
            calculate_sample_weights=True,
        )

        meta_labels = meta_labeler.generate_meta_labels(features, primary_preds)

        if "sample_weight" in meta_labels.columns:
            weights = meta_labels["sample_weight"]
            assert all(w > 0 for w in weights.dropna())
            assert not weights.isna().all()


@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestMetaLabelingValidator:
    """Test Meta-labeling validation and performance assessment."""

    def test_meta_labeling_validator_creation(self):
        """Test MetaLabelingValidator creation."""
        validator = MetaLabelingValidator()
        assert validator is not None

    def test_primary_vs_meta_comparison(self, financial_time_series_data):
        """Test comparison between primary and meta-labeling performance."""
        features = financial_time_series_data

        # Create ground truth labels
        future_returns = features["returns"].shift(-3).fillna(0)
        true_labels = (future_returns > 0.005).astype(int)

        # Create primary model predictions
        primary_preds = np.random.choice([0, 1], size=len(features))

        # Create meta-labeling approach
        meta_labeler = AFMLMetaLabeler(
            labeling_method="fixed_horizon", horizon=3, threshold=0.005
        )

        validator = MetaLabelingValidator()

        # Compare performances
        comparison = validator.compare_primary_vs_meta(
            features, primary_preds, true_labels, meta_labeler
        )

        assert isinstance(comparison, dict)
        assert "primary_accuracy" in comparison
        assert "meta_accuracy" in comparison

    def test_meta_labeling_cross_validation(self, financial_time_series_data):
        """Test cross-validation for meta-labeling."""
        features = financial_time_series_data

        # Create labels using triple barrier
        labeler = TripleBarrierLabeling(pt_sl=[0.015, 0.015], min_ret=0.003, num_days=3)
        labels_data = labeler.generate_labels(features)

        # Align features and labels
        common_index = features.index.intersection(labels_data.index)
        X = features.loc[common_index].values
        y = labels_data.loc[common_index].values

        # Create primary predictions
        primary_preds = np.random.choice([0, 1], size=len(X))

        validator = MetaLabelingValidator()

        # Perform cross-validation
        cv_results = validator.cross_validate_meta_labeling(X, y, primary_preds, cv=3)

        assert isinstance(cv_results, dict)
        assert "accuracy_scores" in cv_results


# Integration tests
@pytest.mark.skipif(
    not META_LABELING_AVAILABLE, reason="Meta-labeling module not available"
)
class TestMetaLabelingIntegration:
    """Integration tests for Meta-labeling components."""

    @pytest.fixture
    def comprehensive_financial_dataset(self):
        """Create comprehensive financial dataset for integration testing."""
        np.random.seed(42)
        n_samples = 2000

        # Create realistic time series
        dates = pd.date_range("2019-01-01", periods=n_samples, freq="1H")

        # Multiple asset returns with correlation
        n_assets = 3
        correlation_matrix = np.array(
            [[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]
        )

        returns = np.random.multivariate_normal(
            [0, 0, 0], correlation_matrix * 0.02, n_samples
        )

        # Add regime changes
        regime_changes = [500, 1000, 1500]
        for change_point in regime_changes:
            returns[change_point:] *= 1.5

        # Create features DataFrame
        features = pd.DataFrame(
            {
                "asset1_return": returns[:, 0],
                "asset2_return": returns[:, 1],
                "asset3_return": returns[:, 2],
                "asset1_price": 100 * np.exp(np.cumsum(returns[:, 0])),
                "asset2_price": 100 * np.exp(np.cumsum(returns[:, 1])),
                "asset3_price": 100 * np.exp(np.cumsum(returns[:, 2])),
                "market_volatility": np.random.exponential(0.02, n_samples),
                "volume_asset1": np.random.exponential(1000, n_samples),
                "sentiment": np.random.uniform(-1, 1, n_samples),
                "economic_indicator": np.random.normal(0, 1, n_samples),
            },
            index=dates,
        )

        return features

    def test_complete_meta_labeling_pipeline(self, comprehensive_financial_dataset):
        """Test complete meta-labeling pipeline."""
        features = comprehensive_financial_dataset

        # Step 1: Create primary model
        from sklearn.ensemble import RandomForestClassifier

        primary_model = RandomForestClassifier(n_estimators=50, random_state=42)

        # Step 2: Generate primary predictions
        # Create simple target for primary model training
        primary_target = (features["asset1_return"] > 0).astype(int)

        # Split data for primary model
        split_point = int(0.7 * len(features))
        primary_train_X = features.iloc[:split_point].values
        primary_train_y = primary_target.iloc[:split_point].values
        primary_test_X = features.iloc[split_point:].values

        # Train primary model
        primary_model.fit(primary_train_X, primary_train_y)
        primary_predictions = primary_model.predict(primary_test_X)

        # Step 3: Apply meta-labeling
        test_features = features.iloc[split_point:]
        test_primary_preds = pd.Series(primary_predictions, index=test_features.index)

        meta_labeler = AFMLMetaLabeler(
            labeling_method="triple_barrier",
            pt_sl=[0.02, 0.02],
            min_ret=0.005,
            num_days=5,
        )

        meta_labels = meta_labeler.generate_meta_labels(
            test_features, test_primary_preds
        )

        assert isinstance(meta_labels, pd.DataFrame)
        assert len(meta_labels) > 0
        assert "primary_pred" in meta_labels.columns
        assert "meta_label" in meta_labels.columns

    def test_meta_labeling_with_multiple_methods(self, comprehensive_financial_dataset):
        """Test meta-labeling with multiple labeling methods."""
        features = comprehensive_financial_dataset

        # Create primary predictions
        primary_preds = pd.Series(
            np.random.choice([0, 1], size=len(features)), index=features.index
        )

        # Test multiple labeling methods
        methods = [
            (
                "triple_barrier",
                {"pt_sl": [0.015, 0.015], "min_ret": 0.003, "num_days": 3},
            ),
            ("fixed_horizon", {"horizon": 5, "threshold": 0.01}),
            ("volatility_adjusted", {"base_threshold": 0.01, "volatility_window": 20}),
        ]

        results = {}

        for method_name, params in methods:
            meta_labeler = create_meta_labeler(method=method_name, **params)

            if hasattr(meta_labeler, "generate_meta_labels"):
                meta_labels = meta_labeler.generate_meta_labels(features, primary_preds)
            else:
                # For basic labelers
                meta_labels = meta_labeler.generate_labels(features)

            results[method_name] = meta_labels

        # All methods should produce results
        assert len(results) == len(methods)
        for method_name, labels in results.items():
            assert labels is not None
            assert len(labels) > 0

    def test_meta_labeling_performance_assessment(
        self, comprehensive_financial_dataset
    ):
        """Test performance assessment of meta-labeling strategies."""
        features = comprehensive_financial_dataset

        # Create realistic scenario
        future_returns = features["asset1_return"].shift(-5).fillna(0)
        true_direction = (future_returns > 0.01).astype(int)

        # Primary model with some skill
        primary_skill_mask = np.random.random(len(features)) < 0.6
        primary_preds = np.where(
            primary_skill_mask,
            true_direction,  # Correct prediction
            1 - true_direction,  # Wrong prediction
        )
        primary_preds = pd.Series(primary_preds, index=features.index)

        # Apply meta-labeling
        meta_labeler = AFMLMetaLabeler(
            labeling_method="triple_barrier", pt_sl=[0.02, 0.02], min_ret=0.005
        )

        meta_labels = meta_labeler.generate_meta_labels(features, primary_preds)

        # Evaluate improvement
        validator = MetaLabelingValidator()

        # Extract common indices for evaluation
        common_index = features.index.intersection(meta_labels.index).intersection(
            true_direction.index
        )

        if len(common_index) > 100:  # Sufficient data for evaluation
            evaluation = validator.evaluate_meta_labeling_improvement(
                primary_preds.loc[common_index],
                meta_labels.loc[common_index],
                true_direction.loc[common_index],
            )

            assert isinstance(evaluation, dict)
            assert (
                "improvement_metrics" in evaluation or "baseline_accuracy" in evaluation
            )


if __name__ == "__main__":
    pytest.main([__file__])
