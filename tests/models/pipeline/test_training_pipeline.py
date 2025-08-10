"""
Test Model Training Pipeline

Comprehensive tests for the automated model training pipeline including
data preparation, model training, evaluation, and ensemble creation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch
from pathlib import Path

from src.models.pipeline.training_pipeline import (
    ModelTrainingPipeline,
    ModelTrainingConfig,
)


class TestModelTrainingConfig:
    """Test ModelTrainingConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ModelTrainingConfig()

        assert config.test_size == 0.2
        assert config.validation_size == 0.2
        assert config.time_series_cv is True
        assert config.cv_splits == 5
        assert config.scaler_type == "standard"
        assert config.save_models is True
        assert config.random_state == 42

        # Check default models
        expected_models = [
            "random_forest",
            "xgboost",
            "svm",
            "lstm",
            "gru",
            "transformer",
        ]
        assert config.models_to_train == expected_models

    def test_custom_config(self):
        """Test custom configuration values"""
        custom_models = ["random_forest", "xgboost"]

        config = ModelTrainingConfig(
            models_to_train=custom_models,
            test_size=0.3,
            scaler_type="minmax",
            save_models=False,
        )

        assert config.models_to_train == custom_models
        assert config.test_size == 0.3
        assert config.scaler_type == "minmax"
        assert config.save_models is False


class TestModelTrainingPipeline:
    """Test ModelTrainingPipeline class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000

        # Create date range
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

        # Create sample financial data
        data = pd.DataFrame(
            {
                "price": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
                "volume": np.random.randint(1000, 10000, n_samples),
                "high": 100
                + np.cumsum(np.random.randn(n_samples) * 0.1)
                + np.random.rand(n_samples),
                "low": 100
                + np.cumsum(np.random.randn(n_samples) * 0.1)
                - np.random.rand(n_samples),
            },
            index=dates,
        )

        # Calculate features
        data["returns"] = data["price"].pct_change()
        data["volatility"] = data["returns"].rolling(20).std()
        data["sma_20"] = data["price"].rolling(20).mean()
        data["sma_50"] = data["price"].rolling(50).mean()
        data["rsi"] = self._calculate_rsi(data["price"], 14)

        # Create target variable (classification: up/down trend)
        data["target"] = (data["returns"].shift(-1) > 0).astype(int)

        # Drop NaN values
        data = data.dropna()

        return data

    def _calculate_rsi(self, prices, window):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_rsi(self, prices, window=14):
        """Helper to calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for model saving"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create ModelTrainingPipeline instance"""
        config = ModelTrainingConfig(
            models_to_train=["random_forest"],  # Single model for faster testing
            save_models=True,
            model_save_dir=temp_dir,
        )
        return ModelTrainingPipeline(config)

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.config is not None
        assert pipeline.scaler is not None
        assert isinstance(pipeline.trained_models, dict)
        assert isinstance(pipeline.model_performance, dict)
        assert isinstance(pipeline.ensemble_models, dict)

    def test_get_scaler(self, temp_dir):
        """Test scaler selection"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        # Test standard scaler
        config = ModelTrainingConfig(scaler_type="standard", model_save_dir=temp_dir)
        pipeline = ModelTrainingPipeline(config)
        assert isinstance(pipeline.scaler, StandardScaler)

        # Test minmax scaler
        config = ModelTrainingConfig(scaler_type="minmax", model_save_dir=temp_dir)
        pipeline = ModelTrainingPipeline(config)
        assert isinstance(pipeline.scaler, MinMaxScaler)

        # Test robust scaler
        config = ModelTrainingConfig(scaler_type="robust", model_save_dir=temp_dir)
        pipeline = ModelTrainingPipeline(config)
        assert isinstance(pipeline.scaler, RobustScaler)

        # Test invalid scaler (should default to standard)
        config = ModelTrainingConfig(scaler_type="invalid", model_save_dir=temp_dir)
        pipeline = ModelTrainingPipeline(config)
        assert isinstance(pipeline.scaler, StandardScaler)

    def test_get_model_instance(self, pipeline):
        """Test model instance creation"""
        # Test valid models
        rf_classifier = pipeline._get_model_instance("random_forest", "classification")
        assert rf_classifier is not None

        rf_regressor = pipeline._get_model_instance("random_forest", "regression")
        assert rf_regressor is not None

        # Test invalid task type
        with pytest.raises(ValueError, match="Unknown task type"):
            pipeline._get_model_instance("random_forest", "invalid_task")

        # Test invalid model name
        with pytest.raises(ValueError, match="Unknown model"):
            pipeline._get_model_instance("invalid_model", "classification")

    def test_prepare_data(self, pipeline, sample_data):
        """Test data preparation"""
        feature_columns = ["returns", "volatility", "sma_20", "sma_50", "rsi"]

        features, target, feature_names = pipeline.prepare_data(
            sample_data, "target", feature_columns
        )

        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert isinstance(feature_names, list)
        assert len(features) == len(target)
        assert len(feature_names) == len(feature_columns)
        assert not features.isnull().any().any()  # No missing values

    def test_prepare_data_auto_features(self, pipeline, sample_data):
        """Test data preparation with auto feature generation"""
        with patch.object(
            pipeline.feature_pipeline, "generate_features"
        ) as mock_generate:
            # Mock feature generation
            mock_features = sample_data[["returns", "volatility", "sma_20"]].copy()
            mock_generate.return_value = mock_features

            features, target, feature_names = pipeline.prepare_data(
                sample_data, "target"
            )

            assert mock_generate.called
            assert len(feature_names) == 3

    def test_split_data(self, pipeline, sample_data):
        """Test data splitting"""
        features = sample_data[["returns", "volatility", "sma_20", "sma_50", "rsi"]]
        target = sample_data["target"]

        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(
            features, target
        )

        # Check that splits sum to original length
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == len(features)

        # Check time series order is preserved
        assert X_train.index.max() < X_val.index.min()
        assert X_val.index.max() < X_test.index.min()

        # Check proportions are approximately correct
        test_ratio = len(X_test) / len(features)
        assert abs(test_ratio - pipeline.config.test_size) < 0.05

    def test_scale_features(self, pipeline, sample_data):
        """Test feature scaling"""
        features = sample_data[["returns", "volatility", "sma_20", "sma_50", "rsi"]]
        target = sample_data["target"]

        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(
            features, target
        )
        X_train_scaled, X_val_scaled, X_test_scaled = pipeline.scale_features(
            X_train, X_val, X_test
        )

        # Check that scaled data maintains shape
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape

        # Check that training data is approximately normalized (mean ~0, std ~1 for StandardScaler)
        if isinstance(pipeline.scaler, type(pipeline._get_scaler())):
            assert abs(X_train_scaled.mean().mean()) < 0.1
            assert abs(X_train_scaled.std().mean() - 1.0) < 0.1

    @patch("src.models.pipeline.training_pipeline.joblib.dump")
    def test_train_single_model(self, mock_dump, pipeline, sample_data):
        """Test single model training"""
        features = sample_data[["returns", "volatility", "sma_20", "sma_50", "rsi"]]
        target = sample_data["target"]

        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(
            features, target
        )
        X_train_scaled, X_val_scaled, X_test_scaled = pipeline.scale_features(
            X_train, X_val, X_test
        )

        result = pipeline.train_single_model(
            "random_forest",
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train,
            y_val,
            y_test,
            "classification",
        )

        assert "model" in result
        assert "performance" in result
        assert "predictions" in result
        assert result["model"] is not None

        # Check performance metrics for classification
        perf = result["performance"]
        assert "train_accuracy" in perf
        assert "val_accuracy" in perf
        assert "test_accuracy" in perf
        assert "test_precision" in perf
        assert "test_recall" in perf
        assert "test_f1" in perf

        # Check predictions
        pred = result["predictions"]
        assert "train" in pred
        assert "val" in pred
        assert "test" in pred

    def test_train_single_model_regression(self, pipeline, sample_data):
        """Test single model training for regression"""
        # Create continuous target
        sample_data["target_continuous"] = sample_data["returns"].shift(-1)
        sample_data = sample_data.dropna()

        features = sample_data[["returns", "volatility", "sma_20", "sma_50", "rsi"]]
        target = sample_data["target_continuous"]

        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(
            features, target
        )
        X_train_scaled, X_val_scaled, X_test_scaled = pipeline.scale_features(
            X_train, X_val, X_test
        )

        result = pipeline.train_single_model(
            "random_forest",
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train,
            y_val,
            y_test,
            "regression",
        )

        # Check performance metrics for regression
        perf = result["performance"]
        assert "train_rmse" in perf
        assert "val_rmse" in perf
        assert "test_rmse" in perf
        assert "test_r2" in perf

    def test_train_single_model_error_handling(self, pipeline, sample_data):
        """Test error handling in single model training"""
        features = sample_data[["returns", "volatility", "sma_20", "sma_50", "rsi"]]
        target = sample_data["target"]

        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(
            features, target
        )

        # Test with invalid model name
        with patch.object(
            pipeline, "_get_model_instance", side_effect=ValueError("Test error")
        ):
            result = pipeline.train_single_model(
                "invalid_model",
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                "classification",
            )

            assert result["model"] is None
            assert "error" in result["performance"]
            assert result["predictions"] is None

    def test_get_model_comparison(self, pipeline):
        """Test model comparison functionality"""
        # Mock some performance data
        pipeline.model_performance = {
            "random_forest": {"test_accuracy": 0.85, "test_f1": 0.80},
            "xgboost": {"test_accuracy": 0.87, "test_f1": 0.82},
        }

        comparison = pipeline.get_model_comparison()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "model" in comparison.columns
        assert "test_accuracy" in comparison.columns
        assert "test_f1" in comparison.columns

    def test_get_best_model(self, pipeline):
        """Test best model selection"""
        # Mock some models and performance data
        mock_model_rf = Mock()
        mock_model_xgb = Mock()

        pipeline.trained_models = {
            "random_forest": mock_model_rf,
            "xgboost": mock_model_xgb,
        }
        pipeline.model_performance = {
            "random_forest": {"test_accuracy": 0.85, "test_rmse": 0.3},
            "xgboost": {"test_accuracy": 0.87, "test_rmse": 0.25},
        }

        # Test accuracy-based selection (higher is better)
        best_name, best_model = pipeline.get_best_model("test_accuracy")
        assert best_name == "xgboost"
        assert best_model == mock_model_xgb

        # Test RMSE-based selection (lower is better)
        best_name, best_model = pipeline.get_best_model("test_rmse")
        assert best_name == "xgboost"
        assert best_model == mock_model_xgb

        # Test with no models
        pipeline.model_performance = {}
        best_name, best_model = pipeline.get_best_model("test_accuracy")
        assert best_name is None
        assert best_model is None

    def test_save_model(self, pipeline, temp_dir):
        """Test model saving functionality"""
        mock_model = Mock()
        mock_model.save = Mock()

        # Test saving model with custom save method
        with patch("os.path.exists", return_value=True):
            filepath = pipeline._save_model(mock_model, "test_model", "classification")
            assert filepath.endswith(".pkl")
            assert "test_model" in filepath
            assert "classification" in filepath

    @patch("src.models.pipeline.training_pipeline.joblib.dump")
    def test_train_all_models_integration(self, mock_dump, pipeline, sample_data):
        """Test complete training pipeline integration"""
        # Use only random forest for faster testing
        pipeline.config.models_to_train = ["random_forest"]
        pipeline.config.ensemble_models = False

        results = pipeline.train_all_models(
            sample_data,
            "target",
            "classification",
            ["returns", "volatility", "sma_20", "sma_50", "rsi"],
        )

        assert "random_forest" in results
        assert "model" in results["random_forest"]
        assert "performance" in results["random_forest"]
        assert results["random_forest"]["model"] is not None

        # Check that model was added to pipeline state
        assert "random_forest" in pipeline.trained_models
        assert "random_forest" in pipeline.model_performance


if __name__ == "__main__":
    pytest.main([__file__])
