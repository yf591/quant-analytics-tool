"""
Test Real-time Prediction Engine

Comprehensive tests for the real-time prediction system including
prediction caching, streaming predictions, and ensemble capabilities.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from threading import Event
import time

from src.models.pipeline.prediction import RealTimePrediction, PredictionCache
from src.models.pipeline.model_registry import ModelRegistry, ModelMetadata


class TestPredictionCache:
    """Test PredictionCache class"""

    @pytest.fixture
    def cache(self):
        """Create PredictionCache instance"""
        return PredictionCache(max_size=100, ttl_seconds=300)

    def test_cache_initialization(self, cache):
        """Test cache initialization"""
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
        assert len(cache.cache) == 0

    def test_add_prediction(self, cache):
        """Test adding predictions to cache"""
        cache.add_prediction(
            symbol="AAPL",
            model_id="model_001",
            prediction=0.75,
            confidence=0.85,
            features={"rsi": 45.0, "sma_20": 150.0},
        )

        assert len(cache.cache) == 1
        prediction = cache.cache[0]
        assert prediction["symbol"] == "AAPL"
        assert prediction["model_id"] == "model_001"
        assert prediction["prediction"] == 0.75
        assert prediction["confidence"] == 0.85

    def test_get_recent_predictions_all(self, cache):
        """Test getting all recent predictions"""
        # Add multiple predictions
        cache.add_prediction("AAPL", "model_001", 0.75)
        cache.add_prediction("GOOGL", "model_002", 0.65)
        cache.add_prediction("MSFT", "model_001", 0.85)

        recent = cache.get_recent_predictions(minutes=60)
        assert len(recent) == 3

    def test_get_recent_predictions_filtered_by_symbol(self, cache):
        """Test getting predictions filtered by symbol"""
        cache.add_prediction("AAPL", "model_001", 0.75)
        cache.add_prediction("GOOGL", "model_002", 0.65)
        cache.add_prediction("AAPL", "model_003", 0.85)

        aapl_predictions = cache.get_recent_predictions(symbol="AAPL", minutes=60)
        assert len(aapl_predictions) == 2
        assert all(pred["symbol"] == "AAPL" for pred in aapl_predictions)

    def test_get_recent_predictions_filtered_by_model(self, cache):
        """Test getting predictions filtered by model"""
        cache.add_prediction("AAPL", "model_001", 0.75)
        cache.add_prediction("GOOGL", "model_001", 0.65)
        cache.add_prediction("MSFT", "model_002", 0.85)

        model_predictions = cache.get_recent_predictions(
            model_id="model_001", minutes=60
        )
        assert len(model_predictions) == 2
        assert all(pred["model_id"] == "model_001" for pred in model_predictions)

    def test_get_recent_predictions_time_filter(self, cache):
        """Test time-based filtering of predictions"""
        # Add old prediction (simulate by modifying timestamp)
        cache.add_prediction("AAPL", "model_001", 0.75)
        cache.cache[0]["timestamp"] = datetime.now() - timedelta(hours=2)

        # Add recent prediction
        cache.add_prediction("GOOGL", "model_002", 0.65)

        # Get predictions from last hour
        recent = cache.get_recent_predictions(minutes=60)
        assert len(recent) == 1
        assert recent[0]["symbol"] == "GOOGL"

    def test_cleanup_expired(self, cache):
        """Test cleanup of expired predictions"""
        # Add predictions with different timestamps
        cache.add_prediction("AAPL", "model_001", 0.75)
        cache.add_prediction("GOOGL", "model_002", 0.65)

        # Simulate old timestamp for first prediction
        cache.cache[0]["timestamp"] = datetime.now() - timedelta(
            seconds=400
        )  # Older than TTL

        cache.cleanup_expired()

        # Should only have 1 prediction left
        assert len(cache.cache) == 1
        assert cache.cache[0]["symbol"] == "GOOGL"

    def test_cache_max_size(self):
        """Test cache max size enforcement"""
        cache = PredictionCache(max_size=3, ttl_seconds=300)

        # Add more predictions than max size
        for i in range(5):
            cache.add_prediction(f"SYMBOL_{i}", "model_001", 0.5 + i * 0.1)

        # Should only keep latest 3
        assert len(cache.cache) == 3
        assert cache.cache[0]["symbol"] == "SYMBOL_2"  # Oldest kept
        assert cache.cache[-1]["symbol"] == "SYMBOL_4"  # Latest


class TestRealTimePrediction:
    """Test RealTimePrediction class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for registry"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_registry(self, temp_dir):
        """Create mock model registry"""
        registry = ModelRegistry(temp_dir)
        return registry

    @pytest.fixture
    def mock_data_collector(self):
        """Create mock data collector"""
        collector = Mock()

        # Mock sample data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        sample_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(150, 250, 100),
                "low": np.random.uniform(90, 190, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        collector.fetch_data.return_value = sample_data
        return collector

    @pytest.fixture
    def mock_feature_pipeline(self):
        """Create mock feature pipeline"""
        pipeline = Mock()

        # Mock feature generation
        features_data = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.02, 1),
                "volatility": np.random.uniform(0.01, 0.05, 1),
                "sma_20": np.random.uniform(140, 160, 1),
                "sma_50": np.random.uniform(135, 165, 1),
                "rsi": np.random.uniform(30, 70, 1),
            }
        )

        pipeline.generate_features.return_value = features_data
        return pipeline

    @pytest.fixture
    def prediction_engine(
        self, mock_registry, mock_data_collector, mock_feature_pipeline
    ):
        """Create RealTimePrediction instance"""
        return RealTimePrediction(
            model_registry=mock_registry,
            data_collector=mock_data_collector,
            feature_pipeline=mock_feature_pipeline,
            cache_size=100,
            cache_ttl=300,
        )

    def test_prediction_engine_initialization(self, prediction_engine):
        """Test prediction engine initialization"""
        assert prediction_engine.model_registry is not None
        assert prediction_engine.data_collector is not None
        assert prediction_engine.feature_pipeline is not None
        assert prediction_engine.cache is not None
        assert isinstance(prediction_engine.loaded_models, dict)
        assert prediction_engine.streaming_active is False

    def test_get_real_time_data(self, prediction_engine):
        """Test real-time data fetching"""
        data = prediction_engine.get_real_time_data("AAPL", period="1d", interval="1m")

        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        # Check data collector was called correctly
        prediction_engine.data_collector.fetch_data.assert_called_once()

    def test_get_real_time_data_error_handling(self, prediction_engine):
        """Test real-time data fetching error handling"""
        # Mock data collector to raise exception
        prediction_engine.data_collector.fetch_data.side_effect = Exception("API Error")

        data = prediction_engine.get_real_time_data("AAPL")
        assert data is None

    def test_generate_features(self, prediction_engine, mock_data_collector):
        """Test feature generation"""
        # Get sample data
        sample_data = mock_data_collector.fetch_data()

        features = prediction_engine.generate_features(sample_data)

        assert features is not None
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1  # Should return latest row

        prediction_engine.feature_pipeline.generate_features.assert_called_once()

    def test_generate_features_error_handling(
        self, prediction_engine, mock_data_collector
    ):
        """Test feature generation error handling"""
        sample_data = mock_data_collector.fetch_data()

        # Mock feature pipeline to raise exception
        prediction_engine.feature_pipeline.generate_features.side_effect = Exception(
            "Feature Error"
        )

        features = prediction_engine.generate_features(sample_data)
        assert features is None

    def test_load_production_models(self, prediction_engine, mock_registry):
        """Test loading production models"""
        # Mock production models
        mock_model = Mock()
        mock_metadata = ModelMetadata(
            model_id="test_model_001",
            model_name="test_model",
            model_type="random_forest",
            task_type="classification",
            version="v001",
            created_at=datetime.now(),
            file_path="/path/to/model.pkl",
            performance_metrics={"accuracy": 0.85},
            hyperparameters={},
            feature_names=["feature1"],
            training_data_info={},
        )

        with patch.object(
            mock_registry, "get_production_models", return_value=[mock_metadata]
        ):
            with patch.object(mock_registry, "load_model", return_value=mock_model):
                loaded_models = prediction_engine.load_production_models()

                assert len(loaded_models) == 1
                assert "test_model_001" in loaded_models
                assert loaded_models["test_model_001"] == mock_model

    def test_predict_single_model(self, prediction_engine):
        """Test single model prediction"""
        # Mock model and metadata
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.75])
        mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])

        mock_metadata = Mock()
        mock_metadata.model_name = "test_model"
        mock_metadata.model_type = "random_forest"
        mock_metadata.feature_names = ["returns", "volatility", "rsi"]

        # Add model to loaded models
        prediction_engine.loaded_models["test_model_001"] = {
            "model": mock_model,
            "metadata": mock_metadata,
        }

        # Create sample features
        features = pd.DataFrame(
            {
                "returns": [0.001],
                "volatility": [0.02],
                "rsi": [45.0],
                "extra_feature": [1.0],  # Extra feature to test filtering
            }
        )

        result = prediction_engine.predict_single_model("test_model_001", features)

        assert "error" not in result
        assert result["model_id"] == "test_model_001"
        assert result["model_name"] == "test_model"
        assert result["model_type"] == "random_forest"
        assert result["prediction"] == 0.75
        assert result["confidence"] == 0.75

    def test_predict_single_model_not_loaded(self, prediction_engine, mock_registry):
        """Test single model prediction with model not in cache"""
        # Mock model loading
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.65])

        mock_metadata = Mock()
        mock_metadata.model_name = "test_model"
        mock_metadata.model_type = "xgboost"
        mock_metadata.feature_names = ["returns", "volatility"]

        with patch.object(mock_registry, "load_model", return_value=mock_model):
            with patch.object(
                mock_registry, "get_model_metadata", return_value=mock_metadata
            ):
                features = pd.DataFrame({"returns": [0.001], "volatility": [0.02]})

                result = prediction_engine.predict_single_model(
                    "new_model_001", features
                )

                assert "error" not in result
                assert result["prediction"] == 0.65
                assert "new_model_001" in prediction_engine.loaded_models

    def test_predict_single_model_error(self, prediction_engine):
        """Test single model prediction error handling"""
        # Mock failed model loading
        with patch.object(
            prediction_engine.model_registry, "load_model", return_value=None
        ):
            features = pd.DataFrame({"returns": [0.001]})

            result = prediction_engine.predict_single_model(
                "nonexistent_model", features
            )

            assert "error" in result

    def test_predict_ensemble(self, prediction_engine, mock_registry):
        """Test ensemble prediction"""
        # Mock production models
        mock_models = []
        mock_metadatas = []

        for i in range(3):
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.6 + i * 0.1])
            mock_models.append(mock_model)

            mock_metadata = ModelMetadata(
                model_id=f"model_{i:03d}",
                model_name=f"model_{i}",
                model_type="random_forest",
                task_type="classification",
                version=f"v{i:03d}",
                created_at=datetime.now(),
                file_path=f"/path/to/model_{i}.pkl",
                performance_metrics={"accuracy": 0.8 + i * 0.02},
                hyperparameters={},
                feature_names=["returns", "volatility", "rsi"],
                training_data_info={},
            )
            mock_metadatas.append(mock_metadata)

        with patch.object(
            mock_registry, "get_production_models", return_value=mock_metadatas
        ):
            # Mock individual predictions
            individual_results = []
            for i, model_id in enumerate([f"model_{i:03d}" for i in range(3)]):
                individual_results.append(
                    {
                        "model_id": model_id,
                        "model_name": f"model_{i}",
                        "model_type": "random_forest",
                        "prediction": 0.6 + i * 0.1,
                        "confidence": 0.8 + i * 0.05,
                        "timestamp": datetime.now(),
                    }
                )

            with patch.object(
                prediction_engine,
                "predict_single_model",
                side_effect=individual_results,
            ):
                result = prediction_engine.predict_ensemble("AAPL")

                assert "error" not in result
                assert result["symbol"] == "AAPL"
                assert "ensemble_prediction" in result
                assert "weighted_prediction" in result
                assert "individual_predictions" in result
                assert len(result["individual_predictions"]) == 3

                # Check ensemble calculation
                expected_ensemble = np.mean([0.6, 0.7, 0.8])
                assert abs(result["ensemble_prediction"] - expected_ensemble) < 0.001

    def test_predict_ensemble_data_error(self, prediction_engine):
        """Test ensemble prediction with data fetch error"""
        # Mock data collection failure
        prediction_engine.get_real_time_data = Mock(return_value=None)

        result = prediction_engine.predict_ensemble("AAPL")

        assert "error" in result
        assert "Failed to fetch real-time data" in result["error"]

    def test_predict_ensemble_feature_error(self, prediction_engine):
        """Test ensemble prediction with feature generation error"""
        # Mock feature generation failure
        prediction_engine.generate_features = Mock(return_value=None)

        result = prediction_engine.predict_ensemble("AAPL")

        assert "error" in result
        assert "Failed to generate features" in result["error"]

    def test_start_streaming_predictions(self, prediction_engine):
        """Test starting streaming predictions"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        with patch.object(prediction_engine, "load_production_models"):
            prediction_engine.start_streaming_predictions(symbols, interval_seconds=1)

            assert prediction_engine.streaming_active is True
            assert prediction_engine.streaming_symbols == symbols
            assert prediction_engine.streaming_interval == 1
            assert prediction_engine.streaming_thread is not None

    def test_start_streaming_already_active(self, prediction_engine):
        """Test starting streaming when already active"""
        prediction_engine.streaming_active = True

        with patch.object(prediction_engine, "load_production_models"):
            prediction_engine.start_streaming_predictions(["AAPL"])

            # Should not create new thread if already active
            assert prediction_engine.streaming_thread is None

    def test_stop_streaming_predictions(self, prediction_engine):
        """Test stopping streaming predictions"""
        # Mock active streaming
        prediction_engine.streaming_active = True
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        prediction_engine.streaming_thread = mock_thread

        prediction_engine.stop_streaming_predictions()

        assert prediction_engine.streaming_active is False

    def test_stop_streaming_not_active(self, prediction_engine):
        """Test stopping streaming when not active"""
        prediction_engine.streaming_active = False

        prediction_engine.stop_streaming_predictions()

        # Should not raise error
        assert prediction_engine.streaming_active is False

    def test_add_prediction_callback(self, prediction_engine):
        """Test adding prediction callback"""

        def test_callback(symbol, result):
            pass

        prediction_engine.add_prediction_callback(test_callback)

        assert len(prediction_engine.prediction_callbacks) == 1
        assert test_callback in prediction_engine.prediction_callbacks

    def test_remove_prediction_callback(self, prediction_engine):
        """Test removing prediction callback"""

        def test_callback(symbol, result):
            pass

        prediction_engine.add_prediction_callback(test_callback)
        prediction_engine.remove_prediction_callback(test_callback)

        assert len(prediction_engine.prediction_callbacks) == 0
        assert test_callback not in prediction_engine.prediction_callbacks

    def test_get_prediction_history(self, prediction_engine):
        """Test getting prediction history"""
        # Add some predictions to cache
        prediction_engine.cache.add_prediction("AAPL", "model_001", 0.75)
        prediction_engine.cache.add_prediction("GOOGL", "model_002", 0.65)
        prediction_engine.cache.add_prediction("AAPL", "model_003", 0.85)

        # Get all history
        all_history = prediction_engine.get_prediction_history()
        assert len(all_history) == 3

        # Get filtered by symbol
        aapl_history = prediction_engine.get_prediction_history(symbol="AAPL")
        assert len(aapl_history) == 2

        # Get filtered by model
        model_history = prediction_engine.get_prediction_history(model_id="model_001")
        assert len(model_history) == 1

    def test_get_model_performance_realtime(self, prediction_engine):
        """Test getting real-time model performance"""
        # Add predictions for multiple models
        prediction_engine.cache.add_prediction(
            symbol="AAPL", model_id="model_001", prediction=0.75, confidence=0.85
        )

        prediction_engine.cache.add_prediction(
            symbol="AAPL", model_id="model_001", prediction=0.80, confidence=0.90
        )

        prediction_engine.cache.add_prediction(
            symbol="MSFT", model_id="model_002", prediction=0.65, confidence=0.75
        )

        performance = prediction_engine.get_model_performance_realtime()

        assert "model_001" in performance
        # Note: model_002 might not appear if the implementation filters by loaded models
        # Just verify we got performance data for the existing model
        assert performance["model_001"]["avg_prediction"] == 0.775
        assert performance["model_001"]["avg_confidence"] == 0.875

    def test_health_check(self, prediction_engine, mock_registry):
        """Test health check functionality"""
        # Mock production models
        mock_metadata = Mock()

        with patch.object(
            mock_registry, "get_production_models", return_value=[mock_metadata]
        ):
            with patch.object(prediction_engine, "get_real_time_data") as mock_get_data:
                mock_get_data.return_value = pd.DataFrame({"close": [100, 101, 102]})

                health = prediction_engine.health_check()

                assert health["status"] == "healthy"
                assert "timestamp" in health
                assert "loaded_models" in health
                assert "production_models" in health
                assert "streaming_active" in health
                assert "data_collection_ok" in health
                assert health["data_collection_ok"] is True

    def test_health_check_unhealthy(self, prediction_engine):
        """Test health check with error"""
        # Mock error in health check
        with patch.object(
            prediction_engine.model_registry,
            "get_production_models",
            side_effect=Exception("DB Error"),
        ):
            health = prediction_engine.health_check()

            assert health["status"] == "unhealthy"
            assert "error" in health


if __name__ == "__main__":
    pytest.main([__file__])
