"""
Real-time Prediction Engine

This module provides real-time prediction capabilities for financial ML models
with streaming data support and prediction caching.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import logging

import pandas as pd
import numpy as np
from threading import Lock

from .model_registry import ModelRegistry
from ...data.collectors import YFinanceCollector
from ...features.pipeline import FeaturePipeline
from ..base import BaseFinancialModel


class PredictionCache:
    """Cache for storing recent predictions"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize prediction cache.

        Args:
            max_size: Maximum number of cached predictions
            ttl_seconds: Time-to-live for cached predictions in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = deque(maxlen=max_size)
        self.lock = Lock()

    def add_prediction(
        self,
        symbol: str,
        model_id: str,
        prediction: Any,
        confidence: float = None,
        features: Dict[str, float] = None,
    ):
        """Add prediction to cache"""
        with self.lock:
            prediction_data = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "model_id": model_id,
                "prediction": prediction,
                "confidence": confidence,
                "features": features,
            }
            self.cache.append(prediction_data)

    def get_recent_predictions(
        self, symbol: str = None, model_id: str = None, minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent predictions from cache"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self.lock:
            recent_predictions = []
            for pred in self.cache:
                if pred["timestamp"] >= cutoff_time:
                    if symbol and pred["symbol"] != symbol:
                        continue
                    if model_id and pred["model_id"] != model_id:
                        continue
                    recent_predictions.append(pred)

            return recent_predictions

    def cleanup_expired(self):
        """Remove expired predictions from cache"""
        cutoff_time = datetime.now() - timedelta(seconds=self.ttl_seconds)

        with self.lock:
            # Convert to list to avoid deque modification during iteration
            valid_predictions = [
                pred for pred in self.cache if pred["timestamp"] >= cutoff_time
            ]

            self.cache.clear()
            self.cache.extend(valid_predictions)


class RealTimePrediction:
    """
    Real-time prediction engine for financial ML models.

    Features:
    - Real-time data fetching and prediction
    - Multiple model ensemble predictions
    - Prediction caching and history
    - Streaming prediction updates
    - Confidence estimation
    - Performance monitoring
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        data_collector: YFinanceCollector = None,
        feature_pipeline: FeaturePipeline = None,
        cache_size: int = 1000,
        cache_ttl: int = 300,
    ):
        """
        Initialize real-time prediction engine.

        Args:
            model_registry: Model registry instance
            data_collector: Data collector for real-time data
            feature_pipeline: Feature engineering pipeline
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
        """
        self.model_registry = model_registry
        self.data_collector = data_collector or YFinanceCollector()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()

        # Prediction cache
        self.cache = PredictionCache(cache_size, cache_ttl)

        # Loaded models cache
        self.loaded_models = {}
        self.model_lock = Lock()

        # Streaming configuration
        self.streaming_active = False
        self.streaming_thread = None
        self.streaming_symbols = []
        self.streaming_interval = 60  # seconds
        self.prediction_callbacks = []

        self.logger = logging.getLogger(__name__)

    def load_production_models(self) -> Dict[str, BaseFinancialModel]:
        """Load all production models into memory"""
        production_models = self.model_registry.get_production_models()

        with self.model_lock:
            for metadata in production_models:
                if metadata.model_id not in self.loaded_models:
                    model = self.model_registry.load_model(metadata.model_id)
                    if model:
                        self.loaded_models[metadata.model_id] = {
                            "model": model,
                            "metadata": metadata,
                        }
                        self.logger.info(
                            f"Loaded production model: {metadata.model_id}"
                        )

        return {mid: data["model"] for mid, data in self.loaded_models.items()}

    def get_real_time_data(
        self, symbol: str, period: str = "1d", interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch real-time market data.

        Args:
            symbol: Trading symbol
            period: Data period
            interval: Data interval

        Returns:
            Real-time market data or None if failed
        """
        try:
            # Calculate start date based on period
            if period == "1d":
                start_date = datetime.now().strftime("%Y-%m-%d")
            elif period == "5d":
                start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            elif period == "1mo":
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            end_date = datetime.now().strftime("%Y-%m-%d")

            data = self.data_collector.fetch_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )

            if data is not None and not data.empty:
                return data
            else:
                self.logger.warning(f"No data received for {symbol}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return None

    def generate_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate features for prediction.

        Args:
            data: Market data

        Returns:
            Feature data or None if failed
        """
        try:
            features = self.feature_pipeline.generate_features(data)

            # Get the latest row for prediction
            if not features.empty:
                return features.iloc[[-1]]  # Latest row as DataFrame
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return None

    def predict_single_model(
        self, model_id: str, features: pd.DataFrame, return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction using a single model.

        Args:
            model_id: Model identifier
            features: Feature data
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary with prediction results
        """
        try:
            if model_id not in self.loaded_models:
                # Load model if not in cache
                model = self.model_registry.load_model(model_id)
                metadata = self.model_registry.get_model_metadata(model_id)

                if model and metadata:
                    with self.model_lock:
                        self.loaded_models[model_id] = {
                            "model": model,
                            "metadata": metadata,
                        }
                else:
                    return {"error": f"Failed to load model {model_id}"}

            model_data = self.loaded_models[model_id]
            model = model_data["model"]
            metadata = model_data["metadata"]

            # Ensure features match expected feature names
            expected_features = metadata.feature_names
            if len(expected_features) > 0:
                # Select only expected features in correct order
                available_features = [
                    f for f in expected_features if f in features.columns
                ]
                if len(available_features) < len(expected_features):
                    self.logger.warning(
                        f"Missing features for model {model_id}. "
                        f"Expected: {len(expected_features)}, Available: {len(available_features)}"
                    )

                features_subset = features[available_features]
            else:
                features_subset = features

            # Make prediction
            prediction = model.predict(features_subset)

            # Get prediction confidence if available
            confidence = None
            if return_confidence and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features_subset)
                    if proba is not None and len(proba) > 0:
                        confidence = float(np.max(proba[0]))
                except:
                    pass
            elif return_confidence and hasattr(model, "uncertainty_estimation"):
                try:
                    confidence = model.uncertainty_estimation(features_subset)
                except:
                    pass

            # Format prediction
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    prediction = prediction[0]
                if len(prediction) == 1:
                    prediction = prediction[0]

            return {
                "model_id": model_id,
                "model_name": metadata.model_name,
                "model_type": metadata.model_type,
                "prediction": (
                    float(prediction)
                    if isinstance(prediction, (int, float, np.number))
                    else prediction
                ),
                "confidence": confidence,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(
                f"Error making prediction with model {model_id}: {str(e)}"
            )
            return {"error": str(e)}

    def predict_ensemble(
        self,
        symbol: str,
        model_ids: List[str] = None,
        period: str = "1d",
        interval: str = "1m",
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction using multiple models.

        Args:
            symbol: Trading symbol
            model_ids: List of model IDs (use production models if None)
            period: Data period for fetching
            interval: Data interval

        Returns:
            Dictionary with ensemble prediction results
        """
        try:
            # Get real-time data
            data = self.get_real_time_data(symbol, period, interval)
            if data is None:
                return {"error": "Failed to fetch real-time data"}

            # Generate features
            features = self.generate_features(data)
            if features is None:
                return {"error": "Failed to generate features"}

            # Get models to use
            if model_ids is None:
                production_models = self.model_registry.get_production_models()
                model_ids = [m.model_id for m in production_models]

            if not model_ids:
                return {"error": "No models available for prediction"}

            # Make predictions with each model
            individual_predictions = []
            for model_id in model_ids:
                pred_result = self.predict_single_model(model_id, features)
                if "error" not in pred_result:
                    individual_predictions.append(pred_result)

            if not individual_predictions:
                return {"error": "All model predictions failed"}

            # Calculate ensemble prediction
            predictions = [pred["prediction"] for pred in individual_predictions]
            confidences = [
                pred["confidence"]
                for pred in individual_predictions
                if pred["confidence"] is not None
            ]

            # Ensemble methods
            ensemble_prediction = np.mean(predictions)
            ensemble_confidence = np.mean(confidences) if confidences else None

            # Weighted ensemble (if confidences available)
            if confidences and len(confidences) == len(predictions):
                weights = np.array(confidences)
                weights = weights / np.sum(weights)  # Normalize
                weighted_prediction = np.sum(np.array(predictions) * weights)
            else:
                weighted_prediction = ensemble_prediction

            # Create result
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "data_period": period,
                "data_interval": interval,
                "ensemble_prediction": float(ensemble_prediction),
                "weighted_prediction": float(weighted_prediction),
                "ensemble_confidence": (
                    float(ensemble_confidence) if ensemble_confidence else None
                ),
                "individual_predictions": individual_predictions,
                "features_used": features.columns.tolist(),
                "latest_price": (
                    float(data["close"].iloc[-1]) if "close" in data.columns else None
                ),
            }

            # Cache prediction
            self.cache.add_prediction(
                symbol=symbol,
                model_id="ensemble",
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                features=features.iloc[0].to_dict(),
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Error making ensemble prediction for {symbol}: {str(e)}"
            )
            return {"error": str(e)}

    def start_streaming_predictions(
        self,
        symbols: List[str],
        interval_seconds: int = 60,
        model_ids: List[str] = None,
    ):
        """
        Start streaming predictions for specified symbols.

        Args:
            symbols: List of trading symbols
            interval_seconds: Prediction interval in seconds
            model_ids: List of model IDs (use production models if None)
        """
        if self.streaming_active:
            self.logger.warning("Streaming predictions already active")
            return

        self.streaming_symbols = symbols
        self.streaming_interval = interval_seconds
        self.streaming_active = True

        # Load production models
        self.load_production_models()

        # Start streaming thread
        self.streaming_thread = threading.Thread(
            target=self._streaming_worker, args=(model_ids,), daemon=True
        )
        self.streaming_thread.start()

        self.logger.info(
            f"Started streaming predictions for {symbols} every {interval_seconds}s"
        )

    def stop_streaming_predictions(self):
        """Stop streaming predictions"""
        if not self.streaming_active:
            return

        self.streaming_active = False

        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=5)

        self.logger.info("Stopped streaming predictions")

    def _streaming_worker(self, model_ids: List[str] = None):
        """Worker thread for streaming predictions"""
        while self.streaming_active:
            try:
                for symbol in self.streaming_symbols:
                    if not self.streaming_active:
                        break

                    # Make ensemble prediction
                    prediction_result = self.predict_ensemble(
                        symbol=symbol, model_ids=model_ids, period="1d", interval="5m"
                    )

                    # Call prediction callbacks
                    for callback in self.prediction_callbacks:
                        try:
                            callback(symbol, prediction_result)
                        except Exception as e:
                            self.logger.error(f"Error in prediction callback: {str(e)}")

                # Cleanup expired cache entries
                self.cache.cleanup_expired()

                # Wait for next interval
                time.sleep(self.streaming_interval)

            except Exception as e:
                self.logger.error(f"Error in streaming worker: {str(e)}")
                time.sleep(self.streaming_interval)

    def add_prediction_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Add callback function for streaming predictions.

        Args:
            callback: Function that takes (symbol, prediction_result) as arguments
        """
        self.prediction_callbacks.append(callback)

    def remove_prediction_callback(self, callback: Callable):
        """Remove prediction callback"""
        if callback in self.prediction_callbacks:
            self.prediction_callbacks.remove(callback)

    def get_prediction_history(
        self, symbol: str = None, model_id: str = None, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get prediction history from cache.

        Args:
            symbol: Filter by symbol
            model_id: Filter by model ID
            hours: Number of hours to look back

        Returns:
            List of historical predictions
        """
        return self.cache.get_recent_predictions(
            symbol=symbol, model_id=model_id, minutes=hours * 60
        )

    def get_model_performance_realtime(self, hours: int = 24) -> Dict[str, Any]:
        """
        Calculate real-time model performance metrics.

        Args:
            hours: Hours to look back for performance calculation

        Returns:
            Dictionary with performance metrics
        """
        predictions = self.get_prediction_history(hours=hours)

        # Group by model_id
        model_predictions = {}
        for pred in predictions:
            model_id = pred["model_id"]
            if model_id not in model_predictions:
                model_predictions[model_id] = []
            model_predictions[model_id].append(pred)

        # Calculate metrics for each model
        performance = {}
        for model_id, preds in model_predictions.items():
            if len(preds) > 1:
                # Calculate prediction consistency and other metrics
                predictions_values = [p["prediction"] for p in preds]
                confidences = [
                    p["confidence"] for p in preds if p["confidence"] is not None
                ]

                performance[model_id] = {
                    "total_predictions": len(preds),
                    "avg_prediction": np.mean(predictions_values),
                    "prediction_std": np.std(predictions_values),
                    "avg_confidence": np.mean(confidences) if confidences else None,
                    "latest_prediction": preds[-1]["prediction"],
                    "latest_timestamp": preds[-1]["timestamp"],
                }

        return performance

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the prediction engine.

        Returns:
            Health status information
        """
        try:
            # Check loaded models
            with self.model_lock:
                loaded_model_count = len(self.loaded_models)

            # Check production models
            production_models = self.model_registry.get_production_models()
            production_model_count = len(production_models)

            # Check cache status
            recent_predictions = len(self.cache.get_recent_predictions(minutes=60))

            # Test data collection
            test_symbol = "AAPL"
            test_data = self.get_real_time_data(test_symbol, period="1d", interval="1h")
            data_collection_ok = test_data is not None and not test_data.empty

            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "loaded_models": loaded_model_count,
                "production_models": production_model_count,
                "streaming_active": self.streaming_active,
                "streaming_symbols": self.streaming_symbols,
                "recent_predictions": recent_predictions,
                "data_collection_ok": data_collection_ok,
                "cache_size": len(self.cache.cache),
            }

        except Exception as e:
            return {"status": "unhealthy", "timestamp": datetime.now(), "error": str(e)}
