"""
Model Pipeline Package

Complete end-to-end machine learning pipeline for financial models
based on "Advances in Financial Machine Learning" methodologies.

Components:
- ModelTrainingPipeline: Automated model training with ensemble support
- ModelRegistry: Model versioning and lifecycle management
- RealTimePrediction: Real-time prediction engine with caching
- ModelDeployment: Production deployment with blue-green and canary strategies
- ModelMonitor: Comprehensive monitoring with drift detection and alerting
"""

from .training_pipeline import ModelTrainingPipeline, ModelTrainingConfig
from .model_registry import ModelRegistry, ModelMetadata
from .prediction import RealTimePrediction, PredictionCache
from .deployment import ModelDeployment, DeploymentConfig
from .monitoring import ModelMonitor, MonitoringConfig, Alert

__all__ = [
    # Training Pipeline
    "ModelTrainingPipeline",
    "ModelTrainingConfig",
    # Model Registry
    "ModelRegistry",
    "ModelMetadata",
    # Real-time Prediction
    "RealTimePrediction",
    "PredictionCache",
    # Model Deployment
    "ModelDeployment",
    "DeploymentConfig",
    # Model Monitoring
    "ModelMonitor",
    "MonitoringConfig",
    "Alert",
]
