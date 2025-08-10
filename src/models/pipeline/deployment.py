"""
Model Deployment Management

This module provides production deployment capabilities for financial ML models
including A/B testing, canary deployments, and deployment monitoring.
"""

import os
import json
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np

from .model_registry import ModelRegistry
from .prediction import RealTimePrediction
from ..base import BaseFinancialModel


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""

    deployment_type: str = "blue_green"  # "blue_green", "canary", "rolling"
    traffic_percentage: float = 100.0  # Percentage of traffic for new model
    rollback_threshold: float = 0.05  # Performance degradation threshold for rollback
    monitoring_window: int = 24  # Hours to monitor before full deployment
    auto_rollback: bool = True  # Automatic rollback on performance degradation
    health_check_interval: int = 300  # Health check interval in seconds


class ModelDeployment:
    """
    Model deployment management for production environments.

    Features:
    - Blue-green and canary deployments
    - A/B testing capabilities
    - Automated rollback on performance degradation
    - Deployment monitoring and health checks
    - Traffic routing and load balancing
    - Deployment history and audit trails
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        prediction_engine: RealTimePrediction,
        deployment_path: str = "deployments",
    ):
        """
        Initialize deployment manager.

        Args:
            model_registry: Model registry instance
            prediction_engine: Real-time prediction engine
            deployment_path: Path for deployment artifacts
        """
        self.model_registry = model_registry
        self.prediction_engine = prediction_engine
        self.deployment_path = deployment_path

        # Create deployment directory
        os.makedirs(deployment_path, exist_ok=True)

        # Track active deployments
        self.active_deployments = {}
        self.deployment_history = []

        self.logger = logging.getLogger(__name__)

        # Load existing deployments
        self._load_deployment_state()

    def _load_deployment_state(self):
        """Load deployment state from disk"""
        state_file = os.path.join(self.deployment_path, "deployment_state.json")

        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)

                self.active_deployments = state.get("active_deployments", {})
                self.deployment_history = state.get("deployment_history", [])

                self.logger.info(
                    f"Loaded deployment state: {len(self.active_deployments)} active deployments"
                )

            except Exception as e:
                self.logger.error(f"Error loading deployment state: {str(e)}")

    def _save_deployment_state(self):
        """Save deployment state to disk"""
        state_file = os.path.join(self.deployment_path, "deployment_state.json")

        state = {
            "active_deployments": self.active_deployments,
            "deployment_history": self.deployment_history,
            "last_updated": datetime.now().isoformat(),
        }

        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving deployment state: {str(e)}")

    def create_deployment(
        self,
        model_id: str,
        deployment_name: str,
        config: DeploymentConfig = None,
        description: str = "",
    ) -> str:
        """
        Create a new model deployment.

        Args:
            model_id: Model identifier to deploy
            deployment_name: Name for the deployment
            config: Deployment configuration
            description: Deployment description

        Returns:
            Deployment ID
        """
        config = config or DeploymentConfig()

        # Validate model exists and is production-ready
        model_metadata = self.model_registry.get_model_metadata(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")

        current_stage = self.model_registry.get_model_stage(model_id)
        if current_stage != "production":
            raise ValueError(
                f"Model {model_id} is not in production stage (current: {current_stage})"
            )

        # Generate deployment ID
        deployment_id = f"{deployment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create deployment directory
        deployment_dir = os.path.join(self.deployment_path, deployment_id)
        os.makedirs(deployment_dir, exist_ok=True)

        # Copy model artifacts
        self._copy_model_artifacts(model_metadata.file_path, deployment_dir)

        # Create deployment metadata
        deployment_metadata = {
            "deployment_id": deployment_id,
            "deployment_name": deployment_name,
            "model_id": model_id,
            "model_name": model_metadata.model_name,
            "model_version": model_metadata.version,
            "deployment_type": config.deployment_type,
            "traffic_percentage": config.traffic_percentage,
            "config": config.__dict__,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "deployment_dir": deployment_dir,
        }

        # Save deployment metadata
        metadata_file = os.path.join(deployment_dir, "deployment_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(deployment_metadata, f, indent=2, default=str)

        # Add to active deployments
        self.active_deployments[deployment_id] = deployment_metadata

        # Add to deployment history
        self.deployment_history.append(
            {
                "deployment_id": deployment_id,
                "action": "created",
                "timestamp": datetime.now().isoformat(),
                "user": "system",
            }
        )

        # Save state
        self._save_deployment_state()

        self.logger.info(f"Created deployment {deployment_id} for model {model_id}")

        return deployment_id

    def _copy_model_artifacts(self, source_path: str, target_dir: str):
        """Copy model artifacts to deployment directory"""
        try:
            if os.path.isdir(source_path):
                # Deep learning model directory
                shutil.copytree(source_path, os.path.join(target_dir, "model"))
            else:
                # Single model file
                shutil.copy2(source_path, os.path.join(target_dir, "model.pkl"))

        except Exception as e:
            self.logger.error(f"Error copying model artifacts: {str(e)}")
            raise

    def deploy_blue_green(
        self, deployment_id: str, switch_traffic: bool = True
    ) -> bool:
        """
        Execute blue-green deployment.

        Args:
            deployment_id: Deployment identifier
            switch_traffic: Whether to switch traffic to new deployment

        Returns:
            True if successful, False otherwise
        """
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.active_deployments[deployment_id]

        try:
            # Load and validate new model
            model_path = self._get_model_path(deployment)
            new_model = self._load_model_from_deployment(
                model_path, deployment["model_id"]
            )

            if new_model is None:
                raise ValueError("Failed to load new model")

            # Health check on new model
            if not self._health_check_model(new_model):
                raise ValueError("New model failed health check")

            if switch_traffic:
                # Switch traffic to new model (Blue-Green)
                old_production_models = self.prediction_engine.loaded_models.copy()

                # Update prediction engine with new model
                self.prediction_engine.loaded_models[deployment["model_id"]] = {
                    "model": new_model,
                    "metadata": self.model_registry.get_model_metadata(
                        deployment["model_id"]
                    ),
                }

                # Store rollback information
                deployment["rollback_models"] = old_production_models
                deployment["status"] = "active"
                deployment["activated_at"] = datetime.now().isoformat()

                self.logger.info(f"Blue-green deployment {deployment_id} activated")

            else:
                deployment["status"] = "staged"
                self.logger.info(
                    f"Blue-green deployment {deployment_id} staged (traffic not switched)"
                )

            # Update deployment history
            self.deployment_history.append(
                {
                    "deployment_id": deployment_id,
                    "action": "deployed_blue_green",
                    "timestamp": datetime.now().isoformat(),
                    "traffic_switched": switch_traffic,
                }
            )

            # Save state
            self._save_deployment_state()

            return True

        except Exception as e:
            self.logger.error(
                f"Blue-green deployment failed for {deployment_id}: {str(e)}"
            )
            deployment["status"] = "failed"
            deployment["error"] = str(e)
            self._save_deployment_state()
            return False

    def deploy_canary(
        self, deployment_id: str, traffic_percentage: float = 10.0
    ) -> bool:
        """
        Execute canary deployment.

        Args:
            deployment_id: Deployment identifier
            traffic_percentage: Percentage of traffic for canary

        Returns:
            True if successful, False otherwise
        """
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.active_deployments[deployment_id]

        try:
            # Load new model
            model_path = self._get_model_path(deployment)
            new_model = self._load_model_from_deployment(
                model_path, deployment["model_id"]
            )

            if new_model is None:
                raise ValueError("Failed to load new model")

            # Health check
            if not self._health_check_model(new_model):
                raise ValueError("New model failed health check")

            # Set up canary deployment
            deployment["traffic_percentage"] = traffic_percentage
            deployment["status"] = "canary"
            deployment["canary_started_at"] = datetime.now().isoformat()

            # Note: In production, this would involve load balancer configuration
            # For now, we'll simulate by storing the canary model
            deployment["canary_model"] = new_model

            self.logger.info(
                f"Canary deployment {deployment_id} started with {traffic_percentage}% traffic"
            )

            # Update deployment history
            self.deployment_history.append(
                {
                    "deployment_id": deployment_id,
                    "action": "deployed_canary",
                    "timestamp": datetime.now().isoformat(),
                    "traffic_percentage": traffic_percentage,
                }
            )

            # Save state
            self._save_deployment_state()

            return True

        except Exception as e:
            self.logger.error(f"Canary deployment failed for {deployment_id}: {str(e)}")
            deployment["status"] = "failed"
            deployment["error"] = str(e)
            self._save_deployment_state()
            return False

    def promote_canary(self, deployment_id: str) -> bool:
        """
        Promote canary deployment to full production.

        Args:
            deployment_id: Deployment identifier

        Returns:
            True if successful, False otherwise
        """
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.active_deployments[deployment_id]

        if deployment["status"] != "canary":
            raise ValueError(f"Deployment {deployment_id} is not in canary state")

        try:
            # Move canary to full production
            canary_model = deployment.get("canary_model")
            if canary_model:
                # Update prediction engine
                self.prediction_engine.loaded_models[deployment["model_id"]] = {
                    "model": canary_model,
                    "metadata": self.model_registry.get_model_metadata(
                        deployment["model_id"]
                    ),
                }

            deployment["status"] = "active"
            deployment["traffic_percentage"] = 100.0
            deployment["promoted_at"] = datetime.now().isoformat()

            # Clean up canary-specific data
            if "canary_model" in deployment:
                del deployment["canary_model"]

            self.logger.info(
                f"Canary deployment {deployment_id} promoted to full production"
            )

            # Update deployment history
            self.deployment_history.append(
                {
                    "deployment_id": deployment_id,
                    "action": "promoted_canary",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Save state
            self._save_deployment_state()

            return True

        except Exception as e:
            self.logger.error(f"Canary promotion failed for {deployment_id}: {str(e)}")
            return False

    def rollback_deployment(self, deployment_id: str, reason: str = "") -> bool:
        """
        Rollback a deployment.

        Args:
            deployment_id: Deployment identifier
            reason: Reason for rollback

        Returns:
            True if successful, False otherwise
        """
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.active_deployments[deployment_id]

        try:
            # Restore previous models if available
            if "rollback_models" in deployment:
                self.prediction_engine.loaded_models = deployment["rollback_models"]
                self.logger.info(
                    f"Restored previous models for rollback of {deployment_id}"
                )

            deployment["status"] = "rolled_back"
            deployment["rollback_at"] = datetime.now().isoformat()
            deployment["rollback_reason"] = reason

            self.logger.info(f"Rolled back deployment {deployment_id}: {reason}")

            # Update deployment history
            self.deployment_history.append(
                {
                    "deployment_id": deployment_id,
                    "action": "rolled_back",
                    "timestamp": datetime.now().isoformat(),
                    "reason": reason,
                }
            )

            # Save state
            self._save_deployment_state()

            return True

        except Exception as e:
            self.logger.error(f"Rollback failed for {deployment_id}: {str(e)}")
            return False

    def _get_model_path(self, deployment: Dict[str, Any]) -> str:
        """Get model path from deployment"""
        deployment_dir = deployment["deployment_dir"]

        # Check for model directory (deep learning) or file (traditional ML)
        model_dir = os.path.join(deployment_dir, "model")
        model_file = os.path.join(deployment_dir, "model.pkl")

        if os.path.exists(model_dir):
            return model_dir
        elif os.path.exists(model_file):
            return model_file
        else:
            raise ValueError(f"Model artifacts not found in {deployment_dir}")

    def _load_model_from_deployment(
        self, model_path: str, model_id: str
    ) -> Optional[BaseFinancialModel]:
        """Load model from deployment artifacts"""
        try:
            if os.path.isdir(model_path):
                # Deep learning model
                from tensorflow import keras

                return keras.models.load_model(model_path)
            else:
                # Traditional ML model
                import joblib

                return joblib.load(model_path)

        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None

    def _health_check_model(self, model: BaseFinancialModel) -> bool:
        """Perform health check on model"""
        try:
            # Create dummy data for health check with correct feature names
            import pandas as pd

            # Use the standard feature names from our pipeline
            dummy_data = pd.DataFrame(
                {
                    "returns": [0.001],
                    "volatility": [0.02],
                    "sma_20": [150.0],
                    "sma_50": [148.0],
                    "rsi": [45.0],
                }
            )

            # Try to make a prediction
            prediction = model.predict(dummy_data)

            # Check if prediction is valid
            if prediction is not None and not np.isnan(prediction).any():
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Model health check failed: {str(e)}")
            return False

    def monitor_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Monitor deployment performance.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Monitoring metrics
        """
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.active_deployments[deployment_id]

        try:
            # Get deployment model performance
            model_id = deployment["model_id"]

            # Get recent predictions from prediction engine
            recent_predictions = self.prediction_engine.get_prediction_history(
                model_id=model_id, hours=24
            )

            # Calculate performance metrics
            if recent_predictions:
                predictions = [p["prediction"] for p in recent_predictions]
                confidences = [
                    p["confidence"]
                    for p in recent_predictions
                    if p["confidence"] is not None
                ]

                metrics = {
                    "total_predictions": len(predictions),
                    "avg_prediction": np.mean(predictions),
                    "prediction_std": np.std(predictions),
                    "avg_confidence": np.mean(confidences) if confidences else None,
                    "min_confidence": np.min(confidences) if confidences else None,
                    "max_confidence": np.max(confidences) if confidences else None,
                }
            else:
                metrics = {
                    "total_predictions": 0,
                    "avg_prediction": None,
                    "prediction_std": None,
                    "avg_confidence": None,
                }

            # Add deployment status
            metrics.update(
                {
                    "deployment_id": deployment_id,
                    "deployment_status": deployment["status"],
                    "traffic_percentage": deployment.get("traffic_percentage", 100.0),
                    "uptime_hours": self._calculate_uptime(deployment),
                    "health_status": (
                        "healthy" if deployment["status"] == "active" else "unknown"
                    ),
                }
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error monitoring deployment {deployment_id}: {str(e)}")
            return {"error": str(e)}

    def _calculate_uptime(self, deployment: Dict[str, Any]) -> float:
        """Calculate deployment uptime in hours"""
        if deployment["status"] in ["active", "canary"]:
            start_time_key = (
                "activated_at" if "activated_at" in deployment else "canary_started_at"
            )
            if start_time_key in deployment:
                start_time = datetime.fromisoformat(deployment[start_time_key])
                uptime = datetime.now() - start_time
                return uptime.total_seconds() / 3600

        return 0.0

    def list_deployments(self, status: str = None) -> List[Dict[str, Any]]:
        """
        List deployments with optional status filter.

        Args:
            status: Filter by deployment status

        Returns:
            List of deployment information
        """
        deployments = []

        for deployment_id, deployment in self.active_deployments.items():
            if status is None or deployment["status"] == status:
                deployment_info = deployment.copy()

                # Add monitoring metrics
                try:
                    monitoring = self.monitor_deployment(deployment_id)
                    deployment_info["monitoring"] = monitoring
                except:
                    deployment_info["monitoring"] = {
                        "error": "Failed to get monitoring data"
                    }

                deployments.append(deployment_info)

        # Sort by creation time (newest first)
        deployments.sort(key=lambda x: x["created_at"], reverse=True)

        return deployments

    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Deployment status information
        """
        if deployment_id not in self.active_deployments:
            return {"error": f"Deployment {deployment_id} not found"}

        deployment = self.active_deployments[deployment_id].copy()

        # Add monitoring data
        try:
            deployment["monitoring"] = self.monitor_deployment(deployment_id)
        except Exception as e:
            deployment["monitoring"] = {"error": str(e)}

        # Add deployment history for this deployment
        deployment["history"] = [
            h for h in self.deployment_history if h["deployment_id"] == deployment_id
        ]

        return deployment

    def cleanup_old_deployments(self, keep_days: int = 30):
        """
        Clean up old deployments.

        Args:
            keep_days: Number of days to keep deployments
        """
        cutoff_date = datetime.now() - timedelta(days=keep_days)

        deployments_to_remove = []

        for deployment_id, deployment in self.active_deployments.items():
            created_at = datetime.fromisoformat(deployment["created_at"])

            if created_at < cutoff_date and deployment["status"] in [
                "rolled_back",
                "failed",
                "inactive",
            ]:

                deployments_to_remove.append(deployment_id)

        # Remove old deployments
        for deployment_id in deployments_to_remove:
            deployment = self.active_deployments[deployment_id]

            # Remove deployment directory
            deployment_dir = deployment["deployment_dir"]
            if os.path.exists(deployment_dir):
                try:
                    shutil.rmtree(deployment_dir)
                    self.logger.info(f"Removed deployment directory: {deployment_dir}")
                except Exception as e:
                    self.logger.error(f"Error removing deployment directory: {str(e)}")

            # Remove from active deployments
            del self.active_deployments[deployment_id]

        # Clean up deployment history
        self.deployment_history = [
            h
            for h in self.deployment_history
            if datetime.fromisoformat(h["timestamp"]) >= cutoff_date
        ]

        # Save state
        self._save_deployment_state()

        self.logger.info(f"Cleaned up {len(deployments_to_remove)} old deployments")
