"""
Test Model Deployment System

Comprehensive tests for the model deployment system including
Blue-Green deployment, Canary deployment, and deployment management.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.models.pipeline.deployment import ModelDeployment, DeploymentConfig
from src.models.pipeline.model_registry import ModelRegistry, ModelMetadata
from src.models.pipeline.prediction import RealTimePrediction


class TestDeploymentConfig:
    """Test DeploymentConfig class"""

    def test_default_config(self):
        """Test default deployment configuration"""
        config = DeploymentConfig()

        assert config.deployment_type == "blue_green"
        assert config.traffic_percentage == 100.0
        assert config.rollback_threshold == 0.05
        assert config.monitoring_window == 24
        assert config.auto_rollback is True
        assert config.health_check_interval == 300

    def test_custom_config(self):
        """Test custom deployment configuration"""
        config = DeploymentConfig(
            deployment_type="canary",
            traffic_percentage=50.0,
            rollback_threshold=0.1,
            auto_rollback=False,
        )

        assert config.deployment_type == "canary"
        assert config.traffic_percentage == 50.0
        assert config.rollback_threshold == 0.1
        assert config.auto_rollback is False


class TestModelDeployment:
    """Test ModelDeployment class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for deployments"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_registry(self, temp_dir):
        """Create mock model registry"""
        registry_dir = os.path.join(temp_dir, "registry")
        os.makedirs(registry_dir, exist_ok=True)
        return ModelRegistry(registry_dir)

    @pytest.fixture
    def mock_prediction_engine(self):
        """Create mock prediction engine"""
        engine = Mock(spec=RealTimePrediction)
        engine.loaded_models = {}
        return engine

    @pytest.fixture
    def deployment_manager(self, mock_registry, mock_prediction_engine, temp_dir):
        """Create ModelDeployment instance"""
        deployment_path = os.path.join(temp_dir, "deployments")
        return ModelDeployment(
            model_registry=mock_registry,
            prediction_engine=mock_prediction_engine,
            deployment_path=deployment_path,
        )

    @pytest.fixture
    def sample_model_metadata(self):
        """Create sample model metadata"""
        return ModelMetadata(
            model_id="test_model_001",
            model_name="test_model",
            model_type="random_forest",
            task_type="classification",
            version="v001",
            created_at=datetime.now(),
            file_path="/path/to/model.pkl",
            performance_metrics={"accuracy": 0.85},
            hyperparameters={"n_estimators": 100},
            feature_names=["feature1", "feature2"],
            training_data_info={"samples": 1000},
        )

    def test_deployment_manager_initialization(self, deployment_manager, temp_dir):
        """Test deployment manager initialization"""
        assert deployment_manager.model_registry is not None
        assert deployment_manager.prediction_engine is not None
        assert isinstance(deployment_manager.active_deployments, dict)
        assert isinstance(deployment_manager.deployment_history, list)

        # Check deployment directory exists
        deployment_path = os.path.join(temp_dir, "deployments")
        assert os.path.exists(deployment_path)

    def test_save_and_load_deployment_state(self, deployment_manager):
        """Test saving and loading deployment state"""
        # Add some state
        deployment_manager.active_deployments["test_deployment"] = {
            "deployment_id": "test_deployment",
            "status": "active",
        }
        deployment_manager.deployment_history.append(
            {
                "deployment_id": "test_deployment",
                "action": "created",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save state
        deployment_manager._save_deployment_state()

        # Clear and reload
        deployment_manager.active_deployments = {}
        deployment_manager.deployment_history = []
        deployment_manager._load_deployment_state()

        # Check state was restored
        assert "test_deployment" in deployment_manager.active_deployments
        assert len(deployment_manager.deployment_history) == 1

    def test_create_deployment(
        self, deployment_manager, mock_registry, sample_model_metadata
    ):
        """Test creating a new deployment"""
        # Mock registry methods
        with patch.object(
            mock_registry, "get_model_metadata", return_value=sample_model_metadata
        ):
            with patch.object(
                mock_registry, "get_model_stage", return_value="production"
            ):
                with patch.object(deployment_manager, "_copy_model_artifacts"):

                    deployment_id = deployment_manager.create_deployment(
                        model_id="test_model_001",
                        deployment_name="test_deployment",
                        description="Test deployment",
                    )

                    assert deployment_id is not None
                    assert "test_deployment" in deployment_id
                    assert deployment_id in deployment_manager.active_deployments

                    deployment = deployment_manager.active_deployments[deployment_id]
                    assert deployment["model_id"] == "test_model_001"
                    assert deployment["deployment_name"] == "test_deployment"
                    assert deployment["status"] == "created"

    def test_create_deployment_model_not_found(self, deployment_manager, mock_registry):
        """Test creating deployment with non-existent model"""
        with patch.object(mock_registry, "get_model_metadata", return_value=None):
            with pytest.raises(ValueError, match="Model .* not found"):
                deployment_manager.create_deployment(
                    model_id="nonexistent_model", deployment_name="test_deployment"
                )

    def test_create_deployment_not_production(
        self, deployment_manager, mock_registry, sample_model_metadata
    ):
        """Test creating deployment with non-production model"""
        with patch.object(
            mock_registry, "get_model_metadata", return_value=sample_model_metadata
        ):
            with patch.object(mock_registry, "get_model_stage", return_value="staging"):
                with pytest.raises(ValueError, match="not in production stage"):
                    deployment_manager.create_deployment(
                        model_id="test_model_001", deployment_name="test_deployment"
                    )

    def test_copy_model_artifacts_file(self, deployment_manager, temp_dir):
        """Test copying model artifact file"""
        # Create source file
        source_file = os.path.join(temp_dir, "source_model.pkl")
        with open(source_file, "w") as f:
            f.write("mock model data")

        target_dir = os.path.join(temp_dir, "target")
        os.makedirs(target_dir, exist_ok=True)

        deployment_manager._copy_model_artifacts(source_file, target_dir)

        # Check file was copied
        target_file = os.path.join(target_dir, "model.pkl")
        assert os.path.exists(target_file)

    def test_copy_model_artifacts_directory(self, deployment_manager, temp_dir):
        """Test copying model artifact directory"""
        # Create source directory
        source_dir = os.path.join(temp_dir, "source_model")
        os.makedirs(source_dir, exist_ok=True)
        with open(os.path.join(source_dir, "model.json"), "w") as f:
            f.write('{"model": "data"}')

        target_dir = os.path.join(temp_dir, "target")
        os.makedirs(target_dir, exist_ok=True)

        deployment_manager._copy_model_artifacts(source_dir, target_dir)

        # Check directory was copied
        target_model_dir = os.path.join(target_dir, "model")
        assert os.path.exists(target_model_dir)
        assert os.path.exists(os.path.join(target_model_dir, "model.json"))

    def test_get_model_path(self, deployment_manager, temp_dir):
        """Test getting model path from deployment"""
        # Test file path
        deployment_dir = os.path.join(temp_dir, "test_deployment")
        os.makedirs(deployment_dir, exist_ok=True)
        model_file = os.path.join(deployment_dir, "model.pkl")
        with open(model_file, "w") as f:
            f.write("model")

        deployment = {"deployment_dir": deployment_dir}
        path = deployment_manager._get_model_path(deployment)
        assert path == model_file

        # Test directory path
        os.remove(model_file)
        model_dir = os.path.join(deployment_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        path = deployment_manager._get_model_path(deployment)
        assert path == model_dir

    def test_get_model_path_not_found(self, deployment_manager, temp_dir):
        """Test getting model path when no artifacts exist"""
        deployment_dir = os.path.join(temp_dir, "empty_deployment")
        os.makedirs(deployment_dir, exist_ok=True)

        deployment = {"deployment_dir": deployment_dir}

        with pytest.raises(ValueError, match="Model artifacts not found"):
            deployment_manager._get_model_path(deployment)

    def test_load_model_from_deployment_pkl(self, deployment_manager):
        """Test loading model from pkl file"""
        mock_model = Mock()

        with patch("joblib.load", return_value=mock_model):
            loaded_model = deployment_manager._load_model_from_deployment(
                "/path/to/model.pkl", "test_model_001"
            )

            assert loaded_model == mock_model

    def test_load_model_from_deployment_error(self, deployment_manager):
        """Test error handling in model loading"""
        with patch("joblib.load", side_effect=Exception("Load error")):
            loaded_model = deployment_manager._load_model_from_deployment(
                "/path/to/model.pkl", "test_model_001"
            )

            assert loaded_model is None

    def test_health_check_model(self, deployment_manager):
        """Test model health check"""
        # Test successful health check
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.5])

        result = deployment_manager._health_check_model(mock_model)
        assert result is True

        # Test failed health check (NaN prediction)
        mock_model.predict.return_value = np.array([np.nan])
        result = deployment_manager._health_check_model(mock_model)
        assert result is False

        # Test health check with exception
        mock_model.predict.side_effect = Exception("Prediction error")
        result = deployment_manager._health_check_model(mock_model)
        assert result is False

    def test_deploy_blue_green_success(self, deployment_manager, temp_dir):
        """Test successful blue-green deployment"""
        # Create deployment
        deployment_id = "test_deployment_001"
        deployment_dir = os.path.join(temp_dir, "deployments", deployment_id)
        os.makedirs(deployment_dir, exist_ok=True)

        # Create model file
        model_file = os.path.join(deployment_dir, "model.pkl")
        with open(model_file, "w") as f:
            f.write("model data")

        deployment = {
            "deployment_id": deployment_id,
            "deployment_dir": deployment_dir,
            "model_id": "test_model_001",
        }
        deployment_manager.active_deployments[deployment_id] = deployment

        # Mock model loading and health check
        mock_model = Mock()
        with patch.object(
            deployment_manager, "_load_model_from_deployment", return_value=mock_model
        ):
            with patch.object(
                deployment_manager, "_health_check_model", return_value=True
            ):
                with patch.object(
                    deployment_manager.model_registry,
                    "get_model_metadata",
                    return_value=Mock(),
                ):

                    success = deployment_manager.deploy_blue_green(
                        deployment_id, switch_traffic=True
                    )

                    assert success is True
                    assert deployment["status"] == "active"
                    assert "activated_at" in deployment
                    assert "rollback_models" in deployment

    def test_deploy_blue_green_health_check_failure(self, deployment_manager, temp_dir):
        """Test blue-green deployment with health check failure"""
        deployment_id = "test_deployment_001"
        deployment_dir = os.path.join(temp_dir, "deployments", deployment_id)
        os.makedirs(deployment_dir, exist_ok=True)

        model_file = os.path.join(deployment_dir, "model.pkl")
        with open(model_file, "w") as f:
            f.write("model data")

        deployment = {
            "deployment_id": deployment_id,
            "deployment_dir": deployment_dir,
            "model_id": "test_model_001",
        }
        deployment_manager.active_deployments[deployment_id] = deployment

        # Mock model loading but failed health check
        mock_model = Mock()
        with patch.object(
            deployment_manager, "_load_model_from_deployment", return_value=mock_model
        ):
            with patch.object(
                deployment_manager, "_health_check_model", return_value=False
            ):

                success = deployment_manager.deploy_blue_green(deployment_id)

                assert success is False
                assert deployment["status"] == "failed"
                assert "error" in deployment

    def test_deploy_blue_green_nonexistent_deployment(self, deployment_manager):
        """Test blue-green deployment with non-existent deployment"""
        with pytest.raises(ValueError, match="Deployment .* not found"):
            deployment_manager.deploy_blue_green("nonexistent_deployment")

    def test_deploy_canary_success(self, deployment_manager, temp_dir):
        """Test successful canary deployment"""
        deployment_id = "test_deployment_001"
        deployment_dir = os.path.join(temp_dir, "deployments", deployment_id)
        os.makedirs(deployment_dir, exist_ok=True)

        model_file = os.path.join(deployment_dir, "model.pkl")
        with open(model_file, "w") as f:
            f.write("model data")

        deployment = {
            "deployment_id": deployment_id,
            "deployment_dir": deployment_dir,
            "model_id": "test_model_001",
        }
        deployment_manager.active_deployments[deployment_id] = deployment

        # Mock model loading and health check
        mock_model = Mock()
        with patch.object(
            deployment_manager, "_load_model_from_deployment", return_value=mock_model
        ):
            with patch.object(
                deployment_manager, "_health_check_model", return_value=True
            ):

                success = deployment_manager.deploy_canary(
                    deployment_id, traffic_percentage=20.0
                )

                assert success is True
                assert deployment["status"] == "canary"
                assert deployment["traffic_percentage"] == 20.0
                assert "canary_started_at" in deployment
                assert "canary_model" in deployment

    def test_promote_canary_success(self, deployment_manager):
        """Test successful canary promotion"""
        deployment_id = "test_deployment_001"
        mock_model = Mock()

        deployment = {
            "deployment_id": deployment_id,
            "status": "canary",
            "model_id": "test_model_001",
            "canary_model": mock_model,
        }
        deployment_manager.active_deployments[deployment_id] = deployment

        with patch.object(
            deployment_manager.model_registry, "get_model_metadata", return_value=Mock()
        ):
            success = deployment_manager.promote_canary(deployment_id)

            assert success is True
            assert deployment["status"] == "active"
            assert deployment["traffic_percentage"] == 100.0
            assert "promoted_at" in deployment
            assert "canary_model" not in deployment

    def test_promote_canary_not_canary(self, deployment_manager):
        """Test promoting non-canary deployment"""
        deployment_id = "test_deployment_001"
        deployment = {"deployment_id": deployment_id, "status": "active"}
        deployment_manager.active_deployments[deployment_id] = deployment

        with pytest.raises(ValueError, match="not in canary state"):
            deployment_manager.promote_canary(deployment_id)

    def test_rollback_deployment(self, deployment_manager):
        """Test deployment rollback"""
        deployment_id = "test_deployment_001"
        rollback_models = {"model_001": Mock()}

        deployment = {
            "deployment_id": deployment_id,
            "status": "active",
            "rollback_models": rollback_models,
        }
        deployment_manager.active_deployments[deployment_id] = deployment

        success = deployment_manager.rollback_deployment(
            deployment_id, reason="Performance degradation"
        )

        assert success is True
        assert deployment["status"] == "rolled_back"
        assert "rollback_at" in deployment
        assert deployment["rollback_reason"] == "Performance degradation"
        assert deployment_manager.prediction_engine.loaded_models == rollback_models

    def test_rollback_nonexistent_deployment(self, deployment_manager):
        """Test rollback of non-existent deployment"""
        with pytest.raises(ValueError, match="Deployment .* not found"):
            deployment_manager.rollback_deployment("nonexistent_deployment")

    def test_monitor_deployment(self, deployment_manager):
        """Test deployment monitoring"""
        deployment_id = "test_deployment_001"
        deployment = {
            "deployment_id": deployment_id,
            "model_id": "test_model_001",
            "status": "active",
            "traffic_percentage": 100.0,
        }
        deployment_manager.active_deployments[deployment_id] = deployment

        # Mock prediction history
        mock_predictions = [
            {"prediction": 0.75, "confidence": 0.85},
            {"prediction": 0.80, "confidence": 0.90},
            {"prediction": 0.70, "confidence": 0.80},
        ]

        with patch.object(
            deployment_manager.prediction_engine,
            "get_prediction_history",
            return_value=mock_predictions,
        ):
            with patch.object(
                deployment_manager, "_calculate_uptime", return_value=24.5
            ):

                metrics = deployment_manager.monitor_deployment(deployment_id)

                assert metrics["deployment_id"] == deployment_id
                assert metrics["deployment_status"] == "active"
                assert metrics["traffic_percentage"] == 100.0
                assert metrics["total_predictions"] == 3
                assert abs(metrics["avg_prediction"] - 0.75) < 0.01
                assert abs(metrics["avg_confidence"] - 0.85) < 0.01
                assert metrics["uptime_hours"] == 24.5

    def test_monitor_nonexistent_deployment(self, deployment_manager):
        """Test monitoring non-existent deployment"""
        with pytest.raises(ValueError, match="Deployment .* not found"):
            deployment_manager.monitor_deployment("nonexistent_deployment")

    def test_calculate_uptime(self, deployment_manager):
        """Test uptime calculation"""
        now = datetime.now()

        # Test active deployment
        deployment = {
            "status": "active",
            "activated_at": (now - timedelta(hours=5)).isoformat(),
        }
        uptime = deployment_manager._calculate_uptime(deployment)
        assert abs(uptime - 5.0) < 0.1

        # Test canary deployment
        deployment = {
            "status": "canary",
            "canary_started_at": (now - timedelta(hours=2)).isoformat(),
        }
        uptime = deployment_manager._calculate_uptime(deployment)
        assert abs(uptime - 2.0) < 0.1

        # Test deployment without timestamps
        deployment = {"status": "created"}
        uptime = deployment_manager._calculate_uptime(deployment)
        assert uptime == 0.0

    def test_list_deployments(self, deployment_manager):
        """Test listing deployments"""
        # Add sample deployments
        deployment_manager.active_deployments = {
            "deploy_001": {
                "deployment_id": "deploy_001",
                "status": "active",
                "created_at": datetime.now().isoformat(),
            },
            "deploy_002": {
                "deployment_id": "deploy_002",
                "status": "failed",
                "created_at": datetime.now().isoformat(),
            },
            "deploy_003": {
                "deployment_id": "deploy_003",
                "status": "active",
                "created_at": datetime.now().isoformat(),
            },
        }

        with patch.object(
            deployment_manager,
            "monitor_deployment",
            return_value={"monitoring": "data"},
        ):
            # Test listing all deployments
            all_deployments = deployment_manager.list_deployments()
            assert len(all_deployments) == 3

            # Test filtering by status
            active_deployments = deployment_manager.list_deployments(status="active")
            assert len(active_deployments) == 2

            failed_deployments = deployment_manager.list_deployments(status="failed")
            assert len(failed_deployments) == 1

    def test_get_deployment_status(self, deployment_manager):
        """Test getting deployment status"""
        deployment_id = "test_deployment_001"
        deployment = {
            "deployment_id": deployment_id,
            "status": "active",
            "model_id": "test_model_001",
        }
        deployment_manager.active_deployments[deployment_id] = deployment
        deployment_manager.deployment_history = [
            {
                "deployment_id": deployment_id,
                "action": "created",
                "timestamp": datetime.now().isoformat(),
            }
        ]

        with patch.object(
            deployment_manager, "monitor_deployment", return_value={"metrics": "data"}
        ):
            status = deployment_manager.get_deployment_status(deployment_id)

            assert status["deployment_id"] == deployment_id
            assert status["status"] == "active"
            assert "monitoring" in status
            assert "history" in status
            assert len(status["history"]) == 1

    def test_get_nonexistent_deployment_status(self, deployment_manager):
        """Test getting status of non-existent deployment"""
        status = deployment_manager.get_deployment_status("nonexistent_deployment")

        assert "error" in status
        assert "not found" in status["error"]

    def test_cleanup_old_deployments(self, deployment_manager, temp_dir):
        """Test cleanup of old deployments"""
        now = datetime.now()
        old_date = (now - timedelta(days=35)).isoformat()
        recent_date = (now - timedelta(days=5)).isoformat()

        # Create deployment directories
        old_deploy_dir = os.path.join(temp_dir, "deployments", "old_deployment")
        recent_deploy_dir = os.path.join(temp_dir, "deployments", "recent_deployment")
        os.makedirs(old_deploy_dir, exist_ok=True)
        os.makedirs(recent_deploy_dir, exist_ok=True)

        # Add deployments
        deployment_manager.active_deployments = {
            "old_deployment": {
                "deployment_id": "old_deployment",
                "status": "failed",
                "created_at": old_date,
                "deployment_dir": old_deploy_dir,
            },
            "recent_deployment": {
                "deployment_id": "recent_deployment",
                "status": "active",
                "created_at": recent_date,
                "deployment_dir": recent_deploy_dir,
            },
        }

        # Add history
        deployment_manager.deployment_history = [
            {"deployment_id": "old_deployment", "timestamp": old_date},
            {"deployment_id": "recent_deployment", "timestamp": recent_date},
        ]

        deployment_manager.cleanup_old_deployments(keep_days=30)

        # Check old deployment was removed
        assert "old_deployment" not in deployment_manager.active_deployments
        assert "recent_deployment" in deployment_manager.active_deployments
        assert not os.path.exists(old_deploy_dir)
        assert os.path.exists(recent_deploy_dir)

        # Check history was cleaned
        assert len(deployment_manager.deployment_history) == 1
        assert (
            deployment_manager.deployment_history[0]["deployment_id"]
            == "recent_deployment"
        )


if __name__ == "__main__":
    pytest.main([__file__])
