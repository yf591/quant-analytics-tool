"""
Test Model Registry

Comprehensive tests for the model registry system including
model registration, versioning, lifecycle management, and metadata storage.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import sqlite3
from datetime import datetime
from unittest.mock import Mock, patch
from pathlib import Path

from src.models.pipeline.model_registry import ModelRegistry, ModelMetadata


class TestModelMetadata:
    """Test ModelMetadata class"""

    def test_metadata_creation(self):
        """Test ModelMetadata creation and initialization"""
        metadata = ModelMetadata(
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
            description="Test model",
            tags=["test"],
        )

        assert metadata.model_id == "test_model_001"
        assert metadata.model_name == "test_model"
        assert metadata.model_type == "random_forest"
        assert metadata.task_type == "classification"
        assert metadata.version == "v001"
        assert metadata.performance_metrics == {"accuracy": 0.85}
        assert metadata.tags == ["test"]
        assert metadata.is_active is True

    def test_metadata_post_init(self):
        """Test ModelMetadata post-initialization"""
        metadata = ModelMetadata(
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

        # Should initialize empty tags list
        assert metadata.tags == []

    def test_metadata_to_dict(self):
        """Test ModelMetadata serialization to dictionary"""
        now = datetime.now()
        metadata = ModelMetadata(
            model_id="test_model_001",
            model_name="test_model",
            model_type="random_forest",
            task_type="classification",
            version="v001",
            created_at=now,
            file_path="/path/to/model.pkl",
            performance_metrics={"accuracy": 0.85},
            hyperparameters={"n_estimators": 100},
            feature_names=["feature1", "feature2"],
            training_data_info={"samples": 1000},
        )

        data_dict = metadata.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict["model_id"] == "test_model_001"
        assert data_dict["created_at"] == now.isoformat()
        assert data_dict["performance_metrics"] == {"accuracy": 0.85}

    def test_metadata_from_dict(self):
        """Test ModelMetadata deserialization from dictionary"""
        now = datetime.now()
        data_dict = {
            "model_id": "test_model_001",
            "model_name": "test_model",
            "model_type": "random_forest",
            "task_type": "classification",
            "version": "v001",
            "created_at": now.isoformat(),
            "file_path": "/path/to/model.pkl",
            "performance_metrics": {"accuracy": 0.85},
            "hyperparameters": {"n_estimators": 100},
            "feature_names": ["feature1", "feature2"],
            "training_data_info": {"samples": 1000},
            "description": "Test model",
            "tags": ["test"],
            "is_active": True,
        }

        metadata = ModelMetadata.from_dict(data_dict)

        assert metadata.model_id == "test_model_001"
        assert metadata.created_at == now
        assert metadata.performance_metrics == {"accuracy": 0.85}


class TestModelRegistry:
    """Test ModelRegistry class"""

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for registry"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def registry(self, temp_registry_dir):
        """Create ModelRegistry instance"""
        return ModelRegistry(temp_registry_dir)

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.5, 0.7, 0.3]))
        return model

    def test_registry_initialization(self, registry, temp_registry_dir):
        """Test registry initialization"""
        assert registry.registry_path == temp_registry_dir
        assert Path(registry.models_path).exists()
        assert Path(registry.db_path).exists()

        # Check database tables were created
        with sqlite3.connect(registry.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            assert "models" in tables
            assert "model_stages" in tables

    def test_generate_version(self, registry):
        """Test version generation"""
        # First model should be v001
        version = registry._generate_version("test_model")
        assert version == "v001"

        # Mock existing models in database
        with sqlite3.connect(registry.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO models (model_id, model_name, model_type, task_type, version, created_at, file_path, performance_metrics, hyperparameters, feature_names, training_data_info, description, tags, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "test_1",
                    "test_model",
                    "rf",
                    "class",
                    "v001",
                    datetime.now().isoformat(),
                    "/path",
                    "{}",
                    "{}",
                    "[]",
                    "{}",
                    "",
                    "[]",
                    1,
                ),
            )
            cursor.execute(
                "INSERT INTO models (model_id, model_name, model_type, task_type, version, created_at, file_path, performance_metrics, hyperparameters, feature_names, training_data_info, description, tags, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "test_2",
                    "test_model",
                    "rf",
                    "class",
                    "v002",
                    datetime.now().isoformat(),
                    "/path",
                    "{}",
                    "{}",
                    "[]",
                    "{}",
                    "",
                    "[]",
                    1,
                ),
            )
            conn.commit()

        # Next version should be v003
        version = registry._generate_version("test_model")
        assert version == "v003"

    @patch("src.models.pipeline.model_registry.joblib.dump")
    def test_save_model_artifact(self, mock_dump, registry, mock_model):
        """Test model artifact saving"""
        model_id = "test_model_001"

        # Test traditional ML model saving
        filepath = registry._save_model_artifact(mock_model, model_id, "random_forest")

        assert filepath.endswith(f"{model_id}.pkl")
        mock_dump.assert_called_once_with(mock_model, filepath)

    @patch("src.models.pipeline.model_registry.joblib.dump")
    def test_register_model(self, mock_dump, registry, mock_model):
        """Test model registration"""
        model_id = registry.register_model(
            model=mock_model,
            model_name="test_model",
            model_type="random_forest",
            task_type="classification",
            performance_metrics={"accuracy": 0.85, "f1": 0.82},
            feature_names=["feature1", "feature2"],
            training_data_info={"samples": 1000, "features": 2},
            description="Test model registration",
        )

        assert model_id is not None
        assert "test_model" in model_id
        assert "v001" in model_id

        # Check model was saved
        mock_dump.assert_called_once()

        # Check metadata was stored in database
        metadata = registry.get_model_metadata(model_id)
        assert metadata is not None
        assert metadata.model_name == "test_model"
        assert metadata.model_type == "random_forest"
        assert metadata.task_type == "classification"
        assert metadata.performance_metrics["accuracy"] == 0.85

        # Check model stage was set to staging
        stage = registry.get_model_stage(model_id)
        assert stage == "staging"

    def test_register_model_with_custom_version(self, registry, mock_model):
        """Test model registration with custom version"""
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id = registry.register_model(
                model=mock_model,
                model_name="test_model",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["feature1"],
                training_data_info={"samples": 1000},
                version="v999",
            )

            assert "v999" in model_id

    @patch("src.models.pipeline.model_registry.joblib.load")
    def test_load_model(self, mock_load, registry, mock_model):
        """Test model loading"""
        mock_load.return_value = mock_model

        # First register a model
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id = registry.register_model(
                model=mock_model,
                model_name="test_model",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["feature1"],
                training_data_info={"samples": 1000},
            )

        # Load the model
        loaded_model = registry.load_model(model_id)

        assert loaded_model is not None
        mock_load.assert_called_once()

    def test_load_nonexistent_model(self, registry):
        """Test loading non-existent model"""
        loaded_model = registry.load_model("nonexistent_model_id")
        assert loaded_model is None

    def test_get_model_metadata(self, registry, mock_model):
        """Test getting model metadata"""
        # Register a model first
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id = registry.register_model(
                model=mock_model,
                model_name="test_model",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["feature1", "feature2"],
                training_data_info={"samples": 1000},
            )

        # Get metadata
        metadata = registry.get_model_metadata(model_id)

        assert metadata is not None
        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_id == model_id
        assert metadata.model_name == "test_model"
        assert metadata.performance_metrics["accuracy"] == 0.85
        assert metadata.feature_names == ["feature1", "feature2"]

    def test_get_nonexistent_metadata(self, registry):
        """Test getting metadata for non-existent model"""
        metadata = registry.get_model_metadata("nonexistent_model_id")
        assert metadata is None

    def test_list_models(self, registry, mock_model):
        """Test listing models with filters"""
        # Register multiple models
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id1 = registry.register_model(
                model=mock_model,
                model_name="test_model_1",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["feature1"],
                training_data_info={"samples": 1000},
            )

            model_id2 = registry.register_model(
                model=mock_model,
                model_name="test_model_2",
                model_type="xgboost",
                task_type="regression",
                performance_metrics={"rmse": 0.15},
                feature_names=["feature1"],
                training_data_info={"samples": 1000},
            )

        # Test listing all models
        all_models = registry.list_models()
        assert len(all_models) == 2

        # Test filtering by model_name
        rf_models = registry.list_models(model_name="test_model_1")
        assert len(rf_models) == 1
        assert rf_models[0].model_name == "test_model_1"

        # Test filtering by model_type
        xgb_models = registry.list_models(model_type="xgboost")
        assert len(xgb_models) == 1
        assert xgb_models[0].model_type == "xgboost"

        # Test filtering by task_type
        class_models = registry.list_models(task_type="classification")
        assert len(class_models) == 1
        assert class_models[0].task_type == "classification"

    def test_set_model_stage(self, registry, mock_model):
        """Test setting model stage"""
        # Register a model first
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id = registry.register_model(
                model=mock_model,
                model_name="test_model",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["feature1"],
                training_data_info={"samples": 1000},
            )

        # Test setting to production
        registry.set_model_stage(model_id, "production")
        stage = registry.get_model_stage(model_id)
        assert stage == "production"

        # Test setting to archived
        registry.set_model_stage(model_id, "archived")
        stage = registry.get_model_stage(model_id)
        assert stage == "archived"

        # Test invalid stage
        with pytest.raises(ValueError, match="Invalid stage"):
            registry.set_model_stage(model_id, "invalid_stage")

    def test_get_production_models(self, registry, mock_model):
        """Test getting production models"""
        # Register multiple models and set different stages
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id1 = registry.register_model(
                model=mock_model,
                model_name="model1",
                model_type="rf",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["f1"],
                training_data_info={},
            )
            model_id2 = registry.register_model(
                model=mock_model,
                model_name="model2",
                model_type="xgb",
                task_type="classification",
                performance_metrics={"accuracy": 0.87},
                feature_names=["f1"],
                training_data_info={},
            )
            model_id3 = registry.register_model(
                model=mock_model,
                model_name="model3",
                model_type="svm",
                task_type="classification",
                performance_metrics={"accuracy": 0.82},
                feature_names=["f1"],
                training_data_info={},
            )

        # Set stages
        registry.set_model_stage(model_id1, "production")
        registry.set_model_stage(model_id2, "production")
        registry.set_model_stage(model_id3, "archived")

        # Get production models
        production_models = registry.get_production_models()

        assert len(production_models) == 2
        production_ids = {model.model_id for model in production_models}
        assert model_id1 in production_ids
        assert model_id2 in production_ids
        assert model_id3 not in production_ids

    def test_compare_models(self, registry, mock_model):
        """Test model comparison functionality"""
        # Register multiple models
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id1 = registry.register_model(
                model=mock_model,
                model_name="model1",
                model_type="rf",
                task_type="classification",
                performance_metrics={"accuracy": 0.85, "f1": 0.82},
                feature_names=["f1"],
                training_data_info={},
            )
            model_id2 = registry.register_model(
                model=mock_model,
                model_name="model2",
                model_type="xgb",
                task_type="classification",
                performance_metrics={"accuracy": 0.87, "f1": 0.84},
                feature_names=["f1"],
                training_data_info={},
            )

        # Compare all models
        comparison_df = registry.compare_models()

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert "model_id" in comparison_df.columns
        assert "accuracy" in comparison_df.columns
        assert "f1" in comparison_df.columns

        # Compare specific models
        specific_comparison = registry.compare_models([model_id1, model_id2])
        assert len(specific_comparison) == 2

        # Test sorting by metric
        sorted_comparison = registry.compare_models(metric="accuracy")
        assert (
            sorted_comparison.iloc[0]["accuracy"]
            >= sorted_comparison.iloc[1]["accuracy"]
        )

    def test_delete_model(self, registry, mock_model, temp_registry_dir):
        """Test model deletion"""
        # Register a model first
        with patch.object(
            registry,
            "_save_model_artifact",
            return_value=f"{temp_registry_dir}/test_model.pkl",
        ):
            # Create actual file for deletion test
            test_file = Path(temp_registry_dir) / "test_model.pkl"
            test_file.touch()

            model_id = registry.register_model(
                model=mock_model,
                model_name="test_model",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["feature1"],
                training_data_info={"samples": 1000},
            )

        # Verify model exists
        assert registry.get_model_metadata(model_id) is not None
        assert test_file.exists()

        # Delete model
        registry.delete_model(model_id, remove_files=True)

        # Verify model is deleted
        assert registry.get_model_metadata(model_id) is None
        assert not test_file.exists()

    def test_archive_old_models(self, registry, mock_model):
        """Test archiving old model versions"""
        # Register multiple versions of the same model
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_ids = []
            for i in range(5):
                model_id = registry.register_model(
                    model=mock_model,
                    model_name="test_model",
                    model_type="random_forest",
                    task_type="classification",
                    performance_metrics={"accuracy": 0.8 + i * 0.01},
                    feature_names=["feature1"],
                    training_data_info={"samples": 1000},
                )
                model_ids.append(model_id)

        # Archive old models, keeping latest 2
        registry.archive_old_models(keep_latest=2)

        # Check that only latest 2 are active
        active_models = registry.list_models(model_name="test_model", is_active=True)
        assert len(active_models) == 2

        # Check that older models are archived
        for model_id in model_ids[:-2]:  # All but last 2
            stage = registry.get_model_stage(model_id)
            assert stage == "archived"

    def test_export_model_metadata(self, registry, mock_model, temp_registry_dir):
        """Test model metadata export"""
        # Register a model
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id = registry.register_model(
                model=mock_model,
                model_name="test_model",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["feature1"],
                training_data_info={"samples": 1000},
            )

        # Export metadata
        export_path = Path(temp_registry_dir) / "export.json"
        registry.export_model_metadata(str(export_path))

        # Verify export file exists and contains data
        assert export_path.exists()

        with open(export_path, "r") as f:
            export_data = json.load(f)

        assert "export_timestamp" in export_data
        assert "total_models" in export_data
        assert "models" in export_data
        assert len(export_data["models"]) == 1
        assert export_data["models"][0]["model_id"] == model_id

    def test_get_registry_stats(self, registry, mock_model):
        """Test registry statistics"""
        # Register models of different types and stages
        with patch.object(
            registry, "_save_model_artifact", return_value="/path/to/model.pkl"
        ):
            model_id1 = registry.register_model(
                model=mock_model,
                model_name="model1",
                model_type="random_forest",
                task_type="classification",
                performance_metrics={"accuracy": 0.85},
                feature_names=["f1"],
                training_data_info={},
            )
            model_id2 = registry.register_model(
                model=mock_model,
                model_name="model2",
                model_type="xgboost",
                task_type="regression",
                performance_metrics={"rmse": 0.15},
                feature_names=["f1"],
                training_data_info={},
            )

        # Set different stages
        registry.set_model_stage(model_id1, "production")
        registry.set_model_stage(model_id2, "staging")

        # Get stats
        stats = registry.get_registry_stats()

        assert isinstance(stats, dict)
        assert "total_models" in stats
        assert "active_models" in stats
        assert "models_by_type" in stats
        assert "models_by_stage" in stats
        assert "registry_path" in stats

        assert stats["total_models"] == 2
        assert stats["active_models"] == 2
        assert stats["models_by_type"]["random_forest"] == 1
        assert stats["models_by_type"]["xgboost"] == 1
        assert stats["models_by_stage"]["production"] == 1
        assert stats["models_by_stage"]["staging"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
