"""
Model Registry for Version Management and Model Tracking

This module provides model versioning, metadata storage, and model lifecycle management
based on MLOps best practices for financial ML systems.
"""

import os
import json
import pickle
import joblib
import sqlite3
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error, r2_score

from ..base import BaseFinancialModel


@dataclass
class ModelMetadata:
    """Metadata for registered models"""

    model_id: str
    model_name: str
    model_type: str
    task_type: str
    version: str
    created_at: datetime
    file_path: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    training_data_info: Dict[str, Any]
    description: str = ""
    tags: List[str] = None
    is_active: bool = True

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary"""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ModelRegistry:
    """
    Model registry for managing trained models, versions, and metadata.

    Features:
    - Model versioning and metadata tracking
    - Model performance comparison
    - Model lifecycle management (staging, production, archived)
    - Model artifact storage and retrieval
    - SQLite database for metadata persistence
    """

    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to store registry database and model artifacts
        """
        self.registry_path = registry_path
        self.models_path = os.path.join(registry_path, "artifacts")
        self.db_path = os.path.join(registry_path, "registry.db")

        # Create directories
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        # Initialize database
        self._init_database()

        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize SQLite database for metadata storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create models table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    performance_metrics TEXT,
                    hyperparameters TEXT,
                    feature_names TEXT,
                    training_data_info TEXT,
                    description TEXT,
                    tags TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """
            )

            # Create model stages table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_stages (
                    model_id TEXT,
                    stage TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """
            )

            conn.commit()

    def register_model(
        self,
        model: BaseFinancialModel,
        model_name: str,
        model_type: str,
        task_type: str,
        performance_metrics: Dict[str, float],
        feature_names: List[str],
        training_data_info: Dict[str, Any],
        hyperparameters: Dict[str, Any] = None,
        description: str = "",
        tags: List[str] = None,
        version: str = None,
    ) -> str:
        """
        Register a new model in the registry.

        Args:
            model: Trained model instance
            model_name: Name of the model
            model_type: Type of model (e.g., "random_forest", "lstm")
            task_type: "classification" or "regression"
            performance_metrics: Model performance metrics
            feature_names: List of feature names used for training
            training_data_info: Information about training data
            hyperparameters: Model hyperparameters
            description: Model description
            tags: List of tags for the model
            version: Model version (auto-generated if None)

        Returns:
            Model ID
        """
        # Generate model ID and version
        if version is None:
            version = self._generate_version(model_name)

        model_id = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save model artifact
        file_path = self._save_model_artifact(model, model_id, model_type)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            task_type=task_type,
            version=version,
            created_at=datetime.now(),
            file_path=file_path,
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters or {},
            feature_names=feature_names,
            training_data_info=training_data_info,
            description=description,
            tags=tags or [],
        )

        # Store in database
        self._store_metadata(metadata)

        # Set as staging stage
        self.set_model_stage(model_id, "staging")

        self.logger.info(f"Model {model_id} registered successfully")

        return model_id

    def _generate_version(self, model_name: str) -> str:
        """Generate version number for model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM models WHERE model_name = ?", (model_name,)
            )
            count = cursor.fetchone()[0]

        return f"v{count + 1:03d}"

    def _save_model_artifact(
        self, model: BaseFinancialModel, model_id: str, model_type: str
    ) -> str:
        """Save model artifact to disk"""
        file_path = os.path.join(self.models_path, f"{model_id}.pkl")

        try:
            if hasattr(model, "save") and model_type in ["lstm", "gru", "transformer"]:
                # Deep learning models with custom save method
                model_dir = os.path.join(self.models_path, model_id)
                os.makedirs(model_dir, exist_ok=True)
                model.save(model_dir)
                file_path = model_dir
            else:
                # Traditional ML models
                joblib.dump(model, file_path)

            return file_path

        except Exception as e:
            self.logger.error(f"Error saving model artifact: {str(e)}")
            raise

    def _store_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO models (
                    model_id, model_name, model_type, task_type, version,
                    created_at, file_path, performance_metrics, hyperparameters,
                    feature_names, training_data_info, description, tags, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.model_id,
                    metadata.model_name,
                    metadata.model_type,
                    metadata.task_type,
                    metadata.version,
                    metadata.created_at.isoformat(),
                    metadata.file_path,
                    json.dumps(metadata.performance_metrics),
                    json.dumps(metadata.hyperparameters),
                    json.dumps(metadata.feature_names),
                    json.dumps(metadata.training_data_info),
                    metadata.description,
                    json.dumps(metadata.tags),
                    metadata.is_active,
                ),
            )

            conn.commit()

    def load_model(self, model_id: str) -> Optional[BaseFinancialModel]:
        """
        Load model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Loaded model instance or None if not found
        """
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            return None

        try:
            if os.path.isdir(metadata.file_path):
                # Deep learning model
                from tensorflow import keras

                return keras.models.load_model(metadata.file_path)
            else:
                # Traditional ML model
                return joblib.load(metadata.file_path)

        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {str(e)}")
            return None

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()

        if not row:
            return None

        # Convert row to dictionary
        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))

        # Parse JSON fields
        data["performance_metrics"] = json.loads(data["performance_metrics"])
        data["hyperparameters"] = json.loads(data["hyperparameters"])
        data["feature_names"] = json.loads(data["feature_names"])
        data["training_data_info"] = json.loads(data["training_data_info"])
        data["tags"] = json.loads(data["tags"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])

        return ModelMetadata(**data)

    def list_models(
        self,
        model_name: str = None,
        model_type: str = None,
        task_type: str = None,
        is_active: bool = True,
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            task_type: Filter by task type
            is_active: Filter by active status

        Returns:
            List of model metadata
        """
        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)

        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)

        if is_active is not None:
            query += " AND is_active = ?"
            params.append(is_active)

        query += " ORDER BY created_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        models = []
        for row in rows:
            data = dict(zip(columns, row))

            # Parse JSON fields
            data["performance_metrics"] = json.loads(data["performance_metrics"])
            data["hyperparameters"] = json.loads(data["hyperparameters"])
            data["feature_names"] = json.loads(data["feature_names"])
            data["training_data_info"] = json.loads(data["training_data_info"])
            data["tags"] = json.loads(data["tags"])
            data["created_at"] = datetime.fromisoformat(data["created_at"])

            models.append(ModelMetadata(**data))

        return models

    def compare_models(
        self, model_ids: List[str] = None, metric: str = "test_accuracy"
    ) -> pd.DataFrame:
        """
        Compare models by performance metrics.

        Args:
            model_ids: List of model IDs to compare (all if None)
            metric: Primary metric for comparison

        Returns:
            DataFrame with model comparison
        """
        if model_ids:
            models = [self.get_model_metadata(mid) for mid in model_ids]
            models = [m for m in models if m is not None]
        else:
            models = self.list_models()

        comparison_data = []
        for model in models:
            row = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "version": model.version,
                "created_at": model.created_at,
            }
            row.update(model.performance_metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by primary metric if available
        if metric in df.columns:
            ascending = metric.lower() in [
                "rmse",
                "mae",
                "mse",
            ]  # Lower is better for these metrics
            df = df.sort_values(metric, ascending=ascending)

        return df

    def set_model_stage(self, model_id: str, stage: str, updated_by: str = "system"):
        """
        Set model stage (staging, production, archived).

        Args:
            model_id: Model identifier
            stage: Model stage ("staging", "production", "archived")
            updated_by: User who updated the stage
        """
        valid_stages = ["staging", "production", "archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_stages (model_id, stage, updated_at, updated_by)
                VALUES (?, ?, ?, ?)
            """,
                (model_id, stage, datetime.now().isoformat(), updated_by),
            )
            conn.commit()

        self.logger.info(f"Model {model_id} stage set to {stage}")

    def get_model_stage(self, model_id: str) -> Optional[str]:
        """Get current stage of a model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT stage FROM model_stages 
                WHERE model_id = ? 
                ORDER BY updated_at DESC 
                LIMIT 1
            """,
                (model_id,),
            )
            row = cursor.fetchone()

        return row[0] if row else None

    def get_production_models(self) -> List[ModelMetadata]:
        """Get all models in production stage"""
        production_model_ids = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT model_id FROM model_stages
                WHERE model_id IN (
                    SELECT model_id FROM model_stages
                    WHERE stage = 'production'
                    AND (model_id, updated_at) IN (
                        SELECT model_id, MAX(updated_at)
                        FROM model_stages
                        GROUP BY model_id
                    )
                )
            """
            )
            production_model_ids = [row[0] for row in cursor.fetchall()]

        return [self.get_model_metadata(mid) for mid in production_model_ids]

    def delete_model(self, model_id: str, remove_files: bool = True):
        """
        Delete model from registry.

        Args:
            model_id: Model identifier
            remove_files: Whether to remove model files from disk
        """
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            self.logger.warning(f"Model {model_id} not found")
            return

        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
            cursor.execute("DELETE FROM model_stages WHERE model_id = ?", (model_id,))
            conn.commit()

        # Remove files if requested
        if remove_files and os.path.exists(metadata.file_path):
            try:
                if os.path.isdir(metadata.file_path):
                    import shutil

                    shutil.rmtree(metadata.file_path)
                else:
                    os.remove(metadata.file_path)

                self.logger.info(f"Model files for {model_id} removed")
            except Exception as e:
                self.logger.error(f"Error removing model files: {str(e)}")

        self.logger.info(f"Model {model_id} deleted from registry")

    def archive_old_models(self, keep_latest: int = 3):
        """
        Archive old model versions, keeping only the latest N versions per model name.

        Args:
            keep_latest: Number of latest versions to keep active
        """
        model_names = set()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT model_name FROM models")
            model_names = {row[0] for row in cursor.fetchall()}

        for model_name in model_names:
            models = self.list_models(model_name=model_name)
            models.sort(key=lambda x: x.created_at, reverse=True)

            # Archive old versions
            for model in models[keep_latest:]:
                self.set_model_stage(model.model_id, "archived")

                # Deactivate model
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE models SET is_active = 0 WHERE model_id = ?",
                        (model.model_id,),
                    )
                    conn.commit()

        self.logger.info(f"Archived old models, keeping latest {keep_latest} versions")

    def export_model_metadata(self, output_path: str):
        """Export all model metadata to JSON file"""
        models = self.list_models(is_active=None)  # Include inactive models

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_models": len(models),
            "models": [model.to_dict() for model in models],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Model metadata exported to {output_path}")

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total models
            cursor.execute("SELECT COUNT(*) FROM models")
            total_models = cursor.fetchone()[0]

            # Active models
            cursor.execute("SELECT COUNT(*) FROM models WHERE is_active = 1")
            active_models = cursor.fetchone()[0]

            # Models by type
            cursor.execute(
                """
                SELECT model_type, COUNT(*) 
                FROM models 
                WHERE is_active = 1 
                GROUP BY model_type
            """
            )
            models_by_type = dict(cursor.fetchall())

            # Models by stage
            cursor.execute(
                """
                SELECT stage, COUNT(DISTINCT model_id) 
                FROM model_stages 
                WHERE (model_id, updated_at) IN (
                    SELECT model_id, MAX(updated_at)
                    FROM model_stages
                    GROUP BY model_id
                )
                GROUP BY stage
            """
            )
            models_by_stage = dict(cursor.fetchall())

        return {
            "total_models": total_models,
            "active_models": active_models,
            "models_by_type": models_by_type,
            "models_by_stage": models_by_stage,
            "registry_path": self.registry_path,
        }
