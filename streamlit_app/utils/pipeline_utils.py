"""
Pipeline Utilities

This module provides utility functions for automated model training pipeline,
separated from UI components for better testability and maintainability.

Design Principles:
- Separation of Concerns: UI logic vs Business logic
- Progress Tracking: Real-time updates during pipeline execution
- Error Handling: Robust error management for production use
- Flexibility: Support for various model types and configurations
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Import from src.models.pipeline
    from src.models.pipeline.training_pipeline import (
        ModelTrainingPipeline,
        ModelTrainingConfig,
    )
    from src.models.pipeline.model_registry import ModelRegistry
    from src.config import settings

    # Mark that imports are successful
    IMPORTS_AVAILABLE = True

except ImportError as e:
    # Handle import errors gracefully for testing
    print(f"Import warning in pipeline_utils: {e}")
    IMPORTS_AVAILABLE = False


class PipelineManager:
    """Manager class for automated training pipeline operations"""

    def __init__(self):
        """Initialize pipeline manager"""
        self.logger = logging.getLogger(__name__)
        self.model_registry = None

        # Check if imports are available
        if not IMPORTS_AVAILABLE:
            raise ImportError(
                "Pipeline components are not available. Please check src.models.pipeline imports."
            )

        try:
            self.model_registry = ModelRegistry()
        except Exception as e:
            self.logger.warning(f"Could not initialize model registry: {e}")

    def run_training_pipeline(
        self,
        feature_key: str,
        task_type: str,
        selected_models: List[str],
        training_config: Dict[str, Any],
        session_state: Dict,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline

        Args:
            feature_key: Key to feature data in session state
            task_type: Type of ML task ('classification' or 'regression')
            selected_models: List of model names to train
            training_config: Configuration for training process
            session_state: Streamlit session state
            progress_callback: Function to call for progress updates

        Returns:
            Dictionary containing results of the pipeline
        """

        start_time = time.time()

        try:
            # Step 1: Prepare data
            if progress_callback:
                progress_callback(1, 6, "Preparing data...")

            feature_data, target_column = self._prepare_pipeline_data(
                feature_key, task_type, session_state
            )

            # Step 2: Configure pipeline
            if progress_callback:
                progress_callback(2, 6, "Configuring pipeline...")

            pipeline_config = self._create_pipeline_config(
                selected_models, training_config
            )

            # Step 3: Initialize pipeline
            if progress_callback:
                progress_callback(3, 6, "Initializing pipeline...")

            pipeline = ModelTrainingPipeline(config=pipeline_config)

            # Step 4: Execute training
            if progress_callback:
                progress_callback(4, 6, "Training models...")

            model_results = pipeline.train_all_models(
                data=feature_data,
                target_column=target_column,
                task_type=task_type,
                feature_columns=None,  # Use all features
            )

            # Step 5: Generate comparison
            if progress_callback:
                progress_callback(5, 6, "Generating comparison...")

            comparison_df = pipeline.get_model_comparison()
            best_model_name, best_model = pipeline.get_best_model()

            # Step 6: Finalize results
            if progress_callback:
                progress_callback(6, 6, "Finalizing results...")

            total_time = time.time() - start_time

            # Package results
            results = {
                "model_results": model_results,
                "comparison_df": comparison_df,
                "best_model": {
                    "name": best_model_name,
                    "model": best_model,
                    "test_score": self._extract_best_score(
                        comparison_df, best_model_name
                    ),
                },
                "total_training_time": total_time,
                "pipeline_config": (
                    pipeline_config.__dict__
                    if hasattr(pipeline_config, "__dict__")
                    else {}
                ),
                "feature_key": feature_key,
                "task_type": task_type,
                "timestamp": datetime.now(),
            }

            return results

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def _prepare_pipeline_data(
        self, feature_key: str, task_type: str, session_state: Dict
    ) -> Tuple[pd.DataFrame, str]:
        """
        Prepare data for pipeline execution

        Args:
            feature_key: Key to feature data in session state
            task_type: Type of ML task
            session_state: Streamlit session state

        Returns:
            Tuple of (feature_data, target_column)
        """

        # Get feature data
        if (
            "feature_cache" not in session_state
            or feature_key not in session_state["feature_cache"]
        ):
            raise ValueError(f"Feature key '{feature_key}' not found in cache")

        cached_data = session_state["feature_cache"][feature_key]

        # Handle different cache structures
        if isinstance(cached_data, pd.DataFrame):
            # Direct DataFrame storage (from feature engineering)
            feature_data = cached_data.copy()
        elif isinstance(cached_data, dict) and "features" in cached_data:
            # Dictionary with features key
            feature_data = cached_data["features"].copy()
        else:
            raise ValueError(f"Unexpected feature cache structure for '{feature_key}'")

        # Determine target column based on task type
        target_column = self._determine_target_column(feature_data, task_type)

        # Validate data
        if target_column not in feature_data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in feature data"
            )

        # Clean data
        feature_data = self._clean_pipeline_data(feature_data)

        return feature_data, target_column

    def _determine_target_column(
        self, feature_data: pd.DataFrame, task_type: str
    ) -> str:
        """Determine the appropriate target column based on task type"""

        available_columns = feature_data.columns.tolist()

        # Define potential target columns by task type
        classification_targets = [
            "label_direction",
            "label_breakout",
            "label_trend",
            "binary_label",
            "signal",
        ]

        regression_targets = [
            "label_return",
            "future_return",
            "target_return",
            "return_target",
            "price_change",
        ]

        if task_type == "classification":
            targets = classification_targets
        else:
            targets = regression_targets

        # Find the first available target column
        for target in targets:
            if target in available_columns:
                return target

        # Fallback: create a simple target if none found
        if task_type == "classification":
            # Create binary target based on price direction
            if "close" in available_columns:
                feature_data["label_direction"] = (
                    feature_data["close"].pct_change().shift(-1) > 0
                ).astype(int)
                return "label_direction"
        else:
            # Create return target
            if "close" in available_columns:
                feature_data["label_return"] = (
                    feature_data["close"].pct_change().shift(-1)
                )
                return "label_return"

        raise ValueError(f"No suitable target column found for {task_type} task")

    def _clean_pipeline_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data for pipeline execution"""

        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN in target columns
        target_like_columns = [
            col
            for col in data.columns
            if any(
                target_word in col.lower()
                for target_word in ["label", "target", "return"]
            )
        ]

        if target_like_columns:
            data = data.dropna(subset=target_like_columns)

        # Forward fill remaining NaN values
        data = data.ffill()

        # Drop any remaining NaN rows
        data = data.dropna()

        return data

    def _create_pipeline_config(
        self, selected_models: List[str], training_config: Dict[str, Any]
    ) -> ModelTrainingConfig:
        """Create pipeline configuration from UI settings"""

        config = ModelTrainingConfig(
            models_to_train=selected_models,
            test_size=training_config.get("test_size", 0.2),
            validation_size=training_config.get("validation_size", 0.2),
            time_series_cv=training_config.get("time_series_cv", True),
            cv_splits=training_config.get("cv_splits", 5),
            scaler_type=training_config.get("scaler_type", "standard"),
            feature_selection=True,  # Always enable
            hyperparameter_tuning=training_config.get("hyperparameter_tuning", True),
            ensemble_models=training_config.get("ensemble_models", True),
            save_models=True,  # Always save models
            random_state=training_config.get("random_state", 42),
        )

        return config

    def _extract_best_score(
        self, comparison_df: pd.DataFrame, best_model_name: str
    ) -> float:
        """Extract the best score from comparison dataframe"""

        if comparison_df is None or comparison_df.empty:
            return 0.0

        try:
            # Find the row for the best model
            if "model_name" in comparison_df.columns:
                model_row = comparison_df[
                    comparison_df["model_name"] == best_model_name
                ]
            else:
                # If no model_name column, assume first row is best
                model_row = comparison_df.iloc[[0]]

            if not model_row.empty:
                # Try different score column names
                score_columns = ["test_score", "test_accuracy", "score", "accuracy"]
                for col in score_columns:
                    if col in model_row.columns:
                        return float(model_row[col].iloc[0])

            return 0.0

        except Exception as e:
            self.logger.warning(f"Could not extract best score: {e}")
            return 0.0

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models organized by category"""

        return {
            "traditional": ["random_forest", "xgboost", "svm"],
            "deep_learning": ["lstm", "gru"],
            "advanced": ["transformer"],
        }

    def validate_pipeline_config(
        self,
        feature_key: str,
        task_type: str,
        selected_models: List[str],
        training_config: Dict[str, Any],
        session_state: Dict,
    ) -> Tuple[bool, List[str]]:
        """
        Validate pipeline configuration before execution

        Returns:
            Tuple of (is_valid, error_messages)
        """

        errors = []

        # Check feature data
        if feature_key not in session_state.get("feature_cache", {}):
            errors.append(f"Feature data '{feature_key}' not found")

        # Check models
        available_models = self.get_available_models()
        all_available = []
        for category_models in available_models.values():
            all_available.extend(category_models)

        invalid_models = [
            model for model in selected_models if model not in all_available
        ]
        if invalid_models:
            errors.append(f"Invalid models: {invalid_models}")

        # Check training config
        if not isinstance(training_config.get("test_size"), (int, float)):
            errors.append("Invalid test_size in training config")

        if not isinstance(training_config.get("cv_splits"), int):
            errors.append("Invalid cv_splits in training config")

        return len(errors) == 0, errors

    def get_pipeline_status(self, session_state: Dict) -> Dict[str, Any]:
        """Get current pipeline execution status"""

        return {
            "is_running": session_state.get("pipeline_running", False),
            "has_results": "pipeline_results" in session_state
            and session_state.pipeline_results is not None,
            "last_execution": session_state.get("pipeline_results", {}).get(
                "timestamp", None
            ),
        }

    def cleanup_pipeline_cache(self, session_state: Dict) -> None:
        """Clean up pipeline-related cache data"""

        keys_to_remove = ["pipeline_results", "pipeline_progress"]

        for key in keys_to_remove:
            if key in session_state:
                del session_state[key]

    def export_pipeline_results(
        self, results: Dict[str, Any], export_format: str = "json"
    ) -> str:
        """
        Export pipeline results in specified format

        Args:
            results: Pipeline results dictionary
            export_format: Export format ('json', 'csv', 'pickle')

        Returns:
            Serialized results string
        """

        if export_format == "json":
            import json

            # Create JSON-serializable copy
            export_data = {
                "comparison_df": results.get("comparison_df", pd.DataFrame()).to_dict(),
                "pipeline_config": results.get("pipeline_config", {}),
                "feature_key": results.get("feature_key"),
                "task_type": results.get("task_type"),
                "total_training_time": results.get("total_training_time"),
                "timestamp": results.get("timestamp", datetime.now()).isoformat(),
            }

            return json.dumps(export_data, indent=2)

        elif export_format == "csv":
            comparison_df = results.get("comparison_df")
            if comparison_df is not None and not comparison_df.empty:
                return comparison_df.to_csv(index=False)
            else:
                return "No comparison data available"

        else:
            raise ValueError(f"Unsupported export format: {export_format}")
