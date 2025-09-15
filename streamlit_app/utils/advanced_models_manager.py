#!/usr/bin/env python3
"""
Advanced Models Training Manager
For use in the Advanced Models Lab
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import uuid
import traceback
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")

# Import advanced models
try:
    from src.models.advanced.ensemble import (
        FinancialRandomForest,
        TimeSeriesBagging,
        StackingEnsemble,
        VotingEnsemble,
        EnsembleConfig,
    )
    from src.models.advanced.transformer import (
        TransformerClassifier,
        TransformerRegressor,
        TransformerConfig,
    )
    from src.models.advanced.attention import (
        create_attention_model,
        AttentionVisualizer,
    )
    from src.models.advanced.meta_labeling import (
        MetaLabelingModel,
        TripleBarrierLabeling,
        MetaLabelingConfig,
    )
    from src.models.advanced.interpretation import (
        FinancialModelInterpreter,
        FeatureImportanceAnalyzer,
        SHAPAnalyzer,
        PartialDependenceAnalyzer,
        InterpretationConfig,
    )

    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Advanced models import error: {e}")
    ADVANCED_MODELS_AVAILABLE = False

# Import sklearn models for ensemble base estimators
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AdvancedModelsManager:
    """Advanced models training manager for ensemble, transformer, and specialized models"""

    def __init__(self):
        self.scaler = StandardScaler()

    def train_ensemble_model(
        self,
        feature_key: str,
        ensemble_type: str,
        task_type: str,
        hyperparams: Dict[str, Any],
        training_config: Dict[str, Any],
        session_state: Any,
    ) -> Tuple[bool, str, str]:
        """Train an ensemble model"""

        try:
            # Get and prepare data
            success, message, data = self._prepare_data(
                feature_key, session_state, training_config
            )
            if not success:
                return False, message, None

            X_train, X_test, y_train, y_test = data

            # Convert target for classification tasks
            if task_type == "classification":
                # Convert continuous targets to binary classification
                y_train_binary = (y_train > np.median(y_train)).astype(int)
                y_test_binary = (y_test > np.median(y_test)).astype(int)

                # Store original for potential regression fallback
                y_train_orig = y_train.copy()
                y_test_orig = y_test.copy()

                y_train = y_train_binary
                y_test = y_test_binary

                st.info(
                    f"🔄 Converted continuous target to binary classification: {np.unique(y_train)}"
                )

            # Create ensemble configuration with correct parameter names
            config = EnsembleConfig(
                n_estimators=hyperparams.get("n_estimators", 100),
                max_depth=hyperparams.get("max_depth", None),
                min_samples_split=hyperparams.get("min_samples_split", 2),
                min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
                max_features=hyperparams.get("max_features", "sqrt"),
                bootstrap=hyperparams.get("bootstrap", True),
                voting=hyperparams.get("voting_type", "soft"),
                meta_model=hyperparams.get("meta_model_type", "logistic"),
                time_split_n_splits=hyperparams.get("cv_splits", 5),
                purge_length=hyperparams.get("purge_embargo_periods", 0),
                embargo_length=hyperparams.get("purge_embargo_periods", 0),
                random_state=42,
            )

            # Create model based on ensemble type
            if ensemble_type == "financial_rf":
                model = FinancialRandomForest(config=config, task_type=task_type)
            elif ensemble_type == "bagging":
                if task_type == "classification":
                    base_estimator = RandomForestClassifier(
                        n_estimators=50, random_state=42
                    )
                else:
                    base_estimator = RandomForestRegressor(
                        n_estimators=50, random_state=42
                    )
                model = TimeSeriesBagging(
                    base_estimator=base_estimator, config=config, task_type=task_type
                )
            elif ensemble_type == "stacking":
                # Create base estimators
                if task_type == "classification":
                    base_estimators = [
                        RandomForestClassifier(n_estimators=50, random_state=42),
                        LogisticRegression(random_state=42, max_iter=1000),
                        SVC(probability=True, random_state=42),
                    ]
                else:
                    base_estimators = [
                        RandomForestRegressor(n_estimators=50, random_state=42),
                        LinearRegression(),
                        SVR(),
                    ]
                model = StackingEnsemble(
                    base_estimators=base_estimators, config=config, task_type=task_type
                )
            elif ensemble_type == "voting":
                # Create estimators for voting
                if task_type == "classification":
                    estimators = [
                        (
                            "rf",
                            RandomForestClassifier(n_estimators=50, random_state=42),
                        ),
                        ("lr", LogisticRegression(random_state=42, max_iter=1000)),
                        ("dt", DecisionTreeClassifier(random_state=42)),
                    ]
                else:
                    estimators = [
                        ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
                        ("lr", LinearRegression()),
                        ("dt", DecisionTreeRegressor(random_state=42)),
                    ]
                model = VotingEnsemble(
                    estimators=estimators, config=config, task_type=task_type
                )
            else:
                return False, f"Unknown ensemble type: {ensemble_type}", None

            # Train model
            model.fit(X_train, y_train)

            # Evaluate model
            evaluation_results = self._evaluate_model(
                model, X_train, X_test, y_train, y_test, task_type
            )

            # Generate unique model ID
            model_id = (
                f"{ensemble_type}_{task_type}_{feature_key}_{str(uuid.uuid4())[:8]}"
            )

            # Store results
            self._store_model_results(
                session_state,
                model_id,
                model,
                evaluation_results,
                hyperparams,
                training_config,
                ensemble_type,
                task_type,
            )

            return (
                True,
                f"✅ {ensemble_type.title()} model trained successfully!",
                model_id,
            )

        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            print(f"Debug: {traceback.format_exc()}")
            return False, error_msg, None

    def train_transformer_model(
        self,
        feature_key: str,
        task_type: str,
        hyperparams: Dict[str, Any],
        training_config: Dict[str, Any],
        session_state: Any,
    ) -> Tuple[bool, str, str]:
        """Train a transformer model"""

        try:
            # Get and prepare data
            success, message, data = self._prepare_data(
                feature_key, session_state, training_config
            )
            if not success:
                return False, message, None

            X_train, X_test, y_train, y_test = data

            # Create transformer configuration
            config = TransformerConfig(
                model_type="transformer",
                hyperparameters=hyperparams,
                training_config=training_config,
                validation_config={},
                feature_config={},
            )

            # Set transformer-specific parameters
            config.d_model = hyperparams.get("d_model", 64)
            config.num_heads = hyperparams.get("num_heads", 8)
            config.num_layers = hyperparams.get("num_layers", 4)
            config.sequence_length = hyperparams.get("sequence_length", 60)
            config.dff = hyperparams.get("dff", 256)
            config.dropout_rate = hyperparams.get("dropout_rate", 0.1)
            config.learning_rate = hyperparams.get("learning_rate", 0.001)
            config.batch_size = hyperparams.get("batch_size", 32)
            config.epochs = hyperparams.get("epochs", 100)

            # Create model based on task type
            if task_type == "classification":
                model = TransformerClassifier(config=config)
            else:
                model = TransformerRegressor(config=config)

            # Prepare DataFrame format (transformers expect DataFrame input with target column)
            # Add temporary target column for transformer processing
            target_col_name = "target_temp"

            X_train_df = (
                pd.DataFrame(X_train)
                if not isinstance(X_train, pd.DataFrame)
                else X_train.copy()
            )
            X_test_df = (
                pd.DataFrame(X_test)
                if not isinstance(X_test, pd.DataFrame)
                else X_test.copy()
            )

            # Add target column to DataFrames for transformer processing
            X_train_df[target_col_name] = y_train
            X_test_df[target_col_name] = y_test

            y_train_series = (
                pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
            )
            y_test_series = (
                pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
            )

            # Train model (pass X with target column and separate y for interface compatibility)
            model_results = model.fit(
                X_train_df, y_train_series, validation_data=(X_test_df, y_test_series)
            )

            # For prediction, remove target column
            X_train_pred_df = X_train_df.drop(columns=[target_col_name])
            X_test_pred_df = X_test_df.drop(columns=[target_col_name])

            # Get predictions for evaluation
            y_train_pred = model.predict(X_train_pred_df)
            y_test_pred = model.predict(X_test_pred_df)

            # Create evaluation results
            evaluation_results = {
                "train_metrics": self._calculate_metrics(
                    y_train, y_train_pred, task_type
                ),
                "test_metrics": self._calculate_metrics(y_test, y_test_pred, task_type),
                "training_history": (
                    model_results.training_history
                    if hasattr(model_results, "training_history")
                    else {}
                ),
            }

            # Generate unique model ID
            model_id = f"transformer_{task_type}_{feature_key}_{str(uuid.uuid4())[:8]}"

            # Store results
            self._store_model_results(
                session_state,
                model_id,
                model,
                evaluation_results,
                hyperparams,
                training_config,
                "transformer",
                task_type,
            )

            return True, "✅ Transformer model trained successfully!", model_id

        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            print(f"Debug: {traceback.format_exc()}")
            return False, error_msg, None

    def train_attention_model(
        self,
        feature_key: str,
        task_type: str,
        attention_type: str,
        hyperparams: Dict[str, Any],
        training_config: Dict[str, Any],
        session_state: Any,
    ) -> Tuple[bool, str, str]:
        """Train an attention-based model"""

        try:
            # Get and prepare data
            success, message, data = self._prepare_data(
                feature_key, session_state, training_config
            )
            if not success:
                return False, message, None

            X_train, X_test, y_train, y_test = data

            # Ensure data is properly shaped for attention models
            if len(X_train.shape) == 1:
                X_train = X_train.reshape(-1, 1)
            if len(X_test.shape) == 1:
                X_test = X_test.reshape(-1, 1)

            # Reshape data for attention model (needs 3D input: samples, time_steps, features)
            sequence_length = hyperparams.get("sequence_length", 60)

            # Validate sequence length against data size
            if len(X_train) < sequence_length + 1:
                return (
                    False,
                    f"❌ Insufficient data for sequence length {sequence_length}. Need at least {sequence_length + 1} samples, got {len(X_train)}.",
                    None,
                )

            # Create sequences
            X_train_seq, y_train_seq = self._create_sequences(
                X_train, y_train, sequence_length
            )
            X_test_seq, y_test_seq = self._create_sequences(
                X_test, y_test, sequence_length
            )

            if X_train_seq.shape[0] == 0:
                return (
                    False,
                    "❌ Not enough data to create sequences. Try reducing sequence_length.",
                    None,
                )

            # Validate sequence shapes before model creation
            st.info(
                f"📊 Sequence shapes - X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}"
            )

            # Ensure sequences have the correct 3D shape
            if len(X_train_seq.shape) != 3:
                return (
                    False,
                    f"❌ Invalid sequence shape: expected 3D (samples, time_steps, features), got {X_train_seq.shape}",
                    None,
                )

            # Validate that we have sufficient data for training
            if X_train_seq.shape[0] < hyperparams.get("batch_size", 32):
                st.warning(
                    f"⚠️ Not enough samples ({X_train_seq.shape[0]}) for batch size ({hyperparams.get('batch_size', 32)}). Using smaller batch size."
                )
                # Adjust batch_size to be at most half of available samples
                adjusted_batch_size = max(1, X_train_seq.shape[0] // 2)
                hyperparams = hyperparams.copy()
                hyperparams["batch_size"] = adjusted_batch_size

            # Create attention model
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

            # Determine output dimension based on task type and data
            if task_type == "classification":
                unique_labels = np.unique(y_train_seq)
                output_dim = 1 if len(unique_labels) == 2 else len(unique_labels)
            else:
                output_dim = 1

            from tensorflow import keras

            model = create_attention_model(
                input_shape=input_shape,
                attention_type=attention_type,
                units=hyperparams.get("units", 64),
                num_heads=hyperparams.get("num_heads", 8),
                output_dim=output_dim,
                dropout_rate=hyperparams.get("dropout_rate", 0.1),
            )

            # Compile model
            if task_type == "classification":
                if output_dim == 1:
                    model.compile(
                        optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                else:
                    model.compile(
                        optimizer="adam",
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )
            else:
                model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            # Train model with adjusted parameters
            st.info(f"🔧 Training with batch_size: {hyperparams.get('batch_size', 32)}")
            st.info(
                f"📊 Final training shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}"
            )

            history = model.fit(
                X_train_seq,
                y_train_seq,
                validation_data=(X_test_seq, y_test_seq),
                epochs=hyperparams.get("epochs", 100),
                batch_size=hyperparams.get("batch_size", 32),
                verbose=0,
            )

            # Get predictions
            st.info("🔮 Making predictions...")
            st.info(
                f"📊 Prediction input shapes - X_train: {X_train_seq.shape}, X_test: {X_test_seq.shape}"
            )

            y_train_pred = model.predict(X_train_seq)
            y_test_pred = model.predict(X_test_seq)

            # Flatten predictions for evaluation
            if task_type == "classification" and output_dim == 1:
                y_train_pred = (y_train_pred > 0.5).astype(int).flatten()
                y_test_pred = (y_test_pred > 0.5).astype(int).flatten()
            elif task_type == "classification":
                y_train_pred = np.argmax(y_train_pred, axis=1)
                y_test_pred = np.argmax(y_test_pred, axis=1)
            else:
                y_train_pred = y_train_pred.flatten()
                y_test_pred = y_test_pred.flatten()

            # Create evaluation results
            evaluation_results = {
                "train_metrics": self._calculate_metrics(
                    y_train_seq, y_train_pred, task_type
                ),
                "test_metrics": self._calculate_metrics(
                    y_test_seq, y_test_pred, task_type
                ),
                "training_history": {
                    "loss": history.history.get("loss", []),
                    "val_loss": history.history.get("val_loss", []),
                    "accuracy": history.history.get("accuracy", []),
                    "val_accuracy": history.history.get("val_accuracy", []),
                },
            }

            # Generate unique model ID
            model_id = f"attention_{attention_type}_{task_type}_{feature_key}_{str(uuid.uuid4())[:8]}"

            # Store results
            self._store_model_results(
                session_state,
                model_id,
                model,
                evaluation_results,
                hyperparams,
                training_config,
                f"attention_{attention_type}",
                task_type,
            )

            return (
                True,
                f"✅ {attention_type.title()} attention model trained successfully!",
                model_id,
            )

        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            print(f"Debug: {traceback.format_exc()}")
            return False, error_msg, None

    def analyze_model_interpretation(
        self,
        feature_key: str,
        model_key: str,
        interpretation_type: str,
        session_state: Any,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Perform model interpretation analysis"""

        try:
            # Get trained model
            if (
                not hasattr(session_state, "model_cache")
                or model_key not in session_state.model_cache
            ):
                return False, "❌ Model not found. Please train a model first.", {}

            model_info = session_state.model_cache[model_key]
            model = model_info["model"]

            # Get feature data
            success, message, data = self._prepare_data(feature_key, session_state, {})
            if not success:
                return False, message, {}

            X_train, X_test, y_train, y_test = data

            # Create interpretation configuration with correct parameter names
            config = InterpretationConfig(
                max_features_display=20,
                shap_explainer_type="auto",
                max_shap_samples=min(1000, len(X_test)),
                n_repeats=10,
                random_state=42,
            )

            # Create interpreter
            interpreter = FinancialModelInterpreter(config=config)

            # Get feature names
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

            # Perform analysis
            if interpretation_type == "comprehensive":
                # Determine task type more accurately based on model info
                model_info = session_state.model_cache[model_key]
                actual_task_type = model_info.get("task_type", "classification")

                # Further validate based on target values
                if actual_task_type == "classification":
                    # Ensure targets are properly formatted for classification
                    unique_y_train = np.unique(y_train)
                    unique_y_test = np.unique(y_test)

                    # If targets are continuous, convert to binary
                    if len(unique_y_train) > 10 or np.any(
                        unique_y_train != unique_y_train.astype(int)
                    ):
                        st.warning(
                            "Converting continuous targets to binary for interpretation analysis"
                        )
                        y_train_binary = (y_train > np.median(y_train)).astype(int)
                        y_test_binary = (y_test > np.median(y_test)).astype(int)
                        y_train, y_test = y_train_binary, y_test_binary

                results = interpreter.comprehensive_analysis(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_names=feature_names,
                    task_type=actual_task_type,
                )
            else:
                # Individual analysis components
                results = {}

                # Determine task type and ensure proper target format
                model_info = session_state.model_cache[model_key]
                actual_task_type = model_info.get("task_type", "classification")

                # Validate and convert targets if needed
                if actual_task_type == "classification":
                    unique_y_test = np.unique(y_test)
                    if len(unique_y_test) > 10 or np.any(
                        unique_y_test != unique_y_test.astype(int)
                    ):
                        y_test = (y_test > np.median(y_test)).astype(int)

                if interpretation_type == "feature_importance":
                    importance_analyzer = FeatureImportanceAnalyzer(config)
                    if hasattr(model, "feature_importances_"):
                        results["tree_importance"] = (
                            importance_analyzer.analyze_tree_importance(
                                model, feature_names
                            )
                        )

                    # Use appropriate scoring for task type
                    scoring = (
                        "accuracy" if actual_task_type == "classification" else "r2"
                    )
                    results["permutation_importance"] = (
                        importance_analyzer.analyze_permutation_importance(
                            model, X_test, y_test, feature_names, scoring=scoring
                        )
                    )

                elif interpretation_type == "shap_analysis":
                    shap_analyzer = SHAPAnalyzer(config)
                    shap_analyzer.create_explainer(
                        model, X_train[:100]
                    )  # Use subset for speed
                    results["shap_values"] = shap_analyzer.calculate_shap_values(
                        X_test[:100]
                    )

                elif interpretation_type == "partial_dependence":
                    pd_analyzer = PartialDependenceAnalyzer(config)
                    # Select top features for PD analysis
                    important_features = list(range(min(5, X_train.shape[1])))
                    results["partial_dependence"] = {
                        "features": important_features,
                        "feature_names": [feature_names[i] for i in important_features],
                    }

            return (
                True,
                f"✅ {interpretation_type.title()} analysis completed!",
                results,
            )

        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            print(f"Debug: {traceback.format_exc()}")
            return False, error_msg, {}

    def _prepare_data(
        self, feature_key: str, session_state: Any, training_config: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Tuple]]:
        """Prepare training data"""

        try:
            # Get feature data
            if (
                not hasattr(session_state, "feature_cache")
                or feature_key not in session_state.feature_cache
            ):
                return False, "❌ Feature dataset not found.", None

            dataset_info = session_state.feature_cache[feature_key]

            # Handle different data formats
            if isinstance(dataset_info, pd.DataFrame):
                features_df = dataset_info
            elif isinstance(dataset_info, dict) and "features" in dataset_info:
                features_df = dataset_info["features"]
            else:
                return False, "❌ Invalid dataset format.", None

            # Handle target variable
            target_column = training_config.get("target_column", "target")
            if target_column not in features_df.columns:
                # Try to find a suitable target column
                possible_targets = [
                    "target",
                    "label",
                    "y",
                    "close",
                    "return",
                    "returns",
                    "price",
                    "future_return",
                ]
                target_column = None
                for col in possible_targets:
                    if col in features_df.columns:
                        target_column = col
                        break

                # If still not found, try columns containing return/target keywords
                if target_column is None:
                    for col in features_df.columns:
                        if any(
                            keyword in col.lower()
                            for keyword in ["return", "target", "label", "close"]
                        ):
                            target_column = col
                            break

                # If still not found, try numeric columns (excluding technical indicators)
                if target_column is None:
                    numeric_cols = features_df.select_dtypes(
                        include=[np.number]
                    ).columns
                    # Exclude common technical indicator columns
                    exclude_patterns = [
                        "sma",
                        "ema",
                        "rsi",
                        "macd",
                        "bb",
                        "volume",
                        "ma_",
                    ]
                    candidate_cols = []
                    for col in numeric_cols:
                        if not any(
                            pattern in col.lower() for pattern in exclude_patterns
                        ):
                            candidate_cols.append(col)

                    if candidate_cols:
                        target_column = candidate_cols[
                            0
                        ]  # Take the first suitable numeric column

                if target_column is None:
                    available_cols = list(features_df.columns)
                    return (
                        False,
                        f"❌ No suitable target column found. Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}",
                        None,
                    )

            # Separate features and target
            y = features_df[target_column]
            X = features_df.drop(columns=[target_column])

            # Create target for classification if needed (binary up/down based on returns)
            if "return" in target_column.lower() or "close" in target_column.lower():
                # For classification tasks, convert returns to binary labels (up=1, down=0)
                y_binary = (y > 0).astype(int)
                # Store both continuous and binary versions
                y_continuous = y.copy()
            else:
                y_binary = y.copy()
                y_continuous = y.copy()

            # Handle missing values
            X = X.fillna(method="ffill").fillna(method="bfill")
            y = y.fillna(method="ffill").fillna(method="bfill")
            y_binary = y_binary.fillna(method="ffill").fillna(method="bfill")
            y_continuous = y_continuous.fillna(method="ffill").fillna(method="bfill")

            # Split data
            test_size = training_config.get("test_size", 0.2)
            random_state = training_config.get("random_state", 42)

            # Use appropriate target based on task type (if specified in session)
            # This is a bit of a hack, but we'll check if we're in a classification context
            try:
                # Try to infer task type from the calling context
                # For now, we'll use the continuous target and let the models handle conversion
                final_y = y
            except:
                final_y = y

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                final_y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            return (
                True,
                f"✅ Data prepared successfully. Target column: {target_column}",
                (X_train_scaled, X_test_scaled, y_train.values, y_test.values),
            )

        except Exception as e:
            return False, f"❌ Data preparation failed: {str(e)}", None

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models with proper shape validation"""

        # Ensure inputs are 2D arrays
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        X_seq, y_seq = [], []

        for i in range(len(X) - sequence_length + 1):
            # Create sequence of features
            X_seq.append(X[i : i + sequence_length])
            # Take target at the end of sequence
            y_seq.append(y[i + sequence_length - 1])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Ensure proper 3D shape for sequences: (samples, time_steps, features)
        if len(X_seq.shape) == 2:
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)

        # Ensure y_seq is 1D
        if len(y_seq.shape) > 1:
            y_seq = y_seq.squeeze()

        return X_seq, y_seq

    def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
    ) -> Dict[str, Any]:
        """Evaluate model performance"""

        try:
            # Get predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, task_type)
            test_metrics = self._calculate_metrics(y_test, y_test_pred, task_type)

            return {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }

        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"train_metrics": {}, "test_metrics": {}}

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str
    ) -> Dict[str, float]:
        """Calculate performance metrics"""

        metrics = {}

        try:
            if task_type == "classification":
                # Ensure predictions are integers for classification
                if y_pred.dtype in [np.float32, np.float64, float]:
                    y_pred = y_pred.astype(int)

                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision"] = precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                metrics["recall"] = recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                metrics["f1_score"] = f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
            else:
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["mae"] = mean_absolute_error(y_true, y_pred)
                metrics["r2"] = r2_score(y_true, y_pred)

        except Exception as e:
            print(f"Metrics calculation error: {e}")

        return metrics

    def _store_model_results(
        self,
        session_state: Any,
        model_id: str,
        model: Any,
        evaluation_results: Dict[str, Any],
        hyperparams: Dict[str, Any],
        training_config: Dict[str, Any],
        model_type: str,
        task_type: str,
    ):
        """Store model and results in session state"""

        # Initialize caches if needed
        if not hasattr(session_state, "model_cache"):
            session_state.model_cache = {}
        if not hasattr(session_state, "training_results"):
            session_state.training_results = {}

        # Store model
        session_state.model_cache[model_id] = {
            "model": model,
            "model_type": model_type,
            "task_type": task_type,
            "hyperparams": hyperparams,
            "training_config": training_config,
            "timestamp": datetime.now(),
        }

        # Store results
        session_state.training_results[model_id] = {
            "evaluation_results": evaluation_results,
            "model_type": model_type,
            "task_type": task_type,
            "timestamp": datetime.now(),
        }
