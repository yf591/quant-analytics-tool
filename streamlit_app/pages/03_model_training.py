#!/usr/bin/env python3
"""
Model Training Page - Complete Phase 3 Week 7-10 Integration
"""

import os
import sys
import traceback
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Enhanced imports with Week 10 Pipeline support
try:
    from src.models.pipeline.training_pipeline import ModelTrainingPipeline
    from src.models.pipeline.model_registry import ModelRegistry

    # ModelTrainingManager is already imported from model_utils
    WEEK10_PIPELINE_AVAILABLE = True
except ImportError:
    WEEK10_PIPELINE_AVAILABLE = False

# Traditional ML Models (Week 7)
try:
    # Import your custom financial ML models
    from src.models.traditional.random_forest import (
        QuantRandomForestClassifier,
        QuantRandomForestRegressor,
    )
    from src.models.traditional.xgboost_model import (
        QuantXGBoostClassifier,
        QuantXGBoostRegressor,
    )
    from src.models.traditional.svm_model import (
        QuantSVMClassifier,
        QuantSVMRegressor,
    )

    # Also import sklearn models for comparison if needed
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    try:
        import xgboost as xgb

        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False

    try:
        import lightgbm as lgb

        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False

    TRADITIONAL_ML_AVAILABLE = True
except ImportError:
    TRADITIONAL_ML_AVAILABLE = False

# Deep Learning Models (Week 8)
try:
    from src.models.deep_learning import (
        QuantLSTMClassifier,
        QuantLSTMRegressor,
        QuantGRUClassifier,
        QuantGRURegressor,
        LSTMDataPreprocessor,
    )

    # Try importing TensorFlow to ensure deep learning is available
    import tensorflow as tf

    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"Deep Learning models not available: {e}")
    DEEP_LEARNING_AVAILABLE = False

# Advanced Models (Week 9)
try:
    from src.models.advanced.ensemble import (
        StackingEnsemble,
        VotingEnsemble,
        FinancialRandomForest,
    )
    from src.models.advanced.attention import (
        AttentionLayer,
        MultiHeadAttention,
        TemporalAttention,
    )
    from src.models.advanced.meta_labeling import (
        MetaLabelingModel,
        TripleBarrierLabeling,
    )
    from src.models.advanced.transformer import (
        TransformerClassifier,
        TransformerRegressor,
    )

    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

# Import UI components
from streamlit_app.components.charts import (
    create_model_performance_chart,
    create_confusion_matrix_chart,
    create_learning_curve_chart,
    create_model_comparison_chart,
)

from streamlit_app.components.data_display import (
    display_model_metrics,
    display_training_progress,
    display_model_comparison,
)

from streamlit_app.components.forms import (
    create_model_selection_form,
    create_hyperparameter_form,
    create_training_config_form,
)

from streamlit_app.utils.model_utils import ModelTrainingManager


# Dummy classes for graceful fallbacks
class DummyModel:
    """Dummy model for unavailable classes"""

    def __init__(self, **kwargs):
        self.name = "DummyModel"

    def fit(self, X, y):
        self.is_fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X))


def generate_test_feature_data():
    """Generate synthetic test feature data for model training demos"""
    try:
        # Create synthetic financial time series data
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        n_samples = len(dates)

        # Generate synthetic price data
        np.random.seed(42)
        price_base = 100
        returns = np.random.normal(0.001, 0.02, n_samples)
        prices = [price_base]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create DataFrame with technical indicators
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        data["volume"] = np.random.lognormal(10, 0.5, n_samples)
        data["high"] = data["close"] * (1 + np.random.uniform(0, 0.03, n_samples))
        data["low"] = data["close"] * (1 - np.random.uniform(0, 0.03, n_samples))
        data["open"] = data["close"].shift(1) * (
            1 + np.random.normal(0, 0.01, n_samples)
        )

        # Technical indicators
        data["sma_20"] = data["close"].rolling(20).mean()
        data["sma_50"] = data["close"].rolling(50).mean()
        data["rsi"] = np.random.uniform(20, 80, n_samples)  # Simplified RSI
        data["macd"] = np.random.normal(0, 0.5, n_samples)
        data["bollinger_upper"] = data["close"] * 1.02
        data["bollinger_lower"] = data["close"] * 0.98
        data["returns"] = data["close"].pct_change()
        data["volatility"] = data["returns"].rolling(20).std()

        # Clean data
        data = data.dropna()

        # Store in session state
        st.session_state.feature_cache["test_data"] = data

        return True

    except Exception as e:
        st.error(f"Failed to generate test data: {e}")
        return False


def main():
    """Main Model Training Page"""
    st.title("🤖 Model Training")
    st.markdown(
        "Train and evaluate machine learning models with comprehensive Phase 3 Week 7-10 integration"
    )

    # Initialize session state
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}
    if "training_history" not in st.session_state:
        st.session_state.training_history = []
    if "feature_cache" not in st.session_state:
        st.session_state.feature_cache = {}
    if "model_registry" not in st.session_state:
        try:
            if WEEK10_PIPELINE_AVAILABLE:
                st.session_state.model_registry = ModelRegistry()
            else:
                st.session_state.model_registry = None
        except Exception:
            st.session_state.model_registry = None

    # Main interface
    show_model_training()


def show_model_training():
    """Display the main model training interface"""

    # Check for feature data
    if not st.session_state.feature_cache:
        st.warning(
            "⚠️ No feature data available. Please generate features first on the Feature Engineering page."
        )

        # Add test data option
        if st.button("🧪 Generate Test Data for Demo"):
            generate_test_feature_data()
            st.success("✅ Test feature data generated!")
            st.rerun()

        return

    # Display available advanced capabilities
    with st.expander("🔧 Advanced Capabilities", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("**Week 10 Pipeline:**")
            if WEEK10_PIPELINE_AVAILABLE:
                st.success("✅ Available")
                st.caption("• Training Pipeline\n• Model Registry\n• Deployment")
            else:
                st.warning("❌ Not Available")
                st.caption("Missing: pipeline modules")

        with col2:
            st.write("**Traditional ML:**")
            if TRADITIONAL_ML_AVAILABLE:
                st.success("✅ Available")
                models_list = ["• Random Forest", "• SVM", "• Decision Tree"]
                if XGBOOST_AVAILABLE:
                    models_list.append("• XGBoost")
                if LIGHTGBM_AVAILABLE:
                    models_list.append("• LightGBM")
                st.caption("\n".join(models_list))
            else:
                st.warning("❌ Not Available")
                st.caption("Missing: scikit-learn")

        with col3:
            st.write("**Deep Learning:**")
            if DEEP_LEARNING_AVAILABLE:
                st.success("✅ Available")
                st.caption("• LSTM Models\n• GRU Models\n• TensorFlow Backend")
            else:
                st.warning("❌ Not Available")
                st.caption("Missing: deep learning modules")

        with col4:
            st.write("**Advanced Models:**")
            if ADVANCED_MODELS_AVAILABLE:
                st.success("✅ Available")
                st.caption("• Ensemble Methods\n• Attention Models\n• Transformers")
            else:
                st.warning("❌ Not Available")
                st.caption("Missing: advanced modules")

    # Feature selection
    st.subheader("📊 Feature Data Selection")

    feature_options = list(st.session_state.feature_cache.keys())
    if not feature_options:
        st.error("No feature data available. Please generate features first.")
        return

    selected_feature_key = st.selectbox(
        "Select Feature Set",
        feature_options,
        help="Choose the feature set to use for training",
    )

    # Display feature info
    feature_data = st.session_state.feature_cache[selected_feature_key]
    if isinstance(feature_data, dict) and "config" in feature_data:
        config = feature_data["config"]
        st.info(
            f"📈 Feature Set: {config.get('name', 'Unknown')} | "
            f"Indicators: {len(config.get('indicators', []))} | "
            f"Window: {config.get('window_size', 'N/A')}"
        )
    elif isinstance(feature_data, pd.DataFrame):
        st.info(
            f"📈 Feature DataFrame: {feature_data.shape[0]} samples, {feature_data.shape[1]} features"
        )

    # Model selection and configuration
    st.subheader("🤖 Model Configuration")

    # Model category selection
    model_category = st.selectbox(
        "Model Category",
        ["Traditional ML", "Deep Learning", "Advanced Models", "Attention Models"],
        help="Select the category of machine learning model",
    )

    # Model selection based on category
    model_config = get_model_selection(model_category)

    if not model_config:
        st.error("Please select a valid model.")
        return

    # Task type selection
    task_type = st.selectbox(
        "Task Type",
        ["Classification", "Regression"],
        help="Select the type of machine learning task",
    )

    model_config["task_type"] = task_type

    # Hyperparameter configuration
    st.subheader("⚙️ Hyperparameter Configuration")

    # Get default hyperparameters
    # Initialize model training manager
    if "model_training_manager" not in st.session_state:
        st.session_state.model_training_manager = ModelTrainingManager()

    model_training_manager = st.session_state.model_training_manager

    # Get default hyperparameters
    default_hyperparams = model_training_manager.get_default_hyperparams(
        model_config["model_class"]
    )

    # Create hyperparameter form
    hyperparams = create_hyperparameter_form(
        model_config["model_class"], default_hyperparams
    )

    # Training configuration
    st.subheader("🎯 Training Configuration")

    training_config = create_training_config_form()

    # Training button and process
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            train_model(
                selected_feature_key, model_config, hyperparams, training_config
            )

    # Display training status
    if "current_training" in st.session_state:
        display_training_status()

    # Training history and model management
    display_training_history()
    display_model_management()


def get_model_selection(category: str) -> Dict:
    """Get model selection based on category"""

    model_config = {}

    if category == "Traditional ML":
        # Week 7 Traditional ML Models - Use your custom financial ML models
        model_options = {}

        if TRADITIONAL_ML_AVAILABLE:
            model_options.update(
                {
                    "Quant Random Forest Classifier": "QuantRandomForestClassifier",
                    "Quant Random Forest Regressor": "QuantRandomForestRegressor",
                    "Quant SVM Classifier": "QuantSVMClassifier",
                    "Quant SVM Regressor": "QuantSVMRegressor",
                    # Also include basic sklearn models for comparison
                    "Random Forest": "RandomForestClassifier",
                    "Gradient Boosting": "GradientBoostingClassifier",
                    "SVM": "SVC",
                    "Logistic Regression": "LogisticRegression",
                    "Decision Tree": "DecisionTreeClassifier",
                }
            )

        if XGBOOST_AVAILABLE:
            model_options.update(
                {
                    "Quant XGBoost Classifier": "QuantXGBoostClassifier",
                    "Quant XGBoost Regressor": "QuantXGBoostRegressor",
                    "XGBoost": "XGBClassifier",
                }
            )

        if LIGHTGBM_AVAILABLE:
            model_options["LightGBM"] = "LGBMClassifier"

        if not model_options:
            st.error("No traditional ML models available. Please install scikit-learn.")
            return {}

        selected_model = st.selectbox(
            "Select Traditional ML Model", list(model_options.keys())
        )
        model_config = {
            "model_type": selected_model,
            "model_class": model_options[selected_model],
            "category": "Traditional ML",
        }

    elif category == "Deep Learning":
        # Week 8 Deep Learning Models
        if not DEEP_LEARNING_AVAILABLE:
            st.error(
                "Deep learning models not available. Please implement Week 8 models."
            )
            return {}

        model_options = {
            "LSTM Classifier": "QuantLSTMClassifier",
            "LSTM Regressor": "QuantLSTMRegressor",
            "GRU Classifier": "QuantGRUClassifier",
            "GRU Regressor": "QuantGRURegressor",
        }

        selected_model = st.selectbox(
            "Select Deep Learning Model", list(model_options.keys())
        )
        model_config = {
            "model_type": selected_model,
            "model_class": model_options[selected_model],
            "category": "Deep Learning",
        }

    elif category == "Advanced Models":
        # Week 9 Advanced Models
        if not ADVANCED_MODELS_AVAILABLE:
            st.error("Advanced models not available. Please implement Week 9 models.")
            return {}

        model_options = {
            "Stacking Ensemble": "StackingEnsemble",
            "Voting Ensemble": "VotingEnsemble",
            "Financial Random Forest": "FinancialRandomForest",
            "Meta Labeling": "MetaLabelingModel",
            "Transformer Classifier": "TransformerClassifier",
            "Transformer Regressor": "TransformerRegressor",
        }

        selected_model = st.selectbox(
            "Select Advanced Model", list(model_options.keys())
        )
        model_config = {
            "model_type": selected_model,
            "model_class": model_options[selected_model],
            "category": "Advanced Models",
        }

    elif category == "Attention Models":
        # Week 9 Attention Models
        if not ADVANCED_MODELS_AVAILABLE:
            st.error("Attention models not available. Please implement Week 9 models.")
            return {}

        model_options = {
            "Attention Layer": "AttentionLayer",
            "Multi-Head Attention": "MultiHeadAttention",
            "Temporal Attention": "TemporalAttention",
        }

        selected_model = st.selectbox(
            "Select Attention Model", list(model_options.keys())
        )
        model_config = {
            "model_type": selected_model,
            "model_class": model_options[selected_model],
            "category": "Attention Models",
        }

    return model_config


def train_model(
    feature_key: str, model_config: Dict, hyperparams: Dict, training_config: Dict
):
    """Train a model with the specified configuration - Enhanced with Week 10 Pipeline"""

    try:
        # Initialize Week 10 Training Pipeline
        try:
            if WEEK10_PIPELINE_AVAILABLE:
                training_pipeline = ModelTrainingPipeline()
                model_registry = st.session_state.model_registry
                model_manager = ModelTrainingManager()
                st.info("🔧 Using advanced Week 10 Training Pipeline")
            else:
                training_pipeline = None
                model_registry = st.session_state.model_registry
                model_manager = None
                st.info("📊 Using basic training pipeline")
        except Exception as e:
            st.warning(f"Advanced pipeline not available: {e}. Using basic training.")
            training_pipeline = None
            model_registry = st.session_state.model_registry
            model_manager = None

        # Get feature data
        feature_data = st.session_state.feature_cache[feature_key]

        # Handle different feature data types
        if isinstance(feature_data, dict):
            # Check if it's old format dict with actual data or just config
            if "features" in feature_data and "data" in feature_data:
                # Old format: Convert to DataFrame
                features_dict = feature_data["features"]
                original_data = feature_data["data"]

                feature_df = pd.DataFrame(index=original_data.index)
                for name, values in features_dict.items():
                    if isinstance(values, pd.Series):
                        feature_df[name] = values

                if len(feature_df.columns) == 0:
                    st.error(
                        "⚠️ No valid features found in old format data. Please regenerate features."
                    )
                    return

                feature_data = feature_df
            else:
                # Configuration dict
                st.error(
                    "⚠️ Feature data is configuration only. Please generate actual features first."
                )
                return
        elif not isinstance(feature_data, pd.DataFrame):
            st.error("⚠️ Unexpected feature data format. Please regenerate features.")
            return

        # Start training process
        training_start = datetime.now()
        model_id = (
            f"{model_config['model_class']}_{training_start.strftime('%Y%m%d_%H%M%S')}"
        )

        st.session_state.current_training = {
            "status": "initializing",
            "start_time": training_start,
            "model_id": model_id,
            "model_config": model_config,
            "hyperparams": hyperparams,
            "training_config": training_config,
        }

        # Create progress containers
        progress_container = st.container()

        with progress_container:
            st.info(f"🚀 Starting training for {model_config['model_type']} model...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Data preparation
            status_text.text("📊 Preparing data...")
            progress_bar.progress(0.1)
            time.sleep(0.5)

            # Enhanced data preparation
            if len(feature_data.columns) == 0:
                st.error("No features available for training.")
                return

            # Week 10 Pipeline: Advanced target creation
            if training_pipeline and model_manager:
                # Use ModelTrainingManager for sophisticated data preparation
                try:
                    X, y = model_manager._prepare_training_data(
                        feature_data, training_config
                    )
                    if X is None or y is None:
                        st.error(
                            "Failed to prepare training data using advanced pipeline."
                        )
                        return

                    st.success(
                        f"✅ Advanced data preparation completed: {X.shape[0]} samples, {X.shape[1]} features"
                    )

                except Exception as e:
                    st.warning(
                        f"Advanced data preparation failed: {e}. Using basic method."
                    )
                    # Fallback to basic method
                    X, y = _basic_data_preparation(
                        feature_data, model_config, training_config
                    )
            else:
                # Basic data preparation
                X, y = _basic_data_preparation(
                    feature_data, model_config, training_config
                )

            if X is None or y is None:
                st.error("Data preparation failed.")
                return

            # Step 2: Data splitting
            status_text.text("🔀 Splitting data...")
            progress_bar.progress(0.3)
            time.sleep(0.5)

            # Week 10 Pipeline: Advanced data splitting
            if training_pipeline and model_manager:
                try:
                    train_test_split = model_manager._split_data(X, y, training_config)
                    if train_test_split is None:
                        st.error("Failed to split data using advanced pipeline.")
                        return
                    X_train, X_test, y_train, y_test = train_test_split
                    st.success(
                        f"✅ Advanced data splitting: Train={len(X_train)}, Test={len(X_test)}"
                    )
                except Exception as e:
                    st.warning(
                        f"Advanced data splitting failed: {e}. Using basic method."
                    )
                    # Fallback to basic splitting
                    X_train, X_test, y_train, y_test = _basic_data_splitting(
                        X, y, training_config
                    )
            else:
                # Basic data splitting
                X_train, X_test, y_train, y_test = _basic_data_splitting(
                    X, y, training_config
                )

            # Step 3: Model creation
            status_text.text("🤖 Creating model...")
            progress_bar.progress(0.5)
            time.sleep(0.5)

            # Week 10 Pipeline: Advanced model creation
            if training_pipeline and model_manager:
                try:
                    model = model_manager._get_model_instance(
                        model_config["model_class"],
                        model_config["task_type"],
                        hyperparams,
                    )
                    if model is None:
                        st.error("Failed to create model using advanced pipeline.")
                        return
                    st.success(
                        f"✅ Advanced model creation: {model_config['model_class']}"
                    )
                except Exception as e:
                    st.warning(
                        f"Advanced model creation failed: {e}. Using basic method."
                    )
                    # Fallback to basic method
                    model = get_model_instance(
                        model_config["model_class"],
                        model_config["task_type"],
                        hyperparams,
                    )
            else:
                # Basic model creation
                model = get_model_instance(
                    model_config["model_class"], model_config["task_type"], hyperparams
                )

            if model is None:
                st.error("Failed to create model.")
                return

            # Step 4: Model training
            status_text.text("🎯 Training model...")
            progress_bar.progress(0.7)

            # Update training status
            st.session_state.current_training["status"] = "training"

            # Train the model
            try:
                model.fit(X_train, y_train)
                st.success(f"✅ Model training completed successfully!")
            except Exception as e:
                st.error(f"Model training failed: {e}")
                st.error(f"Details: {traceback.format_exc()}")
                return

            # Step 5: Model evaluation
            status_text.text("📈 Evaluating model...")
            progress_bar.progress(0.9)

            # Week 10 Pipeline: Advanced evaluation
            if training_pipeline and model_manager:
                try:
                    evaluation = model_manager._evaluate_model(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        model_config["task_type"],
                    )
                    st.success(f"✅ Advanced evaluation completed")
                except Exception as e:
                    st.warning(f"Advanced evaluation failed: {e}. Using basic method.")
                    evaluation = _basic_model_evaluation(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        model_config["task_type"],
                    )
            else:
                # Basic evaluation
                evaluation = _basic_model_evaluation(
                    model, X_train, X_test, y_train, y_test, model_config["task_type"]
                )

            # Calculate feature importance
            feature_importance = None
            if hasattr(model, "feature_importances_"):
                feature_importance = pd.Series(
                    model.feature_importances_,
                    index=(
                        X.columns
                        if hasattr(X, "columns")
                        else [f"feature_{i}" for i in range(X.shape[1])]
                    ),
                )

            # Training completion
            training_end = datetime.now()
            training_duration = training_end - training_start

            # Step 6: Model storage and registry
            status_text.text("💾 Storing model...")
            progress_bar.progress(1.0)

            # Store model results
            model_info = {
                "model": model,
                "model_id": model_id,
                "name": f"{model_config['model_type']} ({model_config['task_type']})",
                "model_class": model_config["model_class"],
                "model_type": model_config["model_type"],
                "task_type": model_config["task_type"],
                "hyperparameters": hyperparams,
                "training_config": training_config,
                "feature_key": feature_key,
                "evaluation": evaluation,
                "feature_importance": feature_importance,
                "training_time": training_duration.total_seconds(),
                "trained_at": training_start,
                "data_shape": X.shape,
                "test_data": {"X_test": X_test, "y_test": y_test},
                "status": "completed",
            }

            # Store in cache
            st.session_state.model_cache[model_id] = model_info

            # Add to training history
            st.session_state.training_history.append(
                {
                    "model_id": model_id,
                    "model_class": model_config["model_class"],
                    "model_type": model_config["model_type"],
                    "task_type": model_config["task_type"],
                    "training_score": evaluation.get("train_score", 0),
                    "test_score": evaluation.get("test_score", 0),
                    "trained_at": training_start,
                    "duration": training_duration.total_seconds(),
                }
            )

            # Week 10 Pipeline: Advanced model registry
            try:
                if model_registry and hasattr(model_registry, "register_model"):
                    model_registry.register_model(
                        model=model,
                        model_name=model_config["model_type"],
                        model_type=model_config["model_class"],
                        task_type=model_config["task_type"],
                        performance_metrics=evaluation,
                        feature_names=(
                            list(X.columns)
                            if hasattr(X, "columns")
                            else [f"feature_{i}" for i in range(X.shape[1])]
                        ),
                        training_data_info={
                            "n_samples": X.shape[0],
                            "n_features": X.shape[1],
                            "feature_key": feature_key,
                        },
                        hyperparameters=hyperparams,
                        description=f"Trained {model_config['model_type']} model using Week 10 pipeline",
                    )
                    st.success("✅ Model registered in advanced registry")
            except Exception as e:
                st.warning(f"Could not register model in advanced registry: {e}")

            # Clear current training status
            if "current_training" in st.session_state:
                del st.session_state.current_training

            st.success(f"🎉 Model training completed successfully!")
            st.balloons()

            # Display results summary
            with st.expander("📊 Training Results Summary", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Training Time", f"{training_duration.total_seconds():.2f}s"
                    )

                with col2:
                    if model_config["task_type"] == "Classification":
                        score_label = "Accuracy"
                        score_value = evaluation.get("test_score", 0)
                    else:
                        score_label = "R² Score"
                        score_value = evaluation.get("test_score", 0)
                    st.metric(score_label, f"{score_value:.4f}")

                with col3:
                    st.metric("Data Points", f"{X.shape[0]:,}")

                with col4:
                    st.metric("Features", X.shape[1])

                # Performance details
                if evaluation:
                    st.subheader("📈 Performance Metrics")

                    if model_config["task_type"] == "Classification":
                        if "accuracy" in evaluation:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Train Accuracy",
                                    f"{evaluation.get('train_score', 0):.4f}",
                                )
                            with col2:
                                st.metric(
                                    "Test Accuracy",
                                    f"{evaluation.get('test_score', 0):.4f}",
                                )
                            with col3:
                                st.metric(
                                    "Metric Type",
                                    evaluation.get("metric_type", "accuracy"),
                                )
                    else:
                        if "r2_score" in evaluation:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Train R²",
                                    f"{evaluation.get('train_score', 0):.4f}",
                                )
                            with col2:
                                st.metric(
                                    "Test R²", f"{evaluation.get('test_score', 0):.4f}"
                                )
                            with col3:
                                st.metric(
                                    "Metric Type",
                                    evaluation.get("metric_type", "r2_score"),
                                )

                # Feature importance
                if feature_importance is not None and len(feature_importance) > 0:
                    st.subheader("🎯 Feature Importance")
                    top_features = feature_importance.nlargest(10)

                    fig = go.Figure(
                        go.Bar(
                            x=top_features.values,
                            y=top_features.index,
                            orientation="h",
                            marker_color="lightblue",
                        )
                    )

                    fig.update_layout(
                        title="Top 10 Feature Importance",
                        xaxis_title="Importance Score",
                        height=400,
                        template="plotly_white",
                    )

                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in training process: {e}")
        st.error(f"Details: {traceback.format_exc()}")

    finally:
        # Clean up current training status
        if "current_training" in st.session_state:
            del st.session_state.current_training


def _basic_data_preparation(
    feature_data: pd.DataFrame, model_config: Dict, training_config: Dict
):
    """Basic data preparation fallback method"""
    try:
        # Simple target creation
        price_col = None
        for col in feature_data.columns:
            if "close" in col.lower() or "price" in col.lower():
                price_col = col
                break

        if price_col is None:
            # Use first numeric column as proxy price
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None, None
            price_col = numeric_cols[0]

        # Create target variable (next period return)
        target = feature_data[price_col].pct_change().shift(-1)

        # For classification task, convert to binary signals
        if model_config["task_type"] == "Classification":
            target = (target > 0).astype(int)

        # Features: exclude price columns
        feature_cols = [col for col in feature_data.columns if col != price_col]
        if len(feature_cols) == 0:
            feature_cols = feature_data.columns.tolist()

        X = feature_data[feature_cols]
        y = target

        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 10:
            return None, None

        return X, y

    except Exception as e:
        st.error(f"Basic data preparation failed: {e}")
        return None, None


def _basic_data_splitting(X, y, training_config: Dict):
    """Basic data splitting fallback method"""
    try:
        from sklearn.model_selection import train_test_split

        test_size = training_config.get("test_size", 0.2)
        random_state = training_config.get("random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        st.error(f"Basic data splitting failed: {e}")
        return None, None, None, None


def _basic_model_evaluation(model, X_train, X_test, y_train, y_test, task_type: str):
    """Basic model evaluation fallback method"""
    try:
        from sklearn.metrics import accuracy_score, r2_score

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results = {}

        if task_type == "Classification":
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            results = {
                "train_score": train_score,
                "test_score": test_score,
                "metric_type": "accuracy",
            }
        else:
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            results = {
                "train_score": train_score,
                "test_score": test_score,
                "metric_type": "r2_score",
            }

        return results

    except Exception as e:
        st.error(f"Basic model evaluation failed: {e}")
        return {"train_score": 0, "test_score": 0, "metric_type": "unknown"}


def get_model_instance(model_class: str, task_type: str, hyperparams: Dict):
    """Create model instance with proper error handling and Phase 3 support"""

    try:
        # Custom Financial ML Models (Week 7)
        if TRADITIONAL_ML_AVAILABLE:
            # Quant Random Forest Models
            if model_class == "QuantRandomForestClassifier":
                return QuantRandomForestClassifier(**hyperparams)
            elif model_class == "QuantRandomForestRegressor":
                return QuantRandomForestRegressor(**hyperparams)

            # Quant XGBoost Models
            elif model_class == "QuantXGBoostClassifier":
                return QuantXGBoostClassifier(**hyperparams)
            elif model_class == "QuantXGBoostRegressor":
                return QuantXGBoostRegressor(**hyperparams)

            # Quant SVM Models
            elif model_class == "QuantSVMClassifier":
                return QuantSVMClassifier(**hyperparams)
            elif model_class == "QuantSVMRegressor":
                return QuantSVMRegressor(**hyperparams)

            # Basic sklearn models for comparison
            elif (
                model_class == "RandomForestClassifier"
                and task_type == "Classification"
            ):
                return RandomForestClassifier(**hyperparams)
            elif model_class == "RandomForestRegressor" and task_type == "Regression":
                return RandomForestRegressor(**hyperparams)
            elif (
                model_class == "GradientBoostingClassifier"
                and task_type == "Classification"
            ):
                return GradientBoostingClassifier(**hyperparams)
            elif (
                model_class == "GradientBoostingRegressor" and task_type == "Regression"
            ):
                return GradientBoostingRegressor(**hyperparams)
            elif model_class == "SVC" and task_type == "Classification":
                return SVC(**hyperparams)
            elif model_class == "SVR" and task_type == "Regression":
                return SVR(**hyperparams)
            elif model_class == "LogisticRegression" and task_type == "Classification":
                return LogisticRegression(**hyperparams)
            elif model_class == "LinearRegression" and task_type == "Regression":
                return LinearRegression(**hyperparams)
            elif (
                model_class == "DecisionTreeClassifier"
                and task_type == "Classification"
            ):
                return DecisionTreeClassifier(**hyperparams)
            elif model_class == "DecisionTreeRegressor" and task_type == "Regression":
                return DecisionTreeRegressor(**hyperparams)

        # XGBoost Models (both custom and standard)
        if XGBOOST_AVAILABLE:
            if model_class == "XGBClassifier" and task_type == "Classification":
                return xgb.XGBClassifier(**hyperparams)
            elif model_class == "XGBRegressor" and task_type == "Regression":
                return xgb.XGBRegressor(**hyperparams)

        # LightGBM Models
        if LIGHTGBM_AVAILABLE:
            if model_class == "LGBMClassifier" and task_type == "Classification":
                return lgb.LGBMClassifier(**hyperparams)
            elif model_class == "LGBMRegressor" and task_type == "Regression":
                return lgb.LGBMRegressor(**hyperparams)

        # Deep Learning Models (Week 8)
        if DEEP_LEARNING_AVAILABLE:
            if model_class == "QuantLSTMClassifier":
                return QuantLSTMClassifier(**hyperparams)
            elif model_class == "QuantLSTMRegressor":
                return QuantLSTMRegressor(**hyperparams)
            elif model_class == "QuantGRUClassifier":
                return QuantGRUClassifier(**hyperparams)
            elif model_class == "QuantGRURegressor":
                return QuantGRURegressor(**hyperparams)

        # Advanced Models (Week 9)
        if ADVANCED_MODELS_AVAILABLE:
            if model_class == "StackingEnsemble":
                return StackingEnsemble(**hyperparams)
            elif model_class == "VotingEnsemble":
                return VotingEnsemble(**hyperparams)
            elif model_class == "FinancialRandomForest":
                return FinancialRandomForest(**hyperparams)
            elif model_class == "MetaLabelingModel":
                return MetaLabelingModel(**hyperparams)
            elif model_class == "TransformerClassifier":
                return TransformerClassifier(**hyperparams)
            elif model_class == "TransformerRegressor":
                return TransformerRegressor(**hyperparams)
            elif model_class == "AttentionLayer":
                return AttentionLayer(**hyperparams)
            elif model_class == "MultiHeadAttention":
                return MultiHeadAttention(**hyperparams)
            elif model_class == "TemporalAttention":
                return TemporalAttention(**hyperparams)

        # Fallback: Return dummy model
        st.warning(f"Model class {model_class} not available. Using dummy model.")
        return DummyModel(**hyperparams)

    except Exception as e:
        st.error(f"Error creating model instance for {model_class}: {e}")
        st.error(f"Details: {traceback.format_exc()}")
        return None


def display_training_status():
    """Display current training status"""

    if "current_training" not in st.session_state:
        return

    training_info = st.session_state.current_training

    st.subheader("🔄 Training in Progress")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", training_info["model_config"]["model_type"])

    with col2:
        st.metric("Status", training_info["status"].title())

    with col3:
        elapsed = datetime.now() - training_info["start_time"]
        st.metric("Elapsed Time", f"{elapsed.total_seconds():.1f}s")

    # Progress indicator
    if training_info["status"] == "training":
        st.info("🎯 Model training in progress...")


def display_training_history():
    """Display training history"""

    if not st.session_state.training_history:
        return

    st.subheader("📜 Training History")

    # Create history DataFrame
    history_df = pd.DataFrame(st.session_state.training_history)

    # Sort by training date
    history_df = history_df.sort_values("trained_at", ascending=False)

    # Format for display
    display_df = history_df.copy()
    display_df["trained_at"] = display_df["trained_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["duration"] = display_df["duration"].round(2)

    # Display table
    st.dataframe(
        display_df[
            [
                "model_type",
                "task_type",
                "training_score",
                "test_score",
                "duration",
                "trained_at",
            ]
        ],
        use_container_width=True,
    )


def display_model_management():
    """Display model management interface"""

    if not st.session_state.model_cache:
        return

    st.subheader("🗂️ Model Management")

    # Model selection
    model_options = {
        info["name"]: model_id
        for model_id, info in st.session_state.model_cache.items()
    }

    selected_model_name = st.selectbox(
        "Select Model for Analysis", list(model_options.keys())
    )

    if selected_model_name:
        model_id = model_options[selected_model_name]
        model_info = st.session_state.model_cache[model_id]

        # Display model details
        with st.expander(f"📊 {selected_model_name} Details", expanded=True):

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model Type", model_info["model_type"])

            with col2:
                st.metric("Task Type", model_info["task_type"])

            with col3:
                st.metric("Training Time", f"{model_info['training_time']:.2f}s")

            with col4:
                test_score = model_info["evaluation"].get("test_score", 0)
                st.metric("Test Score", f"{test_score:.4f}")

            # Model actions
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(f"📈 View Performance", key=f"perf_{model_id}"):
                    display_model_performance(model_info)

            with col2:
                if st.button(f"📋 View Details", key=f"details_{model_id}"):
                    display_model_details(model_info)

            with col3:
                if st.button(f"🗑️ Delete Model", key=f"delete_{model_id}"):
                    del st.session_state.model_cache[model_id]
                    st.success("Model deleted successfully!")
                    st.rerun()


def display_model_performance(model_info: Dict):
    """Display detailed model performance"""

    st.subheader(f"📈 Performance Analysis: {model_info['name']}")

    evaluation = model_info["evaluation"]

    # Performance metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Score", f"{evaluation.get('train_score', 0):.4f}")

    with col2:
        st.metric("Test Score", f"{evaluation.get('test_score', 0):.4f}")

    # Feature importance if available
    if model_info.get("feature_importance") is not None:
        st.subheader("🎯 Feature Importance")

        importance = model_info["feature_importance"]
        top_features = importance.nlargest(15)

        fig = go.Figure(
            go.Bar(
                x=top_features.values,
                y=top_features.index,
                orientation="h",
                marker_color="lightcoral",
            )
        )

        fig.update_layout(
            title="Top 15 Feature Importance",
            xaxis_title="Importance Score",
            height=500,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)


def display_model_details(model_info: Dict):
    """Display detailed model information"""

    st.subheader(f"📋 Model Details: {model_info['name']}")

    # Basic information
    st.write("**Basic Information:**")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"- **Model ID:** {model_info['model_id']}")
        st.write(f"- **Model Class:** {model_info['model_class']}")
        st.write(f"- **Task Type:** {model_info['task_type']}")

    with col2:
        st.write(f"- **Training Time:** {model_info['training_time']:.2f}s")
        st.write(f"- **Data Shape:** {model_info['data_shape']}")
        st.write(f"- **Status:** {model_info['status']}")

    # Hyperparameters
    st.write("**Hyperparameters:**")
    st.json(model_info["hyperparameters"])

    # Training configuration
    st.write("**Training Configuration:**")
    st.json(model_info["training_config"])


if __name__ == "__main__":
    main()
