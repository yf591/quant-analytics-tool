#!/usr/bin/env python3
"""
Traditional ML Models Lab - Phase 3 Week 7 Implementation
Individual exploration and experimentation with traditional ML models.

Design Philosophy:
- Deep model exploration: Focus on understanding and experimenting with ONE model at a time
- Interactive learning: Immediate feedback and visualization of model behavior
- State preservation: All results stored in session_state for cross-session persistence
- Component-based UI: Reusable components for consistent user experience
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import traditional ML models
try:
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
    from src.models.evaluation import ModelEvaluator

    TRADITIONAL_ML_AVAILABLE = True
except ImportError as e:
    st.error(f"Traditional ML models not available: {e}")
    TRADITIONAL_ML_AVAILABLE = False

# Import simple model manager
try:
    from streamlit_app.utils.simple_model_manager import SimpleModelTrainingManager

    MANAGER_AVAILABLE = True
except ImportError as e:
    st.warning(f"Model manager not available: {e}")
    MANAGER_AVAILABLE = False


def main():
    """Main function for Traditional ML Models Lab"""

    # Page configuration
    st.set_page_config(
        page_title="Traditional ML Models Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🔬 Traditional ML Models Lab")
    st.markdown(
        """
    **Individual Model Exploration & Experimentation**
    
    This lab focuses on deep exploration of traditional machine learning models.
    Each model has its own dedicated workspace for hyperparameter tuning,
    training, and comprehensive analysis.
    """
    )

    # Check prerequisites
    if not check_prerequisites():
        return

    # Dataset selection
    feature_key = select_dataset()
    if not feature_key:
        st.info("👆 Please select a feature dataset to begin model training.")
        return

    # Main model tabs
    show_model_labs(feature_key)


def initialize_session_state():
    """Initialize session state for traditional models"""

    # Model cache for storing trained models
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}

    # Training results cache
    if "training_results" not in st.session_state:
        st.session_state.training_results = {}

    # Current experiment tracking
    if "current_experiment" not in st.session_state:
        st.session_state.current_experiment = {}

    # Model comparison data
    if "model_comparison" not in st.session_state:
        st.session_state.model_comparison = []


def check_prerequisites():
    """Check if all required components are available"""

    if not TRADITIONAL_ML_AVAILABLE:
        st.error(
            "❌ Traditional ML models are not available. Please check your installation."
        )
        return False

    if not MANAGER_AVAILABLE:
        st.warning("⚠️ Model manager not available. Functionality may be limited.")

    # Check for feature data
    if "feature_cache" not in st.session_state or not st.session_state.feature_cache:
        st.warning("⚠️ No feature data found. Please run Feature Engineering first.")
        with st.expander("📋 How to get feature data"):
            st.markdown(
                """
            1. Go to **🛠️ Feature Engineering** page
            2. Load or acquire market data
            3. Generate technical indicators and features
            4. Return to this page to train models
            """
            )
        return False

    return True


def select_dataset():
    """Dataset selection interface"""

    st.sidebar.markdown("### 📊 Dataset Selection")

    # Get available feature datasets
    feature_cache = st.session_state.get("feature_cache", {})

    if not feature_cache:
        st.sidebar.error("No feature datasets available")
        return None

    # Dataset selection
    dataset_options = list(feature_cache.keys())
    selected_dataset = st.sidebar.selectbox(
        "Select Feature Dataset:",
        options=dataset_options,
        index=0,
        help="Choose the feature dataset for model training",
    )

    # Dataset information
    if selected_dataset and selected_dataset in feature_cache:
        dataset_info = feature_cache[selected_dataset]

        with st.sidebar.expander("📋 Dataset Info", expanded=False):
            if "metadata" in dataset_info:
                metadata = dataset_info["metadata"]
                st.write(f"**Symbol**: {metadata.get('symbol', 'N/A')}")
                st.write(f"**Features**: {metadata.get('feature_count', 'N/A')}")
                st.write(f"**Samples**: {metadata.get('sample_count', 'N/A')}")
                st.write(f"**Created**: {metadata.get('timestamp', 'N/A')}")

            if "feature_names" in dataset_info:
                st.write("**Available Features**:")
                features = dataset_info["feature_names"][:10]  # Show first 10
                for feature in features:
                    st.write(f"- {feature}")
                if len(dataset_info["feature_names"]) > 10:
                    st.write(f"... and {len(dataset_info['feature_names']) - 10} more")

    return selected_dataset


def show_model_labs(feature_key: str):
    """Display model laboratory tabs"""

    # Model selection tabs
    tab_rf, tab_xgb, tab_svm = st.tabs(
        ["🌳 Random Forest Lab", "🚀 XGBoost Lab", "⚡ SVM Lab"]
    )

    with tab_rf:
        show_random_forest_lab(feature_key)

    with tab_xgb:
        show_xgboost_lab(feature_key)

    with tab_svm:
        show_svm_lab(feature_key)


def show_random_forest_lab(feature_key: str):
    """Random Forest model laboratory"""

    st.header("🌳 Random Forest Laboratory")
    st.markdown(
        "Explore Random Forest models with comprehensive hyperparameter tuning and analysis."
    )

    # Main layout
    col_control, col_display = st.columns([1, 2])

    with col_control:
        st.subheader("🎛️ Control Panel")

        # Task type selection
        task_type = st.radio(
            "Task Type:",
            options=["Classification", "Regression"],
            index=0,
            help="Select the type of machine learning task",
        )

        # Model selection based on task type
        model_class = (
            QuantRandomForestClassifier
            if task_type == "Classification"
            else QuantRandomForestRegressor
        )

        st.markdown("---")

        # Hyperparameters section
        st.subheader("🔧 Hyperparameters")
        hyperparams = get_random_forest_hyperparameters()

        st.markdown("---")

        # Training configuration
        st.subheader("⚙️ Training Settings")
        training_config = get_training_configuration()

        st.markdown("---")

        # Training button
        if st.button(
            "🚀 Train Random Forest", type="primary", use_container_width=True
        ):
            train_traditional_model(
                feature_key=feature_key,
                model_class=model_class,
                model_name="RandomForest",
                task_type=task_type.lower(),
                hyperparams=hyperparams,
                training_config=training_config,
            )

    with col_display:
        st.subheader("📊 Results & Analysis")

        # Display results if available
        model_key = f"RandomForest_{task_type.lower()}_{feature_key}"
        display_model_results(model_key, "Random Forest")


def show_xgboost_lab(feature_key: str):
    """XGBoost model laboratory"""

    st.header("🚀 XGBoost Laboratory")
    st.markdown("Explore XGBoost models with advanced gradient boosting techniques.")

    # Main layout
    col_control, col_display = st.columns([1, 2])

    with col_control:
        st.subheader("🎛️ Control Panel")

        # Task type selection
        task_type = st.radio(
            "Task Type:",
            options=["Classification", "Regression"],
            index=0,
            help="Select the type of machine learning task",
            key="xgb_task_type",
        )

        # Model selection based on task type
        model_class = (
            QuantXGBoostClassifier
            if task_type == "Classification"
            else QuantXGBoostRegressor
        )

        st.markdown("---")

        # Hyperparameters section
        st.subheader("🔧 Hyperparameters")
        hyperparams = get_xgboost_hyperparameters()

        st.markdown("---")

        # Training configuration
        st.subheader("⚙️ Training Settings")
        training_config = get_training_configuration(key_prefix="xgb")

        st.markdown("---")

        # Training button
        if st.button("🚀 Train XGBoost", type="primary", use_container_width=True):
            train_traditional_model(
                feature_key=feature_key,
                model_class=model_class,
                model_name="XGBoost",
                task_type=task_type.lower(),
                hyperparams=hyperparams,
                training_config=training_config,
            )

    with col_display:
        st.subheader("📊 Results & Analysis")

        # Display results if available
        model_key = f"XGBoost_{task_type.lower()}_{feature_key}"
        display_model_results(model_key, "XGBoost")


def show_svm_lab(feature_key: str):
    """SVM model laboratory"""

    st.header("⚡ SVM Laboratory")
    st.markdown(
        "Explore Support Vector Machines with kernel methods and advanced regularization."
    )

    # Main layout
    col_control, col_display = st.columns([1, 2])

    with col_control:
        st.subheader("🎛️ Control Panel")

        # Task type selection
        task_type = st.radio(
            "Task Type:",
            options=["Classification", "Regression"],
            index=0,
            help="Select the type of machine learning task",
            key="svm_task_type",
        )

        # Model selection based on task type
        model_class = (
            QuantSVMClassifier if task_type == "Classification" else QuantSVMRegressor
        )

        st.markdown("---")

        # Hyperparameters section
        st.subheader("🔧 Hyperparameters")
        hyperparams = get_svm_hyperparameters()

        st.markdown("---")

        # Training configuration
        st.subheader("⚙️ Training Settings")
        training_config = get_training_configuration(key_prefix="svm")

        st.markdown("---")

        # Training button
        if st.button("🚀 Train SVM", type="primary", use_container_width=True):
            train_traditional_model(
                feature_key=feature_key,
                model_class=model_class,
                model_name="SVM",
                task_type=task_type.lower(),
                hyperparams=hyperparams,
                training_config=training_config,
            )

    with col_display:
        st.subheader("📊 Results & Analysis")

        # Display results if available
        model_key = f"SVM_{task_type.lower()}_{feature_key}"
        display_model_results(model_key, "SVM")


def get_random_forest_hyperparameters() -> Dict[str, Any]:
    """Get Random Forest hyperparameters from UI"""

    with st.expander("🌳 Random Forest Parameters", expanded=True):
        n_estimators = st.slider(
            "Number of Trees (n_estimators)",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of trees in the forest",
        )

        max_depth = st.slider(
            "Maximum Depth",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Maximum depth of the trees",
        )

        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            help="Minimum samples required to split an internal node",
        )

        min_samples_leaf = st.slider(
            "Min Samples Leaf",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Minimum samples required to be at a leaf node",
        )

        max_features = st.selectbox(
            "Max Features",
            options=["sqrt", "log2", "auto", 0.5, 0.8, 1.0],
            index=0,
            help="Number of features to consider when looking for the best split",
        )

        bootstrap = st.checkbox(
            "Bootstrap",
            value=True,
            help="Whether bootstrap samples are used when building trees",
        )

        # Advanced parameters
        st.markdown("**Advanced Parameters:**")

        criterion = st.selectbox(
            "Criterion",
            options=["gini", "entropy", "log_loss"],
            index=0,
            help="Function to measure quality of split",
        )

        class_weight = st.selectbox(
            "Class Weight",
            options=[None, "balanced", "balanced_subsample"],
            index=2,
            help="Weights associated with classes",
        )

        verbose = st.slider(
            "Verbose",
            min_value=0,
            max_value=2,
            value=0,
            step=1,
            help="Controls verbosity when fitting and predicting",
        )

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth < 50 else None,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "criterion": criterion,
        "class_weight": class_weight,
        "verbose": verbose,
        "random_state": 42,
        "n_jobs": -1,
    }


def get_xgboost_hyperparameters() -> Dict[str, Any]:
    """Get XGBoost hyperparameters from UI"""

    with st.expander("🚀 XGBoost Parameters", expanded=True):
        n_estimators = st.slider(
            "Number of Estimators",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of boosting rounds",
        )

        max_depth = st.slider(
            "Maximum Depth",
            min_value=1,
            max_value=20,
            value=6,
            step=1,
            help="Maximum depth of trees",
        )

        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Boosting learning rate",
        )

        subsample = st.slider(
            "Subsample",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Subsample ratio of training instances",
        )

        colsample_bytree = st.slider(
            "Column Sample by Tree",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Subsample ratio of columns when constructing each tree",
        )

        reg_alpha = st.slider(
            "L1 Regularization (Alpha)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="L1 regularization term on weights",
        )

        reg_lambda = st.slider(
            "L2 Regularization (Lambda)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="L2 regularization term on weights",
        )

        # Advanced parameters
        st.markdown("**Advanced Parameters:**")

        min_child_weight = st.slider(
            "Min Child Weight",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Minimum sum of instance weight needed in a child",
        )

        gamma = st.slider(
            "Gamma",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            help="Minimum loss reduction required to make split",
        )

        colsample_bylevel = st.slider(
            "Column Sample by Level",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Subsample ratio of columns for each level",
        )

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "colsample_bylevel": colsample_bylevel,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "random_state": 42,
        "n_jobs": -1,
    }


def get_svm_hyperparameters() -> Dict[str, Any]:
    """Get SVM hyperparameters from UI"""

    with st.expander("⚡ SVM Parameters", expanded=True):
        kernel = st.selectbox(
            "Kernel",
            options=["rbf", "linear", "poly", "sigmoid"],
            index=0,
            help="Kernel type to be used in the algorithm",
        )

        C = st.slider(
            "Regularization Parameter (C)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.01,
            help="Regularization parameter",
        )

        if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
            gamma = st.selectbox(
                "Gamma",
                options=["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                index=0,
                help="Kernel coefficient",
            )
        else:
            gamma = "scale"

        if kernel == "poly":
            degree = st.slider(
                "Polynomial Degree",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Degree of polynomial kernel function",
            )
        else:
            degree = 3

        if kernel == "poly" or kernel == "sigmoid":
            coef0 = st.slider(
                "Independent Term (coef0)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                help="Independent term in kernel function",
            )
        else:
            coef0 = 0.0

        scale_features = st.checkbox(
            "Scale Features",
            value=True,
            help="Whether to scale features before training",
        )

        # Advanced parameters
        st.markdown("**Advanced Parameters:**")

        shrinking = st.checkbox(
            "Shrinking",
            value=True,
            help="Whether to use shrinking heuristic",
        )

        probability = st.checkbox(
            "Probability",
            value=True,
            help="Whether to enable probability estimates",
        )

        tol = st.number_input(
            "Tolerance",
            min_value=1e-6,
            max_value=1e-1,
            value=1e-3,
            format="%.6f",
            help="Tolerance for stopping criterion",
        )

        cache_size = st.slider(
            "Cache Size (MB)",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Size of kernel cache",
        )

        max_iter = st.number_input(
            "Max Iterations",
            min_value=-1,
            max_value=10000,
            value=-1,
            help="Hard limit on iterations (-1 for no limit)",
        )

    return {
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "shrinking": shrinking,
        "probability": probability,
        "tol": tol,
        "cache_size": cache_size,
        "max_iter": max_iter,
        "random_state": 42,
        "scale_features": scale_features,
    }


def get_training_configuration(key_prefix: str = "") -> Dict[str, Any]:
    """Get training configuration from UI"""

    with st.expander("⚙️ Training Configuration", expanded=True):
        test_size = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of dataset to include in test split",
            key=f"{key_prefix}_test_size",
        )

        target_column = st.selectbox(
            "Target Column",
            options=["auto", "returns", "direction", "volatility"],
            index=0,
            help="Target variable for prediction",
            key=f"{key_prefix}_target",
        )

        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of cross-validation folds",
            key=f"{key_prefix}_cv",
        )

        scale_features = st.checkbox(
            "Scale Features",
            value=True,
            help="Whether to scale features before training",
            key=f"{key_prefix}_scale",
        )

    return {
        "test_size": test_size,
        "target_column": target_column,
        "cv_folds": cv_folds,
        "scale_features": scale_features,
        "random_state": 42,
    }


def train_traditional_model(
    feature_key: str,
    model_class: type,
    model_name: str,
    task_type: str,
    hyperparams: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Train a traditional ML model"""

    try:
        # Initialize model training manager
        if MANAGER_AVAILABLE:
            manager = SimpleModelTrainingManager()

            # Prepare model configuration
            model_config = {
                "model_type": model_name,
                "task_type": task_type,
                "model_class": model_class.__name__,
            }

            # Start training with progress indication
            with st.spinner(f"Training {model_name} model..."):
                success, message, model_id = manager.train_model(
                    feature_key=feature_key,
                    model_config=model_config,
                    hyperparams=hyperparams,
                    training_config=training_config,
                    session_state=st.session_state,
                )

            if success:
                st.success(f"✅ {model_name} training completed successfully!")
                st.balloons()

                # Store experiment info
                experiment_key = f"{model_name}_{task_type}_{feature_key}"
                st.session_state.current_experiment[experiment_key] = {
                    "model_id": model_id,
                    "timestamp": datetime.now(),
                    "hyperparams": hyperparams,
                    "training_config": training_config,
                    "feature_key": feature_key,
                    "model_name": model_name,
                    "task_type": task_type,
                }

                # Auto-refresh to show results
                st.rerun()
            else:
                st.error(f"❌ Training failed: {message}")

        else:
            # Fallback training without manager
            st.warning("Training manager not available. Using basic training...")
            basic_training_fallback(
                feature_key, model_class, hyperparams, training_config
            )

    except Exception as e:
        st.error(f"❌ Training error: {str(e)}")
        st.exception(e)


def basic_training_fallback(
    feature_key: str,
    model_class: type,
    hyperparams: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Basic training fallback when manager is not available"""

    st.info("Performing basic model training...")

    # This would contain basic training logic
    # For now, just show a placeholder
    st.warning("Basic training implementation needed")


def display_model_results(model_key: str, model_name: str):
    """Display model training results and analysis"""

    # Check if we have results for this model
    if model_key in st.session_state.current_experiment:
        experiment = st.session_state.current_experiment[model_key]
        model_id = experiment.get("model_id")

        if model_id and model_id in st.session_state.model_cache:
            model_info = st.session_state.model_cache[model_id]

            # Display training summary
            st.success(f"✅ {model_name} model trained successfully!")

            # Performance metrics
            if "evaluation" in model_info:
                st.subheader("📈 Performance Metrics")
                metrics = model_info["evaluation"]

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    if "accuracy" in metrics:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    elif "r2_score" in metrics:
                        st.metric("R² Score", f"{metrics['r2_score']:.4f}")

                with col2:
                    if "precision" in metrics:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    elif "mse" in metrics:
                        st.metric("MSE", f"{metrics['mse']:.4f}")

                with col3:
                    if "recall" in metrics:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    elif "mae" in metrics:
                        st.metric("MAE", f"{metrics['mae']:.4f}")

            # Feature importance
            if "feature_importance" in model_info:
                st.subheader("🎯 Feature Importance")
                importance_data = model_info["feature_importance"]

                # Display top 10 features
                if isinstance(importance_data, pd.Series):
                    top_features = importance_data.head(10)
                    st.bar_chart(top_features)

            # Training details
            with st.expander("🔧 Training Details", expanded=False):
                st.write("**Hyperparameters:**")
                st.json(experiment["hyperparams"])

                st.write("**Training Configuration:**")
                st.json(experiment["training_config"])

                st.write("**Training Time:**")
                st.write(f"Completed at: {experiment['timestamp']}")

            # Model actions
            st.subheader("🎯 Model Actions")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Detailed Analysis", use_container_width=True):
                    st.info("Detailed analysis feature coming soon!")

            with col2:
                if st.button("💾 Export Model", use_container_width=True):
                    st.info("Model export feature coming soon!")

            with col3:
                if st.button("🔄 Retrain", use_container_width=True):
                    # Clear current results to allow retraining
                    if model_key in st.session_state.current_experiment:
                        del st.session_state.current_experiment[model_key]
                    st.rerun()

        else:
            st.warning(f"Model data not found for {model_name}")

    else:
        # No results yet - show placeholder
        st.info(
            f"No {model_name} model trained yet. Configure parameters and click Train to begin."
        )

        # Show example placeholder
        st.subheader("📊 Results will appear here")
        st.markdown(
            """
        After training, you'll see:
        - **Performance Metrics**: Accuracy, Precision, Recall, etc.
        - **Feature Importance**: Most influential features
        - **Model Visualization**: Charts and plots
        - **Training Details**: Hyperparameters and configuration
        """
        )


if __name__ == "__main__":
    main()
