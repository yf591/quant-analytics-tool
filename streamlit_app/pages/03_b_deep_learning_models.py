#!/usr/bin/env python3
"""
Deep Learning Models Lab - Phase 4 Implementation
Individual exploration and experimentation with deep learning models.

Design Philosophy:
- Deep model exploration: Focus on understanding and experimenting with ONE model at a time
- Interactive learning: Immediate feedback and visualization of model behavior
- State preservation: All results stored in session_state for cross-session persistence
- Component-based UI: Reusable components for consistent user experience
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import deep learning models
try:
    from src.models.deep_learning.lstm import (
        QuantLSTMClassifier,
        QuantLSTMRegressor,
        LSTMDataPreprocessor,
    )
    from src.models.deep_learning.gru import (
        QuantGRUClassifier,
        QuantGRURegressor,
    )
    from src.models.evaluation import ModelEvaluator

    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    st.error(f"Deep Learning models not available: {e}")
    DEEP_LEARNING_AVAILABLE = False

# Import deep learning model manager
try:
    from streamlit_app.utils.deep_learning_manager import DeepLearningModelManager

    DL_MANAGER_AVAILABLE = True
except ImportError as e:
    st.warning(f"Deep Learning model manager not available: {e}")
    DL_MANAGER_AVAILABLE = False


def main():
    """Main function for Deep Learning Models Lab"""

    # Page configuration
    st.set_page_config(
        page_title="Deep Learning Models Lab",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🧠 Deep Learning Models Lab")
    st.markdown(
        """
    **Advanced Neural Network Exploration & Experimentation**
    
    This lab focuses on deep exploration of deep learning models for financial time series.
    Each model has its own dedicated workspace for architecture design, hyperparameter tuning,
    training, and comprehensive analysis.
    
    📝 **Note**: Deep Learning models use different parameters compared to traditional ML:
    - **epochs**: Number of complete passes through the training data
    - **batch_size**: Number of samples processed before model update
    - **sequence_length**: Number of time steps for time series models
    - **learning_rate**: Step size for gradient descent optimization
    """
    )

    # Debug information panel (expandable)
    show_debug_info()

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
    """Initialize session state for deep learning models"""

    # Model cache for storing trained models
    if "dl_model_cache" not in st.session_state:
        st.session_state.dl_model_cache = {}

    # Training results cache
    if "dl_training_results" not in st.session_state:
        st.session_state.dl_training_results = {}

    # Current experiment tracking
    if "dl_current_experiment" not in st.session_state:
        st.session_state.dl_current_experiment = {}

    # Model comparison data
    if "dl_model_comparison" not in st.session_state:
        st.session_state.dl_model_comparison = []


def check_prerequisites():
    """Check if all required components are available"""

    if not DEEP_LEARNING_AVAILABLE:
        st.error(
            "❌ Deep Learning models are not available. Please check your installation."
        )
        return False

    if not DL_MANAGER_AVAILABLE:
        st.warning(
            "⚠️ Deep Learning model manager not available. Functionality may be limited."
        )

    # Check for feature data
    if "feature_cache" not in st.session_state or not st.session_state.feature_cache:
        st.warning("⚠️ No feature data found. Please run Feature Engineering first.")
        with st.expander("📋 How to get feature data"):
            st.markdown(
                """
            ### Prerequisites for Deep Learning Training:
            
            1. **📊 Data Acquisition**: Load market data for your target symbols
            2. **🛠️ Feature Engineering**: Generate technical indicators and features
            3. **🧠 Deep Learning Training**: Configure and train neural networks
            
            Deep Learning models require time series data for sequence prediction.
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
            if isinstance(dataset_info, dict) and "metadata" in dataset_info:
                metadata = dataset_info["metadata"]
                st.write(f"**Symbol**: {metadata.get('symbol', 'Unknown')}")
                st.write(f"**Period**: {metadata.get('period', 'Unknown')}")
                st.write(f"**Interval**: {metadata.get('interval', 'Unknown')}")
                st.write(f"**Samples**: {metadata.get('n_samples', 'Unknown')}")
                st.write(f"**Features**: {metadata.get('n_features', 'Unknown')}")

                if "feature_names" in dataset_info:
                    st.write("**Available Features**:")
                    features = dataset_info["feature_names"][:10]  # Show first 10
                    for feature in features:
                        st.write(f"- {feature}")
                    if len(dataset_info["feature_names"]) > 10:
                        st.write(
                            f"... and {len(dataset_info['feature_names']) - 10} more"
                        )

    return selected_dataset


def show_model_labs(feature_key: str):
    """Display model laboratory tabs"""

    # Model selection tabs
    tab_lstm, tab_gru = st.tabs(["🔄 LSTM Lab", "⚡ GRU Lab"])

    with tab_lstm:
        show_lstm_lab(feature_key)

    with tab_gru:
        show_gru_lab(feature_key)


def show_lstm_lab(feature_key: str):
    """LSTM model laboratory"""

    st.header("🔄 LSTM Laboratory")
    st.markdown(
        "Explore LSTM models with comprehensive architecture design and hyperparameter tuning."
    )

    # Control Panel Section
    st.subheader("🎛️ Control Panel")

    # Task type selection
    task_type = st.radio(
        "Task Type:",
        options=["Classification", "Regression"],
        index=0,
        help="Select the type of machine learning task",
        key="lstm_task_type",
    )

    # Model selection based on task type
    model_class = (
        QuantLSTMClassifier if task_type == "Classification" else QuantLSTMRegressor
    )

    st.markdown("---")

    # Architecture section
    st.subheader("🏗️ Model Architecture")
    architecture_params = get_lstm_architecture_parameters()

    st.markdown("---")

    # Training parameters section
    st.subheader("🎯 Training Parameters")
    training_params = get_lstm_training_parameters()

    st.markdown("---")

    # Regularization section
    st.subheader("🛡️ Regularization")
    regularization_params = get_lstm_regularization_parameters()

    st.markdown("---")

    # Data preprocessing section
    st.subheader("📊 Data Preprocessing")
    preprocessing_params = get_lstm_preprocessing_parameters()

    st.markdown("---")

    # Combine all parameters
    all_params = {
        **architecture_params,
        **training_params,
        **regularization_params,
        **preprocessing_params,
    }

    # Training configuration
    st.subheader("⚙️ Training Settings")
    training_config = get_deep_learning_training_configuration(key_prefix="lstm")

    st.markdown("---")

    # Training button
    if st.button(
        "🚀 Train LSTM",
        type="primary",
        use_container_width=True,
        key="train_lstm",
    ):
        train_deep_learning_model(
            feature_key=feature_key,
            model_class=model_class,
            model_name="LSTM",
            task_type=task_type.lower(),
            hyperparams=all_params,
            training_config=training_config,
        )

    st.markdown("---")

    # Results & Analysis Section (moved below training controls)
    st.subheader("📊 Results & Analysis")

    # Display results if available
    model_key = f"LSTM_{task_type.lower()}_{feature_key}"
    display_deep_learning_results(model_key, "LSTM")


def show_gru_lab(feature_key: str):
    """GRU model laboratory"""

    st.header("⚡ GRU Laboratory")
    st.markdown("Explore GRU models with advanced recurrent neural network techniques.")

    # Control Panel Section
    st.subheader("🎛️ Control Panel")

    # Task type selection
    task_type = st.radio(
        "Task Type:",
        options=["Classification", "Regression"],
        index=0,
        help="Select the type of machine learning task",
        key="gru_task_type",
    )

    # Model selection based on task type
    model_class = (
        QuantGRUClassifier if task_type == "Classification" else QuantGRURegressor
    )

    st.markdown("---")

    # Architecture section
    st.subheader("🏗️ Model Architecture")
    architecture_params = get_gru_architecture_parameters()

    st.markdown("---")

    # Training parameters section
    st.subheader("🎯 Training Parameters")
    training_params = get_gru_training_parameters()

    st.markdown("---")

    # Regularization section
    st.subheader("🛡️ Regularization")
    regularization_params = get_gru_regularization_parameters()

    st.markdown("---")

    # Data preprocessing section
    st.subheader("📊 Data Preprocessing")
    preprocessing_params = get_gru_preprocessing_parameters()

    st.markdown("---")

    # Combine all parameters
    all_params = {
        **architecture_params,
        **training_params,
        **regularization_params,
        **preprocessing_params,
    }

    # Training configuration
    st.subheader("⚙️ Training Settings")
    training_config = get_deep_learning_training_configuration(key_prefix="gru")

    st.markdown("---")

    # Training button
    if st.button(
        "🚀 Train GRU",
        type="primary",
        use_container_width=True,
        key="train_gru",
    ):
        train_deep_learning_model(
            feature_key=feature_key,
            model_class=model_class,
            model_name="GRU",
            task_type=task_type.lower(),
            hyperparams=all_params,
            training_config=training_config,
        )

    st.markdown("---")

    # Results & Analysis Section (moved below training controls)
    st.subheader("📊 Results & Analysis")

    # Display results if available
    model_key = f"GRU_{task_type.lower()}_{feature_key}"
    display_deep_learning_results(model_key, "GRU")


def get_lstm_architecture_parameters() -> Dict[str, Any]:
    """Get LSTM architecture parameters from UI"""

    with st.expander("🔄 LSTM Architecture", expanded=True):
        # Sequence length
        sequence_length = st.slider(
            "Sequence Length",
            min_value=10,
            max_value=200,
            value=60,
            step=5,
            help="Number of time steps in each sequence",
        )

        # LSTM units
        st.write("**LSTM Layers**")
        num_lstm_layers = st.selectbox(
            "Number of LSTM Layers",
            options=[1, 2, 3],
            index=1,
            help="Number of LSTM layers in the network",
        )

        lstm_units = []
        for i in range(num_lstm_layers):
            units = st.slider(
                f"LSTM Layer {i+1} Units",
                min_value=8,
                max_value=512,
                value=50,
                step=8,
                key=f"lstm_units_{i}",
                help=f"Number of units in LSTM layer {i+1}",
            )
            lstm_units.append(units)

        # Dense layers
        st.write("**Dense Layers**")
        num_dense_layers = st.selectbox(
            "Number of Dense Layers",
            options=[0, 1, 2, 3],
            index=1,
            help="Number of dense layers after LSTM",
        )

        dense_units = []
        for i in range(num_dense_layers):
            units = st.slider(
                f"Dense Layer {i+1} Units",
                min_value=8,
                max_value=256,
                value=25,
                step=8,
                key=f"lstm_dense_units_{i}",
                help=f"Number of units in dense layer {i+1}",
            )
            dense_units.append(units)

        # Bidirectional
        bidirectional = st.checkbox(
            "Bidirectional LSTM",
            value=False,
            help="Use bidirectional LSTM layers",
        )

        # Activations
        col1, col2 = st.columns(2)

        with col1:
            activation = st.selectbox(
                "LSTM Activation",
                options=["tanh", "relu", "sigmoid"],
                index=0,
                help="Activation function for LSTM layers",
            )

        with col2:
            dense_activation = st.selectbox(
                "Dense Activation",
                options=["relu", "tanh", "sigmoid", "elu"],
                index=0,
                help="Activation function for dense layers",
            )

    return {
        "sequence_length": sequence_length,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "bidirectional": bidirectional,
        "activation": activation,
        "dense_activation": dense_activation,
    }


def get_lstm_training_parameters() -> Dict[str, Any]:
    """Get LSTM training parameters from UI"""

    with st.expander("🎯 Training Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            epochs = st.slider(
                "Epochs",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of training epochs",
            )

            batch_size = st.selectbox(
                "Batch Size",
                options=[16, 32, 64, 128, 256],
                index=1,
                help="Training batch size",
            )

            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                step=0.0001,
                format="%.4f",
                help="Learning rate for optimization",
            )

        with col2:
            optimizer = st.selectbox(
                "Optimizer",
                options=["adam", "rmsprop", "sgd"],
                index=0,
                help="Optimization algorithm",
            )

            validation_split = st.slider(
                "Validation Split",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for validation",
            )

            verbose = st.selectbox(
                "Verbosity Level",
                options=[0, 1, 2],
                index=1,
                help="Training verbosity level",
            )

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "validation_split": validation_split,
        "verbose": verbose,
    }


def get_lstm_regularization_parameters() -> Dict[str, Any]:
    """Get LSTM regularization parameters from UI"""

    with st.expander("🛡️ Regularization", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            dropout_rate = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.8,
                value=0.2,
                step=0.05,
                help="Dropout rate for regularization",
            )

            recurrent_dropout = st.slider(
                "Recurrent Dropout",
                min_value=0.0,
                max_value=0.8,
                value=0.2,
                step=0.05,
                help="Recurrent dropout rate",
            )

        with col2:
            l1_reg = st.number_input(
                "L1 Regularization",
                min_value=0.0,
                max_value=0.1,
                value=0.0,
                step=0.001,
                format="%.3f",
                help="L1 regularization strength",
            )

            l2_reg = st.number_input(
                "L2 Regularization",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="L2 regularization strength",
            )

        # Early stopping
        col3, col4 = st.columns(2)

        with col3:
            early_stopping_patience = st.slider(
                "Early Stopping Patience",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
                help="Epochs to wait before early stopping",
            )

        with col4:
            reduce_lr_patience = st.slider(
                "Reduce LR Patience",
                min_value=3,
                max_value=20,
                value=5,
                step=1,
                help="Epochs to wait before reducing learning rate",
            )

        # Batch normalization
        batch_norm = st.checkbox(
            "Batch Normalization",
            value=True,
            help="Use batch normalization layers",
        )

    return {
        "dropout_rate": dropout_rate,
        "recurrent_dropout": recurrent_dropout,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "early_stopping_patience": early_stopping_patience,
        "reduce_lr_patience": reduce_lr_patience,
        "batch_norm": batch_norm,
    }


def get_lstm_preprocessing_parameters() -> Dict[str, Any]:
    """Get LSTM preprocessing parameters from UI"""

    with st.expander("📊 Data Preprocessing", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            feature_scaler = st.selectbox(
                "Feature Scaling",
                options=["standard", "minmax", "none"],
                index=0,
                help="Feature scaling method",
            )

        with col2:
            target_scaler = st.selectbox(
                "Target Scaling",
                options=["standard", "minmax", "none"],
                index=0,
                help="Target scaling method (for regression)",
            )

    return {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def get_gru_architecture_parameters() -> Dict[str, Any]:
    """Get GRU architecture parameters from UI"""

    with st.expander("⚡ GRU Architecture", expanded=True):
        # Sequence length
        sequence_length = st.slider(
            "Sequence Length",
            min_value=10,
            max_value=200,
            value=60,
            step=5,
            help="Number of time steps in each sequence",
            key="gru_sequence_length",
        )

        # GRU units
        st.write("**GRU Layers**")
        num_gru_layers = st.selectbox(
            "Number of GRU Layers",
            options=[1, 2, 3],
            index=1,
            help="Number of GRU layers in the network",
            key="gru_num_layers",
        )

        gru_units = []
        for i in range(num_gru_layers):
            units = st.slider(
                f"GRU Layer {i+1} Units",
                min_value=8,
                max_value=512,
                value=50,
                step=8,
                key=f"gru_units_{i}",
                help=f"Number of units in GRU layer {i+1}",
            )
            gru_units.append(units)

        # Dense layers
        st.write("**Dense Layers**")
        num_dense_layers = st.selectbox(
            "Number of Dense Layers",
            options=[0, 1, 2, 3],
            index=1,
            help="Number of dense layers after GRU",
            key="gru_num_dense_layers",
        )

        dense_units = []
        for i in range(num_dense_layers):
            units = st.slider(
                f"Dense Layer {i+1} Units",
                min_value=8,
                max_value=256,
                value=25,
                step=8,
                key=f"gru_dense_units_{i}",
                help=f"Number of units in dense layer {i+1}",
            )
            dense_units.append(units)

        # Bidirectional
        bidirectional = st.checkbox(
            "Bidirectional GRU",
            value=False,
            help="Use bidirectional GRU layers",
            key="gru_bidirectional",
        )

        # Activations
        col1, col2 = st.columns(2)

        with col1:
            activation = st.selectbox(
                "GRU Activation",
                options=["tanh", "relu", "sigmoid"],
                index=0,
                help="Activation function for GRU layers",
                key="gru_activation",
            )

        with col2:
            dense_activation = st.selectbox(
                "Dense Activation",
                options=["relu", "tanh", "sigmoid", "elu"],
                index=0,
                help="Activation function for dense layers",
                key="gru_dense_activation",
            )

    return {
        "sequence_length": sequence_length,
        "gru_units": gru_units,
        "dense_units": dense_units,
        "bidirectional": bidirectional,
        "activation": activation,
        "dense_activation": dense_activation,
    }


def get_gru_training_parameters() -> Dict[str, Any]:
    """Get GRU training parameters from UI"""

    with st.expander("🎯 Training Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            epochs = st.slider(
                "Epochs",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of training epochs",
                key="gru_epochs",
            )

            batch_size = st.selectbox(
                "Batch Size",
                options=[16, 32, 64, 128, 256],
                index=1,
                help="Training batch size",
                key="gru_batch_size",
            )

            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                step=0.0001,
                format="%.4f",
                help="Learning rate for optimization",
                key="gru_learning_rate",
            )

        with col2:
            optimizer = st.selectbox(
                "Optimizer",
                options=["adam", "rmsprop", "sgd"],
                index=0,
                help="Optimization algorithm",
                key="gru_optimizer",
            )

            validation_split = st.slider(
                "Validation Split",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for validation",
                key="gru_validation_split",
            )

            verbose = st.selectbox(
                "Verbosity Level",
                options=[0, 1, 2],
                index=1,
                help="Training verbosity level",
                key="gru_verbose",
            )

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "validation_split": validation_split,
        "verbose": verbose,
    }


def get_gru_regularization_parameters() -> Dict[str, Any]:
    """Get GRU regularization parameters from UI"""

    with st.expander("🛡️ Regularization", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            dropout_rate = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.8,
                value=0.2,
                step=0.05,
                help="Dropout rate for regularization",
                key="gru_dropout_rate",
            )

            recurrent_dropout = st.slider(
                "Recurrent Dropout",
                min_value=0.0,
                max_value=0.8,
                value=0.2,
                step=0.05,
                help="Recurrent dropout rate",
                key="gru_recurrent_dropout",
            )

        with col2:
            l1_reg = st.number_input(
                "L1 Regularization",
                min_value=0.0,
                max_value=0.1,
                value=0.0,
                step=0.001,
                format="%.3f",
                help="L1 regularization strength",
                key="gru_l1_reg",
            )

            l2_reg = st.number_input(
                "L2 Regularization",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="L2 regularization strength",
                key="gru_l2_reg",
            )

        # Early stopping
        col3, col4 = st.columns(2)

        with col3:
            early_stopping_patience = st.slider(
                "Early Stopping Patience",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
                help="Epochs to wait before early stopping",
                key="gru_early_stopping_patience",
            )

        with col4:
            reduce_lr_patience = st.slider(
                "Reduce LR Patience",
                min_value=3,
                max_value=20,
                value=5,
                step=1,
                help="Epochs to wait before reducing learning rate",
                key="gru_reduce_lr_patience",
            )

        # Batch normalization
        batch_norm = st.checkbox(
            "Batch Normalization",
            value=True,
            help="Use batch normalization layers",
            key="gru_batch_norm",
        )

    return {
        "dropout_rate": dropout_rate,
        "recurrent_dropout": recurrent_dropout,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "early_stopping_patience": early_stopping_patience,
        "reduce_lr_patience": reduce_lr_patience,
        "batch_norm": batch_norm,
    }


def get_gru_preprocessing_parameters() -> Dict[str, Any]:
    """Get GRU preprocessing parameters from UI"""

    with st.expander("📊 Data Preprocessing", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            feature_scaler = st.selectbox(
                "Feature Scaling",
                options=["standard", "minmax", "none"],
                index=0,
                help="Feature scaling method",
                key="gru_feature_scaler",
            )

        with col2:
            target_scaler = st.selectbox(
                "Target Scaling",
                options=["standard", "minmax", "none"],
                index=0,
                help="Target scaling method (for regression)",
                key="gru_target_scaler",
            )

    return {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def get_deep_learning_training_configuration(key_prefix: str = "") -> Dict[str, Any]:
    """Get deep learning training configuration from UI"""

    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for testing",
                key=f"{key_prefix}_test_size",
            )

            target_column = st.selectbox(
                "Target Column",
                options=["auto", "close_direction", "return_direction", "custom"],
                index=0,
                help="Target column for prediction",
                key=f"{key_prefix}_target_column",
            )

        with col2:
            cv_folds = st.slider(
                "CV Folds",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                help="Number of cross-validation folds",
                key=f"{key_prefix}_cv_folds",
            )

            scale_features = st.checkbox(
                "Scale Features",
                value=True,
                help="Apply feature scaling",
                key=f"{key_prefix}_scale_features",
            )

    return {
        "test_size": test_size,
        "target_column": target_column,
        "cv_folds": cv_folds,
        "scale_features": scale_features,
        "random_state": 42,
    }


def train_deep_learning_model(
    feature_key: str,
    model_class: type,
    model_name: str,
    task_type: str,
    hyperparams: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Train a deep learning model"""

    try:
        with st.spinner(f"Training {model_name} {task_type} model..."):

            # Use deep learning manager if available
            if DL_MANAGER_AVAILABLE:
                manager = DeepLearningModelManager()

                model_config = {
                    "model_type": model_name,
                    "task_type": task_type,
                    "model_class": model_class,
                }

                success, message, model_id = manager.train_model(
                    feature_key=feature_key,
                    model_config=model_config,
                    hyperparams=hyperparams,
                    training_config=training_config,
                    session_state=st.session_state,
                )

                if success:
                    st.success(message)

                    # Store experiment info
                    experiment_key = f"{model_name}_{task_type}_{feature_key}"
                    st.session_state.dl_current_experiment[experiment_key] = {
                        "model_name": model_name,
                        "task_type": task_type,
                        "feature_key": feature_key,
                        "hyperparams": hyperparams,
                        "training_config": training_config,
                        "model_id": model_id,
                        "timestamp": datetime.now(),
                    }
                else:
                    st.error(message)
            else:
                # Basic training fallback
                deep_learning_training_fallback(
                    feature_key, model_class, hyperparams, training_config
                )

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")


def deep_learning_training_fallback(
    feature_key: str,
    model_class: type,
    hyperparams: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Basic deep learning training fallback when manager is not available"""

    st.info("Performing basic deep learning model training...")
    st.warning("Deep learning training implementation needed")


def display_deep_learning_results(model_key: str, model_name: str):
    """Display deep learning model training results and analysis"""

    # Check if we have results for this model
    if model_key in st.session_state.dl_current_experiment:
        experiment = st.session_state.dl_current_experiment[model_key]

        # Display experiment information
        st.success(f"✅ {model_name} model training completed!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Type", experiment.get("model_name", "Unknown"))
            st.metric("Task Type", experiment.get("task_type", "Unknown").title())

        with col2:
            if experiment.get("timestamp"):
                st.metric("Trained At", experiment["timestamp"].strftime("%H:%M:%S"))
            if experiment.get("model_id"):
                st.metric("Model ID", experiment["model_id"][:8] + "...")

        with col3:
            # Model actions
            if st.button(f"📊 Detailed Analysis", key=f"detailed_analysis_{model_key}"):
                st.info("Detailed analysis feature coming soon!")

            if st.button(f"💾 Export Model", key=f"export_model_{model_key}"):
                st.info("Model export feature coming soon!")

            if st.button(f"🔄 Retrain", key=f"retrain_{model_key}"):
                st.info("Model retraining feature coming soon!")

        # Show model cache info if available
        if experiment.get("model_id") and hasattr(st.session_state, "dl_model_cache"):
            model_cache = st.session_state.dl_model_cache
            if experiment["model_id"] in model_cache:
                model_info = model_cache[experiment["model_id"]]

                # Display training metrics
                if "evaluation" in model_info:
                    st.subheader("📈 Training Metrics")
                    evaluation = model_info["evaluation"]

                    # Check if evaluation has error
                    if "error" in evaluation:
                        st.error(f"❌ Evaluation Error: {evaluation['error']}")
                        if "basic_score" in evaluation:
                            st.metric("Basic Score", f"{evaluation['basic_score']:.4f}")
                    else:
                        # Display all metrics
                        metrics_col1, metrics_col2 = st.columns(2)

                        with metrics_col1:
                            for metric, value in evaluation.items():
                                if isinstance(
                                    value, (int, float)
                                ) and not metric.startswith("test_"):
                                    # Show training metrics
                                    display_name = (
                                        metric.replace("train_", "")
                                        .replace("_", " ")
                                        .title()
                                    )
                                    st.metric(f"Train {display_name}", f"{value:.4f}")

                        with metrics_col2:
                            for metric, value in evaluation.items():
                                if isinstance(
                                    value, (int, float)
                                ) and metric.startswith("test_"):
                                    # Show test metrics
                                    display_name = (
                                        metric.replace("test_", "")
                                        .replace("_", " ")
                                        .title()
                                    )
                                    st.metric(f"Test {display_name}", f"{value:.4f}")

                        # Additional metrics section
                        st.subheader("📊 Additional Information")
                        add_metrics_col1, add_metrics_col2 = st.columns(2)
                        # Additional metrics section
                        st.subheader("📊 Additional Information")
                        add_metrics_col1, add_metrics_col2 = st.columns(2)

                        with add_metrics_col1:
                            # Display data shape info
                            if "data_shape" in model_info:
                                shape_info = model_info["data_shape"]
                                st.metric(
                                    "Training Samples",
                                    shape_info.get("train_size", "Unknown"),
                                )
                                st.metric(
                                    "Test Samples",
                                    shape_info.get("test_size", "Unknown"),
                                )
                                st.metric(
                                    "Features", shape_info.get("n_features", "Unknown")
                                )

                        with add_metrics_col2:
                            # Display hyperparameters
                            if "hyperparams" in model_info:
                                hyperparams = model_info["hyperparams"]
                                st.metric(
                                    "Epochs", hyperparams.get("epochs", "Unknown")
                                )
                                st.metric(
                                    "Batch Size",
                                    hyperparams.get("batch_size", "Unknown"),
                                )
                                st.metric(
                                    "Learning Rate",
                                    f"{hyperparams.get('learning_rate', 0):.6f}",
                                )

                        # Training history visualization
                        if (
                            "training_history" in model_info
                            and model_info["training_history"]
                        ):
                            st.subheader("📈 Training History")
                            history = model_info["training_history"]

                            # Create loss and accuracy plots
                            hist_col1, hist_col2 = st.columns(2)

                            with hist_col1:
                                if "loss" in history and "val_loss" in history:
                                    loss_data = pd.DataFrame(
                                        {
                                            "Training Loss": history["loss"],
                                            "Validation Loss": history["val_loss"],
                                            "Epoch": range(1, len(history["loss"]) + 1),
                                        }
                                    )
                                    st.line_chart(loss_data.set_index("Epoch"))

                            with hist_col2:
                                if "accuracy" in history and "val_accuracy" in history:
                                    acc_data = pd.DataFrame(
                                        {
                                            "Training Accuracy": history["accuracy"],
                                            "Validation Accuracy": history[
                                                "val_accuracy"
                                            ],
                                            "Epoch": range(
                                                1, len(history["accuracy"]) + 1
                                            ),
                                        }
                                    )
                                    st.line_chart(acc_data.set_index("Epoch"))

                else:
                    st.warning("No evaluation results found.")

    else:
        st.info(
            f"No {model_name} model trained yet. Configure parameters and click train to get started."
        )

        # Show helpful information
        with st.expander("💡 Deep Learning Training Tips", expanded=False):
            st.markdown(
                f"""
            ### {model_name} Model Training Tips:
            
            **Architecture Design:**
            - Start with simpler architectures (1-2 layers)
            - Adjust sequence length based on your data patterns
            - Use bidirectional layers for better pattern recognition
            
            **Training Parameters:**
            - Start with lower epochs (50-100) to test quickly
            - Use appropriate batch size (32-64 for most cases)
            - Monitor validation loss to avoid overfitting
            
            **Regularization:**
            - Apply dropout (0.2-0.5) to prevent overfitting
            - Use early stopping to save training time
            - Consider L2 regularization for complex models
            
            **Data Preprocessing:**
            - Feature scaling is crucial for neural networks
            - Ensure sufficient sequence length for patterns
            - Monitor for data leakage in time series
            """
            )


def show_debug_info():
    """Show debug information about session state and feature cache"""
    with st.expander("🔍 Debug Information", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Session State")

            # Feature cache info
            if "feature_cache" in st.session_state:
                feature_cache = st.session_state.feature_cache
                st.write(f"**Feature Cache**: {len(feature_cache)} datasets")
                for key in feature_cache.keys():
                    st.write(f"- {key}")
            else:
                st.write("**Feature Cache**: Not found")

            # Deep learning model cache info
            if "dl_model_cache" in st.session_state:
                dl_model_cache = st.session_state.dl_model_cache
                st.write(f"**DL Model Cache**: {len(dl_model_cache)} models")
                for key in dl_model_cache.keys():
                    st.write(f"- {key[:8]}...")
            else:
                st.write("**DL Model Cache**: Empty")

        with col2:
            st.subheader("Experiments")

            # Current experiments
            if "dl_current_experiment" in st.session_state:
                experiments = st.session_state.dl_current_experiment
                st.write(f"**Current Experiments**: {len(experiments)}")
                for key, exp in experiments.items():
                    st.write(f"- {key}: {exp.get('model_name', 'Unknown')}")
            else:
                st.write("**Current Experiments**: None")

            # System info
            st.subheader("System Status")
            st.write(f"**Deep Learning Available**: {DEEP_LEARNING_AVAILABLE}")
            st.write(f"**DL Manager Available**: {DL_MANAGER_AVAILABLE}")

            # TensorFlow status
            try:
                import tensorflow as tf

                st.write(f"**TensorFlow Version**: {tf.__version__}")
                st.write(
                    f"**GPU Available**: {len(tf.config.list_physical_devices('GPU')) > 0}"
                )
            except ImportError:
                st.write("**TensorFlow**: Not available")


if __name__ == "__main__":
    main()
