#!/usr/bin/env python3
"""
Advanced Models Lab - Phase 4 Implementation
Individual exploration and experimentation with advanced ML models.

Design Philosophy:
- Advanced model exploration: Focus on understanding cutting-edge models
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
    st.error(f"Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

# Import advanced models manager
try:
    from streamlit_app.utils.advanced_models_manager import AdvancedModelsManager

    MANAGER_AVAILABLE = True
except ImportError as e:
    st.warning(f"Advanced models manager not available: {e}")
    MANAGER_AVAILABLE = False


def main():
    """Main function for Advanced Models Lab"""

    # Page configuration
    st.set_page_config(
        page_title="Advanced Models Lab",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🧠 Advanced Models Lab")
    st.markdown(
        """
    **Cutting-Edge Model Exploration & Experimentation**
    
    This lab focuses on advanced machine learning models including ensemble methods,
    transformer architectures, attention mechanisms, meta-labeling techniques, and
    comprehensive model interpretation tools.
    
    🧬 **Advanced Features**:
    - **Ensemble Methods**: Financial-aware ensemble techniques
    - **Transformers**: Attention-based sequence models
    - **Meta-Labeling**: Advanced position sizing strategies
    - **Model Interpretation**: SHAP, feature importance, and explainability tools
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
        st.info("👆 Please select a feature dataset to begin advanced model training.")
        return

    # Main model tabs
    show_model_labs(feature_key)


def initialize_session_state():
    """Initialize session state for advanced models"""

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

    # Interpretation results
    if "interpretation_results" not in st.session_state:
        st.session_state.interpretation_results = {}


def check_prerequisites():
    """Check if all required components are available"""

    if not ADVANCED_MODELS_AVAILABLE:
        st.error(
            "❌ Advanced models are not available. Please check your installation."
        )
        return False

    if not MANAGER_AVAILABLE:
        st.warning(
            "⚠️ Advanced models manager not available. Functionality may be limited."
        )

    # Check for feature data
    if "feature_cache" not in st.session_state or not st.session_state.feature_cache:
        st.warning("⚠️ No feature data found. Please run Feature Engineering first.")
        with st.expander("📋 How to get feature data", expanded=True):
            st.markdown(
                """
            ### Steps to prepare data for Advanced Models:
            
            1. **�️ Data Management**
               - Load market data for your target symbols
               - Ensure you have sufficient historical data
            
            2. **🛠️ Feature Engineering** 
               - Generate technical indicators and features
               - Create target variables (returns, price movements)
               - Save feature datasets for training
            
            3. **🧠 Advanced Models Training**
               - Select from ensemble methods, transformers, attention mechanisms
               - Configure hyperparameters and training settings
               - Train and evaluate cutting-edge models
            
            ### Required Data Format:
            - DataFrame with features and target columns
            - Target column: 'target', 'return', 'close', or similar
            - Features: Technical indicators, price data, volume data
            """
            )
        return False

    return True


def select_dataset():
    """Enhanced dataset selection interface with detailed information"""

    st.sidebar.markdown("### 📊 Dataset Selection")

    # Get available feature datasets
    feature_cache = st.session_state.get("feature_cache", {})

    if not feature_cache:
        st.sidebar.error("❌ No feature datasets available")
        st.sidebar.info("💡 Please run Feature Engineering first")
        return None

    # Dataset selection
    dataset_options = list(feature_cache.keys())
    selected_dataset = st.sidebar.selectbox(
        "📂 Select Feature Dataset:",
        options=dataset_options,
        index=0,
        help="Choose the feature dataset for advanced model training",
        key="advanced_dataset_selector"
    )

    # Enhanced dataset information
    if selected_dataset and selected_dataset in feature_cache:
        dataset_info = feature_cache[selected_dataset]
        
        # Try to get the actual DataFrame
        df = None
        if isinstance(dataset_info, pd.DataFrame):
            df = dataset_info
        elif isinstance(dataset_info, dict):
            if "data" in dataset_info:
                df = dataset_info["data"]
            elif "features" in dataset_info:
                df = dataset_info["features"]

        with st.sidebar.expander("📋 Dataset Overview", expanded=True):
            if df is not None:
                # Basic statistics
                st.metric("📊 Total Rows", f"{len(df):,}")
                st.metric("📈 Features", len(df.columns))
                
                # Missing data percentage
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("⚠️ Missing Data", f"{missing_pct:.2f}%")
                
                # Date range info
                if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                    st.write(f"**📅 Period**: {df.index.min()} to {df.index.max()}")
                
                # Target column detection
                target_candidates = [col for col in df.columns if any(
                    keyword in col.lower() for keyword in 
                    ['target', 'return', 'close', 'price', 'label', 'y']
                )]
                
                if target_candidates:
                    st.write("**🎯 Potential Targets**:")
                    for target in target_candidates[:5]:  # Show up to 5
                        st.write(f"- {target}")
                else:
                    st.warning("⚠️ No obvious target columns found")
                
                # Feature preview
                if len(df.columns) > 0:
                    st.write("**🛠️ Sample Features**:")
                    features_to_show = list(df.columns)[:8]  # Show first 8
                    for feature in features_to_show:
                        st.write(f"- {feature}")
                    
                    if len(df.columns) > 8:
                        st.write(f"... and {len(df.columns) - 8} more")
            
            # Legacy metadata support
            elif isinstance(dataset_info, dict) and "metadata" in dataset_info:
                metadata = dataset_info["metadata"]
                st.write(f"**Symbol**: {metadata.get('symbol', 'N/A')}")
                st.write(f"**Features**: {metadata.get('feature_count', 'N/A')}")
                st.write(f"**Samples**: {metadata.get('sample_count', 'N/A')}")
                st.write(f"**Created**: {metadata.get('timestamp', 'N/A')}")

                if "feature_names" in dataset_info:
                    st.write("**Available Features**:")
                    features = dataset_info["feature_names"][:8]  # Show first 8
                    for feature in features:
                        st.write(f"- {feature}")
                    if len(dataset_info["feature_names"]) > 8:
                        st.write(f"... and {len(dataset_info['feature_names']) - 8} more")
            else:
                st.warning("⚠️ Dataset format not recognized")
        
        # Data quality check
        if df is not None:
            with st.sidebar.expander("🔍 Data Quality", expanded=False):
                # Check for sufficient data
                min_samples = 100
                if len(df) < min_samples:
                    st.error(f"⚠️ Insufficient data: {len(df)} samples (minimum: {min_samples})")
                else:
                    st.success(f"✅ Sufficient data: {len(df)} samples")
                
                # Check for missing data
                if missing_pct > 50:
                    st.error(f"⚠️ High missing data: {missing_pct:.1f}%")
                elif missing_pct > 20:
                    st.warning(f"⚠️ Moderate missing data: {missing_pct:.1f}%")
                else:
                    st.success(f"✅ Low missing data: {missing_pct:.1f}%")
                
                # Check for numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    st.error("⚠️ No numeric features found")
                else:
                    st.success(f"✅ {len(numeric_cols)} numeric features")

    return selected_dataset


def show_model_labs(feature_key: str):
    """Display advanced model laboratory tabs"""

    # Model selection tabs
    tab_ensemble, tab_transformer, tab_attention, tab_meta, tab_interpretation = (
        st.tabs(
            [
                "🎯 Ensemble Methods",
                "🤖 Transformer Models",
                "🧠 Attention Mechanisms",
                "🎖️ Meta-Labeling",
                "🔍 Model Interpretation",
            ]
        )
    )

    with tab_ensemble:
        show_ensemble_lab(feature_key)

    with tab_transformer:
        show_transformer_lab(feature_key)

    with tab_attention:
        show_attention_lab(feature_key)

    with tab_meta:
        show_meta_labeling_lab(feature_key)

    with tab_interpretation:
        show_interpretation_lab(feature_key)


def show_ensemble_lab(feature_key: str):
    """Ensemble methods laboratory"""

    st.header("🎯 Ensemble Methods Laboratory")
    st.markdown(
        "Explore advanced ensemble techniques with financial-specific considerations."
    )

    # Control Panel Section
    st.subheader("🎛️ Control Panel")

    col1, col2 = st.columns(2)

    with col1:
        # Task type selection
        task_type = st.radio(
            "Task Type:",
            options=["Classification", "Regression"],
            index=0,
            help="Select the type of machine learning task",
            key="ensemble_task_type",
        )

    with col2:
        # Ensemble method selection
        ensemble_method = st.selectbox(
            "Ensemble Method:",
            options=["financial_rf", "bagging", "stacking", "voting"],
            format_func=lambda x: {
                "financial_rf": "Financial Random Forest",
                "bagging": "Time Series Bagging",
                "stacking": "Stacking Ensemble",
                "voting": "Voting Ensemble",
            }[x],
            help="Select the ensemble method to use",
        )

    st.markdown("---")

    # Hyperparameters section
    st.subheader("🔧 Hyperparameters")
    hyperparams = get_ensemble_hyperparameters(ensemble_method)

    st.markdown("---")

    # Training configuration
    st.subheader("⚙️ Training Settings")
    training_config = get_training_configuration(key_prefix="ensemble")

    st.markdown("---")

    # Training button
    if st.button(
        f"🚀 Train {ensemble_method.title()} Ensemble",
        type="primary",
        use_container_width=True,
        key="train_ensemble",
    ):
        train_ensemble_model(
            feature_key=feature_key,
            ensemble_method=ensemble_method,
            task_type=task_type.lower(),
            hyperparams=hyperparams,
            training_config=training_config,
        )

    st.markdown("---")

    # Results & Analysis Section
    st.subheader("📊 Results & Analysis")

    # Display results if available
    model_key = f"{ensemble_method}_{task_type.lower()}_{feature_key}"
    display_ensemble_results(model_key, f"{ensemble_method.title()} Ensemble")


def show_transformer_lab(feature_key: str):
    """Transformer models laboratory"""

    st.header("🤖 Transformer Models Laboratory")
    st.markdown(
        "Explore transformer architectures for financial time series prediction."
    )

    # Control Panel Section
    st.subheader("🎛️ Control Panel")

    # Task type selection
    task_type = st.radio(
        "Task Type:",
        options=["Classification", "Regression"],
        index=0,
        help="Select the type of machine learning task",
        key="transformer_task_type",
    )

    st.markdown("---")

    # Hyperparameters section
    st.subheader("🔧 Hyperparameters")
    hyperparams = get_transformer_hyperparameters()

    st.markdown("---")

    # Training configuration
    st.subheader("⚙️ Training Settings")
    training_config = get_training_configuration(key_prefix="transformer")

    st.markdown("---")

    # Training button
    if st.button(
        "🚀 Train Transformer",
        type="primary",
        use_container_width=True,
        key="train_transformer",
    ):
        train_transformer_model(
            feature_key=feature_key,
            task_type=task_type.lower(),
            hyperparams=hyperparams,
            training_config=training_config,
        )

    st.markdown("---")

    # Results & Analysis Section
    st.subheader("📊 Results & Analysis")

    # Display results if available
    model_key = f"transformer_{task_type.lower()}_{feature_key}"
    display_transformer_results(model_key, "Transformer")


def show_attention_lab(feature_key: str):
    """Attention mechanisms laboratory"""

    st.header("🧠 Attention Mechanisms Laboratory")
    st.markdown("Explore attention mechanisms for financial pattern recognition.")

    # Control Panel Section
    st.subheader("🎛️ Control Panel")

    col1, col2 = st.columns(2)

    with col1:
        # Task type selection
        task_type = st.radio(
            "Task Type:",
            options=["Classification", "Regression"],
            index=0,
            help="Select the type of machine learning task",
            key="attention_task_type",
        )

    with col2:
        # Attention type selection
        attention_type = st.selectbox(
            "Attention Type:",
            options=["basic", "multi_head", "temporal"],
            format_func=lambda x: {
                "basic": "Basic Attention",
                "multi_head": "Multi-Head Attention",
                "temporal": "Temporal Attention",
            }[x],
            help="Select the attention mechanism type",
        )

    st.markdown("---")

    # Hyperparameters section
    st.subheader("🔧 Hyperparameters")
    hyperparams = get_attention_hyperparameters(attention_type)

    st.markdown("---")

    # Training configuration
    st.subheader("⚙️ Training Settings")
    training_config = get_training_configuration(key_prefix="attention")

    st.markdown("---")

    # Training button
    if st.button(
        f"🚀 Train {attention_type.title()} Attention",
        type="primary",
        use_container_width=True,
        key="train_attention",
    ):
        train_attention_model(
            feature_key=feature_key,
            attention_type=attention_type,
            task_type=task_type.lower(),
            hyperparams=hyperparams,
            training_config=training_config,
        )

    st.markdown("---")

    # Results & Analysis Section
    st.subheader("📊 Results & Analysis")

    # Display results if available
    model_key = f"attention_{attention_type}_{task_type.lower()}_{feature_key}"
    display_attention_results(model_key, f"{attention_type.title()} Attention")


def show_meta_labeling_lab(feature_key: str):
    """Meta-labeling laboratory"""

    st.header("🎖️ Meta-Labeling Laboratory")
    st.markdown("Explore meta-labeling techniques for advanced position sizing.")

    # Control Panel Section
    st.subheader("🎛️ Control Panel")

    st.info(
        """
        📚 **Meta-Labeling**: Advanced technique from AFML Chapter 3
        
        Meta-labeling helps determine:
        1. **When** to act on a primary model's prediction
        2. **What size** position to take
        3. **How confident** we should be in the signal
        """
    )

    # Primary model selection
    primary_model_key = st.selectbox(
        "Primary Model:",
        options=list(st.session_state.get("model_cache", {}).keys()),
        help="Select a trained model to use as the primary predictor",
    )

    if not primary_model_key:
        st.warning(
            "⚠️ Please train a primary model first (e.g., in Traditional ML or Deep Learning labs)"
        )
        return

    st.markdown("---")

    # Hyperparameters section
    st.subheader("🔧 Meta-Labeling Parameters")
    meta_params = get_meta_labeling_parameters()

    st.markdown("---")

    # Training configuration
    st.subheader("⚙️ Training Settings")
    training_config = get_training_configuration(key_prefix="meta")

    st.markdown("---")

    # Training button
    if st.button(
        "🚀 Train Meta-Labeling Model",
        type="primary",
        use_container_width=True,
        key="train_meta",
    ):
        train_meta_labeling_model(
            feature_key=feature_key,
            primary_model_key=primary_model_key,
            meta_params=meta_params,
            training_config=training_config,
        )

    st.markdown("---")

    # Results & Analysis Section
    st.subheader("📊 Results & Analysis")

    # Display results if available
    model_key = f"meta_labeling_{feature_key}"
    display_meta_labeling_results(model_key, "Meta-Labeling")


def show_interpretation_lab(feature_key: str):
    """Model interpretation laboratory"""

    st.header("🔍 Model Interpretation Laboratory")
    st.markdown("Comprehensive model explainability and interpretation tools.")

    # Select model to interpret
    available_models = list(st.session_state.get("model_cache", {}).keys())

    if not available_models:
        st.warning("⚠️ No trained models available. Please train a model first.")
        return

    # Model selection
    model_key = st.selectbox(
        "Model to Interpret:",
        options=available_models,
        help="Select a trained model for interpretation analysis",
    )

    st.markdown("---")

    # Interpretation type selection
    interpretation_type = st.selectbox(
        "Analysis Type:",
        options=[
            "comprehensive",
            "feature_importance",
            "shap_analysis",
            "partial_dependence",
        ],
        format_func=lambda x: {
            "comprehensive": "Comprehensive Analysis",
            "feature_importance": "Feature Importance",
            "shap_analysis": "SHAP Analysis",
            "partial_dependence": "Partial Dependence",
        }[x],
        help="Select the type of interpretation analysis",
    )

    st.markdown("---")

    # Analysis button
    if st.button(
        f"🔍 Run {interpretation_type.title()} Analysis",
        type="primary",
        use_container_width=True,
        key="run_interpretation",
    ):
        run_interpretation_analysis(
            feature_key=feature_key,
            model_key=model_key,
            interpretation_type=interpretation_type,
        )

    st.markdown("---")

    # Results & Analysis Section
    st.subheader("📊 Interpretation Results")

    # Display interpretation results if available
    interpretation_key = f"{model_key}_{interpretation_type}"
    display_interpretation_results(interpretation_key, interpretation_type)


# Hyperparameter functions for each model type
def get_ensemble_hyperparameters(ensemble_method: str) -> Dict[str, Any]:
    """Get ensemble-specific hyperparameters"""

    hyperparams = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Basic Parameters**")
        hyperparams["n_estimators"] = st.slider(
            "Number of Estimators:",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of base estimators in the ensemble",
            key="ensemble_n_estimators",
        )

        hyperparams["max_depth"] = st.slider(
            "Max Depth:",
            min_value=3,
            max_value=20,
            value=None,
            step=1,
            help="Maximum depth of trees (None for unlimited)",
            key="ensemble_max_depth",
        )

    with col2:
        st.markdown("**Financial Parameters**")
        hyperparams["bootstrap_size"] = st.slider(
            "Bootstrap Size:",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Fraction of samples for bootstrap",
            key="ensemble_bootstrap_size",
        )

        hyperparams["purge_embargo_periods"] = st.slider(
            "Purge/Embargo Periods:",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            help="Periods to purge for financial time series",
            key="ensemble_purge_embargo",
        )

    with col3:
        st.markdown("**Advanced Parameters**")
        hyperparams["min_samples_split"] = st.slider(
            "Min Samples Split:",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            help="Minimum samples required to split a node",
            key="ensemble_min_samples_split",
        )

        if ensemble_method in ["stacking", "voting"]:
            hyperparams["cv_splits"] = st.slider(
                "CV Splits:",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                help="Number of cross-validation splits",
                key="ensemble_cv_splits",
            )

        if ensemble_method == "stacking":
            hyperparams["meta_model_type"] = st.selectbox(
                "Meta Model:",
                options=["rf", "lr", "svm"],
                help="Type of meta-model for stacking",
                key="ensemble_meta_model_type",
            )

        if ensemble_method == "voting":
            hyperparams["voting_type"] = st.selectbox(
                "Voting Type:",
                options=["soft", "hard"],
                help="Type of voting (soft uses probabilities)",
                key="ensemble_voting_type",
            )

    return hyperparams


def get_transformer_hyperparameters() -> Dict[str, Any]:
    """Get transformer-specific hyperparameters"""

    hyperparams = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Architecture Parameters**")
        hyperparams["d_model"] = st.slider(
            "Model Dimension:",
            min_value=32,
            max_value=512,
            value=64,
            step=32,
            help="Dimensionality of the model",
            key="transformer_d_model",
        )

        hyperparams["num_heads"] = st.selectbox(
            "Number of Heads:",
            options=[2, 4, 8, 16],
            index=2,
            help="Number of attention heads",
            key="transformer_num_heads",
        )

        hyperparams["num_layers"] = st.slider(
            "Number of Layers:",
            min_value=1,
            max_value=12,
            value=4,
            step=1,
            help="Number of transformer layers",
            key="transformer_num_layers",
        )

    with col2:
        st.markdown("**Sequence Parameters**")
        hyperparams["sequence_length"] = st.slider(
            "Sequence Length:",
            min_value=10,
            max_value=200,
            value=60,
            step=10,
            help="Length of input sequences",
            key="transformer_sequence_length",
        )

        hyperparams["dff"] = st.slider(
            "Feed-Forward Dimension:",
            min_value=64,
            max_value=1024,
            value=256,
            step=64,
            help="Dimension of feed-forward network",
            key="transformer_dff",
        )

    with col3:
        st.markdown("**Training Parameters**")
        hyperparams["dropout_rate"] = st.slider(
            "Dropout Rate:",
            min_value=0.0,
            max_value=0.8,
            value=0.1,
            step=0.1,
            help="Dropout rate for regularization",
            key="transformer_dropout_rate",
        )

        hyperparams["learning_rate"] = st.selectbox(
            "Learning Rate:",
            options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
            index=2,
            help="Learning rate for optimization",
            key="transformer_learning_rate",
        )

        hyperparams["batch_size"] = st.selectbox(
            "Batch Size:",
            options=[16, 32, 64, 128],
            index=1,
            help="Training batch size",
            key="transformer_batch_size",
        )

        hyperparams["epochs"] = st.slider(
            "Epochs:",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of training epochs",
            key="transformer_epochs",
        )

    return hyperparams


def get_attention_hyperparameters(attention_type: str) -> Dict[str, Any]:
    """Get attention-specific hyperparameters"""

    hyperparams = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Attention Parameters**")
        hyperparams["units"] = st.slider(
            "Attention Units:",
            min_value=32,
            max_value=512,
            value=64,
            step=32,
            help="Number of attention units",
            key="attention_units",
        )

        if attention_type == "multi_head":
            hyperparams["num_heads"] = st.selectbox(
                "Number of Heads:",
                options=[2, 4, 8, 16],
                index=2,
                help="Number of attention heads",
                key="attention_num_heads",
            )

    with col2:
        st.markdown("**Sequence Parameters**")
        hyperparams["sequence_length"] = st.slider(
            "Sequence Length:",
            min_value=10,
            max_value=200,
            value=60,
            step=10,
            help="Length of input sequences",
            key="attention_sequence_length",
        )

        if attention_type == "temporal":
            hyperparams["time_steps"] = hyperparams["sequence_length"]

    with col3:
        st.markdown("**Training Parameters**")
        hyperparams["dropout_rate"] = st.slider(
            "Dropout Rate:",
            min_value=0.0,
            max_value=0.8,
            value=0.1,
            step=0.1,
            help="Dropout rate for regularization",
            key="attention_dropout_rate",
        )

        hyperparams["batch_size"] = st.selectbox(
            "Batch Size:",
            options=[16, 32, 64, 128],
            index=1,
            help="Training batch size",
            key="attention_batch_size",
        )

        hyperparams["epochs"] = st.slider(
            "Epochs:",
            min_value=10,
            max_value=300,
            value=100,
            step=10,
            help="Number of training epochs",
            key="attention_epochs",
        )

    return hyperparams


def get_meta_labeling_parameters() -> Dict[str, Any]:
    """Get meta-labeling specific parameters"""

    params = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Barrier Parameters**")
        params["profit_target"] = st.slider(
            "Profit Target (%):",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Profit target for triple barrier",
            key="meta_profit_target",
        )

        params["stop_loss"] = st.slider(
            "Stop Loss (%):",
            min_value=0.5,
            max_value=10.0,
            value=1.5,
            step=0.5,
            help="Stop loss for triple barrier",
            key="meta_stop_loss",
        )

        params["max_holding_period"] = st.slider(
            "Max Holding Period:",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            help="Maximum holding period in periods",
            key="meta_max_holding",
        )

    with col2:
        st.markdown("**Meta-Model Parameters**")
        params["confidence_threshold"] = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Confidence threshold for position taking",
            key="meta_confidence_threshold",
        )

        params["meta_model_type"] = st.selectbox(
            "Meta Model Type:",
            options=["rf", "lr", "xgb"],
            help="Type of meta-model to use",
            key="meta_model_type",
        )

    return params


def get_training_configuration(key_prefix: str = "") -> Dict[str, Any]:
    """Get general training configuration"""

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        config["test_size"] = st.slider(
            "Test Size:",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Fraction of data to use for testing",
            key=f"{key_prefix}_test_size",
        )

        config["random_state"] = st.number_input(
            "Random State:",
            min_value=0,
            max_value=999,
            value=42,
            step=1,
            help="Random seed for reproducibility",
            key=f"{key_prefix}_random_state",
        )

    with col2:
        config["target_column"] = st.text_input(
            "Target Column:",
            value="target",
            help="Name of the target column",
            key=f"{key_prefix}_target_column",
        )

    return config


# Training functions for each model type
def train_ensemble_model(
    feature_key: str,
    ensemble_method: str,
    task_type: str,
    hyperparams: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Train ensemble model"""

    if not MANAGER_AVAILABLE:
        st.error("❌ Advanced models manager not available")
        return

    manager = AdvancedModelsManager()

    with st.spinner(f"Training {ensemble_method} ensemble model..."):
        success, message, model_id = manager.train_ensemble_model(
            feature_key=feature_key,
            ensemble_type=ensemble_method,
            task_type=task_type,
            hyperparams=hyperparams,
            training_config=training_config,
            session_state=st.session_state,
        )

    if success:
        st.success(message)
        st.balloons()
    else:
        st.error(message)


def train_transformer_model(
    feature_key: str,
    task_type: str,
    hyperparams: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Train transformer model"""

    if not MANAGER_AVAILABLE:
        st.error("❌ Advanced models manager not available")
        return

    manager = AdvancedModelsManager()

    with st.spinner("Training transformer model..."):
        success, message, model_id = manager.train_transformer_model(
            feature_key=feature_key,
            task_type=task_type,
            hyperparams=hyperparams,
            training_config=training_config,
            session_state=st.session_state,
        )

    if success:
        st.success(message)
        st.balloons()
    else:
        st.error(message)


def train_attention_model(
    feature_key: str,
    attention_type: str,
    task_type: str,
    hyperparams: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Train attention model"""

    if not MANAGER_AVAILABLE:
        st.error("❌ Advanced models manager not available")
        return

    manager = AdvancedModelsManager()

    with st.spinner(f"Training {attention_type} attention model..."):
        success, message, model_id = manager.train_attention_model(
            feature_key=feature_key,
            task_type=task_type,
            attention_type=attention_type,
            hyperparams=hyperparams,
            training_config=training_config,
            session_state=st.session_state,
        )

    if success:
        st.success(message)
        st.balloons()
    else:
        st.error(message)


def train_meta_labeling_model(
    feature_key: str,
    primary_model_key: str,
    meta_params: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Train meta-labeling model"""

    st.info("🚧 Meta-labeling implementation is in progress...")
    # TODO: Implement meta-labeling training


def run_interpretation_analysis(
    feature_key: str,
    model_key: str,
    interpretation_type: str,
):
    """Run model interpretation analysis"""

    if not MANAGER_AVAILABLE:
        st.error("❌ Advanced models manager not available")
        return

    manager = AdvancedModelsManager()

    with st.spinner(f"Running {interpretation_type} analysis..."):
        success, message, results = manager.analyze_model_interpretation(
            feature_key=feature_key,
            model_key=model_key,
            interpretation_type=interpretation_type,
            session_state=st.session_state,
        )

    if success:
        st.success(message)
        # Store results
        interpretation_key = f"{model_key}_{interpretation_type}"
        st.session_state.interpretation_results[interpretation_key] = {
            "results": results,
            "timestamp": datetime.now(),
            "interpretation_type": interpretation_type,
        }
    else:
        st.error(message)


# Result display functions
def display_ensemble_results(model_key: str, model_name: str):
    """Display ensemble model results"""

    if model_key not in st.session_state.get("training_results", {}):
        st.info("🎯 Train an ensemble model to see results here.")
        return

    results = st.session_state.training_results[model_key]["evaluation_results"]

    # Metrics display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Training Metrics")
        train_metrics = results.get("train_metrics", {})
        for metric, value in train_metrics.items():
            st.metric(label=metric.title(), value=f"{value:.4f}")

    with col2:
        st.markdown("### 📈 Test Metrics")
        test_metrics = results.get("test_metrics", {})
        for metric, value in test_metrics.items():
            st.metric(label=metric.title(), value=f"{value:.4f}")


def display_transformer_results(model_key: str, model_name: str):
    """Display transformer model results"""

    if model_key not in st.session_state.get("training_results", {}):
        st.info("🤖 Train a transformer model to see results here.")
        return

    results = st.session_state.training_results[model_key]["evaluation_results"]

    # Metrics display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Training Metrics")
        train_metrics = results.get("train_metrics", {})
        for metric, value in train_metrics.items():
            st.metric(label=metric.title(), value=f"{value:.4f}")

    with col2:
        st.markdown("### 📈 Test Metrics")
        test_metrics = results.get("test_metrics", {})
        for metric, value in test_metrics.items():
            st.metric(label=metric.title(), value=f"{value:.4f}")

    # Training history if available
    if "training_history" in results:
        st.markdown("### 📈 Training History")
        history = results["training_history"]

        if history:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Loss", "Accuracy/Metrics"],
            )

            # Loss plot
            if "loss" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["loss"],
                        name="Training Loss",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )

            if "val_loss" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["val_loss"],
                        name="Validation Loss",
                        line=dict(color="red"),
                    ),
                    row=1,
                    col=1,
                )

            # Accuracy plot
            if "accuracy" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["accuracy"],
                        name="Training Accuracy",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=2,
                )

            if "val_accuracy" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["val_accuracy"],
                        name="Validation Accuracy",
                        line=dict(color="orange"),
                    ),
                    row=1,
                    col=2,
                )

            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)


def display_attention_results(model_key: str, model_name: str):
    """Display attention model results"""

    if model_key not in st.session_state.get("training_results", {}):
        st.info("🧠 Train an attention model to see results here.")
        return

    results = st.session_state.training_results[model_key]["evaluation_results"]

    # Metrics display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Training Metrics")
        train_metrics = results.get("train_metrics", {})
        for metric, value in train_metrics.items():
            st.metric(label=metric.title(), value=f"{value:.4f}")

    with col2:
        st.markdown("### 📈 Test Metrics")
        test_metrics = results.get("test_metrics", {})
        for metric, value in test_metrics.items():
            st.metric(label=metric.title(), value=f"{value:.4f}")

    # Training history if available
    if "training_history" in results:
        st.markdown("### 📈 Training History")
        history = results["training_history"]

        if history:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Loss", "Accuracy/Metrics"],
            )

            # Loss plot
            if "loss" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["loss"],
                        name="Training Loss",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )

            if "val_loss" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["val_loss"],
                        name="Validation Loss",
                        line=dict(color="red"),
                    ),
                    row=1,
                    col=1,
                )

            # Accuracy plot
            if "accuracy" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["accuracy"],
                        name="Training Accuracy",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=2,
                )

            if "val_accuracy" in history:
                fig.add_trace(
                    go.Scatter(
                        y=history["val_accuracy"],
                        name="Validation Accuracy",
                        line=dict(color="orange"),
                    ),
                    row=1,
                    col=2,
                )

            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)


def display_meta_labeling_results(model_key: str, model_name: str):
    """Display meta-labeling results"""

    st.info("🚧 Meta-labeling results display is in progress...")


def display_interpretation_results(interpretation_key: str, interpretation_type: str):
    """Display model interpretation results"""

    if interpretation_key not in st.session_state.get("interpretation_results", {}):
        st.info("🔍 Run an interpretation analysis to see results here.")
        return

    results_data = st.session_state.interpretation_results[interpretation_key]
    results = results_data["results"]

    if interpretation_type == "feature_importance":
        st.markdown("### 🎯 Feature Importance Results")

        if "tree_importance" in results:
            st.markdown("**Tree-based Importance:**")
            importance_data = results["tree_importance"]
            if importance_data:
                importance_df = pd.DataFrame(
                    list(importance_data.items()), columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                st.bar_chart(importance_df.set_index("Feature"))

        if "permutation_importance" in results:
            st.markdown("**Permutation Importance:**")
            perm_data = results["permutation_importance"]
            if perm_data:
                perm_df = pd.DataFrame(perm_data).T
                st.dataframe(perm_df)

    elif interpretation_type == "comprehensive":
        st.markdown("### 🔍 Comprehensive Analysis Results")

        # Display summary statistics
        if "summary" in results:
            st.json(results["summary"])

        # Display feature importance if available
        if "feature_importance" in results:
            st.markdown("**Feature Importance:**")
            fi_data = results["feature_importance"]
            if isinstance(fi_data, dict):
                fi_df = pd.DataFrame(
                    list(fi_data.items()), columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                st.bar_chart(fi_df.set_index("Feature"))

    else:
        st.markdown(f"### 📊 {interpretation_type.title()} Results")
        st.json(results)


def show_debug_info():
    """Show debug information panel (expandable)"""

    with st.expander("🐛 Debug Information", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Session State Keys:**")
            for key in st.session_state.keys():
                st.write(f"- {key}")

        with col2:
            st.markdown("**Feature Cache:**")
            if "feature_cache" in st.session_state:
                for key, value in st.session_state.feature_cache.items():
                    if isinstance(value, dict) and "metadata" in value:
                        metadata = value["metadata"]
                        st.write(
                            f"- {key}: {metadata.get('sample_count', 'N/A')} samples"
                        )
                    else:
                        st.write(f"- {key}: {type(value).__name__}")

        st.markdown("**Model Cache:**")
        if "model_cache" in st.session_state:
            for key, value in st.session_state.model_cache.items():
                model_type = value.get("model_type", "Unknown")
                task_type = value.get("task_type", "Unknown")
                timestamp = value.get("timestamp", "Unknown")
                st.write(f"- {key}: {model_type} ({task_type}) - {timestamp}")


if __name__ == "__main__":
    main()
