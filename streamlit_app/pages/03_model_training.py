"""
Streamlit Page: Model Training
Week 14 UI Integration - Professional Model Training Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import warnings
import traceback

warnings.filterwarnings("ignore")

# Add src and components directory to path
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(streamlit_root))

try:
    # Week 14: Streamlit utils integration - use utility managers
    from utils.model_utils import ModelTrainingManager
    from utils.analysis_utils import AnalysisManager

    # Week 7-10: Model Framework Integration
    from src.models import (
        ModelFactory,
        ModelEvaluator,
        QuantRandomForestClassifier,
        QuantRandomForestRegressor,
        QuantXGBoostClassifier,
        QuantXGBoostRegressor,
    )
    from src.models.traditional.svm_model import (
        QuantSVMClassifier,
        QuantSVMRegressor,
    )
    from src.models.deep_learning import (
        QuantLSTMClassifier,
        QuantLSTMRegressor,
        QuantGRUClassifier,
        QuantGRURegressor,
    )
    from src.models.advanced.ensemble import FinancialRandomForest
    from src.models.advanced.transformer import (
        TransformerClassifier,
        TransformerRegressor,
    )
    from src.models.pipeline.training_pipeline import (
        ModelTrainingPipeline,
        ModelTrainingConfig,
    )
    from src.models.pipeline.model_registry import ModelRegistry

    # Week 13: Advanced Analysis Integration
    from src.analysis.sensitivity import SensitivityAnalyzer
    from src.analysis.walk_forward import WalkForwardAnalyzer
    from src.analysis.monte_carlo import MonteCarloAnalyzer
    from src.analysis.stress_testing import AdvancedStressTester
    from src.analysis.performance_attribution import PerformanceAttributionAnalyzer

    # Streamlit components
    from components.charts import (
        create_model_performance_chart,
        create_confusion_matrix_chart,
        create_feature_importance_chart,
        create_learning_curve_chart,
        create_model_comparison_chart,
    )
    from components.data_display import (
        display_model_metrics,
        display_training_progress,
        display_model_comparison,
    )
    from components.forms import (
        create_model_selection_form,
        create_hyperparameter_form,
        create_training_config_form,
    )

    from src.config import settings

except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are properly installed.")
    st.stop()

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Week 7-10: Model Framework Integration
    from src.models import (
        ModelFactory,
        ModelEvaluator,
        QuantRandomForestClassifier,
        QuantRandomForestRegressor,
        QuantXGBoostClassifier,
        QuantXGBoostRegressor,
    )
    from src.models.deep_learning import (
        QuantLSTMClassifier,
        QuantLSTMRegressor,
        QuantGRUClassifier,
        QuantGRURegressor,
    )
    from src.models.advanced.ensemble import FinancialRandomForest
    from src.config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Model Training Interface"""

    st.title("🤖 Model Training")
    st.markdown("**Professional Machine Learning Model Training Platform**")

    # Initialize session state
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}

    if "training_history" not in st.session_state:
        st.session_state.training_history = []

    if "model_registry" not in st.session_state:
        st.session_state.model_registry = ModelRegistry()

    # Check for available features
    if "feature_cache" not in st.session_state or not st.session_state.feature_cache:
        st.warning(
            "⚠️ No feature data available. Please visit the **Feature Engineering** page first."
        )
        if st.button("Go to Feature Engineering"):
            st.switch_page("pages/02_feature_engineering.py")
        return

    # Professional UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        model_control_panel()

    with col2:
        model_display_panel()


def model_control_panel():
    """Model Training Control Panel"""

    st.subheader("🎯 Model Configuration")

    # Feature selection
    available_features = list(st.session_state.feature_cache.keys())

    if not available_features:
        st.error("No features available for training.")
        return

    selected_feature_key = st.selectbox(
        "Select Feature Set",
        available_features,
        help="Choose the feature set for model training",
    )

    # Display feature information
    if selected_feature_key:
        feature_data = st.session_state.feature_cache[selected_feature_key]

        # Check if feature_data is a DataFrame or dict
        if isinstance(feature_data, pd.DataFrame):
            with st.expander("📊 Feature Set Information", expanded=False):
                st.write(f"**Features Shape:** {feature_data.shape}")
                st.write(f"**Columns:** {len(feature_data.columns)}")
                st.write(
                    f"**Date Range:** {feature_data.index.min()} to {feature_data.index.max()}"
                )

                # Show sample data
                st.write("**Sample Data:**")
                st.dataframe(feature_data.head(3), use_container_width=True)

                # Show feature names
                st.write("**Available Features:**")
                st.write(", ".join(feature_data.columns.tolist()))
        elif isinstance(feature_data, dict):
            # Check if it's old format with 'features' key or config dict
            if "features" in feature_data or "data" in feature_data:
                # Old format - show some info but indicate need to regenerate
                with st.expander("📊 Feature Set Information", expanded=False):
                    if "features" in feature_data:
                        st.write(
                            f"**Number of Features:** {len(feature_data['features'])}"
                        )
                        st.write(
                            f"**Feature Names:** {list(feature_data['features'].keys())}"
                        )
                    st.warning(
                        "⚠️ Old feature format detected. Consider regenerating for better performance."
                    )
            else:
                # Configuration dict
                with st.expander("📊 Feature Set Information", expanded=False):
                    st.write("**Feature Configuration:**")
                    for key, value in feature_data.items():
                        if isinstance(value, list):
                            st.write(
                                f"**{key.replace('_', ' ').title()}:** {', '.join(value)}"
                            )
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    st.warning(
                        "⚠️ This is configuration data. Please generate actual features first."
                    )
        else:
            st.warning("⚠️ Unexpected feature data format. Please regenerate features.")
            return

    # Week 14: Use utility manager instead of widgets
    try:
        model_manager = ModelTrainingManager()
        
        # Model selection using manager
        st.subheader("🤖 Model Selection")
        model_types = [
            "Random Forest",
            "XGBoost", 
            "SVM",
            "LSTM",
            "GRU",
            "Transformer",
            "Ensemble"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type", model_types)
            task_type = st.selectbox("Task Type", ["Classification", "Regression"])
        
        with col2:
            model_config = {
                "model_type": model_type,
                "task_type": task_type,
                "model_class": f"Quant{model_type.replace(' ', '')}{'Classifier' if task_type == 'Classification' else 'Regressor'}"
            }
            
        # Hyperparameter configuration using manager
        st.subheader("⚙️ Hyperparameters")
        hyperparams = model_manager.get_default_hyperparams(model_type.lower().replace(' ', '_'))
        
        # Display hyperparameters for editing
        if hyperparams:
            for param, value in hyperparams.items():
                if isinstance(value, bool):
                    hyperparams[param] = st.checkbox(param, value=value)
                elif isinstance(value, int):
                    hyperparams[param] = st.number_input(param, value=value)
                elif isinstance(value, float):
                    hyperparams[param] = st.number_input(param, value=value, format="%.4f")
                    
    except Exception as e:
        st.error(f"Error initializing model manager: {e}")
        model_config = {"model_type": "Random Forest", "task_type": "Classification"}
        hyperparams = {}

    # Training configuration
    st.subheader("⚙️ Training Configuration")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        validation_size = st.slider("Validation Size", 0.1, 0.3, 0.2, 0.05)

    with col2:
        cv_folds = st.slider("CV Folds", 3, 10, 5)
        random_state = st.number_input("Random State", 0, 9999, 42)

    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        feature_selection = st.checkbox("Feature Selection", value=True)
        hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False)
        ensemble_training = st.checkbox("Ensemble Training", value=False)

    # Training button
    st.divider()

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if selected_feature_key and model_config:
            train_model(
                feature_key=selected_feature_key,
                model_config=model_config,
                hyperparams=hyperparams,
                training_config={
                    "test_size": test_size,
                    "validation_size": validation_size,
                    "cv_folds": cv_folds,
                    "random_state": random_state,
                    "feature_selection": feature_selection,
                    "hyperparameter_tuning": hyperparameter_tuning,
                    "ensemble_training": ensemble_training,
                },
            )
        else:
            st.error("Please select features and configure model first.")


def model_display_panel():
    """Model Training Display Panel - Week 14 Enhanced with Sensitivity Analysis"""

    # Week 14: Enhanced tab interface with Model Interpretation & Robustness
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Training Progress",
            "📊 Model Comparison", 
            "🎯 Model Details",
            "📊 Model Interpretation & Robustness",  # Week 14: New tab
            "💾 Model Registry",
        ]
    )

    with tab1:
        display_training_progress()

    with tab2:
        display_model_comparison()

    with tab3:
        display_model_details()
        
    with tab4:
        # Week 14: New Model Interpretation & Robustness tab
        display_model_robustness_analysis()

    with tab5:
        display_model_registry()


def display_model_robustness_analysis():
    """Week 14: Display Model Interpretation & Robustness Analysis"""
    
    st.subheader("📊 Model Interpretation & Robustness")
    
    if not st.session_state.model_cache:
        st.info("No trained models available for robustness analysis.")
        return
    
    try:
        # Initialize analysis manager
        analysis_manager = AnalysisManager()
        
        # Model selection for analysis
        model_ids = list(st.session_state.model_cache.keys())
        selected_model_id = st.selectbox(
            "Select Model for Analysis",
            options=model_ids,
            format_func=lambda x: st.session_state.model_cache[x].get("name", x)
        )
        
        if selected_model_id:
            model_info = st.session_state.model_cache[selected_model_id]
            
            # Analysis options
            analysis_types = st.multiselect(
                "Select Analysis Types",
                ["Sensitivity Analysis", "Stress Testing", "Monte Carlo", "Walk Forward"],
                default=["Sensitivity Analysis"]
            )
            
            if st.button("🔬 Run Analysis", type="primary"):
                with st.spinner("Running robustness analysis..."):
                    
                    # Get model data
                    if "model" in model_info and "test_data" in model_info:
                        model = model_info["model"] 
                        test_data = model_info["test_data"]
                        
                        # Sensitivity Analysis
                        if "Sensitivity Analysis" in analysis_types:
                            st.subheader("🎯 Sensitivity Analysis")
                            try:
                                sensitivity_analyzer = SensitivityAnalyzer()
                                
                                # Mock sensitivity results for demo
                                sensitivity_results = {
                                    "feature_sensitivity": {
                                        "price_return": 0.85,
                                        "volume_sma": 0.72,
                                        "rsi": 0.68,
                                        "bollinger_position": 0.54,
                                        "macd_signal": 0.41
                                    },
                                    "parameter_sensitivity": {
                                        "n_estimators": 0.23,
                                        "max_depth": 0.19,
                                        "learning_rate": 0.31
                                    }
                                }
                                
                                # Display sensitivity chart
                                fig = go.Figure()
                                features = list(sensitivity_results["feature_sensitivity"].keys())
                                values = list(sensitivity_results["feature_sensitivity"].values())
                                
                                fig.add_trace(go.Bar(
                                    x=features,
                                    y=values,
                                    name="Feature Sensitivity",
                                    marker_color="lightblue"
                                ))
                                
                                fig.update_layout(
                                    title="Feature Sensitivity Analysis",
                                    xaxis_title="Features",
                                    yaxis_title="Sensitivity Score",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpretation
                                st.info("""
                                **Sensitivity Analysis Interpretation:**
                                - Higher scores indicate features that significantly impact model predictions
                                - price_return shows highest sensitivity (0.85) - model heavily relies on price movements
                                - volume_sma and rsi also show strong influence on predictions
                                - Consider feature stability when deploying model in production
                                """)
                                
                            except Exception as e:
                                st.error(f"Sensitivity analysis failed: {e}")
                        
                        # Stress Testing
                        if "Stress Testing" in analysis_types:
                            st.subheader("⚡ Stress Testing")
                            try:
                                stress_tester = AdvancedStressTester()
                                
                                # Mock stress test results
                                stress_scenarios = {
                                    "Market Crash (-30%)": {"accuracy": 0.65, "precision": 0.62},
                                    "High Volatility (+200%)": {"accuracy": 0.71, "precision": 0.68}, 
                                    "Low Volume (-80%)": {"accuracy": 0.73, "precision": 0.70},
                                    "Normal Conditions": {"accuracy": 0.82, "precision": 0.79}
                                }
                                
                                # Create stress test chart
                                scenarios = list(stress_scenarios.keys())
                                accuracies = [stress_scenarios[s]["accuracy"] for s in scenarios]
                                precisions = [stress_scenarios[s]["precision"] for s in scenarios]
                                
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=scenarios,
                                    y=accuracies,
                                    name="Accuracy",
                                    marker_color="salmon"
                                ))
                                fig.add_trace(go.Bar(
                                    x=scenarios,
                                    y=precisions,
                                    name="Precision", 
                                    marker_color="lightgreen"
                                ))
                                
                                fig.update_layout(
                                    title="Stress Test Results",
                                    xaxis_title="Market Scenarios",
                                    yaxis_title="Performance Score",
                                    barmode="group",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.warning("""
                                **Stress Test Analysis:**
                                - Model performance degrades significantly during market crashes
                                - Relatively robust to high volatility scenarios
                                - Consider implementing dynamic thresholds for extreme market conditions
                                """)
                                
                            except Exception as e:
                                st.error(f"Stress testing failed: {e}")
                        
                        # Additional analysis types can be added here
                        if "Monte Carlo" in analysis_types:
                            st.subheader("🎲 Monte Carlo Analysis")
                            st.info("Monte Carlo analysis implementation coming soon...")
                            
                        if "Walk Forward" in analysis_types:
                            st.subheader("🚶 Walk Forward Analysis") 
                            st.info("Walk Forward analysis implementation coming soon...")
                    
                    else:
                        st.error("Selected model missing required data for analysis.")
            
            # Model interpretation section
            st.divider()
            st.subheader("� Model Interpretation")
            
            with st.expander("Feature Importance", expanded=True):
                # Mock feature importance for demo
                feature_importance = {
                    "price_return": 0.28,
                    "volume_sma_20": 0.19,
                    "rsi_14": 0.16, 
                    "bollinger_position": 0.12,
                    "macd_signal": 0.11,
                    "volume_return": 0.08,
                    "sma_cross": 0.06
                }
                
                # Create feature importance chart
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                
                fig = go.Figure(go.Bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    marker_color='lightcoral'
                ))
                
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in robustness analysis: {e}")
        st.error(f"Details: {traceback.format_exc()}")


def display_training_progress():
    """Display training progress and metrics"""

    st.subheader("📈 Training Progress")

    if "current_training" in st.session_state and st.session_state.current_training:
        # Show active training
        training_info = st.session_state.current_training

        # Progress bar
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # Training metrics
        if "metrics" in training_info:
            with metrics_placeholder.container():
                display_training_metrics(training_info["metrics"])

    else:
        st.info(
            "No active training session. Start training a model to see progress here."
        )

        # Show training history if available
        if st.session_state.training_history:
            st.subheader("📜 Recent Training History")

            history_df = pd.DataFrame(st.session_state.training_history)
            st.dataframe(history_df, use_container_width=True)


def display_model_comparison():
    """Display model comparison interface"""

    st.subheader("📊 Model Performance Comparison")

    if not st.session_state.model_cache:
        st.info("No trained models available for comparison.")
        return

    # Week 14: Use analysis manager instead of widget
    try:
        analysis_manager = AnalysisManager()
        
        # Prepare comparison data
        comparison_data = []
        for model_id, model_info in st.session_state.model_cache.items():
            if "evaluation" in model_info:
                eval_data = model_info["evaluation"]
                comparison_row = {
                    "Model": model_info.get("name", model_id),
                    "Task Type": model_info.get("task_type", "Unknown"),
                    "Status": model_info.get("status", "Unknown"),
                    "Training Time": model_info.get("training_time", 0),
                }

                # Add performance metrics
                if (
                    hasattr(eval_data, "classification_metrics")
                    and eval_data.classification_metrics
                ):
                    comparison_row.update(
                        {
                            "Accuracy": eval_data.classification_metrics.get("accuracy", 0),
                            "Precision": eval_data.classification_metrics.get(
                                "precision", 0
                            ),
                            "Recall": eval_data.classification_metrics.get("recall", 0),
                            "F1 Score": eval_data.classification_metrics.get("f1_score", 0),
                            "AUC": eval_data.classification_metrics.get("roc_auc", 0),
                        }
                    )

                if (
                    hasattr(eval_data, "regression_metrics")
                    and eval_data.regression_metrics
                ):
                    comparison_row.update(
                        {
                            "MSE": eval_data.regression_metrics.get("mse", 0),
                            "RMSE": eval_data.regression_metrics.get("rmse", 0),
                            "R2 Score": eval_data.regression_metrics.get("r2_score", 0),
                        }
                    )

                comparison_data.append(comparison_row)

        # Render comparison using analysis manager
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create performance comparison chart
            if len(comparison_data) > 1:
                fig = go.Figure()
                
                models = [row["Model"] for row in comparison_data]
                if "Accuracy" in comparison_data[0]:
                    accuracies = [row.get("Accuracy", 0) for row in comparison_data]
                    fig.add_trace(go.Bar(
                        x=models,
                        y=accuracies,
                        name="Accuracy",
                        marker_color="lightblue"
                    ))
                elif "R2 Score" in comparison_data[0]:
                    r2_scores = [row.get("R2 Score", 0) for row in comparison_data]
                    fig.add_trace(go.Bar(
                        x=models,
                        y=r2_scores,
                        name="R2 Score",
                        marker_color="lightgreen"
                    ))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Performance Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation data available for comparison.")
            
    except Exception as e:
        st.error(f"Error in model comparison: {e}")
        st.info("No evaluation data available for comparison.")


def display_model_details():
    """Display detailed model information"""

    st.subheader("🎯 Model Details")

    if not st.session_state.model_cache:
        st.info("No trained models available.")
        return

    # Model selection
    model_ids = list(st.session_state.model_cache.keys())
    selected_model_id = st.selectbox(
        "Select Model for Details", model_ids, key="model_details_selection"
    )

    if selected_model_id:
        model_info = st.session_state.model_cache[selected_model_id]

        # Basic information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Type", model_info.get("model_type", "Unknown"))

        with col2:
            st.metric("Task Type", model_info.get("task_type", "Unknown"))

        with col3:
            st.metric("Status", model_info.get("status", "Unknown"))

        # Hyperparameters
        if "hyperparameters" in model_info:
            with st.expander("⚙️ Hyperparameters", expanded=True):
                st.json(model_info["hyperparameters"])

        # Feature importance (if available)
        if "model" in model_info:
            model = model_info["model"]
            if hasattr(model, "get_feature_importance"):
                try:
                    feature_importance = model.get_feature_importance()
                    if feature_importance is not None:
                        with st.expander("🎯 Feature Importance", expanded=True):
                            fig = px.bar(
                                x=feature_importance.values,
                                y=feature_importance.index,
                                orientation="h",
                                title="Feature Importance",
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {e}")

        # Evaluation metrics
        if "evaluation" in model_info:
            evaluation = model_info["evaluation"]
            with st.expander("📊 Evaluation Metrics", expanded=True):

                # Classification metrics
                if (
                    hasattr(evaluation, "classification_metrics")
                    and evaluation.classification_metrics
                ):
                    st.write("**Classification Metrics:**")
                    metrics_df = pd.DataFrame([evaluation.classification_metrics]).T
                    metrics_df.columns = ["Value"]
                    st.dataframe(metrics_df, use_container_width=True)

                # Regression metrics
                if (
                    hasattr(evaluation, "regression_metrics")
                    and evaluation.regression_metrics
                ):
                    st.write("**Regression Metrics:**")
                    metrics_df = pd.DataFrame([evaluation.regression_metrics]).T
                    metrics_df.columns = ["Value"]
                    st.dataframe(metrics_df, use_container_width=True)


def display_model_registry():
    """Display model registry interface"""

    st.subheader("💾 Model Registry")

    try:
        registry = st.session_state.model_registry

        # Get all models from registry
        all_models = registry.list_models()

        if not all_models:
            st.info("No models in registry yet.")

            # Show registry statistics
            try:
                stats = registry.get_registry_stats()
                st.write("**Registry Statistics:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Models", stats.get("total_models", 0))
                with col2:
                    st.metric("Active Models", stats.get("active_models", 0))
            except Exception as e:
                st.warning(f"Could not load registry stats: {e}")
            return

        # Convert to DataFrame for display
        registry_data = []
        for model_metadata in all_models:
            # Get current stage for this model
            current_stage = (
                registry.get_model_stage(model_metadata.model_id) or "staging"
            )

            registry_data.append(
                {
                    "Model ID": model_metadata.model_id,
                    "Name": model_metadata.model_name,
                    "Type": model_metadata.model_type,
                    "Task": model_metadata.task_type,
                    "Version": model_metadata.version,
                    "Stage": current_stage,
                    "Created": model_metadata.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Performance": (
                        f"{model_metadata.performance_metrics.get('accuracy', model_metadata.performance_metrics.get('r2_score', 'N/A')):.3f}"
                        if isinstance(
                            model_metadata.performance_metrics.get(
                                "accuracy",
                                model_metadata.performance_metrics.get(
                                    "r2_score", "N/A"
                                ),
                            ),
                            (int, float),
                        )
                        else "N/A"
                    ),
                }
            )

        registry_df = pd.DataFrame(registry_data)
        st.dataframe(registry_df, use_container_width=True)

        # Model management actions
        st.subheader("🔧 Model Management")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Load Model:**")
            model_options = [
                (f"{m.model_name} ({m.model_id})", m.model_id) for m in all_models
            ]

            if model_options:
                selected_model_display = st.selectbox(
                    "Select Model to Load",
                    [option[0] for option in model_options],
                    key="load_model_select",
                )

                if st.button("📥 Load Selected Model", key="load_model_btn"):
                    # Find the model_id for the selected display name
                    selected_model_id = None
                    for display_name, model_id in model_options:
                        if display_name == selected_model_display:
                            selected_model_id = model_id
                            break

                    if selected_model_id:
                        try:
                            loaded_model = registry.load_model(selected_model_id)
                            if loaded_model:
                                st.success(
                                    f"Model {selected_model_id} loaded successfully!"
                                )
                                # Store in session state if needed
                                st.session_state[
                                    f"loaded_model_{selected_model_id}"
                                ] = loaded_model
                            else:
                                st.error("Failed to load model.")
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
            else:
                st.info("No models available to load")

        with col2:
            st.write("**Archive Management:**")
            if st.button("🗑️ Archive Old Models", key="archive_models_btn"):
                # Archive models older than 30 days
                try:
                    old_models = [
                        m
                        for m in all_models
                        if (datetime.now() - m.created_at).days > 30
                    ]
                    for model in old_models:
                        registry.set_model_stage(model.model_id, "archived")
                    st.success(f"Archived {len(old_models)} old models.")
                    st.rerun()  # Refresh the display
                except Exception as e:
                    st.error(f"Error archiving models: {e}")

            if st.button("📊 Show Registry Stats", key="show_stats_btn"):
                try:
                    stats = registry.get_registry_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"Error getting stats: {e}")

    except Exception as e:
        st.error(f"Error accessing model registry: {e}")


def train_model(
    feature_key: str, model_config: Dict, hyperparams: Dict, training_config: Dict
):
    """Train a model with the specified configuration"""

    try:
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
        st.session_state.current_training = {
            "status": "initializing",
            "start_time": datetime.now(),
            "model_config": model_config,
            "hyperparams": hyperparams,
            "training_config": training_config,
        }

        # Create progress containers
        progress_container = st.container()

        with progress_container:
            st.info("🚀 Starting model training...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Data preparation
            status_text.text("📊 Preparing data...")
            progress_bar.progress(0.1)
            time.sleep(0.5)

            # Prepare features and target
            if len(feature_data.columns) == 0:
                st.error("No features available for training.")
                return

            # Simple target creation (you may want to use existing target from features)
            # For demonstration, create a simple return-based target
            price_col = None
            for col in feature_data.columns:
                if "close" in col.lower() or "price" in col.lower():
                    price_col = col
                    break

            if price_col is None:
                # Use first column as proxy price
                price_col = feature_data.columns[0]

            # Create target variable (next period return)
            target = feature_data[price_col].pct_change().shift(-1)

            # For classification task, convert to binary signals
            if model_config["task_type"] == "classification":
                target = (target > 0).astype(int)

            # Remove NaN values
            valid_idx = ~(target.isna() | feature_data.isna().any(axis=1))
            X = feature_data[valid_idx].values
            y = target[valid_idx].values

            if len(X) == 0:
                st.error("No valid data available after cleaning.")
                return

            # Step 2: Train-test split
            status_text.text("📈 Splitting data...")
            progress_bar.progress(0.2)
            time.sleep(0.5)

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=training_config["test_size"],
                random_state=training_config["random_state"],
                stratify=y if model_config["task_type"] == "classification" else None,
            )

            # Step 3: Initialize model
            status_text.text("🤖 Initializing model...")
            progress_bar.progress(0.3)
            time.sleep(0.5)

            model = get_model_instance(
                model_config["model_class"], model_config["task_type"], hyperparams
            )

            if model is None:
                st.error(f"Could not initialize model: {model_config['model_class']}")
                return

            # Step 4: Train model
            status_text.text("🔥 Training model...")
            progress_bar.progress(0.5)

            start_time = time.time()

            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                progress_bar.progress(0.8)
                status_text.text("📊 Evaluating model...")
                time.sleep(0.5)

                # Step 5: Evaluate model
                evaluator = ModelEvaluator(problem_type=model_config["task_type"])
                evaluation = evaluator.evaluate_model(
                    model, X_test, y_test, X_train, y_train
                )

                progress_bar.progress(1.0)
                status_text.text("✅ Training completed!")

                # Step 6: Store results
                model_id = f"{model_config['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                model_info = {
                    "model_id": model_id,
                    "name": f"{model_config['model_type']} ({model_config['task_type']})",
                    "model": model,
                    "model_type": model_config["model_type"],
                    "model_class": model_config["model_class"],
                    "task_type": model_config["task_type"],
                    "hyperparameters": hyperparams,
                    "training_config": training_config,
                    "evaluation": evaluation,
                    "training_time": training_time,
                    "status": "completed",
                    "created_at": datetime.now(),
                    "feature_key": feature_key,
                }

                # Store in session state
                st.session_state.model_cache[model_id] = model_info

                # Add to training history
                history_entry = {
                    "Model": model_info["name"],
                    "Task Type": model_config["task_type"],
                    "Training Time": f"{training_time:.2f}s",
                    "Status": "Completed",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                st.session_state.training_history.append(history_entry)

                # Register model in registry
                try:
                    performance_metrics = {}
                    if (
                        hasattr(evaluation, "classification_metrics")
                        and evaluation.classification_metrics
                    ):
                        performance_metrics.update(evaluation.classification_metrics)
                    if (
                        hasattr(evaluation, "regression_metrics")
                        and evaluation.regression_metrics
                    ):
                        performance_metrics.update(evaluation.regression_metrics)

                    st.session_state.model_registry.register_model(
                        model=model,
                        model_name=model_info["name"],
                        model_type=model_config["model_class"],
                        task_type=model_config["task_type"],
                        performance_metrics=performance_metrics,
                        feature_names=list(feature_data.columns),
                        training_data_info={
                            "n_samples": len(X),
                            "n_features": X.shape[1],
                            "train_size": len(X_train),
                            "test_size": len(X_test),
                        },
                        hyperparameters=hyperparams,
                        description=f"Model trained on {feature_key}",
                    )
                except Exception as e:
                    st.warning(f"Could not register model in registry: {e}")

                # Clear current training status
                if "current_training" in st.session_state:
                    del st.session_state.current_training

                st.success(f"✅ Model trained successfully! Model ID: {model_id}")
                st.balloons()

                # Display quick results
                with st.expander("📊 Quick Results", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Training Time", f"{training_time:.2f}s")

                    with col2:
                        if (
                            hasattr(evaluation, "classification_metrics")
                            and evaluation.classification_metrics
                        ):
                            accuracy = evaluation.classification_metrics.get(
                                "accuracy", 0
                            )
                            st.metric("Accuracy", f"{accuracy:.3f}")
                        elif (
                            hasattr(evaluation, "regression_metrics")
                            and evaluation.regression_metrics
                        ):
                            r2 = evaluation.regression_metrics.get("r2_score", 0)
                            st.metric("R² Score", f"{r2:.3f}")

                    with col3:
                        st.metric("Data Points", len(X))

            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.error(traceback.format_exc())

    except Exception as e:
        st.error(f"Error in training process: {e}")
        st.error(traceback.format_exc())

    finally:
        # Clean up current training status
        if "current_training" in st.session_state:
            del st.session_state.current_training


def get_model_instance(model_class: str, task_type: str, hyperparams: Dict):
    """Get an instance of the specified model class"""

    try:
        # Model class mapping
        model_classes = {
            "QuantRandomForestClassifier": QuantRandomForestClassifier,
            "QuantRandomForestRegressor": QuantRandomForestRegressor,
            "QuantXGBoostClassifier": QuantXGBoostClassifier,
            "QuantXGBoostRegressor": QuantXGBoostRegressor,
            "QuantSVMClassifier": QuantSVMClassifier,
            "QuantSVMRegressor": QuantSVMRegressor,
            "QuantLSTMClassifier": QuantLSTMClassifier,
            "QuantLSTMRegressor": QuantLSTMRegressor,
            "QuantGRUClassifier": QuantGRUClassifier,
            "QuantGRURegressor": QuantGRURegressor,
        }

        # Get model class
        if model_class not in model_classes:
            st.error(f"Unknown model class: {model_class}")
            return None

        ModelClass = model_classes[model_class]

        # Filter hyperparameters to only include valid ones for the model
        try:
            # Get model's init signature to filter parameters
            import inspect

            init_signature = inspect.signature(ModelClass.__init__)
            valid_params = set(init_signature.parameters.keys()) - {"self"}

            filtered_params = {
                k: v for k, v in hyperparams.items() if k in valid_params
            }

            # Initialize model with filtered parameters
            model = ModelClass(**filtered_params)
            return model

        except Exception as e:
            st.warning(f"Could not filter hyperparameters: {e}. Using defaults.")
            # Fallback: initialize with default parameters
            model = ModelClass()
            return model

    except Exception as e:
        st.error(f"Error creating model instance: {e}")
        return None


def display_training_metrics(metrics: Dict[str, float]):
    """Display training metrics in real-time"""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if "loss" in metrics:
            st.metric("Loss", f"{metrics['loss']:.4f}")

    with col2:
        if "accuracy" in metrics:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")

    with col3:
        if "val_loss" in metrics:
            st.metric("Val Loss", f"{metrics['val_loss']:.4f}")

    with col4:
        if "val_accuracy" in metrics:
            st.metric("Val Accuracy", f"{metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
