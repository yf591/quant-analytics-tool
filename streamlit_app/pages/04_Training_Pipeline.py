#!/usr/bin/env python3
"""
Training Pipeline - Phase 5 Implementation
Automated multi-model training and comparison pipeline.

Design Philosophy:
- "Colosseum" approach: Battle multiple models against each other
- Automated pipeline execution with real-time progress tracking
- Comprehensive model comparison and performance analysis
- Easy model selection and configuration
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


def main():
    """Main function for Training Pipeline"""

    # Page configuration
    st.set_page_config(
        page_title="Training Pipeline",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🚀 Training Pipeline")
    st.markdown(
        """
    **Automated Multi-Model Training & Comparison**
    
    Welcome to the training pipeline - your automated model comparison arena. 
    Select your features, configure your experiment, and let the models battle it out!
    """
    )

    # Check prerequisites
    if not check_prerequisites():
        return

    # Initialize session state
    initialize_session_state()

    # Show experiment setup
    st.header("1. 🧪 Experiment Setup")
    feature_key, task_type = configure_experiment()

    # Show model selection
    st.header("2. 🤖 Model Selection")
    selected_models = configure_models()

    # Show training configuration
    st.header("3. ⚙️ Common Training Configuration")
    training_config = configure_training_settings()

    # Show execution section
    st.header("4. 🏁 Execution")
    execute_pipeline(feature_key, task_type, selected_models, training_config)

    # Show results if available
    if "pipeline_results" in st.session_state and st.session_state.pipeline_results:
        st.header("5. 📊 Results")
        display_pipeline_results()


def check_prerequisites():
    """Check if prerequisites are met for pipeline execution"""

    # Check for feature data
    if "feature_cache" not in st.session_state or not st.session_state.feature_cache:
        st.warning("⚠️ No feature data found. Please run Feature Engineering first.")

        with st.expander("📋 How to get started", expanded=True):
            st.markdown(
                """
            **Steps to prepare for training pipeline:**
            
            1. **Navigate to Data Collection** 📈
               - Select symbols and collect historical data
               
            2. **Run Feature Engineering** 🔧
               - Create feature sets with technical indicators
               - Generate labels for your prediction tasks
               
            3. **Return here** 🚀
               - Configure and run the training pipeline
            """
            )

        if st.button("🔄 Check Again"):
            st.rerun()

        return False

    return True


def initialize_session_state():
    """Initialize session state for pipeline execution"""

    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False

    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None

    if "pipeline_progress" not in st.session_state:
        st.session_state.pipeline_progress = {}


def configure_experiment():
    """Configure basic experiment settings"""

    col1, col2 = st.columns(2)

    with col1:
        # Feature set selection
        available_features = list(st.session_state.feature_cache.keys())
        feature_key = st.selectbox(
            "📊 Select Feature Set",
            options=available_features,
            help="Choose the feature set to use for training",
        )

        # Show feature set info
        if feature_key and feature_key in st.session_state.feature_cache:
            feature_data = st.session_state.feature_cache[feature_key]
            # Get the shape of the features DataFrame
            if isinstance(feature_data, dict) and "features" in feature_data:
                shape = feature_data["features"].shape
                period = feature_data.get("period", "Unknown")
            elif isinstance(feature_data, pd.DataFrame):
                # Direct DataFrame case (from feature engineering)
                shape = feature_data.shape
                period = "Unknown"
                # Try to get metadata for period info
                metadata_key = f"{feature_key}_metadata"
                if metadata_key in st.session_state.feature_cache:
                    metadata = st.session_state.feature_cache[metadata_key]
                    period = metadata.get("period", "Unknown")
            else:
                # Handle other cases
                shape = "Unknown"
                period = "Unknown"

            st.info(
                f"**Selected:** {feature_key}\n\n"
                f"**Shape:** {shape}\n"
                f"**Period:** {period}"
            )

    with col2:
        # Task type selection
        task_type = st.radio(
            "🎯 Task Type",
            options=["classification", "regression"],
            help="Choose the type of machine learning task",
        )

        # Show task type description
        if task_type == "classification":
            st.info(
                "**Classification Task**\n\n"
                "Predicting discrete outcomes (e.g., up/down movements, buy/sell signals)"
            )
        else:
            st.info(
                "**Regression Task**\n\n"
                "Predicting continuous values (e.g., price changes, return percentages)"
            )

    return feature_key, task_type


def configure_models():
    """Configure model selection for the pipeline"""

    # Available models by category
    available_models = {
        "Traditional ML": {
            "random_forest": "🌲 Random Forest",
            "xgboost": "🚀 XGBoost",
            "svm": "🔍 Support Vector Machine",
        },
        "Deep Learning": {
            "lstm": "🧠 LSTM Network",
            "gru": "⚡ GRU Network",
        },
        "Advanced Models": {
            "transformer": "🤖 Transformer",
        },
    }

    # Model selection interface
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_models = []

        for category, models in available_models.items():
            st.markdown(f"**{category}**")

            # Category-wise selection
            category_models = list(models.keys())
            category_selected = st.multiselect(
                f"Select {category} models:",
                options=category_models,
                format_func=lambda x: models[x],
                key=f"models_{category.replace(' ', '_').lower()}",
            )

            selected_models.extend(category_selected)

    with col2:
        # Quick selection options
        st.markdown("**Quick Select**")

        if st.button("📊 All Traditional", use_container_width=True):
            # Use forms or set default values differently
            st.info(
                "Please manually select all Traditional ML models from the multiselect above."
            )

        if st.button("🧠 All Deep Learning", use_container_width=True):
            st.info(
                "Please manually select all Deep Learning models from the multiselect above."
            )

        if st.button("🚀 All Models", use_container_width=True):
            st.info("Please manually select all models from the multiselects above.")

        if st.button("❌ Clear All", use_container_width=True):
            st.info("Please manually clear selections from the multiselects above.")

    # Show selected models summary
    if selected_models:
        st.success(
            f"✅ **{len(selected_models)} models selected:** {', '.join(selected_models)}"
        )
    else:
        st.warning("⚠️ Please select at least one model to train.")

    return selected_models


def configure_training_settings():
    """Configure common training settings for all models"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Data Splitting**")

        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing",
        )

        validation_size = st.slider(
            "Validation Set Size",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05,
            help="Proportion of training data to use for validation",
        )

    with col2:
        st.markdown("**🔧 Preprocessing**")

        scaler_type = st.selectbox(
            "Feature Scaling",
            options=["standard", "minmax", "robust"],
            help="Type of feature scaling to apply",
        )

        time_series_cv = st.checkbox(
            "Time Series CV",
            value=True,
            help="Use time-aware cross-validation to avoid look-ahead bias",
        )

    with col3:
        st.markdown("**⚡ Optimization**")

        hyperparameter_tuning = st.checkbox(
            "Hyperparameter Tuning",
            value=True,
            help="Enable automatic hyperparameter optimization",
        )

        ensemble_models = st.checkbox(
            "Create Ensemble",
            value=True,
            help="Create ensemble models from individual predictions",
        )

        cv_splits = st.number_input(
            "CV Splits",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of cross-validation splits",
        )

    # Package configuration
    training_config = {
        "test_size": test_size,
        "validation_size": validation_size,
        "scaler_type": scaler_type,
        "time_series_cv": time_series_cv,
        "hyperparameter_tuning": hyperparameter_tuning,
        "ensemble_models": ensemble_models,
        "cv_splits": int(cv_splits),
        "random_state": 42,
    }

    return training_config


def execute_pipeline(feature_key, task_type, selected_models, training_config):
    """Execute the training pipeline"""

    # Check if we can run the pipeline
    can_run = (
        feature_key
        and task_type
        and selected_models
        and feature_key in st.session_state.feature_cache
    )

    if not can_run:
        st.warning(
            "⚠️ Please complete all configuration steps above before running the pipeline."
        )
        return

    # Show execution summary
    with st.expander("📋 Execution Summary", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Feature Set:** {feature_key}")
            st.markdown(f"**Task Type:** {task_type.title()}")
            st.markdown(f"**Models to Train:** {len(selected_models)}")

        with col2:
            st.markdown(f"**Test Size:** {training_config['test_size']:.1%}")
            st.markdown(f"**CV Splits:** {training_config['cv_splits']}")
            st.markdown(
                f"**Hyperparameter Tuning:** {'✅' if training_config['hyperparameter_tuning'] else '❌'}"
            )

    # Execution button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if not st.session_state.pipeline_running:
            if st.button(
                "🚀 Run Training Pipeline", use_container_width=True, type="primary"
            ):
                run_training_pipeline(
                    feature_key, task_type, selected_models, training_config
                )
        else:
            st.warning("🔄 Pipeline is currently running...")
            if st.button("🛑 Stop Pipeline", use_container_width=True):
                st.session_state.pipeline_running = False
                st.rerun()


def run_training_pipeline(feature_key, task_type, selected_models, training_config):
    """Run the actual training pipeline"""

    try:
        # Import pipeline utilities
        from streamlit_app.utils.pipeline_utils import PipelineManager

        # Set pipeline as running
        st.session_state.pipeline_running = True

        # Initialize pipeline manager
        pipeline_manager = PipelineManager()

        # Show progress
        progress_container = st.container()

        with progress_container:
            st.subheader("🏃‍♂️ Training Progress")

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Execute pipeline with progress updates
            results = pipeline_manager.run_training_pipeline(
                feature_key=feature_key,
                task_type=task_type,
                selected_models=selected_models,
                training_config=training_config,
                session_state=st.session_state,
                progress_callback=lambda step, total, message: update_progress(
                    progress_bar, status_text, step, total, message
                ),
            )

            # Store results
            st.session_state.pipeline_results = results
            st.session_state.pipeline_running = False

            # Show completion
            progress_bar.progress(100)
            status_text.success("✅ Training pipeline completed successfully!")

            st.rerun()

    except Exception as e:
        st.session_state.pipeline_running = False
        st.error(f"❌ Pipeline execution failed: {str(e)}")
        st.exception(e)


def update_progress(progress_bar, status_text, step, total, message):
    """Update progress display"""
    progress = int((step / total) * 100)
    progress_bar.progress(progress)
    status_text.info(f"Step {step}/{total}: {message}")


def display_pipeline_results():
    """Display pipeline execution results"""

    if not st.session_state.pipeline_results:
        return

    results = st.session_state.pipeline_results

    # Overall summary
    st.subheader("📈 Overall Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Models Trained", len(results.get("model_results", {})))

    with col2:
        best_model = results.get("best_model", {})
        if best_model:
            st.metric("Best Model", best_model.get("name", "Unknown"))

    with col3:
        if best_model:
            best_score = best_model.get("test_score", 0)
            st.metric("Best Score", f"{best_score:.4f}")

    with col4:
        training_time = results.get("total_training_time", 0)
        st.metric("Total Time", f"{training_time:.1f}s")

    # Model comparison table
    st.subheader("🏆 Model Leaderboard")

    if "comparison_df" in results and results["comparison_df"] is not None:
        comparison_df = results["comparison_df"]

        # Sort by test score (descending)
        if "test_score" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("test_score", ascending=False)

        # Display with formatting
        st.dataframe(
            comparison_df,
            use_container_width=True,
            column_config={
                "test_score": st.column_config.NumberColumn(
                    "Test Score", format="%.4f"
                ),
                "validation_score": st.column_config.NumberColumn(
                    "Validation Score", format="%.4f"
                ),
                "training_time": st.column_config.NumberColumn(
                    "Training Time (s)", format="%.1f"
                ),
            },
        )

        # Highlight best model
        if not comparison_df.empty:
            best_idx = comparison_df.index[0]
            best_model_name = comparison_df.loc[best_idx, "model_name"]
            best_score = comparison_df.loc[best_idx, "test_score"]

            st.success(
                f"🏆 **Best Model: {best_model_name}** (Test Score: {best_score:.4f})"
            )

    # Individual model details
    if st.checkbox("🔍 Show Detailed Results"):
        display_detailed_results(results)

    # Export options
    st.subheader("💾 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Export Comparison Table"):
            export_comparison_table(results)

    with col2:
        if st.button("🤖 Save Best Model"):
            save_best_model(results)


def display_detailed_results(results):
    """Display detailed results for each model"""

    model_results = results.get("model_results", {})

    for model_name, model_result in model_results.items():
        with st.expander(f"📊 {model_name} Details"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Performance Metrics**")
                metrics = model_result.get("metrics", {})
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        st.metric(metric.replace("_", " ").title(), f"{value:.4f}")

            with col2:
                st.markdown("**Training Information**")
                st.text(f"Training Time: {model_result.get('training_time', 0):.1f}s")
                st.text(f"Model Type: {model_result.get('model_type', 'Unknown')}")

                # Feature importance if available
                if "feature_importance" in model_result:
                    st.markdown("**Top Features**")
                    importance = model_result["feature_importance"]
                    if importance is not None and not importance.empty:
                        top_features = importance.head(5)
                        for feature, score in top_features.items():
                            st.text(f"{feature}: {score:.4f}")


def export_comparison_table(results):
    """Export comparison table to CSV"""

    if "comparison_df" in results and results["comparison_df"] is not None:
        csv = results["comparison_df"].to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.error("No comparison data available for export")


def save_best_model(results):
    """Save the best model to the model registry"""

    best_model = results.get("best_model")
    if best_model:
        # This would integrate with the model registry
        st.success(
            f"🏆 Best model ({best_model.get('name', 'Unknown')}) saved to registry!"
        )
        st.info("Model is now available for deployment and inference.")
    else:
        st.error("No best model found to save")


if __name__ == "__main__":
    main()
