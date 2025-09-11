"""
Model UI Widgets for Streamlit Application

This module provides specialized UI widgets for model training interface
including model selection, hyperparameter adjustment, and model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class ModelSelectionWidget:
    """Widget for model selection and configuration"""

    def __init__(self):
        self.model_categories = {
            "Traditional ML": {
                "Random Forest": {
                    "classifier": "QuantRandomForestClassifier",
                    "regressor": "QuantRandomForestRegressor",
                },
                "XGBoost": {
                    "classifier": "QuantXGBoostClassifier",
                    "regressor": "QuantXGBoostRegressor",
                },
                "SVM": {
                    "classifier": "QuantSVMClassifier",
                    "regressor": "QuantSVMRegressor",
                },
            },
            "Deep Learning": {
                "LSTM": {
                    "classifier": "QuantLSTMClassifier",
                    "regressor": "QuantLSTMRegressor",
                },
                "GRU": {
                    "classifier": "QuantGRUClassifier",
                    "regressor": "QuantGRURegressor",
                },
            },
            "Advanced Models": {
                "Transformer": {
                    "classifier": "TransformerClassifier",
                    "regressor": "TransformerRegressor",
                },
                "Financial Random Forest": {
                    "classifier": "FinancialRandomForest",
                    "regressor": "FinancialRandomForest",
                },
            },
        }

    def render_model_selection(self, key_prefix: str = "") -> Dict[str, Any]:
        """Render model selection interface"""
        st.subheader("🎯 Model Selection")

        # Task type selection
        task_type = st.selectbox(
            "Select Task Type",
            ["Classification", "Regression"],
            key=f"{key_prefix}_task_type",
            help="Choose the type of machine learning task",
        )

        # Model category selection
        category = st.selectbox(
            "Select Model Category",
            list(self.model_categories.keys()),
            key=f"{key_prefix}_category",
            help="Choose the category of machine learning models",
        )

        # Model type selection
        available_models = list(self.model_categories[category].keys())
        model_type = st.selectbox(
            "Select Model Type",
            available_models,
            key=f"{key_prefix}_model_type",
            help="Choose the specific model algorithm",
        )

        # Get the actual model class name
        task_key = "classifier" if task_type == "Classification" else "regressor"
        model_class = self.model_categories[category][model_type][task_key]

        return {
            "task_type": task_type.lower(),
            "category": category,
            "model_type": model_type,
            "model_class": model_class,
        }


class HyperparameterWidget:
    """Widget for hyperparameter configuration"""

    def __init__(self):
        self.hyperparameters = {
            "QuantRandomForestClassifier": {
                "n_estimators": {
                    "type": "slider",
                    "min": 10,
                    "max": 500,
                    "default": 100,
                },
                "max_depth": {"type": "slider", "min": 3, "max": 20, "default": 10},
                "min_samples_split": {
                    "type": "slider",
                    "min": 2,
                    "max": 20,
                    "default": 2,
                },
                "min_samples_leaf": {
                    "type": "slider",
                    "min": 1,
                    "max": 10,
                    "default": 1,
                },
                "max_features": {
                    "type": "selectbox",
                    "options": ["sqrt", "log2", "auto"],
                    "default": "sqrt",
                },
                "class_weight": {
                    "type": "selectbox",
                    "options": ["balanced", "balanced_subsample", None],
                    "default": "balanced_subsample",
                },
            },
            "QuantRandomForestRegressor": {
                "n_estimators": {
                    "type": "slider",
                    "min": 10,
                    "max": 500,
                    "default": 100,
                },
                "max_depth": {"type": "slider", "min": 3, "max": 20, "default": 10},
                "min_samples_split": {
                    "type": "slider",
                    "min": 2,
                    "max": 20,
                    "default": 2,
                },
                "min_samples_leaf": {
                    "type": "slider",
                    "min": 1,
                    "max": 10,
                    "default": 1,
                },
                "max_features": {
                    "type": "selectbox",
                    "options": ["sqrt", "log2", "auto"],
                    "default": "sqrt",
                },
            },
            "QuantXGBoostClassifier": {
                "n_estimators": {
                    "type": "slider",
                    "min": 10,
                    "max": 500,
                    "default": 100,
                },
                "max_depth": {"type": "slider", "min": 3, "max": 15, "default": 6},
                "learning_rate": {
                    "type": "slider",
                    "min": 0.01,
                    "max": 0.3,
                    "default": 0.1,
                },
                "subsample": {"type": "slider", "min": 0.5, "max": 1.0, "default": 1.0},
                "colsample_bytree": {
                    "type": "slider",
                    "min": 0.5,
                    "max": 1.0,
                    "default": 1.0,
                },
                "reg_alpha": {"type": "slider", "min": 0.0, "max": 1.0, "default": 0.0},
                "reg_lambda": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 1.0,
                },
            },
            "QuantXGBoostRegressor": {
                "n_estimators": {
                    "type": "slider",
                    "min": 10,
                    "max": 500,
                    "default": 100,
                },
                "max_depth": {"type": "slider", "min": 3, "max": 15, "default": 6},
                "learning_rate": {
                    "type": "slider",
                    "min": 0.01,
                    "max": 0.3,
                    "default": 0.1,
                },
                "subsample": {"type": "slider", "min": 0.5, "max": 1.0, "default": 1.0},
                "colsample_bytree": {
                    "type": "slider",
                    "min": 0.5,
                    "max": 1.0,
                    "default": 1.0,
                },
                "reg_alpha": {"type": "slider", "min": 0.0, "max": 1.0, "default": 0.0},
                "reg_lambda": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 1.0,
                },
            },
            "QuantSVMClassifier": {
                "C": {"type": "slider", "min": 0.1, "max": 100.0, "default": 1.0},
                "kernel": {
                    "type": "selectbox",
                    "options": ["rbf", "linear", "poly", "sigmoid"],
                    "default": "rbf",
                },
                "gamma": {
                    "type": "selectbox",
                    "options": ["scale", "auto"],
                    "default": "scale",
                },
                "degree": {"type": "slider", "min": 2, "max": 6, "default": 3},
            },
            "QuantSVMRegressor": {
                "C": {"type": "slider", "min": 0.1, "max": 100.0, "default": 1.0},
                "kernel": {
                    "type": "selectbox",
                    "options": ["rbf", "linear", "poly", "sigmoid"],
                    "default": "rbf",
                },
                "gamma": {
                    "type": "selectbox",
                    "options": ["scale", "auto"],
                    "default": "scale",
                },
                "epsilon": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1},
            },
            "QuantLSTMClassifier": {
                "lstm_units": {"type": "slider", "min": 32, "max": 256, "default": 64},
                "sequence_length": {
                    "type": "slider",
                    "min": 10,
                    "max": 100,
                    "default": 60,
                },
                "dropout_rate": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 0.5,
                    "default": 0.2,
                },
                "epochs": {"type": "slider", "min": 10, "max": 200, "default": 50},
                "batch_size": {"type": "slider", "min": 16, "max": 128, "default": 32},
                "bidirectional": {"type": "checkbox", "default": False},
            },
            "QuantLSTMRegressor": {
                "lstm_units": {"type": "slider", "min": 32, "max": 256, "default": 64},
                "sequence_length": {
                    "type": "slider",
                    "min": 10,
                    "max": 100,
                    "default": 60,
                },
                "dropout_rate": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 0.5,
                    "default": 0.2,
                },
                "epochs": {"type": "slider", "min": 10, "max": 200, "default": 50},
                "batch_size": {"type": "slider", "min": 16, "max": 128, "default": 32},
                "bidirectional": {"type": "checkbox", "default": False},
            },
            "QuantGRUClassifier": {
                "gru_units": {"type": "slider", "min": 32, "max": 256, "default": 64},
                "sequence_length": {
                    "type": "slider",
                    "min": 10,
                    "max": 100,
                    "default": 60,
                },
                "dropout_rate": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 0.5,
                    "default": 0.2,
                },
                "epochs": {"type": "slider", "min": 10, "max": 200, "default": 50},
                "batch_size": {"type": "slider", "min": 16, "max": 128, "default": 32},
                "bidirectional": {"type": "checkbox", "default": False},
            },
            "QuantGRURegressor": {
                "gru_units": {"type": "slider", "min": 32, "max": 256, "default": 64},
                "sequence_length": {
                    "type": "slider",
                    "min": 10,
                    "max": 100,
                    "default": 60,
                },
                "dropout_rate": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 0.5,
                    "default": 0.2,
                },
                "epochs": {"type": "slider", "min": 10, "max": 200, "default": 50},
                "batch_size": {"type": "slider", "min": 16, "max": 128, "default": 32},
                "bidirectional": {"type": "checkbox", "default": False},
            },
        }

    def render_hyperparameters(
        self, model_class: str, key_prefix: str = ""
    ) -> Dict[str, Any]:
        """Render hyperparameter configuration interface"""
        st.subheader("⚙️ Hyperparameter Configuration")

        if model_class not in self.hyperparameters:
            st.warning(f"Hyperparameters not configured for {model_class}")
            return {}

        params = {}
        hyperparams = self.hyperparameters[model_class]

        col1, col2 = st.columns(2)

        for i, (param_name, config) in enumerate(hyperparams.items()):
            # Alternate between columns
            current_col = col1 if i % 2 == 0 else col2

            with current_col:
                if config["type"] == "slider":
                    if "step" in config:
                        step = config["step"]
                    else:
                        # Auto-determine step based on data type
                        min_val, max_val = config["min"], config["max"]
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            step = 1
                        else:
                            step = 0.01

                    params[param_name] = st.slider(
                        param_name.replace("_", " ").title(),
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"],
                        step=step,
                        key=f"{key_prefix}_{param_name}",
                    )

                elif config["type"] == "selectbox":
                    params[param_name] = st.selectbox(
                        param_name.replace("_", " ").title(),
                        config["options"],
                        index=(
                            config["options"].index(config["default"])
                            if config["default"] in config["options"]
                            else 0
                        ),
                        key=f"{key_prefix}_{param_name}",
                    )

                elif config["type"] == "checkbox":
                    params[param_name] = st.checkbox(
                        param_name.replace("_", " ").title(),
                        value=config["default"],
                        key=f"{key_prefix}_{param_name}",
                    )

        return params


class ModelComparisonWidget:
    """Widget for model comparison and visualization"""

    def __init__(self):
        pass

    def render_comparison_table(
        self, comparison_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Render model comparison table"""
        if not comparison_data:
            st.info("No models to compare yet.")
            return pd.DataFrame()

        df = pd.DataFrame(comparison_data)

        # Reorder columns for better presentation
        preferred_order = [
            "Model",
            "Task Type",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC",
            "MSE",
            "RMSE",
            "R2 Score",
            "Training Time",
            "Status",
        ]

        # Reorder columns that exist in the dataframe
        existing_cols = [col for col in preferred_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in existing_cols]
        df = df[existing_cols + other_cols]

        st.subheader("📊 Model Comparison")
        st.dataframe(df, use_container_width=True)

        return df

    def render_performance_charts(self, comparison_data: List[Dict[str, Any]]):
        """Render performance comparison charts"""
        if not comparison_data or len(comparison_data) < 2:
            return

        df = pd.DataFrame(comparison_data)

        # Check if we have classification or regression metrics
        has_classification = any(
            col in df.columns for col in ["Accuracy", "Precision", "Recall", "F1 Score"]
        )
        has_regression = any(col in df.columns for col in ["MSE", "RMSE", "R2 Score"])

        if has_classification:
            self._render_classification_charts(df)

        if has_regression:
            self._render_regression_charts(df)

        # Training time comparison
        if "Training Time" in df.columns:
            self._render_training_time_chart(df)

    def _render_classification_charts(self, df: pd.DataFrame):
        """Render classification performance charts"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Accuracy", "Precision", "Recall", "F1 Score"),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for metric, (row, col) in zip(metrics, positions):
            if metric in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df["Model"],
                        y=df[metric],
                        name=metric,
                        showlegend=False,
                        text=df[metric].round(3),
                        textposition="auto",
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title="Classification Performance Comparison",
            height=600,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_regression_charts(self, df: pd.DataFrame):
        """Render regression performance charts"""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("MSE", "RMSE", "R² Score"),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        )

        metrics = ["MSE", "RMSE", "R2 Score"]
        for i, metric in enumerate(metrics, 1):
            if metric in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df["Model"],
                        y=df[metric],
                        name=metric,
                        showlegend=False,
                        text=df[metric].round(3),
                        textposition="auto",
                    ),
                    row=1,
                    col=i,
                )

        fig.update_layout(
            title="Regression Performance Comparison", height=400, showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_training_time_chart(self, df: pd.DataFrame):
        """Render training time comparison chart"""
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df["Model"],
                y=df["Training Time"],
                name="Training Time",
                text=df["Training Time"].round(2),
                textposition="auto",
                marker_color="lightblue",
            )
        )

        fig.update_layout(
            title="Training Time Comparison",
            xaxis_title="Model",
            yaxis_title="Training Time (seconds)",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)


class ProgressWidget:
    """Widget for displaying training progress"""

    def __init__(self):
        pass

    def create_progress_container(self):
        """Create a container for progress display"""
        return st.container()

    def update_progress(
        self,
        container,
        step: int,
        total_steps: int,
        message: str = "",
        details: Dict[str, Any] = None,
    ):
        """Update progress display"""
        with container:
            # Clear previous content
            st.empty()

            # Progress bar
            progress = step / total_steps
            st.progress(progress)

            # Progress text
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{message}")
            with col2:
                st.text(f"{step}/{total_steps}")

            # Additional details
            if details:
                with st.expander("Training Details", expanded=False):
                    for key, value in details.items():
                        st.text(f"{key}: {value}")

    def display_training_metrics(self, metrics: Dict[str, float]):
        """Display real-time training metrics"""
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
