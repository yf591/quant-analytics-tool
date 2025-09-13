"""
Forms Component Module

This module provides form components for the Streamlit application.
Includes data selection forms, parameter configuration forms, and file upload forms.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta


def create_data_selection_form(
    available_data: List[str],
    default_selection: Optional[str] = None,
    title: str = "Data Selection",
) -> Optional[str]:
    """
    Create a form for selecting data from available options.

    Args:
        available_data: List of available data identifiers
        default_selection: Default selection
        title: Form title

    Returns:
        Selected data identifier or None
    """
    try:
        st.subheader(title)

        if not available_data:
            st.warning("No data available. Please collect data first.")
            return None

        # Data selection
        selected_data = st.selectbox(
            "Select Dataset:",
            options=available_data,
            index=(
                available_data.index(default_selection)
                if default_selection in available_data
                else 0
            ),
            help="Choose the dataset to work with",
        )

        return selected_data

    except Exception as e:
        st.error(f"Error creating data selection form: {str(e)}")
        return None


def create_parameter_form(
    parameter_config: Dict[str, Dict[str, Any]], title: str = "Parameter Configuration"
) -> Dict[str, Any]:
    """
    Create a dynamic parameter configuration form.

    Args:
        parameter_config: Configuration for parameters
        title: Form title

    Returns:
        Dictionary of parameter values
    """
    try:
        st.subheader(title)

        parameters = {}

        for param_name, config in parameter_config.items():
            param_type = config.get("type", "number")
            label = config.get("label", param_name)
            default = config.get("default", 0)
            help_text = config.get("help", "")

            if param_type == "number":
                min_val = config.get("min", 0)
                max_val = config.get("max", 100)
                step = config.get("step", 1)
                parameters[param_name] = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    step=step,
                    help=help_text,
                )

            elif param_type == "slider":
                min_val = config.get("min", 0)
                max_val = config.get("max", 100)
                step = config.get("step", 1)
                parameters[param_name] = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    step=step,
                    help=help_text,
                )

            elif param_type == "checkbox":
                parameters[param_name] = st.checkbox(
                    label, value=default, help=help_text
                )

            elif param_type == "selectbox":
                options = config.get("options", [])
                parameters[param_name] = st.selectbox(
                    label,
                    options=options,
                    index=options.index(default) if default in options else 0,
                    help=help_text,
                )

            elif param_type == "multiselect":
                options = config.get("options", [])
                parameters[param_name] = st.multiselect(
                    label,
                    options=options,
                    default=default if isinstance(default, list) else [],
                    help=help_text,
                )

            elif param_type == "text":
                parameters[param_name] = st.text_input(
                    label, value=str(default), help=help_text
                )

        return parameters

    except Exception as e:
        st.error(f"Error creating parameter form: {str(e)}")
        return {}


def create_technical_indicators_form() -> Dict[str, Any]:
    """
    Create a specialized form for technical indicators configuration.

    Returns:
        Dictionary of technical indicator parameters
    """
    try:
        st.subheader("🔧 Technical Indicators Configuration")

        indicators = {}

        # Moving Averages
        with st.expander("Moving Averages", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                indicators["sma_enabled"] = st.checkbox(
                    "Simple Moving Average (SMA)", value=True
                )
                if indicators["sma_enabled"]:
                    indicators["sma_period"] = st.slider("SMA Period", 5, 50, 20)

            with col2:
                indicators["ema_enabled"] = st.checkbox(
                    "Exponential Moving Average (EMA)", value=True
                )
                if indicators["ema_enabled"]:
                    indicators["ema_period"] = st.slider("EMA Period", 5, 50, 12)

        # Momentum Indicators
        with st.expander("Momentum Indicators", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                indicators["rsi_enabled"] = st.checkbox("RSI", value=True)
                if indicators["rsi_enabled"]:
                    indicators["rsi_period"] = st.slider("RSI Period", 5, 30, 14)

            with col2:
                indicators["momentum_enabled"] = st.checkbox("Momentum", value=False)
                if indicators["momentum_enabled"]:
                    indicators["momentum_period"] = st.slider(
                        "Momentum Period", 5, 20, 10
                    )

        # MACD
        with st.expander("MACD", expanded=True):
            indicators["macd_enabled"] = st.checkbox("MACD", value=True)
            if indicators["macd_enabled"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    indicators["macd_fast"] = st.slider("Fast Period", 5, 20, 12)
                with col2:
                    indicators["macd_slow"] = st.slider("Slow Period", 15, 35, 26)
                with col3:
                    indicators["macd_signal"] = st.slider("Signal Period", 5, 15, 9)

        # Bollinger Bands
        with st.expander("Bollinger Bands"):
            indicators["bb_enabled"] = st.checkbox("Bollinger Bands", value=False)
            if indicators["bb_enabled"]:
                col1, col2 = st.columns(2)
                with col1:
                    indicators["bb_period"] = st.slider("BB Period", 10, 30, 20)
                with col2:
                    indicators["bb_std"] = st.slider(
                        "Standard Deviations", 1.0, 3.0, 2.0
                    )

        # Volatility Indicators
        with st.expander("Volatility Indicators"):
            indicators["atr_enabled"] = st.checkbox(
                "Average True Range (ATR)", value=False
            )
            if indicators["atr_enabled"]:
                indicators["atr_period"] = st.slider("ATR Period", 5, 25, 14)

        # Oscillators
        with st.expander("Oscillators"):
            col1, col2 = st.columns(2)

            with col1:
                indicators["stoch_enabled"] = st.checkbox("Stochastic", value=False)
                if indicators["stoch_enabled"]:
                    indicators["stoch_k"] = st.slider("Stochastic %K", 5, 20, 14)
                    indicators["stoch_d"] = st.slider("Stochastic %D", 2, 10, 3)

            with col2:
                indicators["williams_enabled"] = st.checkbox("Williams %R", value=False)
                if indicators["williams_enabled"]:
                    indicators["williams_period"] = st.slider(
                        "Williams %R Period", 5, 25, 14
                    )

        return indicators

    except Exception as e:
        st.error(f"Error creating technical indicators form: {str(e)}")
        return {}


def create_advanced_features_form() -> Dict[str, Any]:
    """
    Create a specialized form for advanced features configuration.

    Returns:
        Dictionary of advanced feature parameters
    """
    try:
        st.subheader("🧠 Advanced Features Configuration")

        features = {}

        # Fractal Analysis
        with st.expander("Fractal Analysis", expanded=True):
            features["fractal_enabled"] = st.checkbox("Fractal Dimension", value=True)
            if features["fractal_enabled"]:
                col1, col2 = st.columns(2)
                with col1:
                    features["fractal_window"] = st.slider(
                        "Fractal Window", 20, 200, 100
                    )
                with col2:
                    features["fractal_method"] = st.selectbox(
                        "Fractal Method", options=["higuchi", "box_counting"], index=0
                    )

        # Hurst Exponent
        with st.expander("Hurst Exponent", expanded=True):
            features["hurst_enabled"] = st.checkbox("Hurst Exponent", value=True)
            if features["hurst_enabled"]:
                col1, col2 = st.columns(2)
                with col1:
                    features["hurst_window"] = st.slider("Hurst Window", 50, 300, 100)
                with col2:
                    features["hurst_method"] = st.selectbox(
                        "Hurst Method", options=["rs", "dfa"], index=0
                    )

        # Information Bars
        with st.expander("Information-Driven Bars"):
            features["info_bars_enabled"] = st.checkbox("Information Bars", value=False)
            if features["info_bars_enabled"]:
                col1, col2 = st.columns(2)
                with col1:
                    features["bar_type"] = st.selectbox(
                        "Bar Type", options=["volume", "tick", "dollar"], index=0
                    )
                with col2:
                    features["bar_threshold"] = st.number_input(
                        "Threshold (auto if 0)",
                        min_value=0.0,
                        value=0.0,
                        help="Leave 0 for automatic threshold calculation",
                    )

        # Fractional Differentiation
        with st.expander("Fractional Differentiation"):
            features["frac_diff_enabled"] = st.checkbox(
                "Fractional Differentiation", value=False
            )
            if features["frac_diff_enabled"]:
                col1, col2 = st.columns(2)
                with col1:
                    features["frac_diff_d"] = st.slider(
                        "Differentiation Order (d)", 0.0, 1.0, 0.4, 0.1
                    )
                with col2:
                    features["frac_diff_threshold"] = st.slider(
                        "Weight Threshold", 0.001, 0.1, 0.01
                    )

        return features

    except Exception as e:
        st.error(f"Error creating advanced features form: {str(e)}")
        return {}


def create_file_upload_form(
    accepted_types: List[str] = ["csv", "xlsx", "json"], title: str = "File Upload"
) -> Optional[pd.DataFrame]:
    """
    Create a form for file upload and processing.

    Args:
        accepted_types: List of accepted file extensions
        title: Form title

    Returns:
        Uploaded DataFrame or None
    """
    try:
        st.subheader(title)

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=accepted_types,
            help=f"Accepted formats: {', '.join(accepted_types)}",
        )

        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                file_extension = uploaded_file.name.split(".")[-1].lower()

                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension == "xlsx":
                    df = pd.read_excel(uploaded_file)
                elif file_extension == "json":
                    df = pd.read_json(uploaded_file)
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    return None

                # Display file info
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

                # Show preview
                with st.expander("Preview Data"):
                    st.dataframe(df.head())

                return df

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None

        return None

    except Exception as e:
        st.error(f"Error creating file upload form: {str(e)}")
        return None


def create_date_range_form(
    default_start: Optional[datetime] = None,
    default_end: Optional[datetime] = None,
    title: str = "Date Range Selection",
) -> Tuple[datetime, datetime]:
    """
    Create a form for date range selection.

    Args:
        default_start: Default start date
        default_end: Default end date
        title: Form title

    Returns:
        Tuple of (start_date, end_date)
    """
    try:
        st.subheader(title)

        # Set defaults if not provided
        if default_end is None:
            default_end = datetime.now().date()
        if default_start is None:
            default_start = default_end - timedelta(days=365)

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                help="Select the start date for data collection",
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                help="Select the end date for data collection",
            )

        # Validate date range
        if start_date >= end_date:
            st.warning("Start date must be before end date")

        return start_date, end_date

    except Exception as e:
        st.error(f"Error creating date range form: {str(e)}")
        return datetime.now().date() - timedelta(days=365), datetime.now().date()


def create_model_selection_form(
    available_models: List[str] = None, title: str = "Model Selection"
) -> Dict[str, Any]:
    """
    Create a form for selecting model type and configuration.

    Args:
        available_models: List of available model types
        title: Form title

    Returns:
        Dictionary containing model configuration
    """
    try:
        st.subheader(title)

        if available_models is None:
            available_models = [
                "Random Forest",
                "XGBoost",
                "SVM",
                "LSTM",
                "GRU",
                "Transformer",
                "Ensemble",
            ]

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "Model Type",
                options=available_models,
                help="Select the machine learning model type",
            )

        with col2:
            task_type = st.selectbox(
                "Task Type",
                options=["Classification", "Regression"],
                help="Select the type of machine learning task",
            )

        # Model class mapping
        model_class = f"Quant{model_type.replace(' ', '')}{'Classifier' if task_type == 'Classification' else 'Regressor'}"

        return {
            "model_type": model_type,
            "task_type": task_type,
            "model_class": model_class,
        }

    except Exception as e:
        st.error(f"Error creating model selection form: {str(e)}")
        return {
            "model_type": "Random Forest",
            "task_type": "Classification",
            "model_class": "QuantRandomForestClassifier",
        }


def create_hyperparameter_form(
    model_class: str, title: str = "Hyperparameter Configuration"
) -> Dict[str, Any]:
    """
    Create a form for configuring model hyperparameters.

    Args:
        model_class: Model class name
        title: Form title

    Returns:
        Dictionary containing hyperparameter configuration
    """
    try:
        st.subheader(title)

        hyperparams = {}

        # Default hyperparameters based on model type
        if "RandomForest" in model_class:
            col1, col2 = st.columns(2)

            with col1:
                hyperparams["n_estimators"] = st.slider(
                    "Number of Estimators", 10, 500, 100, 10
                )
                hyperparams["max_depth"] = st.slider("Max Depth", 1, 50, 10)

            with col2:
                hyperparams["min_samples_split"] = st.slider(
                    "Min Samples Split", 2, 20, 2
                )
                hyperparams["min_samples_leaf"] = st.slider(
                    "Min Samples Leaf", 1, 20, 1
                )

        elif "XGBoost" in model_class:
            col1, col2 = st.columns(2)

            with col1:
                hyperparams["n_estimators"] = st.slider(
                    "Number of Estimators", 10, 500, 100, 10
                )
                hyperparams["max_depth"] = st.slider("Max Depth", 1, 20, 6)

            with col2:
                hyperparams["learning_rate"] = st.slider(
                    "Learning Rate", 0.01, 1.0, 0.1, 0.01
                )
                hyperparams["subsample"] = st.slider("Subsample", 0.1, 1.0, 0.8, 0.1)

        elif "SVM" in model_class:
            col1, col2 = st.columns(2)

            with col1:
                hyperparams["C"] = st.slider("Regularization (C)", 0.1, 100.0, 1.0, 0.1)

            with col2:
                hyperparams["kernel"] = st.selectbox(
                    "Kernel", ["rbf", "linear", "poly", "sigmoid"]
                )

        elif "LSTM" in model_class or "GRU" in model_class:
            col1, col2 = st.columns(2)

            with col1:
                hyperparams["hidden_size"] = st.slider("Hidden Size", 16, 512, 64, 16)
                hyperparams["num_layers"] = st.slider("Number of Layers", 1, 5, 2)

            with col2:
                hyperparams["dropout"] = st.slider("Dropout", 0.0, 0.8, 0.2, 0.1)
                hyperparams["learning_rate"] = st.slider(
                    "Learning Rate", 0.0001, 0.1, 0.001, 0.0001
                )

        else:
            # Generic hyperparameters
            st.info("Using default hyperparameters for this model type.")

        return hyperparams

    except Exception as e:
        st.error(f"Error creating hyperparameter form: {str(e)}")
        return {}


def create_training_config_form(
    title: str = "Training Configuration",
) -> Dict[str, Any]:
    """
    Create a form for training configuration.

    Args:
        title: Form title

    Returns:
        Dictionary containing training configuration
    """
    try:
        st.subheader(title)

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Test Size",
                0.1,
                0.4,
                0.2,
                0.05,
                help="Proportion of data to use for testing",
            )
            validation_size = st.slider(
                "Validation Size",
                0.1,
                0.3,
                0.2,
                0.05,
                help="Proportion of training data to use for validation",
            )

        with col2:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                3,
                10,
                5,
                help="Number of folds for cross-validation",
            )
            random_state = st.number_input(
                "Random State", 0, 9999, 42, help="Random seed for reproducibility"
            )

        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            feature_selection = st.checkbox(
                "Enable Feature Selection",
                value=True,
                help="Automatically select the most important features",
            )
            hyperparameter_tuning = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=False,
                help="Automatically tune hyperparameters using grid search",
            )
            ensemble_training = st.checkbox(
                "Enable Ensemble Training",
                value=False,
                help="Train multiple models and combine predictions",
            )

        return {
            "test_size": test_size,
            "validation_size": validation_size,
            "cv_folds": cv_folds,
            "random_state": random_state,
            "feature_selection": feature_selection,
            "hyperparameter_tuning": hyperparameter_tuning,
            "ensemble_training": ensemble_training,
        }

    except Exception as e:
        st.error(f"Error creating training config form: {str(e)}")
        return {
            "test_size": 0.2,
            "validation_size": 0.2,
            "cv_folds": 5,
            "random_state": 42,
            "feature_selection": True,
            "hyperparameter_tuning": False,
            "ensemble_training": False,
        }
