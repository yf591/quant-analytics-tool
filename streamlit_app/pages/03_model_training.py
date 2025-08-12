"""
Streamlit Page: Model Training
Week 14 UI Integration - Professional Model Training Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime

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

    # Check for available features
    if "feature_cache" not in st.session_state or not st.session_state.feature_cache:
        st.warning("🛠️ Please generate features first from Feature Engineering page")
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
    selected_features = st.selectbox("Select Features", available_features)

    if not selected_features:
        return

    # Model type selection
    st.subheader("🤖 Model Types")

    tab1, tab2, tab3 = st.tabs(["Traditional", "Deep Learning", "Advanced"])

    with tab1:
        traditional_models_config(selected_features)

    with tab2:
        deep_learning_models_config(selected_features)

    with tab3:
        advanced_models_config(selected_features)


def traditional_models_config(feature_key: str):
    """Traditional ML Models Configuration"""

    st.markdown("**Traditional ML Models**")

    # Model selection
    model_type = st.selectbox(
        "Model Type", ["Random Forest", "XGBoost", "SVM"], key="trad_model_type"
    )

    # Task type
    task_type = st.radio(
        "Task Type", ["Classification", "Regression"], key="trad_task_type"
    )

    # Hyperparameters
    st.markdown("**Hyperparameters**")

    if model_type == "Random Forest":
        n_estimators = st.slider("N Estimators", 10, 200, 100, key="trad_n_est")
        max_depth = st.slider("Max Depth", 3, 20, 10, key="trad_max_depth")
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
        }
    elif model_type == "XGBoost":
        n_estimators = st.slider("N Estimators", 10, 200, 100, key="trad_xgb_n_est")
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, key="trad_lr")
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": 42,
        }
    else:  # SVM
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, key="trad_C")
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="trad_kernel")
        params = {"C": C, "kernel": kernel, "random_state": 42}

    # Train button
    if st.button(
        "🚀 Train Traditional Model", type="primary", use_container_width=True
    ):
        train_traditional_model(feature_key, model_type, task_type, params)


def deep_learning_models_config(feature_key: str):
    """Deep Learning Models Configuration"""

    st.markdown("**Deep Learning Models**")

    # Model selection
    model_type = st.selectbox("Model Type", ["LSTM", "GRU"], key="dl_model_type")

    # Task type
    task_type = st.radio(
        "Task Type", ["Classification", "Regression"], key="dl_task_type"
    )

    # Hyperparameters
    st.markdown("**Hyperparameters**")

    sequence_length = st.slider("Sequence Length", 10, 50, 20, key="dl_seq_len")
    lstm_units = st.slider("LSTM/GRU Units", 16, 128, 64, key="dl_units")
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, key="dl_dropout")
    epochs = st.slider("Epochs", 5, 50, 20, key="dl_epochs")

    params = {
        "sequence_length": sequence_length,
        "lstm_units": [lstm_units, lstm_units // 2] if model_type == "LSTM" else None,
        "gru_units": [lstm_units, lstm_units // 2] if model_type == "GRU" else None,
        "dense_units": [lstm_units // 4],
        "dropout_rate": dropout_rate,
        "epochs": epochs,
        "batch_size": 32,
        "verbose": 0,
    }

    # Train button
    if st.button(
        "🚀 Train Deep Learning Model", type="primary", use_container_width=True
    ):
        train_deep_learning_model(feature_key, model_type, task_type, params)


def advanced_models_config(feature_key: str):
    """Advanced Models Configuration"""

    st.markdown("**Advanced Models (AFML)**")

    # Model selection
    model_type = st.selectbox(
        "Model Type", ["Financial Random Forest", "Ensemble"], key="adv_model_type"
    )

    # Task type
    task_type = st.radio(
        "Task Type", ["Classification", "Regression"], key="adv_task_type"
    )

    # Hyperparameters
    st.markdown("**Hyperparameters**")

    n_estimators = st.slider("N Estimators", 10, 100, 50, key="adv_n_est")
    max_samples = st.slider("Max Samples", 0.5, 1.0, 0.8, key="adv_max_samples")

    params = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "random_state": 42,
    }

    # Train button
    if st.button("🚀 Train Advanced Model", type="primary", use_container_width=True):
        train_advanced_model(feature_key, model_type, task_type, params)


def model_display_panel():
    """Model Display and Comparison Panel"""

    if not st.session_state.model_cache:
        st.info("🤖 Configure and train models to see results")
        return

    # Model selection
    selected_model = st.selectbox(
        "Select Model", list(st.session_state.model_cache.keys())
    )

    if selected_model:
        display_model_overview(selected_model)
        display_model_performance(selected_model)


def train_traditional_model(
    feature_key: str, model_type: str, task_type: str, params: dict
):
    """Train traditional ML model using Week 7 modules"""

    try:
        cached_features = st.session_state.feature_cache[feature_key]
        data = cached_features["data"]
        features = cached_features["features"]

        with st.spinner(f"Training {model_type} {task_type} model..."):
            # Prepare features
            feature_df = pd.DataFrame()
            for name, values in features.items():
                if isinstance(values, pd.Series):
                    feature_df[name] = values
                elif isinstance(values, dict):
                    for sub_name, sub_values in values.items():
                        if isinstance(sub_values, pd.Series):
                            feature_df[f"{name}_{sub_name}"] = sub_values

            if feature_df.empty:
                st.error("No suitable features found for training")
                return

            # Create target
            if task_type == "Classification":
                target = (data["Close"].pct_change() > 0).astype(int)
            else:
                target = data["Close"].pct_change()

            # Align data
            aligned_data = pd.concat(
                [feature_df, target.rename("target")], axis=1
            ).dropna()

            if len(aligned_data) < 50:
                st.error("Insufficient data for training")
                return

            X = aligned_data[feature_df.columns]
            y = aligned_data["target"]

            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Select model using existing Week 7 modules
            if model_type == "Random Forest":
                if task_type == "Classification":
                    model = QuantRandomForestClassifier(**params)
                else:
                    model = QuantRandomForestRegressor(**params)
            elif model_type == "XGBoost":
                if task_type == "Classification":
                    model = QuantXGBoostClassifier(**params)
                else:
                    model = QuantXGBoostRegressor(**params)

            # Train model
            model.fit(X_train, y_train)

            # Evaluate using existing Week 7 modules
            evaluator = ModelEvaluator()
            if task_type == "Classification":
                performance = evaluator.evaluate_classifier(model, X_test, y_test)
            else:
                performance = evaluator.evaluate_regressor(model, X_test, y_test)

            # Store results
            model_key = f"{model_type}_{task_type}_{feature_key}"
            st.session_state.model_cache[model_key] = {
                "model": model,
                "model_type": model_type,
                "task_type": task_type,
                "performance": performance,
                "feature_names": list(X.columns),
                "test_data": (X_test, y_test),
                "trained_at": datetime.now(),
                "category": "traditional",
            }

        st.success(f"✅ Trained {model_type} {task_type} model successfully")
        st.rerun()

    except Exception as e:
        st.error(f"Traditional model training failed: {str(e)}")


def train_deep_learning_model(
    feature_key: str, model_type: str, task_type: str, params: dict
):
    """Train deep learning model using Week 8 modules"""

    try:
        cached_features = st.session_state.feature_cache[feature_key]
        data = cached_features["data"]
        features = cached_features["features"]

        with st.spinner(f"Training {model_type} {task_type} model..."):
            # Prepare features
            feature_df = pd.DataFrame()
            for name, values in features.items():
                if isinstance(values, pd.Series):
                    feature_df[name] = values

            if feature_df.empty:
                st.error("No suitable features found for training")
                return

            # Create target
            if task_type == "Classification":
                target = (data["Close"].pct_change() > 0).astype(int)
            else:
                target = data["Close"].pct_change()

            # Align data
            aligned_data = pd.concat(
                [feature_df, target.rename("target")], axis=1
            ).dropna()

            if len(aligned_data) < 100:
                st.error("Insufficient data for deep learning training")
                return

            X = aligned_data[feature_df.columns].values
            y = aligned_data["target"].values

            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Select model using existing Week 8 modules
            if model_type == "LSTM":
                if task_type == "Classification":
                    model = QuantLSTMClassifier(**params)
                else:
                    model = QuantLSTMRegressor(**params)
            else:  # GRU
                if task_type == "Classification":
                    model = QuantGRUClassifier(**params)
                else:
                    model = QuantGRURegressor(**params)

            # Train model
            model.fit(X_train, y_train)

            # Simple evaluation
            predictions = model.predict(X_test)

            if task_type == "Classification":
                accuracy = np.mean(
                    predictions == y_test[params["sequence_length"] - 1 :]
                )
                performance = {"accuracy": accuracy}
            else:
                y_test_aligned = y_test[params["sequence_length"] - 1 :]
                mse = np.mean((predictions - y_test_aligned) ** 2)
                mae = np.mean(np.abs(predictions - y_test_aligned))
                performance = {"mse": mse, "mae": mae}

            # Store results
            model_key = f"{model_type}_{task_type}_{feature_key}"
            st.session_state.model_cache[model_key] = {
                "model": model,
                "model_type": model_type,
                "task_type": task_type,
                "performance": performance,
                "feature_names": list(feature_df.columns),
                "test_data": (X_test, y_test),
                "trained_at": datetime.now(),
                "category": "deep_learning",
            }

        st.success(f"✅ Trained {model_type} {task_type} model successfully")
        st.rerun()

    except Exception as e:
        st.error(f"Deep learning model training failed: {str(e)}")


def train_advanced_model(
    feature_key: str, model_type: str, task_type: str, params: dict
):
    """Train advanced model using Week 9 modules"""

    try:
        cached_features = st.session_state.feature_cache[feature_key]
        data = cached_features["data"]
        features = cached_features["features"]

        with st.spinner(f"Training {model_type} {task_type} model..."):
            # Prepare features
            feature_df = pd.DataFrame()
            for name, values in features.items():
                if isinstance(values, pd.Series):
                    feature_df[name] = values

            if feature_df.empty:
                st.error("No suitable features found for training")
                return

            # Create target
            if task_type == "Classification":
                target = (data["Close"].pct_change() > 0).astype(int)
            else:
                target = data["Close"].pct_change()

            # Align data
            aligned_data = pd.concat(
                [feature_df, target.rename("target")], axis=1
            ).dropna()

            if len(aligned_data) < 50:
                st.error("Insufficient data for training")
                return

            X = aligned_data[feature_df.columns]
            y = aligned_data["target"]

            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Use existing Week 9 modules
            from src.models.advanced.ensemble import EnsembleConfig

            config = EnsembleConfig(**params)
            model = FinancialRandomForest(config=config)

            # Train model
            model.fit(X_train.values, y_train.values)

            # Simple evaluation
            predictions = model.predict(X_test.values)

            if task_type == "Classification":
                accuracy = np.mean(predictions == y_test.values)
                performance = {"accuracy": accuracy}
            else:
                mse = np.mean((predictions - y_test.values) ** 2)
                mae = np.mean(np.abs(predictions - y_test.values))
                performance = {"mse": mse, "mae": mae}

            # Store results
            model_key = f"{model_type}_{task_type}_{feature_key}"
            st.session_state.model_cache[model_key] = {
                "model": model,
                "model_type": model_type,
                "task_type": task_type,
                "performance": performance,
                "feature_names": list(X.columns),
                "test_data": (X_test, y_test),
                "trained_at": datetime.now(),
                "category": "advanced",
            }

        st.success(f"✅ Trained {model_type} {task_type} model successfully")
        st.rerun()

    except Exception as e:
        st.error(f"Advanced model training failed: {str(e)}")


def display_model_overview(model_key: str):
    """Display model overview with metrics"""

    cached_model = st.session_state.model_cache[model_key]
    model = cached_model["model"]
    model_type = cached_model["model_type"]
    task_type = cached_model["task_type"]
    performance = cached_model["performance"]
    category = cached_model["category"]

    st.subheader(f"🤖 Model Overview: {model_key}")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Type", model_type)

    with col2:
        st.metric("Task", task_type)

    with col3:
        st.metric("Category", category.title())

    with col4:
        trained_time = cached_model["trained_at"]
        st.metric("Trained", trained_time.strftime("%H:%M"))

    # Performance metrics
    st.subheader("📊 Performance Metrics")

    perf_cols = st.columns(len(performance))
    for i, (metric, value) in enumerate(performance.items()):
        with perf_cols[i]:
            st.metric(metric.upper(), f"{value:.4f}")

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        st.subheader("🎯 Feature Importance")

        feature_names = cached_model["feature_names"]
        importances = model.feature_importances_

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        st.dataframe(importance_df.head(10), use_container_width=True)


def display_model_performance(model_key: str):
    """Display model performance visualization"""

    cached_model = st.session_state.model_cache[model_key]
    model = cached_model["model"]
    task_type = cached_model["task_type"]
    X_test, y_test = cached_model["test_data"]

    st.subheader("📈 Model Performance")

    try:
        # Get predictions
        if cached_model["category"] == "deep_learning":
            predictions = model.predict(X_test)
            if task_type == "Classification":
                y_test_aligned = y_test[model.sequence_length - 1 :]
            else:
                y_test_aligned = y_test[model.sequence_length - 1 :]
        else:
            predictions = model.predict(X_test)
            y_test_aligned = y_test.values if hasattr(y_test, "values") else y_test

        # Create visualization
        fig = go.Figure()

        if task_type == "Classification":
            # Classification: Confusion Matrix visualization
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_test_aligned, predictions)

            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=["Predicted 0", "Predicted 1"],
                    y=["Actual 0", "Actual 1"],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                )
            )

            fig.update_layout(title="Confusion Matrix", height=400)

        else:
            # Regression: Actual vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(predictions))),
                    y=y_test_aligned,
                    mode="lines",
                    name="Actual",
                    line=dict(color="blue"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(predictions))),
                    y=predictions,
                    mode="lines",
                    name="Predicted",
                    line=dict(color="red"),
                )
            )

            fig.update_layout(
                title="Actual vs Predicted Values",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400,
            )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")


if __name__ == "__main__":
    main()
