"""
Documentation utility page for Quant Analytics Tool
"""

import streamlit as st


def show_documentation_page():
    """Display the documentation page"""

    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("📝 Documentation")

    # Navigation tabs for documentation sections
    doc_tabs = st.tabs(
        ["🚀 Quick Start", "📚 User Guide", "🔧 API Reference", "💡 Examples", "❓ FAQ"]
    )

    with doc_tabs[0]:
        st.markdown(
            """
        ## 🚀 Quick Start Guide
        
        ### 1. Data Acquisition
        - Navigate to **📈 Data Acquisition** 
        - Enter ticker symbols (e.g., AAPL, MSFT, GOOGL)
        - Select time period and interval
        - Click **Fetch Data** to download market data
        
        ### 2. Feature Engineering
        - Go to **🛠️ Feature Engineering**
        - Select your acquired data
        - Choose technical indicators and AFML features
        - Generate feature sets for modeling
        
        ### 3. Model Training
        - Navigate to **🧠 Model Training**
        - Select feature sets and model types
        - Configure hyperparameters
        - Train and evaluate models
        
        ### 4. Backtesting
        - Go to **🔙 Backtesting** 
        - Select trained models or strategies
        - Configure backtest parameters
        - Run comprehensive performance analysis
        
        ### 5. Risk Management
        - Navigate to **⚖️ Risk Management**
        - Analyze portfolio risk metrics
        - Optimize position sizing
        - Run stress tests
        """
        )

    with doc_tabs[1]:
        st.markdown(
            """
        ## 📚 Comprehensive User Guide
        
        ### Platform Architecture
        The Quant Analytics Tool implements a modular architecture with clear separation of concerns:
        
        - **Data Layer**: Market data acquisition and storage
        - **Feature Layer**: Technical indicators and AFML features
        - **Model Layer**: Machine learning models and training
        - **Strategy Layer**: Trading strategies and backtesting
        - **Risk Layer**: Risk management and portfolio optimization
        
        ### AFML Implementation
        Our implementation follows the methodologies from "Advances in Financial Machine Learning":
        
        - **Chapter 2**: Financial Data Structures
        - **Chapter 3**: Labeling (Triple Barrier Method)
        - **Chapter 4**: Sample Weights
        - **Chapter 5**: Fractional Differentiation
        - **Chapter 6**: Ensemble Methods
        - **Chapter 7**: Cross-Validation in Finance
        - **Chapter 8**: Feature Importance
        
        ### Best Practices
        1. **Data Quality**: Always validate data before feature engineering
        2. **Feature Selection**: Use AFML feature importance methods
        3. **Cross-Validation**: Implement purged cross-validation
        4. **Overfitting**: Monitor for backtest overfitting
        5. **Risk Management**: Always implement proper position sizing
        """
        )

    with doc_tabs[2]:
        st.markdown(
            """
        ## 🔧 API Reference
        
        ### Core Modules
        
        #### Data Acquisition (`src.data`)
        ```python
        from src.data.collectors import YFinanceCollector
        from src.data.storage import SQLiteStorage
        from src.data.validators import DataValidator
        ```
        
        #### Feature Engineering (`src.features`)
        ```python
        from src.features.technical import TechnicalIndicators
        from src.features.advanced import AdvancedFeatures
        from src.features.pipeline import FeaturePipeline
        ```
        
        #### Model Training (`src.models`)
        ```python
        from src.models.traditional.random_forest import QuantRandomForestClassifier
        from src.models.deep_learning import QuantLSTMClassifier
        from src.models.evaluation import ModelEvaluator
        ```
        
        #### Backtesting (`src.backtesting`)
        ```python
        from src.backtesting import BacktestEngine
        from src.backtesting.strategies import MomentumStrategy
        from src.backtesting.portfolio import Portfolio
        ```
        
        ### Utility Functions
        
        #### Session State Management
        ```python
        from streamlit_app.utils.data_utils import DataAcquisitionManager
        from streamlit_app.utils.feature_utils import FeatureEngineeringManager
        from streamlit_app.utils.model_utils import ModelTrainingManager
        ```
        """
        )

    with doc_tabs[3]:
        st.markdown(
            """
        ## 💡 Examples & Tutorials
        
        ### Example 1: Basic Data Acquisition
        ```python
        # Using the Data Acquisition Manager
        data_manager = DataAcquisitionManager()
        data_manager.initialize_session_state(st.session_state)
        
        # Fetch data for multiple symbols
        config = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'period': '1y',
            'interval': '1d'
        }
        
        success, message = data_manager.start_yahoo_finance_collection(
            config, st.session_state
        )
        ```
        
        ### Example 2: Feature Engineering Pipeline
        ```python
        # Using the Feature Engineering Manager
        feature_manager = FeatureEngineeringManager()
        
        # Configure technical indicators
        config = {
            'technical_indicators': [
                'sma_20', 'sma_50', 'rsi', 'macd', 'bollinger_bands'
            ],
            'advanced_features': [
                'fractal_dimension', 'hurst_exponent'
            ]
        }
        
        # Run feature pipeline
        success, message = feature_manager.run_feature_pipeline(
            'AAPL', data, config, st.session_state
        )
        ```
        
        ### Example 3: Model Training
        ```python
        # Using the Model Training Manager
        model_manager = ModelTrainingManager()
        
        # Configure model training
        model_config = {'model_type': 'RandomForest'}
        hyperparams = {'n_estimators': 100, 'max_depth': 10}
        training_config = {'test_size': 0.2, 'cv_folds': 5}
        
        # Train model
        success, message, model_id = model_manager.train_model(
            feature_key, model_config, hyperparams, 
            training_config, st.session_state
        )
        ```
        """
        )

    with doc_tabs[4]:
        st.markdown(
            """
        ## ❓ Frequently Asked Questions
        
        ### General Questions
        
        **Q: What data sources are supported?**
        A: We support Yahoo Finance (free), Alpha Vantage, Polygon.io, and custom CSV uploads.
        
        **Q: Can I use custom indicators?**
        A: Yes, you can implement custom technical indicators using our extensible framework.
        
        **Q: Is real-time data supported?**
        A: Real-time data streaming is planned for future releases.
        
        ### Technical Questions
        
        **Q: How do I handle missing data?**
        A: Our data validators automatically detect and handle missing data using configurable strategies.
        
        **Q: What ML models are available?**
        A: We provide Random Forest, XGBoost, LSTM, GRU, SVM, and ensemble methods.
        
        **Q: How is overfitting prevented?**
        A: We implement purged cross-validation and multiple overfitting detection methods from AFML.
        
        ### Troubleshooting
        
        **Q: Memory usage is high, what can I do?**
        A: Use the Cache Management page to clear unused data and enable garbage collection.
        
        **Q: Models are training slowly, how to optimize?**
        A: Reduce feature dimensions, use smaller datasets for development, or adjust hyperparameters.
        
        **Q: Getting API rate limit errors?**
        A: Implement request delays and consider upgrading to paid API tiers for higher limits.
        """
        )

    # External links
    st.markdown("---")
    st.markdown("### 🔗 External Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        **📖 AFML Book**
        - [Official Website](https://www.afml.com)
        - [GitHub Repository](https://github.com/yf591/quant-analytics-tool)
        """
        )

    with col2:
        st.markdown(
            """
        **🛠️ Technical Resources**
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Plotly Documentation](https://plotly.com/python/)
        """
        )

    with col3:
        st.markdown(
            """
        **📊 Financial Data**
        - [Yahoo Finance](https://finance.yahoo.com)
        - [Alpha Vantage](https://www.alphavantage.co)
        """
        )

    st.markdown("</div>", unsafe_allow_html=True)
