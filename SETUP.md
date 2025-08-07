# Quant Analytics Tool - Setup Guide

## 🚀 Quick Setup

### 1. Create and Activate Virtual Environment

```bash
# Create Python virtual environment
python -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Activate virtual environment (Windows)
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install basic dependencies
pip install --upgrade pip
pip install streamlit pandas numpy yfinance plotly

# Full dependencies (may take time)
# pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Create environment variables file
cp .env.example .env

# Edit .env file to set required API keys
# (It will work with default values initially)
```

### 4. Launch Application

```bash
# Start Streamlit app
streamlit run streamlit_app/main.py
```

You can access the application at `http://localhost:8501` in your browser.

## 📋 Next Steps

### Phase 1: Foundation Features Implementation (Recommended Order)

1. **Data Acquisition Features**
   ```bash
   # Implement data acquisition modules
   touch src/data/collectors.py
   touch src/data/processors.py
   touch src/data/__init__.py
   ```

2. **Basic Technical Indicators**
   ```bash
   # Feature engineering modules
   touch src/features/technical.py
   touch src/features/__init__.py
   ```

3. **Enhanced Visualization Features**
   ```bash
   # Visualization components
   touch src/visualization/charts.py
   touch src/visualization/__init__.py
   ```

### Development Commands

```bash
# Code formatting (after Black installation)
black src/ streamlit_app/

# Linting (after flake8 installation)
flake8 src/ streamlit_app/

# Run tests (after pytest installation)
pytest tests/
```

## 🛠 Recommended Development Flow

1. First, start the Streamlit app to check the basic structure
2. Implement data acquisition features step by step
3. Visualize and test each feature with Streamlit pages
4. Add machine learning models and backtesting features

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Plotly Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
