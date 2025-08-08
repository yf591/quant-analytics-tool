# Feature Pipeline Documentation

## Overview

The Feature Pipeline module provides a comprehensive feature engineering framework for financial machine learning applications. It integrates technical indicators, advanced features, feature selection, scaling, and quality validation into a unified pipeline.

## Architecture

The feature pipeline consists of several key components:

### Core Components

1. **FeaturePipeline**: Main orchestration class
2. **FeatureImportance**: AFML-compliant feature importance analysis
3. **FeatureQualityValidator**: Comprehensive feature quality validation
4. **TechnicalIndicators**: Technical analysis features
5. **AdvancedFeatures**: Advanced financial features

### Pipeline Flow

```
Raw Data → Technical Features → Advanced Features → Feature Combination 
    ↓
Quality Validation → Feature Selection → Feature Scaling → Final Features
```

## Quick Start

### Basic Usage

```python
from src.features import FeaturePipeline
import pandas as pd

# Load your OHLCV data
data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Initialize pipeline
pipeline = FeaturePipeline()

# Generate features
results = pipeline.generate_features(data)

# Access generated features
features = results.features
feature_names = results.feature_names
quality_metrics = results.quality_metrics
```

### Supervised Learning

```python
# Create target variable (e.g., future returns)
target = data['close'].pct_change(5).shift(-5).dropna()

# Generate features with supervised selection
results = pipeline.generate_features(data, target=target)

# Access feature importance
importance_df = results.feature_importance
selected_features = results.features
```

## Configuration

### Default Configuration

The pipeline uses sensible defaults, but can be customized:

```python
config = {
    'technical_indicators': {
        'trend': {
            'sma': {'windows': [5, 10, 20, 50]},
            'ema': {'windows': [5, 10, 20, 50]},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        },
        'momentum': {
            'rsi': {'window': 14},
            'stochastic': {'k_period': 14, 'd_period': 3}
        },
        'volatility': {
            'bollinger_bands': {'window': 20, 'std_dev': 2},
            'atr': {'window': 14}
        }
    },
    'advanced_features': {
        'fractal_dimension': {'window': 100},
        'hurst_exponent': {'window': 100}
    },
    'feature_selection': {
        'method': 'mdi',  # 'mdi', 'mda', 'sfi', 'variance'
        'n_features': 'auto'
    },
    'scaling': {
        'method': 'standard'  # 'standard', 'minmax', 'robust'
    },
    'validation': {
        'check_stationarity': True,
        'check_multicollinearity': True
    }
}

pipeline = FeaturePipeline(config)
```

### Technical Indicators Configuration

Configure which technical indicators to generate:

```python
technical_config = {
    'trend': {
        'sma': {'windows': [10, 20, 50]},        # Simple Moving Average
        'ema': {'windows': [10, 20]},            # Exponential Moving Average
        'macd': {'fast': 12, 'slow': 26, 'signal': 9}  # MACD
    },
    'momentum': {
        'rsi': {'window': 14},                   # RSI
        'stochastic': {'k_period': 14, 'd_period': 3},  # Stochastic
        'williams_r': {'window': 14},            # Williams %R
        'cci': {'window': 20},                   # Commodity Channel Index
        'momentum': {'window': 10}               # Price Momentum
    },
    'volatility': {
        'bollinger_bands': {'window': 20, 'std_dev': 2},  # Bollinger Bands
        'atr': {'window': 14}                    # Average True Range
    }
}
```

## Feature Selection Methods

### Mean Decrease Impurity (MDI)

Based on AFML Chapter 8, MDI measures feature importance using tree-based models:

```python
from src.features import FeatureImportance

analyzer = FeatureImportance()
mdi_scores = analyzer.calculate_mdi_importance(features, target)
```

### Mean Decrease Accuracy (MDA)

MDA measures importance by shuffling feature values and measuring accuracy decrease:

```python
mda_scores = analyzer.calculate_mda_importance(features, target, cv_folds=5)
```

### Single Feature Importance (SFI)

SFI measures the performance of models using only one feature:

```python
sfi_scores = analyzer.calculate_sfi_importance(features, target, cv_folds=5)
```

### Combined Analysis

```python
# Calculate all importance measures
results = analyzer.calculate_all_importance(features, target)

# Access individual measures
mdi = results.mdi_importance
mda = results.mda_importance
sfi = results.sfi_importance

# Combined ranking
ranking = results.feature_ranking
```

## Feature Quality Validation

### Comprehensive Validation

```python
from src.features import FeatureQualityValidator

validator = FeatureQualityValidator()
quality_results = validator.validate_all_features(features)

# Access validation results
stationarity = quality_results.stationarity_results
multicollinearity = quality_results.multicollinearity_results
completeness = quality_results.completeness_results
quality_scores = quality_results.quality_score
recommendations = quality_results.recommendations
```

### Individual Validation Methods

```python
# Test stationarity
stationarity_results = validator.test_stationarity(features)

# Detect multicollinearity
multicollinearity_results = validator.detect_multicollinearity(features)

# Check data completeness
completeness_results = validator.check_completeness(features)

# Detect outliers
outlier_results = validator.detect_outliers(features)
```

## Scaling Options

### Standard Scaling

```python
config = {
    'scaling': {
        'method': 'standard'  # Zero mean, unit variance
    }
}
```

### MinMax Scaling

```python
config = {
    'scaling': {
        'method': 'minmax',
        'feature_range': (0, 1)  # Scale to [0, 1]
    }
}
```

### Robust Scaling

```python
config = {
    'scaling': {
        'method': 'robust'  # Uses median and IQR
    }
}
```

## Advanced Usage

### Custom Feature Pipeline

```python
class CustomPipeline(FeaturePipeline):
    def _generate_custom_features(self, data):
        """Add custom features."""
        custom_features = pd.DataFrame(index=data.index)
        
        # Add your custom features
        custom_features['custom_indicator'] = your_custom_calculation(data)
        
        return custom_features
    
    def _combine_features(self, data, technical_results, advanced_results):
        # Call parent method
        features = super()._combine_features(data, technical_results, advanced_results)
        
        # Add custom features
        custom_features = self._generate_custom_features(data)
        
        return pd.concat([features, custom_features], axis=1)
```

### Batch Processing

```python
# Process multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT']
all_features = {}

for symbol in symbols:
    data = load_data(symbol)
    results = pipeline.generate_features(data)
    all_features[symbol] = results.features
```

### Transform New Data

```python
# Fit pipeline on training data
train_results = pipeline.generate_features(train_data, target=train_target)

# Transform new data using same parameters
test_features = pipeline.transform_new_data(test_data, train_results)
```

## Best Practices

### 1. Data Quality

- Ensure OHLCV data is clean and properly formatted
- Handle missing values before feature generation
- Check for data leakage in target variable creation

### 2. Feature Selection

- Use cross-validation for robust feature selection
- Consider feature stability across different time periods
- Avoid look-ahead bias in feature engineering

### 3. Validation

- Always validate feature quality before modeling
- Check for multicollinearity and remove redundant features
- Test stationarity for time series features

### 4. Performance

- Use caching for expensive computations
- Enable parallel processing for large datasets
- Consider memory usage with many features

### 5. Reproducibility

- Set random seeds for consistent results
- Save pipeline configurations
- Version control feature engineering code

## Error Handling

### Common Issues

1. **Insufficient Data**: Features requiring rolling windows need enough data
2. **Missing Values**: Some indicators can't handle NaN values
3. **Data Types**: Ensure numeric data types for calculations
4. **Memory Issues**: Large feature sets may require chunking

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Check intermediate results
results = pipeline.generate_features(data)
print(f"Technical results: {results.technical_results}")
print(f"Advanced results: {results.advanced_results}")
print(f"Quality metrics: {results.quality_metrics}")
```

## Performance Optimization

### Caching

```python
config = {
    'caching': {
        'enabled': True,
        'cache_dir': 'cache/features',
        'ttl': 3600  # 1 hour
    }
}
```

### Parallel Processing

```python
config = {
    'parallel': {
        'enabled': True,
        'max_workers': 4
    }
}
```

### Memory Management

```python
# For large datasets, process in chunks
chunk_size = 10000
for i in range(0, len(data), chunk_size):
    chunk = data.iloc[i:i+chunk_size]
    chunk_results = pipeline.generate_features(chunk)
    # Process chunk_results
```

## Integration with ML Pipelines

### Scikit-learn Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create ML pipeline
ml_pipeline = Pipeline([
    ('features', pipeline),
    ('classifier', RandomForestClassifier())
])

# Fit and predict
ml_pipeline.fit(train_data, train_target)
predictions = ml_pipeline.predict(test_data)
```

### Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeaturePipelineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.pipeline = FeaturePipeline(config)
        self.results = None
    
    def fit(self, X, y=None):
        self.results = self.pipeline.generate_features(X, y)
        return self
    
    def transform(self, X):
        if self.results is None:
            raise ValueError("Must fit before transform")
        return self.pipeline.transform_new_data(X, self.results)
```

## Examples

See `examples/feature_pipeline_example.py` for comprehensive examples including:

1. Basic pipeline usage
2. Supervised feature selection
3. Feature importance analysis
4. Quality validation
5. Complete workflow

## Testing

Run the test suite:

```bash
python -m pytest tests/test_feature_pipeline.py -v
```

Or run the comprehensive test:

```bash
python tests/test_feature_pipeline.py
```

## API Reference

### FeaturePipeline

- `generate_features(data, target=None, force_recompute=False)`: Generate features
- `transform_new_data(data, pipeline_results)`: Transform new data
- `get_feature_description()`: Get feature descriptions

### FeatureImportance

- `calculate_mdi_importance(X, y, sample_weights=None)`: MDI importance
- `calculate_mda_importance(X, y, sample_weights=None, cv_folds=5)`: MDA importance
- `calculate_sfi_importance(X, y, sample_weights=None, cv_folds=5)`: SFI importance
- `calculate_all_importance(X, y, sample_weights=None, cv_folds=5)`: All methods

### FeatureQualityValidator

- `validate_all_features(features)`: Comprehensive validation
- `test_stationarity(features)`: Stationarity tests
- `detect_multicollinearity(features)`: Multicollinearity detection
- `check_completeness(features)`: Missing data analysis
- `detect_outliers(features)`: Outlier detection

## Troubleshooting

### Common Error Messages

1. **"Insufficient data for feature importance analysis"**
   - Solution: Ensure you have at least 50 samples after cleaning

2. **"Feature scaling failed"**
   - Solution: Check for infinite values or very large numbers

3. **"No features selected"**
   - Solution: Lower feature selection threshold or check data quality

4. **"Stationarity test failed"**
   - Solution: Consider differencing or transformation of features

For more help, check the examples and test files for working code patterns.
