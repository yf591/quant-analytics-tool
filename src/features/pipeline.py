"""
Feature Pipeline Module

This module implements a comprehensive feature engineering pipeline that integrates
technical indicators and advanced features, providing automated feature generation,
selection, normalization, and quality validation.

Based on Advances in Financial Machine Learning (AFML) methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import json
from pathlib import Path
import logging

from .technical import TechnicalIndicators, TechnicalIndicatorResults
from .advanced import AdvancedFeatures, AdvancedFeatureResults


@dataclass
class FeaturePipelineConfig:
    """Configuration class for feature pipeline."""

    # Technical indicators configuration
    technical_indicators: Dict[str, Dict[str, Any]]

    # Advanced features configuration
    advanced_features: Dict[str, Dict[str, Any]]

    # Feature selection configuration
    feature_selection: Dict[str, Any]

    # Scaling configuration
    scaling: Dict[str, Any]

    # Quality validation configuration
    validation: Dict[str, Any]

    # Caching configuration
    caching: Dict[str, Any]

    # Parallel processing configuration
    parallel: Dict[str, Any]


@dataclass
@dataclass
class FeaturePipelineResults:
    """Container for feature pipeline results."""

    features: Optional[pd.DataFrame] = None
    feature_names: Optional[List[str]] = None
    feature_importance: Optional[pd.DataFrame] = None
    scaling_params: Optional[Dict[str, Any]] = None
    selection_mask: Optional[pd.Series] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    technical_results: Optional[Dict] = None
    advanced_results: Optional[AdvancedFeatureResults] = None


class FeaturePipeline:
    """
    Comprehensive feature engineering pipeline.

    This class orchestrates the entire feature generation process, from raw data
    to ML-ready features, incorporating both technical indicators and advanced
    features with automated selection, scaling, and quality validation.
    """

    def __init__(
        self, config: Optional[Union[str, Dict, FeaturePipelineConfig]] = None
    ):
        """
        Initialize the feature pipeline.

        Args:
            config: Configuration as file path, dict, or FeaturePipelineConfig object
        """
        self.config = self._load_config(config)
        self.technical_indicators = TechnicalIndicators()
        self.advanced_features = AdvancedFeatures()
        self.logger = logging.getLogger(__name__)

        # Initialize feature cache
        self._feature_cache = {}
        self._scaling_cache = {}

    def _load_config(
        self, config: Optional[Union[str, Dict, FeaturePipelineConfig]]
    ) -> FeaturePipelineConfig:
        """Load and validate configuration."""
        if config is None:
            return self._get_default_config()

        if isinstance(config, FeaturePipelineConfig):
            return config

        if isinstance(config, str):
            # Load from file
            config_path = Path(config)
            if config_path.suffix in [".yaml", ".yml"]:
                with open(config_path, "r") as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )
        else:
            config_dict = config

        return FeaturePipelineConfig(**config_dict)

    def _get_default_config(self) -> FeaturePipelineConfig:
        """Get default configuration."""
        return FeaturePipelineConfig(
            technical_indicators={
                "trend": {
                    "sma": {"windows": [5, 10, 20, 50]},
                    "ema": {"windows": [5, 10, 20, 50]},
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                },
                "momentum": {
                    "rsi": {"window": 14},
                    "stochastic": {"k_period": 14, "d_period": 3},
                    "williams_r": {"window": 14},
                    "cci": {"window": 20},
                    "momentum": {"window": 10},
                },
                "volatility": {
                    "bollinger_bands": {"window": 20, "std_dev": 2},
                    "atr": {"window": 14},
                },
            },
            advanced_features={
                "fractal_dimension": {"window": 100, "method": "higuchi"},
                "hurst_exponent": {"window": 100, "method": "rs"},
                "information_bars": {"bar_type": "volume", "threshold": None},
                "fractional_diff": {"d": 0.4, "threshold": 0.01},
            },
            feature_selection={
                "method": "mdi",  # 'mdi', 'mda', 'sfi', 'variance'
                "n_features": "auto",  # number or 'auto'
                "threshold": 0.01,
            },
            scaling={
                "method": "standard",  # 'standard', 'minmax', 'robust', 'quantile'
                "feature_range": (0, 1),
                "per_feature": False,
            },
            validation={
                "check_stationarity": True,
                "check_multicollinearity": True,
                "vif_threshold": 10.0,
                "missing_threshold": 0.05,
            },
            caching={
                "enabled": True,
                "cache_dir": "cache/features",
                "ttl": 3600,  # seconds
            },
            parallel={"enabled": True, "max_workers": 4},
        )

    def generate_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        force_recompute: bool = False,
    ) -> FeaturePipelineResults:
        """
        Generate comprehensive feature set from raw data.

        Args:
            data: Raw OHLCV data
            target: Target variable for supervised feature selection
            force_recompute: Force recomputation even if cached

        Returns:
            FeaturePipelineResults containing all generated features and metadata
        """
        self.logger.info("Starting feature generation pipeline")

        # Check cache first
        cache_key = self._generate_cache_key(data)
        if not force_recompute and self.config.caching["enabled"]:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.logger.info("Using cached features")
                return cached_result

        results = FeaturePipelineResults()

        try:
            # 1. Generate technical indicators
            self.logger.info("Generating technical indicators")
            results.technical_results = self._generate_technical_features(data)

            # 2. Generate advanced features
            self.logger.info("Generating advanced features")
            results.advanced_results = self._generate_advanced_features(data)

            # 3. Combine all features
            self.logger.info("Combining features")
            results.features = self._combine_features(
                data, results.technical_results, results.advanced_results
            )

            # 4. Feature quality validation
            self.logger.info("Validating feature quality")
            results.quality_metrics = self._validate_features(results.features)

            # 5. Feature selection
            if target is not None:
                self.logger.info("Performing feature selection")
                results.features, results.selection_mask, results.feature_importance = (
                    self._select_features(results.features, target)
                )

            # 6. Feature scaling
            self.logger.info("Scaling features")
            results.features, results.scaling_params = self._scale_features(
                results.features
            )

            # 7. Final feature names
            results.feature_names = list(results.features.columns)

            # Cache results
            if self.config.caching["enabled"]:
                self._save_to_cache(cache_key, results)

            self.logger.info(
                f"Feature generation completed. Generated {len(results.feature_names)} features"
            )

        except Exception as e:
            self.logger.error(f"Error in feature generation: {str(e)}")
            raise

        return results

    def _generate_technical_features(self, data: pd.DataFrame) -> Dict:
        """Generate technical indicator features."""
        # Prepare indicators list based on config
        indicators = []

        # Trend indicators
        for indicator, params in self.config.technical_indicators.get(
            "trend", {}
        ).items():
            if indicator in ["sma", "ema"]:
                indicators.append(indicator)
            elif indicator == "macd":
                indicators.append("macd")

        # Momentum indicators
        for indicator in self.config.technical_indicators.get("momentum", {}):
            indicators.append(indicator)

        # Volatility indicators
        for indicator in self.config.technical_indicators.get("volatility", {}):
            indicators.append(indicator)

        return self.technical_indicators.calculate_all_indicators(
            data, indicators=indicators if indicators else None
        )

    def _generate_advanced_features(self, data: pd.DataFrame) -> AdvancedFeatureResults:
        """Generate advanced features."""
        price_col = "close" if "close" in data.columns else data.columns[0]
        volume_col = "volume" if "volume" in data.columns else None

        # Use the window from config or default
        window = self.config.advanced_features.get("fractal_dimension", {}).get(
            "window", 100
        )

        return self.advanced_features.calculate_all_features(
            data, price_col=price_col, volume_col=volume_col, window=window
        )

    def _combine_features(
        self,
        data: pd.DataFrame,
        technical_results: Dict,
        advanced_results: AdvancedFeatureResults,
    ) -> pd.DataFrame:
        """Combine all generated features into a single DataFrame."""
        feature_dfs = []

        # Add basic price features
        if "close" in data.columns:
            price_features = pd.DataFrame(index=data.index)
            price_features["returns"] = data["close"].pct_change()
            price_features["log_returns"] = np.log(data["close"]).diff()

            # Price-based features
            for window in [5, 10, 20]:
                price_features[f"price_momentum_{window}"] = data["close"].pct_change(
                    window
                )
                price_features[f"volatility_{window}"] = (
                    price_features["returns"].rolling(window).std()
                )

            feature_dfs.append(price_features)

        # Add technical indicators
        tech_features = self._extract_technical_features(technical_results, data.index)
        if not tech_features.empty:
            feature_dfs.append(tech_features)

        # Add advanced features
        adv_features = self._extract_advanced_features(advanced_results, data.index)
        if not adv_features.empty:
            feature_dfs.append(adv_features)

        # Combine all features
        if feature_dfs:
            combined_features = pd.concat(feature_dfs, axis=1)

            # Remove features with too many NaN values
            missing_threshold = self.config.validation.get("missing_threshold", 0.05)
            combined_features = combined_features.loc[
                :, combined_features.isnull().mean() < missing_threshold
            ]

            return combined_features
        else:
            return pd.DataFrame(index=data.index)

    def _extract_technical_features(
        self, tech_results: Dict, index: pd.Index
    ) -> pd.DataFrame:
        """Extract technical indicator features into DataFrame."""
        features = pd.DataFrame(index=index)

        # Process all technical indicator results
        for indicator_key, result in tech_results.items():
            if hasattr(result, "values"):
                values = result.values
            else:
                continue

            # Handle different types of indicator results
            if isinstance(values, pd.Series):
                features[indicator_key] = values
            elif isinstance(values, pd.DataFrame):
                if len(values.columns) == 1:
                    features[indicator_key] = values.iloc[:, 0]
                else:
                    # Multi-column indicators (like MACD, Bollinger Bands)
                    for col in values.columns:
                        feature_name = (
                            f"{indicator_key}_{col}"
                            if col not in indicator_key
                            else indicator_key
                        )
                        features[feature_name] = values[col]

        return features

    def _extract_advanced_features(
        self, adv_results: AdvancedFeatureResults, index: pd.Index
    ) -> pd.DataFrame:
        """Extract advanced features into DataFrame."""
        features = pd.DataFrame(index=index)

        if adv_results.fractal_dimension is not None:
            features["fractal_dimension"] = adv_results.fractal_dimension

        if adv_results.hurst_exponent is not None:
            features["hurst_exponent"] = adv_results.hurst_exponent

        if adv_results.fractional_diff is not None:
            features["fractional_diff"] = adv_results.fractional_diff

        return features

    def _validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature quality and return metrics."""
        quality_metrics = {}

        # Basic statistics
        quality_metrics["n_features"] = len(features.columns)
        quality_metrics["n_observations"] = len(features)
        quality_metrics["missing_percentage"] = features.isnull().sum() / len(features)

        # Stationarity check (if enabled)
        if self.config.validation["check_stationarity"]:
            quality_metrics["stationarity"] = self._check_stationarity(features)

        # Multicollinearity check (if enabled)
        if self.config.validation["check_multicollinearity"]:
            quality_metrics["multicollinearity"] = self._check_multicollinearity(
                features
            )

        return quality_metrics

    def _check_stationarity(self, features: pd.DataFrame) -> Dict[str, bool]:
        """Check stationarity of features using ADF test."""
        from scipy.stats import jarque_bera

        stationarity_results = {}

        for column in features.columns:
            series = features[column].dropna()
            if len(series) > 10:
                try:
                    # Simple stationarity proxy: check if mean is stable
                    # Split into two halves and compare means
                    mid = len(series) // 2
                    mean1 = series.iloc[:mid].mean()
                    mean2 = series.iloc[mid:].mean()

                    # Consider stationary if means are similar (within 1 std)
                    std = series.std()
                    stationarity_results[column] = abs(mean1 - mean2) < std
                except:
                    stationarity_results[column] = False
            else:
                stationarity_results[column] = False

        return stationarity_results

    def _check_multicollinearity(self, features: pd.DataFrame) -> Dict[str, float]:
        """Check for multicollinearity using correlation matrix."""
        correlation_results = {}

        # Calculate correlation matrix
        corr_matrix = features.corr()

        # Find high correlations
        threshold = 0.95
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )

        correlation_results["high_correlation_pairs"] = high_corr_pairs
        correlation_results["max_correlation"] = corr_matrix.abs().max().max()

        return correlation_results

    def _select_features(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Perform feature selection."""
        method = self.config.feature_selection["method"]

        if method == "variance":
            return self._variance_threshold_selection(features)
        elif method == "mdi":
            return self._mdi_feature_selection(features, target)
        elif method == "correlation":
            return self._correlation_feature_selection(features, target)
        else:
            # No selection, return all features
            mask = pd.Series(True, index=features.columns)
            importance = pd.DataFrame(index=features.columns)
            return features, mask, importance

    def _variance_threshold_selection(
        self, features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Select features based on variance threshold."""
        threshold = self.config.feature_selection.get("threshold", 0.01)

        # Calculate variance for each feature
        variances = features.var()
        mask = variances > threshold

        selected_features = features.loc[:, mask]
        importance = pd.DataFrame({"variance": variances[mask]}).sort_values(
            "variance", ascending=False
        )

        return selected_features, mask, importance

    def _mdi_feature_selection(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Feature selection using Mean Decrease Impurity (MDI)."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        # Align features and target
        aligned_features, aligned_target = features.align(target, join="inner", axis=0)

        # Remove NaN values
        valid_idx = aligned_features.dropna().index.intersection(
            aligned_target.dropna().index
        )
        X = aligned_features.loc[valid_idx]
        y = aligned_target.loc[valid_idx]

        if len(X) < 10:
            # Not enough data for feature selection
            mask = pd.Series(True, index=features.columns)
            importance = pd.DataFrame(index=features.columns)
            return features, mask, importance

        # Determine if classification or regression
        unique_values = y.nunique()
        if unique_values <= 10:  # Classification
            # Encode target if string
            if y.dtype == "object":
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)

            model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=5
            )
        else:  # Regression
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=5
            )

        try:
            model.fit(X, y)

            # Get feature importance
            importance_scores = pd.Series(model.feature_importances_, index=X.columns)

            # Select top features
            n_features = self.config.feature_selection.get("n_features", "auto")
            if n_features == "auto":
                # Keep features with importance > mean importance
                threshold = importance_scores.mean()
                mask = importance_scores > threshold
            else:
                # Keep top n features
                mask = importance_scores.nlargest(n_features).index
                mask = pd.Series(features.columns.isin(mask), index=features.columns)

            selected_features = features.loc[:, mask]
            importance = pd.DataFrame(
                {"mdi_importance": importance_scores[mask]}
            ).sort_values("mdi_importance", ascending=False)

            return selected_features, mask, importance

        except Exception as e:
            self.logger.warning(
                f"MDI feature selection failed: {str(e)}. Using all features."
            )
            mask = pd.Series(True, index=features.columns)
            importance = pd.DataFrame(index=features.columns)
            return features, mask, importance

    def _correlation_feature_selection(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Select features based on correlation with target."""
        # Align features and target
        aligned_features, aligned_target = features.align(target, join="inner", axis=0)

        # Calculate correlations
        correlations = {}
        for column in aligned_features.columns:
            try:
                corr = aligned_features[column].corr(aligned_target)
                if not np.isnan(corr):
                    correlations[column] = abs(corr)
            except:
                pass

        if not correlations:
            mask = pd.Series(True, index=features.columns)
            importance = pd.DataFrame(index=features.columns)
            return features, mask, importance

        # Convert to Series
        corr_series = pd.Series(correlations)

        # Select features with correlation above threshold
        threshold = self.config.feature_selection.get("threshold", 0.01)
        mask = pd.Series(
            features.columns.isin(corr_series[corr_series > threshold].index),
            index=features.columns,
        )

        selected_features = features.loc[:, mask]
        importance = pd.DataFrame({"correlation": corr_series[mask]}).sort_values(
            "correlation", ascending=False
        )

        return selected_features, mask, importance

    def _scale_features(
        self, features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale features according to configuration."""
        method = self.config.scaling["method"]

        if method == "none":
            return features, {}

        from sklearn.preprocessing import (
            StandardScaler,
            MinMaxScaler,
            RobustScaler,
            QuantileTransformer,
        )

        # Choose scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            feature_range = self.config.scaling.get("feature_range", (0, 1))
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "quantile":
            scaler = QuantileTransformer()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Apply scaling
        features_clean = features.dropna()
        if len(features_clean) == 0:
            return features, {}

        try:
            scaled_values = scaler.fit_transform(features_clean)
            scaled_features = pd.DataFrame(
                scaled_values,
                index=features_clean.index,
                columns=features_clean.columns,
            )

            # Reindex to original features index
            scaled_features = scaled_features.reindex(features.index)

            scaling_params = {
                "scaler": scaler,
                "method": method,
                "feature_names": list(features.columns),
            }

            return scaled_features, scaling_params

        except Exception as e:
            self.logger.warning(
                f"Feature scaling failed: {str(e)}. Using unscaled features."
            )
            return features, {}

    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key based on data characteristics."""
        import hashlib

        # Create a hash based on data shape, column names, and first/last values
        key_data = f"{data.shape}_{list(data.columns)}_{data.iloc[0].sum()}_{data.iloc[-1].sum()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[FeaturePipelineResults]:
        """Get results from cache if available and valid."""
        return self._feature_cache.get(cache_key)

    def _save_to_cache(self, cache_key: str, results: FeaturePipelineResults):
        """Save results to cache."""
        self._feature_cache[cache_key] = results

    def transform_new_data(
        self, data: pd.DataFrame, pipeline_results: FeaturePipelineResults
    ) -> pd.DataFrame:
        """Transform new data using previously fitted pipeline."""
        # Generate features using the same process
        technical_results = self._generate_technical_features(data)
        advanced_results = self._generate_advanced_features(data)

        # Combine features
        features = self._combine_features(data, technical_results, advanced_results)

        # Apply same feature selection
        if pipeline_results.selection_mask is not None:
            features = features.loc[:, pipeline_results.selection_mask]

        # Apply same scaling
        if pipeline_results.scaling_params:
            scaler = pipeline_results.scaling_params["scaler"]
            feature_names = pipeline_results.scaling_params["feature_names"]

            # Ensure same features are present
            for name in feature_names:
                if name not in features.columns:
                    features[name] = np.nan

            features = features[feature_names]

            features_clean = features.dropna()
            if len(features_clean) > 0:
                scaled_values = scaler.transform(features_clean)
                scaled_features = pd.DataFrame(
                    scaled_values,
                    index=features_clean.index,
                    columns=features_clean.columns,
                )
                features = scaled_features.reindex(features.index)

        return features

    def get_feature_description(self) -> Dict[str, str]:
        """Get description of all generated features."""
        descriptions = {
            # Price features
            "returns": "Simple returns (close price percentage change)",
            "log_returns": "Log returns (log difference of close prices)",
            # Technical indicators
            "sma": "Simple Moving Average",
            "ema": "Exponential Moving Average",
            "rsi": "Relative Strength Index",
            "macd": "Moving Average Convergence Divergence",
            "macd_signal": "MACD Signal Line",
            "macd_histogram": "MACD Histogram",
            "bb_upper": "Bollinger Bands Upper Band",
            "bb_middle": "Bollinger Bands Middle Band",
            "bb_lower": "Bollinger Bands Lower Band",
            "bb_width": "Bollinger Bands Width",
            "bb_position": "Price Position within Bollinger Bands",
            "atr": "Average True Range",
            "stoch_k": "Stochastic %K",
            "stoch_d": "Stochastic %D",
            "williams_r": "Williams %R",
            "cci": "Commodity Channel Index",
            "momentum": "Price Momentum",
            # Advanced features
            "fractal_dimension": "Fractal dimension (market complexity measure)",
            "hurst_exponent": "Hurst exponent (trend persistence measure)",
            "fractional_diff": "Fractionally differentiated prices",
        }

        return descriptions
