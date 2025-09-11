"""
Feature Engineering Utilities

This module provides utility functions for feature engineering workflows,
separated from UI components for better testability and maintainability.

Design Principles:
- Separation of Concerns: UI logic vs Business logic
- Testability: Pure Python functions without Streamlit dependencies
- Reusability: Functions can be used across different pages
- Maintainability: Easy to modify business logic without touching UI
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback
from pathlib import Path
import sys

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.features.technical import TechnicalIndicators
    from src.features.advanced import AdvancedFeatures
    from src.features.pipeline import FeaturePipeline
    from src.features.importance import FeatureImportance
    from src.config import settings
except ImportError as e:
    # Handle import errors gracefully for testing
    print(f"Import warning in feature_utils: {e}")


class FeatureEngineeringManager:
    """Manager class for feature engineering operations"""

    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.advanced_features = AdvancedFeatures()
        self.feature_pipeline = FeaturePipeline()
        self.feature_importance = FeatureImportance()

    def initialize_session_state(self, session_state: Dict) -> None:
        """Initialize session state for feature engineering"""

        if "feature_cache" not in session_state:
            session_state["feature_cache"] = {}

        if "feature_pipeline_cache" not in session_state:
            session_state["feature_pipeline_cache"] = {}

    def calculate_technical_indicators(
        self,
        ticker: str,
        data: pd.DataFrame,
        config: Dict[str, Any],
        session_state: Dict,
    ) -> Tuple[bool, str]:
        """
        Calculate technical indicators using the TechnicalIndicators framework

        Args:
            ticker: Symbol identifier
            data: Price data DataFrame
            config: Technical indicators configuration
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            # Validate and normalize data columns
            normalized_data = self._validate_and_normalize_data(data)
            if normalized_data is None:
                return False, "Data validation failed"

            # Build indicator list based on config
            indicators_to_calculate = self._build_indicator_list(config)

            if not indicators_to_calculate:
                return False, "No indicators selected"

            # Calculate indicators using lowercase column names
            lowercase_data = normalized_data.copy()
            lowercase_data.columns = [col.lower() for col in lowercase_data.columns]

            all_results = self.technical_indicators.calculate_all_indicators(
                lowercase_data, indicators_to_calculate
            )

            # Process results for chart and table display
            features_for_chart, feature_df_for_table = self._process_technical_results(
                all_results, normalized_data
            )

            if feature_df_for_table.empty:
                return False, "No technical indicators were calculated successfully"

            # Store results in session state
            feature_key = f"{ticker}_technical"

            # Store table data
            session_state["feature_cache"][feature_key] = feature_df_for_table

            # Store metadata with chart-ready data
            session_state["feature_cache"][f"{feature_key}_metadata"] = {
                "original_data": normalized_data,
                "features_dict_for_chart": features_for_chart,
                "type": "technical",
                "config": config,
                "calculated_at": datetime.now(),
            }

            return (
                True,
                f"Successfully calculated {len(feature_df_for_table.columns)} technical indicators",
            )

        except Exception as e:
            return False, f"Technical indicator calculation failed: {str(e)}"

    def calculate_advanced_features(
        self,
        ticker: str,
        data: pd.DataFrame,
        config: Dict[str, Any],
        session_state: Dict,
    ) -> Tuple[bool, str]:
        """
        Calculate advanced features using the AdvancedFeatures framework

        Args:
            ticker: Symbol identifier
            data: Price data DataFrame
            config: Advanced features configuration
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            # Validate data
            clean_data = self._validate_price_data(data)
            if clean_data is None:
                return False, "Data validation failed"

            # Calculate features based on configuration
            results = {}

            # Fractal Dimension
            if config.get("fractal_enabled", False):
                fractal_window = config.get("fractal_window", 100)
                fractal_result = self.advanced_features.calculate_fractal_dimension(
                    clean_data["Close"], window=fractal_window
                )
                results["fractal_dimension"] = fractal_result

            # Hurst Exponent
            if config.get("hurst_enabled", False):
                hurst_window = config.get("hurst_window", 100)
                hurst_result = self.advanced_features.calculate_hurst_exponent(
                    clean_data["Close"], window=hurst_window
                )
                results["hurst_exponent"] = hurst_result

            # Information Bars
            if config.get("info_bars_enabled", False):
                volume_col = self._find_volume_column(clean_data)
                if volume_col:
                    bar_type = config.get("bar_type", "volume")
                    threshold = config.get("bar_threshold")

                    info_bars = self.advanced_features.create_information_bars(
                        clean_data,
                        bar_type=bar_type,
                        threshold=threshold,
                        volume_col=volume_col,
                    )
                    results["information_bars"] = info_bars

            if not results:
                return False, "No advanced features were calculated"

            # Process results
            features_dict, feature_df = self._process_advanced_results(
                results, clean_data
            )

            # Store results
            feature_key = f"{ticker}_advanced"

            session_state["feature_cache"][feature_key] = feature_df
            session_state["feature_cache"][f"{feature_key}_metadata"] = {
                "original_data": clean_data,
                "features_dict": features_dict,
                "type": "advanced",
                "config": config,
                "calculated_at": datetime.now(),
            }

            return (
                True,
                f"Successfully calculated {len([k for k, v in features_dict.items() if isinstance(v, pd.Series)])} advanced features",
            )

        except Exception as e:
            return False, f"Advanced features calculation failed: {str(e)}"

    def run_feature_pipeline(
        self,
        ticker: str,
        data: pd.DataFrame,
        config: Dict[str, Any],
        session_state: Dict,
    ) -> Tuple[bool, str]:
        """
        Run comprehensive feature pipeline

        Args:
            ticker: Symbol identifier
            data: Price data DataFrame
            config: Pipeline configuration
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            # Validate data
            clean_data = self._validate_price_data(data)
            if clean_data is None:
                return False, "Data validation failed"

            # Initialize pipeline
            pipeline_config = {
                "include_technical": config.get("include_technical", True),
                "include_advanced": config.get("include_advanced", True),
                "feature_selection": config.get("feature_selection", True),
                "quality_validation": config.get("quality_validation", True),
                "max_features": config.get("max_features", 50),
                "correlation_threshold": config.get("correlation_threshold", 0.8),
            }

            # Run pipeline
            results = self.feature_pipeline.run_comprehensive_pipeline(
                data=clean_data, config=pipeline_config
            )

            if results is None or results.features is None:
                return False, "Pipeline execution failed"

            # Store results
            pipeline_key = f"{ticker}_pipeline"

            session_state["feature_pipeline_cache"][pipeline_key] = {
                "results": results,
                "config": config,
                "ticker": ticker,
                "calculated_at": datetime.now(),
            }

            feature_count = (
                len(results.features.columns) if results.features is not None else 0
            )
            return (
                True,
                f"Pipeline completed successfully with {feature_count} features",
            )

        except Exception as e:
            return False, f"Feature pipeline failed: {str(e)}"

    def _validate_and_normalize_data(
        self, data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Validate and normalize data columns for technical indicators"""

        try:
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = []

            # Check for standard column names and their lowercase variants
            column_mapping = {}
            for col in required_columns:
                if col in data.columns:
                    column_mapping[col] = col
                elif col.lower() in data.columns:
                    column_mapping[col] = col.lower()
                else:
                    missing_columns.append(col)

            if missing_columns:
                return None

            # Create normalized data with standard column names
            normalized_data = pd.DataFrame(index=data.index)
            for standard_col, actual_col in column_mapping.items():
                normalized_data[standard_col] = data[actual_col]

            # Remove any rows with NaN values in required columns
            normalized_data = normalized_data.dropna()

            if len(normalized_data) < 50:
                return None

            return normalized_data

        except Exception:
            return None

    def _validate_price_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate price data for advanced features"""

        try:
            # Ensure we have at least Close price
            if "Close" not in data.columns and "close" not in data.columns:
                return None

            # Standardize column names
            clean_data = data.copy()

            # Map common column name variations
            column_mapping = {
                "close": "Close",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
            }

            clean_data = clean_data.rename(columns=column_mapping)

            # Ensure Close column exists
            if "Close" not in clean_data.columns:
                return None

            # Remove NaN values
            clean_data = clean_data.dropna()

            if len(clean_data) < 30:
                return None

            return clean_data

        except Exception:
            return None

    def _build_indicator_list(self, config: Dict[str, Any]) -> List[str]:
        """Build list of indicators to calculate based on configuration"""

        indicators_to_calculate = []

        if config.get("sma_enabled", False):
            indicators_to_calculate.append("sma")
        if config.get("ema_enabled", False):
            indicators_to_calculate.append("ema")
        if config.get("rsi_enabled", False):
            indicators_to_calculate.append("rsi")
        if config.get("macd_enabled", False):
            indicators_to_calculate.append("macd")
        if config.get("bb_enabled", False):
            indicators_to_calculate.append("bollinger_bands")
        if config.get("atr_enabled", False):
            indicators_to_calculate.append("atr")
        if config.get("stoch_enabled", False):
            indicators_to_calculate.append("stochastic")
        if config.get("williams_enabled", False):
            indicators_to_calculate.append("williams_r")
        if config.get("momentum_enabled", False):
            indicators_to_calculate.append("momentum")

        return indicators_to_calculate

    def _process_technical_results(
        self, all_results: Dict, normalized_data: pd.DataFrame
    ) -> Tuple[Dict, pd.DataFrame]:
        """Process technical indicator results for storage"""

        # Chart-ready data (preserve original structure)
        features_for_chart = {}
        for name, result_obj in all_results.items():
            if hasattr(result_obj, "values"):
                features_for_chart[name] = result_obj.values

        # Table-ready data (expand multi-column indicators)
        feature_df_for_table = pd.DataFrame(index=normalized_data.index)
        for name, values in features_for_chart.items():
            if isinstance(values, pd.DataFrame):
                for col in values.columns:
                    # Create readable column names
                    readable_name = self._create_readable_indicator_name(name, col)
                    feature_df_for_table[readable_name] = values[col]
            elif isinstance(values, pd.Series):
                readable_name = self._create_readable_indicator_name(name)
                feature_df_for_table[readable_name] = values

        return features_for_chart, feature_df_for_table

    def _process_advanced_results(
        self, results: Dict, clean_data: pd.DataFrame
    ) -> Tuple[Dict, pd.DataFrame]:
        """Process advanced feature results for storage"""

        # Separate Series and DataFrame results
        features_dict = {}
        feature_df = pd.DataFrame(index=clean_data.index)

        for name, values in results.items():
            if isinstance(values, pd.Series):
                features_dict[name] = values
                feature_df[name] = values
            elif isinstance(values, pd.DataFrame):
                # Keep DataFrame as-is for special handling (like information_bars)
                features_dict[name] = values

        return features_dict, feature_df

    def _create_readable_indicator_name(
        self, indicator_name: str, column_name: str = None
    ) -> str:
        """Create readable names for indicators"""

        name_mapping = {
            "bollinger_bands": "BB",
            "stochastic": "Stoch",
            "williams_r": "Williams %R",
            "macd": "MACD",
        }

        base_name = name_mapping.get(indicator_name, indicator_name.upper())

        if column_name:
            return f"{base_name}_{column_name}"
        else:
            return base_name

    def _find_volume_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find volume column in data"""

        volume_candidates = ["Volume", "volume", "Vol", "vol"]

        for candidate in volume_candidates:
            if candidate in data.columns:
                return candidate

        return None

    def get_feature_summary(self, session_state: Dict) -> Dict[str, Any]:
        """Get summary of available features"""

        feature_cache = session_state.get("feature_cache", {})
        pipeline_cache = session_state.get("feature_pipeline_cache", {})

        # Count features (excluding metadata)
        feature_sets = [
            key for key in feature_cache.keys() if not key.endswith("_metadata")
        ]
        pipeline_sets = list(pipeline_cache.keys())

        total_features = 0
        for key in feature_sets:
            if key in feature_cache:
                feature_data = feature_cache[key]
                if isinstance(feature_data, pd.DataFrame):
                    total_features += len(feature_data.columns)

        return {
            "feature_sets": len(feature_sets),
            "pipeline_sets": len(pipeline_sets),
            "total_features": total_features,
            "available_sets": feature_sets + pipeline_sets,
        }

    def get_feature_data_for_chart(
        self, feature_key: str, session_state: Dict
    ) -> Optional[Dict]:
        """Get feature data formatted for chart display"""

        metadata_key = f"{feature_key}_metadata"

        if metadata_key in session_state["feature_cache"]:
            metadata = session_state["feature_cache"][metadata_key]
            feature_type = metadata.get("type", "unknown")

            if feature_type == "technical":
                return metadata.get("features_dict_for_chart", {})
            elif feature_type == "advanced":
                return metadata.get("features_dict", {})

        return None

    def get_feature_statistics(
        self, feature_key: str, session_state: Dict
    ) -> Optional[pd.DataFrame]:
        """Get feature statistics"""

        if feature_key in session_state["feature_cache"]:
            feature_data = session_state["feature_cache"][feature_key]

            if isinstance(feature_data, pd.DataFrame):
                return feature_data.describe()

        elif feature_key in session_state["feature_pipeline_cache"]:
            pipeline_data = session_state["feature_pipeline_cache"][feature_key]
            results = pipeline_data["results"]

            if results.features is not None:
                return results.features.describe()

        return None
