"""
Backtesting Utilities for Streamlit Application

This module provides utility functions for backtesting integration,
data preparation, and result processing for the Streamlit interface.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.backtesting import (
        BacktestEngine,
        BuyAndHoldStrategy,
        MomentumStrategy,
        MeanReversionStrategy,
        ModelBasedStrategy,
        MultiAssetStrategy,
        PerformanceCalculator,
        Portfolio,
        Order,
        OrderSide,
        OrderType,
    )
except ImportError as e:
    print(f"Import error in backtest_utils: {e}")


class BacktestDataPreparer:
    """Prepare data for backtesting from different sources"""

    def __init__(self):
        pass

    def find_metadata_for_features(
        self, feature_key: str, session_state
    ) -> Optional[pd.DataFrame]:
        """
        Find corresponding metadata for a feature DataFrame that lacks price data

        Args:
            feature_key: Key of the feature cache entry
            session_state: Streamlit session state containing feature_cache

        Returns:
            Original price data from metadata if found, None otherwise
        """
        metadata_key = f"{feature_key}_metadata"
        print(f"🔍 Looking for metadata key: {metadata_key}")

        # Debug: Show all available feature cache keys
        if hasattr(session_state, "feature_cache") and session_state.feature_cache:
            available_keys = list(session_state.feature_cache.keys())
            print(f"📋 Available feature cache keys: {available_keys}")

            # Look for keys that contain the feature_key
            matching_keys = [
                k
                for k in available_keys
                if feature_key in k and "metadata" in k.lower()
            ]
            print(f"🎯 Matching metadata keys: {matching_keys}")
        else:
            print("❌ No feature_cache found or it's empty")
            return None

        if (
            hasattr(session_state, "feature_cache")
            and metadata_key in session_state.feature_cache
        ):
            metadata = session_state.feature_cache[metadata_key]
            print(f"✅ Found metadata with type: {type(metadata)}")

            if isinstance(metadata, dict):
                print(f"📊 Metadata keys: {list(metadata.keys())}")

                if "original_data" in metadata:
                    original_data = metadata["original_data"]
                    if isinstance(original_data, pd.DataFrame):
                        print(
                            f"🎉 Found original_data in metadata '{metadata_key}': {original_data.shape}"
                        )
                        print(f"Original data columns: {list(original_data.columns)}")
                        return original_data
                    else:
                        print(
                            f"original_data is not a DataFrame: {type(original_data)}"
                        )
                else:
                    print("No 'original_data' key found in metadata")
            else:
                print("Metadata is not a dictionary")
        else:
            print(f"❌ Metadata key '{metadata_key}' not found in feature cache")
            if hasattr(session_state, "feature_cache"):
                available_keys = list(session_state.feature_cache.keys())
                print(f"📋 Available feature cache keys: {available_keys}")

                # Show all metadata keys
                metadata_keys = [k for k in available_keys if "metadata" in k.lower()]
                print(f"🔍 All metadata keys found: {metadata_keys}")

                # Try to find similar keys (case-insensitive search)
                feature_base = feature_key.lower()
                similar_keys = []
                for key in available_keys:
                    if (
                        key.lower().startswith(feature_base)
                        or feature_base in key.lower()
                    ):
                        similar_keys.append(key)

                if similar_keys:
                    print(f"🎯 Found similar keys: {similar_keys}")
                    # Try to use the most similar metadata key
                    for similar_key in similar_keys:
                        if similar_key.endswith("_metadata"):
                            metadata = session_state.feature_cache[similar_key]
                            print(f"🔍 Trying similar metadata key: {similar_key}")
                            print(f"📊 Metadata type: {type(metadata)}")
                            if isinstance(metadata, dict):
                                print(f"📋 Metadata keys: {list(metadata.keys())}")
                                if "original_data" in metadata:
                                    original_data = metadata["original_data"]
                                    if isinstance(original_data, pd.DataFrame):
                                        print(
                                            f"🎉 Found original_data in similar key '{similar_key}': {original_data.shape}"
                                        )
                                        print(
                                            f"📊 Original data columns: {list(original_data.columns)}"
                                        )
                                        return original_data
                                    else:
                                        print(
                                            f"❌ original_data is not DataFrame: {type(original_data)}"
                                        )
                                else:
                                    print("❌ No 'original_data' key in metadata")
                            else:
                                print(f"❌ Metadata is not dict: {type(metadata)}")
                else:
                    print("❌ No similar keys found")

        return None

    def find_best_metadata_match(
        self, feature_key: str, session_state
    ) -> Optional[pd.DataFrame]:
        """
        Find the best metadata match for a feature key using multiple strategies

        Args:
            feature_key: Key of the feature cache entry
            session_state: Streamlit session state

        Returns:
            Original price data if found, None otherwise
        """
        if not hasattr(session_state, "feature_cache"):
            print("❌ No feature_cache in session_state")
            return None

        available_keys = list(session_state.feature_cache.keys())
        metadata_keys = [k for k in available_keys if "metadata" in k.lower()]

        print(f"🔍 Searching for metadata for feature key: '{feature_key}'")
        print(f"📋 Available metadata keys: {metadata_keys}")

        # Strategy 1: Extract ticker and look for ticker_technical_metadata
        ticker_match = None
        if "_" in feature_key:
            parts = feature_key.split("_")
            ticker = parts[0]  # First part should be ticker (e.g., "AAPL")

            # Look for patterns like "AAPL_technical_metadata"
            for metadata_key in metadata_keys:
                if (
                    metadata_key.startswith(f"{ticker}_")
                    and "technical" in metadata_key
                ):
                    print(f"🎯 Strategy 1 - Found ticker match: {metadata_key}")
                    original_data = self._extract_original_data(
                        metadata_key, session_state
                    )
                    if original_data is not None:
                        return original_data

        # Strategy 2: Look for exact feature_key + _metadata
        exact_metadata_key = f"{feature_key}_metadata"
        if exact_metadata_key in metadata_keys:
            print(f"🎯 Strategy 2 - Found exact match: {exact_metadata_key}")
            original_data = self._extract_original_data(
                exact_metadata_key, session_state
            )
            if original_data is not None:
                return original_data

        # Strategy 3: Fuzzy matching - look for any metadata containing the ticker
        if "_" in feature_key:
            ticker = feature_key.split("_")[0]
            for metadata_key in metadata_keys:
                if ticker.upper() in metadata_key.upper():
                    print(f"🎯 Strategy 3 - Found fuzzy match: {metadata_key}")
                    original_data = self._extract_original_data(
                        metadata_key, session_state
                    )
                    if original_data is not None:
                        return original_data

        print("❌ No metadata match found with any strategy")
        return None

    def _extract_original_data(
        self, metadata_key: str, session_state
    ) -> Optional[pd.DataFrame]:
        """Extract original_data from a metadata entry"""
        try:
            metadata = session_state.feature_cache[metadata_key]
            if isinstance(metadata, dict) and "original_data" in metadata:
                original_data = metadata["original_data"]
                if isinstance(original_data, pd.DataFrame):
                    print(
                        f"✅ Extracted original_data from '{metadata_key}': {original_data.shape}"
                    )
                    return original_data
                else:
                    print(
                        f"❌ original_data is not DataFrame in '{metadata_key}': {type(original_data)}"
                    )
            else:
                print(
                    f"❌ Invalid metadata structure in '{metadata_key}': {type(metadata)}"
                )
        except Exception as e:
            print(f"❌ Error extracting from '{metadata_key}': {e}")
        return None

    def prepare_feature_data(
        self, feature_data: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """
        Prepare feature data for backtesting with enhanced technical indicator support

        Args:
            feature_data: Feature data from feature cache

        Returns:
            Prepared DataFrame with price data or None if no price data found
        """

        # Debug: Print structure to understand data format
        print(f"Processing feature data with type: {type(feature_data)}")

        if isinstance(feature_data, pd.DataFrame):
            print(f"Direct DataFrame with columns: {list(feature_data.columns)}")

            # Check if DataFrame contains actual OHLCV price data (not just technical indicators)
            columns_lower = [col.lower() for col in feature_data.columns]

            print(f"🔍 Analyzing columns for OHLCV data: {list(feature_data.columns)}")

            # Look for exact OHLCV column matches (case insensitive)
            exact_ohlcv_matches = []
            for col in feature_data.columns:
                col_clean = col.lower().strip()
                if col_clean in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "adj_close",
                    "adjusted_close",
                ]:
                    exact_ohlcv_matches.append(col)

            print(f"📊 Exact OHLCV matches found: {exact_ohlcv_matches}")

            # Check for technical indicator patterns to avoid false positives
            technical_patterns = [
                "sma",
                "ema",
                "rsi",
                "macd",
                "bb_",
                "atr",
                "stoch",
                "bollinger",
                "moving_average",
                "momentum",
                "signal",
            ]

            is_technical_indicators = all(
                any(pattern in col.lower() for pattern in technical_patterns)
                for col in feature_data.columns
            )

            print(f"🔧 Contains only technical indicators: {is_technical_indicators}")

            # Determine if this is actual price data or technical indicators
            if (
                len(exact_ohlcv_matches) >= 2 and not is_technical_indicators
            ):  # Need at least 2 OHLCV columns
                print("✅ DataFrame contains actual OHLCV price data")
                return self._validate_price_data(feature_data)
            else:
                print(
                    "🔧 DataFrame contains technical indicators but no actual OHLCV price data"
                )
                print(f"📊 Technical indicators found: {list(feature_data.columns)}")
                print("📋 Will check metadata for original price data")
                # Return None to signal that metadata should be checked
                return None

        elif isinstance(feature_data, dict):
            print(f"Feature data keys: {list(feature_data.keys())}")

            # Check for original_data first (common in metadata)
            if "original_data" in feature_data:
                original_data = feature_data["original_data"]
                if isinstance(original_data, pd.DataFrame):
                    print(
                        f"Found original_data with columns: {list(original_data.columns)}"
                    )
                    return self._validate_price_data(original_data)

            # Check for raw data
            if "raw_data" in feature_data:
                raw_data = feature_data["raw_data"]
                if isinstance(raw_data, pd.DataFrame):
                    print(f"Found raw_data with columns: {list(raw_data.columns)}")
                    return self._validate_price_data(raw_data)

            # Check for data
            if "data" in feature_data:
                data = feature_data["data"]
                if isinstance(data, pd.DataFrame):
                    print(f"Found data with columns: {list(data.columns)}")
                    return self._validate_price_data(data)

            # Check for source data
            if "source_data" in feature_data:
                source_data = feature_data["source_data"]
                if isinstance(source_data, pd.DataFrame):
                    print(
                        f"Found source_data with columns: {list(source_data.columns)}"
                    )
                    return self._validate_price_data(source_data)

            # Try to reconstruct from features
            if "features" in feature_data:
                features = feature_data["features"]
                if isinstance(features, dict):
                    # Look for price-related features
                    price_data = {}

                    print(f"Available feature columns: {list(features.keys())}")

                    # Try to find basic price data with multiple strategies
                    for name, series in features.items():
                        if isinstance(series, pd.Series):
                            name_lower = name.lower()

                            # Primary price matching
                            if any(
                                keyword in name_lower
                                for keyword in ["close", "adj_close", "adjusted_close"]
                            ):
                                price_data["Close"] = series
                            elif (
                                any(keyword in name_lower for keyword in ["open"])
                                and "close" not in name_lower
                            ):
                                price_data["Open"] = series
                            elif (
                                any(keyword in name_lower for keyword in ["high"])
                                and "close" not in name_lower
                            ):
                                price_data["High"] = series
                            elif (
                                any(keyword in name_lower for keyword in ["low"])
                                and "close" not in name_lower
                            ):
                                price_data["Low"] = series
                            elif any(
                                keyword in name_lower for keyword in ["volume", "vol"]
                            ):
                                price_data["Volume"] = series
                            # Fallback: any column with 'price' in name becomes Close
                            elif "price" in name_lower and "Close" not in price_data:
                                price_data["Close"] = series

                    print(f"Extracted price columns: {list(price_data.keys())}")

                    if price_data:
                        df = pd.DataFrame(price_data)
                        return self._validate_price_data(df)

            # Look for any DataFrame in the top level of the dict
            if isinstance(feature_data, dict):
                for key, value in feature_data.items():
                    if isinstance(value, pd.DataFrame):
                        print(
                            f"Found DataFrame in key '{key}', columns: {list(value.columns)}"
                        )
                        # Check if it looks like price data
                        columns_lower = [col.lower() for col in value.columns]
                        if any(
                            col in columns_lower
                            for col in ["close", "open", "high", "low", "price"]
                        ):
                            return self._validate_price_data(value)

            # Final attempt: try to extract the first DataFrame found
            if isinstance(feature_data, dict):
                for key, value in feature_data.items():
                    if isinstance(value, pd.DataFrame) and not value.empty:
                        print(f"Using DataFrame from key '{key}' as fallback")
                        return self._validate_price_data(value)

        elif isinstance(feature_data, pd.DataFrame):
            # Already in DataFrame format - check if it contains price data
            print(
                f"Processing direct DataFrame with columns: {list(feature_data.columns)}"
            )
            columns_lower = [col.lower() for col in feature_data.columns]

            # Check if this looks like price data (OHLCV)
            price_indicators = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "price",
                "adj",
            ]
            has_price_data = any(
                any(indicator in col for indicator in price_indicators)
                for col in columns_lower
            )

            if has_price_data:
                print("DataFrame appears to contain price data")
                return self._validate_price_data(feature_data)
            else:
                print(
                    "DataFrame does not contain obvious price data - cannot use for backtesting"
                )
                raise ValueError(
                    "DataFrame does not contain price data (OHLCV) required for backtesting"
                )

        # If we can't extract proper price data, create a simple synthetic dataset
        # This is a fallback for demonstration purposes
        print("Warning: Cannot extract proper price data, creating synthetic data")
        return self._create_synthetic_price_data()

    def _validate_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean price data for backtesting

        Args:
            data: Raw price data

        Returns:
            Cleaned price data
        """
        # Make a copy to avoid modifying original data
        data = data.copy()

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "Date" in data.columns:
                data.set_index("Date", inplace=True)
            elif "date" in data.columns:
                data.set_index("date", inplace=True)
            elif data.index.dtype == "object":
                try:
                    data.index = pd.to_datetime(data.index)
                except:
                    # If index conversion fails, create a date range
                    data.index = pd.date_range(
                        start="2023-01-01", periods=len(data), freq="D"
                    )

        # Ensure timezone-naive index for backtest engine compatibility
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Look for essential price columns
        essential_cols = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = []

        # Case-insensitive column mapping
        column_mapping = {}
        for col in essential_cols:
            for existing_col in data.columns:
                if (
                    col.lower() in existing_col.lower()
                    or existing_col.lower() in col.lower()
                ):
                    column_mapping[existing_col] = col
                    break

        # Apply column mapping
        data = data.rename(columns=column_mapping)
        available_cols = [col for col in essential_cols if col in data.columns]

        # If we don't have Close price, try to find any price column
        if "Close" not in available_cols:
            price_cols = [
                c
                for c in data.columns
                if any(term in c.lower() for term in ["price", "value", "close", "adj"])
            ]
            if price_cols:
                data["Close"] = pd.to_numeric(data[price_cols[0]], errors="coerce")
                available_cols.append("Close")
            else:
                # If no price data at all, we can't proceed
                print(f"❌ Available columns: {list(data.columns)}")
                print(
                    f"❌ No OHLCV price data found. This appears to be technical indicators data."
                )
                raise ValueError("No price data found in the dataset")

        # Ensure Close is numeric
        if "Close" in data.columns:
            data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

        # Create missing OHLC from Close if needed
        if "Close" in available_cols:
            # Ensure Close is properly numeric
            data["Close"] = pd.to_numeric(data["Close"], errors="coerce").dropna()

            if "Open" not in available_cols:
                data["Open"] = data["Close"].shift(1).fillna(data["Close"])
                available_cols.append("Open")
            else:
                data["Open"] = pd.to_numeric(data["Open"], errors="coerce").fillna(
                    data["Close"]
                )

            if "High" not in available_cols:
                # High should be at least as high as Open and Close
                data["High"] = data[["Open", "Close"]].max(axis=1)
                # Add some random variation to make it more realistic
                np.random.seed(42)  # For reproducibility
                variation = np.random.uniform(1.0, 1.02, len(data))
                data["High"] = data["High"] * variation
                available_cols.append("High")
            else:
                data["High"] = pd.to_numeric(data["High"], errors="coerce").fillna(
                    data[["Open", "Close"]].max(axis=1)
                )

            if "Low" not in available_cols:
                # Low should be at most as low as Open and Close
                data["Low"] = data[["Open", "Close"]].min(axis=1)
                # Add some random variation to make it more realistic
                np.random.seed(42)  # For reproducibility
                variation = np.random.uniform(0.98, 1.0, len(data))
                data["Low"] = data["Low"] * variation
                available_cols.append("Low")
            else:
                data["Low"] = pd.to_numeric(data["Low"], errors="coerce").fillna(
                    data[["Open", "Close"]].min(axis=1)
                )

            if "Volume" not in available_cols:
                # Generate synthetic volume if not available
                data["Volume"] = np.random.uniform(100000, 1000000, len(data))
                available_cols.append("Volume")

        # Ensure all essential columns are present
        for col in essential_cols:
            if col not in data.columns:
                if col == "Volume":
                    data[col] = 1000000  # Default volume
                else:
                    data[col] = data["Close"]  # Use Close as fallback

        # Select only the essential columns
        data = data[essential_cols]

        # Remove NaN values
        data = data.dropna()

        # Sort by date
        data = data.sort_index()

        # Ensure all values are positive
        for col in ["Open", "High", "Low", "Close"]:
            data[col] = data[col].abs()

        data["Volume"] = data["Volume"].abs()

        # Basic validation: High >= Low, High >= max(Open, Close), Low <= min(Open, Close)
        data["High"] = np.maximum(data["High"], np.maximum(data["Open"], data["Close"]))
        data["Low"] = np.minimum(data["Low"], np.minimum(data["Open"], data["Close"]))

        return data

    def _create_synthetic_price_data(self) -> pd.DataFrame:
        """Create synthetic price data for demonstration when real data is not available"""

        # Create a simple random walk price series
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

        # Generate price series
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        prices = [initial_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data["Close"] = prices

        # Create synthetic OHLC from Close
        data["Open"] = data["Close"].shift(1).fillna(data["Close"].iloc[0])

        # Add some intraday variation
        daily_range = np.random.uniform(
            0.005, 0.03, len(data)
        )  # 0.5% to 3% daily range
        high_factor = 1 + daily_range / 2
        low_factor = 1 - daily_range / 2

        data["High"] = np.maximum(data["Open"], data["Close"]) * high_factor
        data["Low"] = np.minimum(data["Open"], data["Close"]) * low_factor

        # Synthetic volume
        data["Volume"] = np.random.uniform(100000, 1000000, len(data))

        return data

    def prepare_model_data(
        self, model_info: Dict, feature_key: str
    ) -> Tuple[Any, pd.DataFrame]:
        """
        Prepare model and corresponding data for model-based backtesting

        Args:
            model_info: Model information from cache
            feature_key: Feature set key

        Returns:
            Tuple of (model, feature_data)
        """
        model = model_info.get("model")
        if model is None:
            raise ValueError("No model found in model_info")

        # Get feature data that was used to train this model
        feature_data = model_info.get("feature_data")
        if feature_data is None:
            raise ValueError("No feature data found for this model")

        return model, feature_data


class StrategyBuilder:
    """Build strategies for backtesting"""

    def __init__(self):
        pass

    def build_strategy(
        self, strategy_config: Dict[str, Any], symbols: List[str]
    ) -> Any:
        """
        Build strategy instance based on configuration

        Args:
            strategy_config: Strategy configuration
            symbols: List of symbols to trade

        Returns:
            Strategy instance
        """
        strategy_type = strategy_config.get("strategy_type")
        parameters = strategy_config.get("parameters", {})

        if strategy_type == "Buy & Hold":
            return BuyAndHoldStrategy(symbols=symbols)

        elif strategy_type == "Momentum":
            return MomentumStrategy(
                symbols=symbols,
                short_window=parameters.get("short_window", 10),
                long_window=parameters.get("long_window", 50),
                **{
                    k: v
                    for k, v in parameters.items()
                    if k not in ["short_window", "long_window"]
                },
            )

        elif strategy_type == "Mean Reversion":
            return MeanReversionStrategy(
                symbols=symbols,
                window=parameters.get("window", 20),
                num_std=parameters.get("num_std", 2.0),
                **{
                    k: v
                    for k, v in parameters.items()
                    if k not in ["window", "num_std"]
                },
            )

        elif strategy_type == "Model-Based":
            return self._build_model_strategy(parameters, symbols)

        elif strategy_type == "Multi-Asset":
            return self._build_multi_asset_strategy(parameters, symbols)

        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _build_model_strategy(self, parameters: Dict, symbols: List[str]):
        """Build model-based strategy using backend implementation"""
        try:
            from src.backtesting.advanced_strategies import ModelBasedStrategy

            # Get model from parameters or session state
            model = parameters.get("model")
            if model is None:
                # Try to get from session state
                import streamlit as st

                model = st.session_state.get("selected_model")

            if model is None:
                raise ValueError("No model provided for model-based strategy")

            # Create strategy with backend class
            return ModelBasedStrategy(
                model=model,
                symbols=symbols,
                confidence_threshold=parameters.get("confidence_threshold", 0.7),
                position_sizing=parameters.get("position_sizing", "Fixed"),
            )

        except ImportError as e:
            raise ImportError(f"Failed to import ModelBasedStrategy from backend: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create model-based strategy: {e}")

    def _build_multi_asset_strategy(self, parameters: Dict, symbols: List[str]):
        """Build multi-asset portfolio strategy using backend implementation"""
        try:
            from src.backtesting.advanced_strategies import MultiAssetStrategy

            # Create strategy with backend class
            return MultiAssetStrategy(
                symbols=symbols,
                rebalance_frequency=parameters.get("rebalance_frequency", "Monthly"),
                risk_model=parameters.get("risk_model", "Equal Weight"),
                max_position=parameters.get("max_position", 0.2),
            )

        except ImportError as e:
            raise ImportError(f"Failed to import MultiAssetStrategy from backend: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create multi-asset strategy: {e}")


class BacktestResultProcessor:
    """Process and format backtest results"""

    def __init__(self):
        self.calculator = PerformanceCalculator()

    def process_results(
        self, engine: BacktestEngine, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process backtest results into standardized format

        Args:
            engine: Completed backtest engine
            config: Backtest configuration

        Returns:
            Processed results dictionary
        """
        # Extract basic results
        portfolio_values = (
            [value[1] for value in engine.portfolio_values]
            if engine.portfolio_values
            else [config["initial_capital"]]
        )

        # Calculate returns
        returns = pd.Series(portfolio_values).pct_change().dropna()

        # Get trades and positions
        trades = self._format_trades(engine.get_trades_summary())
        positions = engine.get_positions_summary()

        # Calculate comprehensive metrics
        try:
            metrics = self.calculator.calculate_comprehensive_metrics(
                returns=returns,
                portfolio_values=pd.Series(portfolio_values),
                trades=trades,
                benchmark_returns=None,
                initial_capital=config["initial_capital"],
            )
        except Exception as e:
            # Create minimal metrics if calculation fails
            metrics = self._create_minimal_metrics(
                returns, portfolio_values, config["initial_capital"]
            )

        # Calculate additional metrics
        win_rate = self._calculate_win_rate(trades)
        drawdown_info = self._calculate_drawdown_info(portfolio_values)

        return {
            "portfolio_values": portfolio_values,
            "returns": returns,
            "metrics": metrics,
            "trades": trades,
            "positions": positions,
            "win_rate": win_rate,
            "drawdown_info": drawdown_info,
            "config": config,
            "summary": self._create_summary(metrics, trades, win_rate),
        }

    def _format_trades(self, trades_summary) -> List[Dict]:
        """Format trades for display"""
        if isinstance(trades_summary, pd.DataFrame) and not trades_summary.empty:
            return trades_summary.to_dict("records")
        elif isinstance(trades_summary, list):
            return trades_summary
        else:
            return []

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate using backend metrics when possible"""
        if not trades:
            return 0.0

        # Use backend PerformanceCalculator if available
        try:
            from src.backtesting.metrics import PerformanceCalculator

            calculator = PerformanceCalculator()
            # Use backend's trade metrics calculation
            trade_metrics = calculator._calculate_trade_metrics(trades)
            return trade_metrics.get("win_rate", 0.0)
        except:
            # Fallback to simple calculation for UI display
            profitable_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
            return profitable_trades / len(trades) if trades else 0.0

    def _calculate_drawdown_info(self, portfolio_values: List[float]) -> Dict[str, Any]:
        """Calculate detailed drawdown information"""
        if len(portfolio_values) < 2:
            return {"max_drawdown": 0, "current_drawdown": 0, "drawdown_duration": 0}

        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max

        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]

        # Calculate drawdown duration
        in_drawdown = drawdown < -0.001  # More than 0.1% drawdown
        if in_drawdown.any():
            drawdown_periods = []
            current_period = 0
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                        current_period = 0

            if current_period > 0:
                drawdown_periods.append(current_period)

            avg_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        else:
            avg_duration = 0

        return {
            "max_drawdown": abs(max_drawdown),
            "current_drawdown": abs(current_drawdown),
            "drawdown_duration": avg_duration,
        }

    def _create_minimal_metrics(
        self, returns: pd.Series, portfolio_values: List[float], initial_capital: float
    ):
        """Create minimal metrics using backend PerformanceCalculator"""

        try:
            # Use backend PerformanceCalculator for proper metrics calculation
            from src.backtesting.metrics import PerformanceCalculator

            calculator = PerformanceCalculator()
            portfolio_series = pd.Series(
                portfolio_values,
                index=pd.date_range(
                    start="2023-01-01", periods=len(portfolio_values), freq="D"
                ),
            )

            # Calculate using backend - no hardcoding
            metrics = calculator.calculate_comprehensive_metrics(
                returns=returns,
                portfolio_values=portfolio_series,
                trades=[],  # No trades data for minimal metrics
                initial_capital=initial_capital,
            )

            return metrics

        except Exception as e:
            print(f"Backend calculation failed, using fallback: {e}")

            # Fallback minimal metrics - still avoiding hardcoding where possible
            class MinimalMetrics:
                def __init__(self, returns, portfolio_values, initial_capital):
                    self.total_return = (
                        (portfolio_values[-1] / initial_capital - 1)
                        if portfolio_values
                        else 0
                    )
                    self.annualized_return = self.total_return  # Simplified
                    self.volatility = (
                        returns.std() if len(returns) > 1 else 0
                    )  # Daily volatility only
                    self.sharpe_ratio = 0.0  # Will be calculated by backend if needed
                    self.sortino_ratio = 0.0  # Will be calculated by backend if needed
                    self.calmar_ratio = 0.0  # Will be calculated by backend if needed
                    self.max_drawdown = abs(self._calculate_max_dd(portfolio_values))

            def _calculate_max_dd(self, values):
                if len(values) < 2:
                    return 0
                portfolio_series = pd.Series(values)
                rolling_max = portfolio_series.expanding().max()
                drawdown = (portfolio_series - rolling_max) / rolling_max
                return drawdown.min()

        return MinimalMetrics(returns, portfolio_values, initial_capital)

    def _create_summary(
        self, metrics, trades: List[Dict], win_rate: float
    ) -> Dict[str, str]:
        """Create human-readable summary"""
        return {
            "total_return": f"{getattr(metrics, 'total_return', 0) * 100:.2f}%",
            "sharpe_ratio": f"{getattr(metrics, 'sharpe_ratio', 0):.3f}",
            "max_drawdown": f"{getattr(metrics, 'max_drawdown', 0) * 100:.2f}%",
            "total_trades": str(len(trades)),
            "win_rate": f"{win_rate * 100:.1f}%",
        }


def get_available_symbols_from_cache() -> List[str]:
    """Get available symbols from feature cache"""
    try:
        import streamlit as st

        if "feature_cache" not in st.session_state:
            return []

        symbols = set()
        for feature_key, feature_data in st.session_state.feature_cache.items():
            # Try to extract symbol from feature key or data
            if isinstance(feature_data, dict):
                if "symbol" in feature_data:
                    symbols.add(feature_data["symbol"])
                elif "metadata" in feature_data:
                    metadata = feature_data["metadata"]
                    if isinstance(metadata, dict) and "symbol" in metadata:
                        symbols.add(metadata["symbol"])
                    elif isinstance(metadata, dict) and "ticker" in metadata:
                        symbols.add(metadata["ticker"])

        return list(symbols) if symbols else ["ASSET"]  # Default symbol

    except Exception:
        return ["ASSET"]  # Default symbol


def validate_backtest_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate backtest configuration

    Args:
        config: Backtest configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required fields
    required_fields = ["initial_capital", "commission_rate", "slippage_rate"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate ranges
    if config.get("initial_capital", 0) <= 0:
        errors.append("Initial capital must be positive")

    if not 0 <= config.get("commission_rate", 0) <= 1:
        errors.append("Commission rate must be between 0 and 1")

    if not 0 <= config.get("slippage_rate", 0) <= 1:
        errors.append("Slippage rate must be between 0 and 1")

    if config.get("leverage", 1) < 1:
        errors.append("Leverage must be at least 1")

    # Date validation
    start_date = config.get("start_date")
    end_date = config.get("end_date")

    if start_date and end_date:
        if start_date >= end_date:
            errors.append("Start date must be before end date")

    return len(errors) == 0, errors
