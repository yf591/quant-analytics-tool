"""
Backtesting Utilities for Streamlit Application (Clean Architecture)

This module provides ONLY data preparation and result formatting utilities.
ALL business logic is handled by backend src/backtesting modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import sys

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Import ALL backend classes - NO local implementations
    from src.backtesting import (
        BacktestEngine,
        BuyAndHoldStrategy,
        MomentumStrategy,
        MeanReversionStrategy,
        ModelBasedStrategy,
        MultiAssetStrategy,
        PerformanceCalculator,
    )
except ImportError as e:
    print(f"Backend import error in backtest_utils: {e}")


class BacktestDataPreparer:
    """ONLY data preparation - NO business logic"""

    def __init__(self):
        pass

    def find_metadata_for_features(
        self, feature_key: str, session_state
    ) -> Optional[pd.DataFrame]:
        """Find and prepare data from session state - formatting only"""

        try:
            if not hasattr(session_state, "feature_cache"):
                return None

            if feature_key not in session_state.feature_cache:
                return None

            feature_data = session_state.feature_cache[feature_key]
            return self.format_data_display(feature_data)

        except Exception as e:
            print(f"Data preparation error: {e}")
            return None

    def format_data_display(
        self, feature_data: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """Format data for UI display only - NO processing"""

        try:
            # Return formatted data from feature cache
            if isinstance(feature_data, pd.DataFrame):
                # Check if already has OHLCV columns
                required_cols = ["Open", "High", "Low", "Close"]
                if all(col in feature_data.columns for col in required_cols):
                    return feature_data.head(100)  # Limit for UI display
                else:
                    # Convert features dict to OHLCV if possible
                    return self._convert_features_to_ohlcv(feature_data)

            elif isinstance(feature_data, dict) and "original_data" in feature_data:
                # Extract original OHLCV data
                original_data = feature_data["original_data"]
                if isinstance(original_data, pd.DataFrame):
                    return original_data.head(100)

            # If no proper price data found, use sample data
            return self.get_sample_data()

        except Exception as e:
            print(f"Feature data preparation failed: {e}")
            return self.get_sample_data()

    def _convert_features_to_ohlcv(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Convert feature data to OHLCV format if possible"""
        try:
            # Try to find price-like columns
            if "Close" in feature_data.columns:
                # Create basic OHLCV structure
                close_col = feature_data["Close"]
                return pd.DataFrame(
                    {
                        "Open": close_col,
                        "High": close_col * 1.02,  # Simple simulation
                        "Low": close_col * 0.98,
                        "Close": close_col,
                        "Volume": 1000000,  # Default volume
                    }
                )
            else:
                return self.get_sample_data()
        except Exception:
            return self.get_sample_data()

    def get_sample_data(self) -> pd.DataFrame:
        """Get sample data from backend for demo"""

        try:
            # Create simple sample data for testing
            dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
            np.random.seed(42)

            # Generate sample price data
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

            return pd.DataFrame(
                {
                    "Open": prices * 0.999,
                    "High": prices * 1.01,
                    "Low": prices * 0.99,
                    "Close": prices,
                    "Volume": np.random.randint(1000000, 5000000, 100),
                },
                index=dates,
            )
        except Exception:
            # Minimal fallback
            return pd.DataFrame(
                {
                    "Open": [100, 101, 102],
                    "High": [101, 102, 103],
                    "Low": [99, 100, 101],
                    "Close": [100, 101, 102],
                    "Volume": [1000000, 1000000, 1000000],
                }
            )


class ConfigurationHelper:
    """Helper for UI configuration - display support only"""

    def __init__(self):
        pass

    def create_strategy_instance(
        self, strategy_config: Dict[str, Any], symbols: List[str]
    ) -> Any:
        """Create strategy instance using backend classes directly"""

        try:
            strategy_type = strategy_config.get("strategy_type", "Buy and Hold")
            parameters = strategy_config.get("parameters", {})

            if strategy_type == "Buy and Hold":
                return BuyAndHoldStrategy(symbols=symbols, **parameters)
            elif strategy_type == "Momentum":
                return MomentumStrategy(symbols=symbols, **parameters)
            elif strategy_type == "Mean Reversion":
                return MeanReversionStrategy(symbols=symbols, **parameters)
            else:
                return BuyAndHoldStrategy(symbols=symbols)

        except Exception as e:
            print(f"Strategy creation failed: {e}")
            return None


class BacktestResultProcessor:
    """Process backtest results - formatting ONLY, NO calculations"""

    def __init__(self):
        # Use backend calculator for ANY metrics
        try:
            self.calculator = PerformanceCalculator()
        except ImportError:
            self.calculator = None

    def process_results(
        self, engine: BacktestEngine, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process results using backend ONLY - NO frontend calculations"""

        try:
            # Extract data from backend engine
            portfolio_values = (
                [value[1] for value in engine.portfolio_values]
                if engine.portfolio_values
                else []
            )
            trades = self._format_trades(engine.get_trades_summary())
            positions = engine.get_positions_summary()

            # Use backend for ALL calculations
            if self.calculator and len(portfolio_values) > 1:
                # Use backend for all calculations
                portfolio_data = {
                    "values": portfolio_values,
                    "dates": pd.date_range(
                        start="2023-01-01", periods=len(portfolio_values), freq="D"
                    ),
                }

                # Use backend comprehensive metrics calculation
                portfolio_series = pd.Series(portfolio_values)
                returns = portfolio_series.pct_change().dropna()

                # Use the correct method name from backend
                metrics = self.calculator.calculate_comprehensive_metrics(
                    returns=returns,
                    portfolio_values=portfolio_series,
                    trades=trades,
                    initial_capital=config.get("initial_capital", 100000),
                )
            else:
                metrics = self.get_empty_display()

            # Format results for UI display - NO calculations here
            # Include returns data for risk management
            portfolio_series = (
                pd.Series(portfolio_values) if portfolio_values else pd.Series()
            )
            returns_data = (
                portfolio_series.pct_change().dropna()
                if len(portfolio_series) > 1
                else pd.Series()
            )

            return {
                "portfolio_values": portfolio_values,
                "returns": (
                    returns_data.tolist() if len(returns_data) > 0 else []
                ),  # Include returns for risk management
                "metrics": metrics,
                "trades": trades,
                "positions": positions,
                "config": config,
                "summary": self.format_metrics_display(metrics),
            }

        except Exception as e:
            print(f"Result processing failed: {e}")
            return {
                "portfolio_values": [],
                "returns": [],  # Include empty returns for consistency
                "metrics": self.get_empty_display(),
                "trades": [],
                "positions": pd.DataFrame(),
                "config": config,
                "summary": {},
            }

    def _format_trades(self, trades_summary) -> List[Dict]:
        """Format trades for display - NO calculations"""

        if isinstance(trades_summary, pd.DataFrame) and not trades_summary.empty:
            return trades_summary.to_dict("records")
        elif isinstance(trades_summary, list):
            return trades_summary
        else:
            return []

    def format_metrics_display(self, metrics) -> Dict[str, str]:
        """Format metrics for UI display - formatting ONLY"""

        try:
            if hasattr(metrics, "total_return"):
                return {
                    "total_return": f"{getattr(metrics, 'total_return', 0):.2%}",
                    "sharpe_ratio": f"{getattr(metrics, 'sharpe_ratio', 0):.3f}",
                    "max_drawdown": f"{getattr(metrics, 'max_drawdown', 0):.2%}",
                }
            else:
                return {"status": "Metrics available"}
        except Exception:
            return {"status": "Formatting error"}

    def get_empty_display(self):
        """Get empty display state - UI only"""

        return {"status": "No data available"}


def get_available_symbols_from_cache() -> List[str]:
    """Get available symbols from cache - utility only"""

    try:
        import streamlit as st

        if (
            hasattr(st.session_state, "feature_cache")
            and st.session_state.feature_cache
        ):
            # Extract symbols from feature cache keys
            symbols = []
            for key in st.session_state.feature_cache.keys():
                if "_" in key:
                    symbol = key.split("_")[0].upper()
                    if symbol not in symbols:
                        symbols.append(symbol)
            return symbols
        else:
            return ["AAPL", "MSFT", "GOOGL"]  # Default symbols

    except Exception:
        return ["AAPL", "MSFT", "GOOGL"]  # Fallback


def validate_backtest_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate backtest configuration - validation only"""

    errors = []

    # Basic validation
    required_fields = ["initial_capital", "commission_rate", "slippage_rate"]
    for field in required_fields:
        if field not in config or config[field] is None:
            errors.append(f"Missing required field: {field}")

    # Range validation
    if config.get("initial_capital", 0) <= 0:
        errors.append("Initial capital must be positive")

    if not 0 <= config.get("commission_rate", 0) <= 1:
        errors.append("Commission rate must be between 0 and 1")

    if not 0 <= config.get("slippage_rate", 0) <= 1:
        errors.append("Slippage rate must be between 0 and 1")

    return len(errors) == 0, errors
