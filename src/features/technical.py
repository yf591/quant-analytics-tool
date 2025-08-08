"""
Technical Indicators Module

This module implements comprehensive technical analysis indicators
following industry standards and AFML methodologies.

Classes:
    TechnicalIndicators: Main class for technical indicator calculations

Functions:
    calculate_sma: Simple Moving Average
    calculate_ema: Exponential Moving Average
    calculate_rsi: Relative Strength Index
    calculate_macd: Moving Average Convergence Divergence
    calculate_bollinger_bands: Bollinger Bands
    calculate_atr: Average True Range
    calculate_stochastic: Stochastic Oscillator
    calculate_williams_r: Williams %R
    calculate_cci: Commodity Channel Index
    calculate_momentum: Price Momentum
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicatorResults:
    """Data class for storing technical indicator results"""

    indicator_name: str
    values: Union[pd.Series, pd.DataFrame]
    parameters: Dict
    timestamp: pd.Timestamp

    def to_dict(self) -> Dict:
        """Convert results to dictionary format"""
        return {
            "indicator_name": self.indicator_name,
            "values": self.values,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
        }


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator

    This class implements various technical analysis indicators commonly
    used in quantitative finance and algorithmic trading.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Technical Indicators calculator

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def calculate_all_indicators(
        self, data: pd.DataFrame, indicators: Optional[List[str]] = None
    ) -> Dict[str, TechnicalIndicatorResults]:
        """
        Calculate multiple technical indicators

        Args:
            data: DataFrame with OHLCV data
            indicators: List of indicators to calculate (default: all)

        Returns:
            Dictionary of indicator results
        """
        if indicators is None:
            indicators = [
                "sma",
                "ema",
                "rsi",
                "macd",
                "bollinger_bands",
                "atr",
                "stochastic",
                "williams_r",
                "cci",
                "momentum",
            ]

        results = {}

        try:
            for indicator in indicators:
                self.logger.debug(f"Calculating {indicator}")

                if indicator == "sma":
                    results["sma_20"] = self._create_result(
                        "SMA_20", calculate_sma(data["close"], 20), {"window": 20}
                    )
                    results["sma_50"] = self._create_result(
                        "SMA_50", calculate_sma(data["close"], 50), {"window": 50}
                    )

                elif indicator == "ema":
                    results["ema_12"] = self._create_result(
                        "EMA_12", calculate_ema(data["close"], 12), {"window": 12}
                    )
                    results["ema_26"] = self._create_result(
                        "EMA_26", calculate_ema(data["close"], 26), {"window": 26}
                    )

                elif indicator == "rsi":
                    results["rsi"] = self._create_result(
                        "RSI", calculate_rsi(data["close"], 14), {"window": 14}
                    )

                elif indicator == "macd":
                    macd_result = calculate_macd(data["close"], 12, 26, 9)
                    results["macd"] = self._create_result(
                        "MACD", macd_result, {"fast": 12, "slow": 26, "signal": 9}
                    )

                elif indicator == "bollinger_bands":
                    bb_result = calculate_bollinger_bands(data["close"], 20, 2.0)
                    results["bollinger_bands"] = self._create_result(
                        "Bollinger_Bands", bb_result, {"window": 20, "num_std": 2.0}
                    )

                elif indicator == "atr":
                    results["atr"] = self._create_result(
                        "ATR",
                        calculate_atr(data["high"], data["low"], data["close"], 14),
                        {"window": 14},
                    )

                elif indicator == "stochastic":
                    stoch_result = calculate_stochastic(
                        data["high"], data["low"], data["close"], 14, 3
                    )
                    results["stochastic"] = self._create_result(
                        "Stochastic", stoch_result, {"k_window": 14, "d_window": 3}
                    )

                elif indicator == "williams_r":
                    results["williams_r"] = self._create_result(
                        "Williams_R",
                        calculate_williams_r(
                            data["high"], data["low"], data["close"], 14
                        ),
                        {"window": 14},
                    )

                elif indicator == "cci":
                    results["cci"] = self._create_result(
                        "CCI",
                        calculate_cci(data["high"], data["low"], data["close"], 20),
                        {"window": 20},
                    )

                elif indicator == "momentum":
                    results["momentum"] = self._create_result(
                        "Momentum",
                        calculate_momentum(data["close"], 10),
                        {"window": 10},
                    )

            self.logger.info(f"Successfully calculated {len(results)} indicators")
            return results

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def _create_result(
        self, name: str, values: Union[pd.Series, pd.DataFrame], params: Dict
    ) -> TechnicalIndicatorResults:
        """Create standardized result object"""
        return TechnicalIndicatorResults(
            indicator_name=name,
            values=values,
            parameters=params,
            timestamp=pd.Timestamp.now(),
        )


def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA)

    Args:
        data: Price series
        window: Number of periods

    Returns:
        SMA values as pandas Series
    """
    if len(data) < window:
        raise ValueError(f"Insufficient data points. Need at least {window} points")

    return data.rolling(window=window, min_periods=window).mean()


def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)

    Args:
        data: Price series
        window: Number of periods

    Returns:
        EMA values as pandas Series
    """
    if len(data) < window:
        raise ValueError(f"Insufficient data points. Need at least {window} points")

    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)

    Args:
        data: Price series
        window: Number of periods (default: 14)

    Returns:
        RSI values as pandas Series (0-100 scale)
    """
    if len(data) < window + 1:
        raise ValueError(f"Insufficient data points. Need at least {window + 1} points")

    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Use exponential moving average for subsequent calculations
    for i in range(window, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Args:
        data: Price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    if len(data) < slow_period:
        raise ValueError(
            f"Insufficient data points. Need at least {slow_period} points"
        )

    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line.dropna(), signal_period)
    histogram = macd_line - signal_line

    result = pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram},
        index=data.index,
    )

    return result


def calculate_bollinger_bands(
    data: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands

    Args:
        data: Price series
        window: SMA period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        DataFrame with Upper, Middle (SMA), and Lower bands
    """
    if len(data) < window:
        raise ValueError(f"Insufficient data points. Need at least {window} points")

    sma = calculate_sma(data, window)
    std = data.rolling(window=window, min_periods=window).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    result = pd.DataFrame(
        {"Upper": upper_band, "Middle": sma, "Lower": lower_band}, index=data.index
    )

    return result


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Number of periods (default: 14)

    Returns:
        ATR values as pandas Series
    """
    if len(high) < window + 1:
        raise ValueError(f"Insufficient data points. Need at least {window + 1} points")

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()

    return atr


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_window: %K period (default: 14)
        d_window: %D period (default: 3)

    Returns:
        DataFrame with %K and %D values
    """
    if len(high) < k_window:
        raise ValueError(f"Insufficient data points. Need at least {k_window} points")

    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()

    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window, min_periods=d_window).mean()

    result = pd.DataFrame({"%K": k_percent, "%D": d_percent}, index=close.index)

    return result


def calculate_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculate Williams %R

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Number of periods (default: 14)

    Returns:
        Williams %R values as pandas Series
    """
    if len(high) < window:
        raise ValueError(f"Insufficient data points. Need at least {window} points")

    highest_high = high.rolling(window=window, min_periods=window).max()
    lowest_low = low.rolling(window=window, min_periods=window).min()

    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

    return williams_r


def calculate_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Number of periods (default: 20)

    Returns:
        CCI values as pandas Series
    """
    if len(high) < window:
        raise ValueError(f"Insufficient data points. Need at least {window} points")

    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window, min_periods=window).mean()
    mean_deviation = (
        (typical_price - sma_tp).abs().rolling(window=window, min_periods=window).mean()
    )

    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

    return cci


def calculate_momentum(data: pd.Series, window: int = 10) -> pd.Series:
    """
    Calculate Price Momentum

    Args:
        data: Price series
        window: Number of periods (default: 10)

    Returns:
        Momentum values as pandas Series
    """
    if len(data) < window + 1:
        raise ValueError(f"Insufficient data points. Need at least {window + 1} points")

    momentum = data - data.shift(window)

    return momentum
