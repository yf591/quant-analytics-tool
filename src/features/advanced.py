"""
Advanced Feature Engineering Module

This module implements advanced quantitative finance techniques based on
Advances in Financial Machine Learning (AFML) methodology.

Features:
- Fractal Dimension calculation
- Hurst Exponent estimation
- Information-driven Bars (Tick, Volume, Dollar bars)
- Triple Barrier Method
- Fractional Differentiation
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class AdvancedFeatureResults:
    """Container for advanced feature engineering results."""

    fractal_dimension: Optional[pd.Series] = None
    hurst_exponent: Optional[pd.Series] = None
    information_bars: Optional[pd.DataFrame] = None
    triple_barrier_labels: Optional[pd.DataFrame] = None
    fractional_diff: Optional[pd.DataFrame] = None


class AdvancedFeatures:
    """
    Advanced Feature Engineering class implementing AFML-based techniques.

    This class provides sophisticated financial data analysis methods including
    fractal analysis, information theory applications, and advanced labeling
    techniques for machine learning in quantitative finance.
    """

    def __init__(self):
        """Initialize the AdvancedFeatures class."""
        pass

    def calculate_fractal_dimension(
        self, series: pd.Series, window: int = 100, method: str = "higuchi"
    ) -> pd.Series:
        """
        Calculate fractal dimension using Higuchi's method.

        Fractal dimension measures the complexity and roughness of a time series.
        Higher values indicate more complex, irregular patterns.

        Args:
            series: Price series to analyze
            window: Rolling window size for calculation
            method: Method to use ('higuchi' or 'box_counting')

        Returns:
            Series of fractal dimension values
        """
        if method not in ["higuchi", "box_counting"]:
            raise ValueError("Method must be 'higuchi' or 'box_counting'")

        def higuchi_fd(data: np.ndarray, k_max: int = 10) -> float:
            """Calculate Higuchi fractal dimension."""
            N = len(data)
            if N < k_max:
                return np.nan

            L = []
            x = []

            for k in range(1, k_max + 1):
                Lk = []
                for m in range(k):
                    Lmk = 0
                    max_i = int((N - 1 - m) / k)

                    for i in range(1, max_i + 1):
                        Lmk += abs(data[m + i * k] - data[m + (i - 1) * k])

                    if max_i > 0:
                        Lmk = Lmk * (N - 1) / (max_i * k * k)
                        Lk.append(Lmk)

                if Lk:
                    L.append(np.mean(Lk))
                    x.append(np.log(1 / k))

            if len(L) < 2:
                return np.nan

            # Linear regression to find slope
            try:
                slope, _, _, _, _ = stats.linregress(x, np.log(L))
                return slope
            except:
                return np.nan

        def box_counting_fd(data: np.ndarray) -> float:
            """Calculate Box-counting fractal dimension."""
            if len(data) < 10:
                return np.nan

            # Normalize data to [0, 1]
            data_min, data_max = np.min(data), np.max(data)
            if data_max == data_min:
                return np.nan
            data_norm = (data - data_min) / (data_max - data_min)

            # Create different box sizes
            scales = np.logspace(-2, -0.1, num=10, base=10)  # Adjusted scale range
            counts = []

            for scale in scales:
                # Number of boxes needed to cover the curve
                bins = max(1, int(1 / scale))  # Ensure bins is at least 1
                try:
                    H, edges = np.histogramdd(data_norm.reshape(-1, 1), bins=bins)
                    counts.append(np.sum(H > 0))
                except:
                    continue

            if len(counts) < 3:
                return np.nan

            # Linear regression in log-log space
            try:
                valid_scales = scales[: len(counts)]
                coeffs = np.polyfit(np.log(1 / valid_scales), np.log(counts), 1)
                return coeffs[0]
            except:
                return np.nan

        result = pd.Series(index=series.index, dtype=float)

        for i in range(window, len(series)):
            data_window = series.iloc[i - window : i].values

            if method == "higuchi":
                fd = higuchi_fd(data_window)
            else:  # box_counting
                fd = box_counting_fd(data_window)

            result.iloc[i] = fd

        return result

    def calculate_hurst_exponent(
        self, series: pd.Series, window: int = 100, method: str = "rs"
    ) -> pd.Series:
        """
        Calculate Hurst exponent to measure long-term memory in time series.

        Hurst exponent values:
        - H = 0.5: Random walk (no correlation)
        - H > 0.5: Persistent (trend-following)
        - H < 0.5: Anti-persistent (mean-reverting)

        Args:
            series: Price series to analyze
            window: Rolling window size for calculation
            method: Method to use ('rs' for R/S analysis, 'dfa' for Detrended Fluctuation Analysis)

        Returns:
            Series of Hurst exponent values
        """
        if method not in ["rs", "dfa"]:
            raise ValueError("Method must be 'rs' or 'dfa'")

        def rs_hurst(data: np.ndarray, lags: Optional[List[int]] = None) -> float:
            """Calculate Hurst exponent using R/S analysis."""
            if len(data) < 20:
                return np.nan

            if lags is None:
                # Create more conservative lag selection
                max_lag = min(len(data) // 4, 50)  # Conservative upper bound
                lags = [lag for lag in range(8, max_lag, 2) if lag <= len(data) // 3]

            if len(lags) < 3:
                return np.nan

            rs_values = []

            for lag in lags:
                if lag >= len(data) // 2:  # More conservative check
                    continue

                # Divide series into chunks
                n_chunks = len(data) // lag
                if n_chunks < 2:
                    continue

                chunk_rs = []

                for i in range(n_chunks):
                    chunk = data[i * lag : (i + 1) * lag]

                    if len(chunk) < lag:  # Ensure full chunk
                        continue

                    # Calculate mean
                    mean_chunk = np.mean(chunk)

                    # Calculate deviations from mean
                    deviations = chunk - mean_chunk

                    # Calculate cumulative deviations
                    cum_deviations = np.cumsum(deviations)

                    # Calculate range
                    R = np.max(cum_deviations) - np.min(cum_deviations)

                    # Calculate standard deviation
                    S = np.std(chunk, ddof=1)

                    # Avoid division by zero and very small S
                    if S > 1e-8 and R > 0:
                        chunk_rs.append(R / S)

                if len(chunk_rs) >= 2:  # Need at least 2 chunks
                    rs_values.append(np.mean(chunk_rs))

            if len(rs_values) < 3:
                return np.nan

            # Linear regression in log-log space
            try:
                valid_lags = lags[: len(rs_values)]
                log_lags = np.log(valid_lags)
                log_rs = np.log(rs_values)

                # Check for valid values
                valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
                if np.sum(valid_mask) < 3:
                    return np.nan

                slope, _, _, _, _ = stats.linregress(
                    log_lags[valid_mask], log_rs[valid_mask]
                )
                return slope
            except:
                return np.nan

        def dfa_hurst(data: np.ndarray) -> float:
            """Calculate Hurst exponent using Detrended Fluctuation Analysis."""
            N = len(data)

            # Integrate the data
            y = np.cumsum(data - np.mean(data))

            # Different window sizes
            scales = np.unique(np.logspace(0.5, np.log10(N // 4), 20).astype(int))
            scales = scales[scales >= 4]

            if len(scales) < 4:
                return np.nan

            fluctuations = []

            for scale in scales:
                # Number of segments
                n_segments = N // scale

                segment_fluctuations = []

                for v in range(n_segments):
                    start_idx = v * scale
                    end_idx = start_idx + scale
                    segment = y[start_idx:end_idx]

                    # Linear detrend
                    x = np.arange(scale)
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)

                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
                    segment_fluctuations.append(fluctuation)

                if segment_fluctuations:
                    fluctuations.append(np.mean(segment_fluctuations))

            if len(fluctuations) < 2:
                return np.nan

            # Linear regression in log-log space
            try:
                log_scales = np.log(scales[: len(fluctuations)])
                log_fluctuations = np.log(fluctuations)
                slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
                return slope
            except:
                return np.nan

        result = pd.Series(index=series.index, dtype=float)

        for i in range(window, len(series)):
            data_window = series.iloc[i - window : i].values

            if method == "rs":
                hurst = rs_hurst(data_window)
            else:  # dfa
                hurst = dfa_hurst(data_window)

            result.iloc[i] = hurst

        return result

    def create_information_bars(
        self,
        data: pd.DataFrame,
        bar_type: str = "volume",
        threshold: Optional[float] = None,
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """
        Create information-driven bars instead of time-based bars.

        Information bars sample data based on information content rather than time,
        leading to more statistically robust features for ML models.

        Args:
            data: DataFrame with OHLCV data
            bar_type: Type of bars ('tick', 'volume', 'dollar')
            threshold: Threshold for bar creation (auto-calculated if None)
            price_col: Name of price column to use
            volume_col: Name of volume column to use

        Returns:
            DataFrame with information-driven bars
        """
        if bar_type not in ["tick", "volume", "dollar"]:
            raise ValueError("bar_type must be 'tick', 'volume', or 'dollar'")

        if threshold is None:
            # Auto-calculate threshold based on data characteristics
            if bar_type == "tick":
                threshold = len(data) // 100  # ~100 bars
            elif bar_type == "volume":
                threshold = data[volume_col].mean()
            else:  # dollar
                threshold = (data[price_col] * data[volume_col]).mean()

        bars = []
        current_bar = {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": 0,
            "vwap": 0,
            "count": 0,
            "start_time": None,
            "end_time": None,
        }

        cumulative_measure = 0
        total_dollar_volume = 0

        for idx, row in data.iterrows():
            price = row[price_col]
            volume = row[volume_col] if volume_col in row else 1

            # Initialize bar if first tick
            if current_bar["open"] is None:
                current_bar["open"] = price
                current_bar["high"] = price
                current_bar["low"] = price
                current_bar["start_time"] = idx

            # Update OHLC
            current_bar["high"] = max(current_bar["high"], price)
            current_bar["low"] = min(current_bar["low"], price)
            current_bar["close"] = price
            current_bar["end_time"] = idx

            # Update volume and count
            current_bar["volume"] += volume
            current_bar["count"] += 1

            # Update VWAP calculation
            dollar_volume = price * volume
            total_dollar_volume += dollar_volume
            current_bar["vwap"] = (
                total_dollar_volume / current_bar["volume"]
                if current_bar["volume"] > 0
                else price
            )

            # Update cumulative measure
            if bar_type == "tick":
                cumulative_measure += 1
            elif bar_type == "volume":
                cumulative_measure += volume
            else:  # dollar
                cumulative_measure += dollar_volume

            # Check if threshold is reached
            if cumulative_measure >= threshold:
                bars.append(current_bar.copy())

                # Reset for next bar
                current_bar = {
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": 0,
                    "vwap": 0,
                    "count": 0,
                    "start_time": None,
                    "end_time": None,
                }
                cumulative_measure = 0
                total_dollar_volume = 0

        # Add final bar if it has data
        if current_bar["open"] is not None:
            bars.append(current_bar)

        # Convert to DataFrame
        result_df = pd.DataFrame(bars)
        if not result_df.empty:
            result_df.set_index("end_time", inplace=True)

        return result_df

    def triple_barrier_method(
        self,
        prices: pd.Series,
        events: pd.Series,
        pt_sl: List[float] = [1, 1],
        molecule: Optional[List] = None,
        vertical_barrier: Optional[pd.Series] = None,
        side: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Implement Triple Barrier Method for meta-labeling.

        The triple barrier method creates labels for supervised learning by
        defining profit-taking and stop-loss barriers, plus a vertical time barrier.

        Args:
            prices: Price series (usually close prices)
            events: Series of event timestamps to label
            pt_sl: [profit_taking_factor, stop_loss_factor] as multiples of volatility
            molecule: Subset of events to process (for parallel processing)
            vertical_barrier: Series of vertical barrier timestamps
            side: Series indicating bet direction (1 for long, -1 for short)

        Returns:
            DataFrame with barrier labels and metadata
        """
        # Get daily volatility estimate
        daily_vol = self._get_daily_vol(prices)

        # Align events with price data
        events = events.dropna().sort_index()

        if molecule is not None:
            events = events.loc[molecule]

        # Initialize barriers
        barriers = pd.DataFrame(index=events.index)
        barriers["t1"] = (
            vertical_barrier.reindex(events.index)
            if vertical_barrier is not None
            else pd.NaT
        )
        barriers["trgt"] = daily_vol.reindex(events.index)
        barriers["side"] = side.reindex(events.index) if side is not None else 1

        # Calculate profit-taking and stop-loss barriers
        barriers["pt"] = pt_sl[0] * barriers["trgt"]
        barriers["sl"] = -pt_sl[1] * barriers["trgt"]

        # Apply barrier method
        labels = self._apply_triple_barrier(prices, barriers)

        return labels

    def _get_daily_vol(self, prices: pd.Series, span: int = 100) -> pd.Series:
        """Calculate daily volatility using exponential weighted moving average."""
        # Calculate returns
        returns = prices.pct_change().dropna()

        # Calculate EWMA volatility
        vol = returns.ewm(span=span).std()

        return vol

    def _apply_triple_barrier(
        self, prices: pd.Series, barriers: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the triple barrier method to generate labels."""
        labels = pd.DataFrame(index=barriers.index)
        labels["barrier"] = 0  # 0: vertical, 1: profit-taking, -1: stop-loss
        labels["ret"] = 0.0
        labels["t1"] = pd.NaT

        for event_time, barrier in barriers.iterrows():
            # Get future prices starting from event
            future_prices = prices.loc[event_time:]

            if len(future_prices) < 2:
                continue

            # Starting price
            p0 = future_prices.iloc[0]

            # Calculate returns
            future_returns = (future_prices / p0 - 1) * barrier["side"]

            # Find first barrier touch
            pt_touch = future_returns[future_returns >= barrier["pt"]]
            sl_touch = future_returns[future_returns <= barrier["sl"]]

            # Determine which barrier was touched first
            first_touch_time = None
            barrier_type = 0  # Default to vertical barrier

            if not pt_touch.empty and not sl_touch.empty:
                if pt_touch.index[0] <= sl_touch.index[0]:
                    first_touch_time = pt_touch.index[0]
                    barrier_type = 1
                else:
                    first_touch_time = sl_touch.index[0]
                    barrier_type = -1
            elif not pt_touch.empty:
                first_touch_time = pt_touch.index[0]
                barrier_type = 1
            elif not sl_touch.empty:
                first_touch_time = sl_touch.index[0]
                barrier_type = -1

            # Check vertical barrier
            if pd.notna(barrier["t1"]):
                if first_touch_time is None or first_touch_time > barrier["t1"]:
                    first_touch_time = barrier["t1"]
                    barrier_type = 0
            else:
                # If no vertical barrier, use end of series
                if first_touch_time is None:
                    first_touch_time = future_returns.index[-1]
                    barrier_type = 0

            # Record results
            if first_touch_time is not None:
                labels.loc[event_time, "barrier"] = barrier_type
                labels.loc[event_time, "ret"] = future_returns.loc[first_touch_time]
                labels.loc[event_time, "t1"] = first_touch_time

        return labels

    def fractional_differentiation(
        self, series: pd.Series, d: float, threshold: float = 0.01
    ) -> pd.Series:
        """
        Apply fractional differentiation to achieve stationarity while preserving memory.

        Fractional differentiation allows for making a series stationary while
        retaining some memory, unlike integer differentiation which removes all memory.

        Args:
            series: Time series to differentiate
            d: Differentiation order (typically between 0 and 1)
            threshold: Weight threshold for computational efficiency

        Returns:
            Fractionally differentiated series
        """
        # Calculate weights
        weights = self._get_weights_ffd(d, threshold)
        width = len(weights) - 1

        # Apply fractional differentiation
        result = pd.Series(index=series.index, dtype=float)

        for i in range(width, len(series)):
            result.iloc[i] = np.dot(weights.T, series.iloc[i - width : i + 1].values)

        return result

    def _get_weights_ffd(self, d: float, threshold: float) -> np.ndarray:
        """Calculate weights for fixed-width fractional differentiation."""
        w = [1.0]
        k = 1

        while True:
            w_new = -w[-1] / k * (d - k + 1)
            if abs(w_new) < threshold:
                break
            w.append(w_new)
            k += 1

            # Safety check to prevent infinite loops
            if k > 1000:
                break

        return np.array(w[::-1]).reshape(-1, 1)

    def calculate_all_features(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume",
        window: int = 100,
    ) -> AdvancedFeatureResults:
        """
        Calculate all advanced features for the given data.

        Args:
            data: DataFrame with OHLCV data
            price_col: Name of price column
            volume_col: Name of volume column
            window: Window size for rolling calculations

        Returns:
            AdvancedFeatureResults object containing all calculated features
        """
        results = AdvancedFeatureResults()

        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")

        prices = data[price_col]

        try:
            # Calculate fractal dimension
            results.fractal_dimension = self.calculate_fractal_dimension(prices, window)

            # Calculate Hurst exponent
            results.hurst_exponent = self.calculate_hurst_exponent(prices, window)

            # Create information-driven bars
            if volume_col in data.columns:
                results.information_bars = self.create_information_bars(
                    data, "volume", price_col=price_col, volume_col=volume_col
                )

            # Apply fractional differentiation
            results.fractional_diff = self.fractional_differentiation(prices, d=0.4)

            # Generate events for triple barrier method (using volatility breakouts)
            returns = prices.pct_change().dropna()
            vol = returns.rolling(window=20).std()
            events = vol[vol > vol.quantile(0.95)].index

            if len(events) > 0:
                results.triple_barrier_labels = self.triple_barrier_method(
                    prices, pd.Series(index=events, data=events)
                )

        except Exception as e:
            warnings.warn(f"Error calculating advanced features: {str(e)}")

        return results
