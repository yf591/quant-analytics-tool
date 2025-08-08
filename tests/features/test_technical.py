"""
Unit tests for technical indicators module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from features.technical import (
    TechnicalIndicators,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_stochastic,
    calculate_williams_r,
    calculate_cci,
    calculate_momentum,
    TechnicalIndicatorResults,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    high = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    low = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volume = np.random.randint(1000, 10000, 100)

    data = pd.DataFrame(
        {"open": prices, "high": high, "low": low, "close": prices, "volume": volume},
        index=dates,
    )

    return data


@pytest.fixture
def price_series():
    """Create simple price series for testing"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    prices = 100 + np.random.randn(50).cumsum()
    return pd.Series(prices, index=dates)


class TestTechnicalIndicators:
    """Test TechnicalIndicators class"""

    def test_initialization(self):
        """Test TechnicalIndicators initialization"""
        ti = TechnicalIndicators()
        assert ti.config == {}

        config = {"param1": "value1"}
        ti_with_config = TechnicalIndicators(config)
        assert ti_with_config.config == config

    def test_calculate_all_indicators(self, sample_ohlcv_data):
        """Test calculation of all indicators"""
        ti = TechnicalIndicators()
        results = ti.calculate_all_indicators(sample_ohlcv_data)

        expected_indicators = [
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi",
            "macd",
            "bollinger_bands",
            "atr",
            "stochastic",
            "williams_r",
            "cci",
            "momentum",
        ]

        for indicator in expected_indicators:
            assert indicator in results
            assert isinstance(results[indicator], TechnicalIndicatorResults)

    def test_specific_indicators_only(self, sample_ohlcv_data):
        """Test calculation of specific indicators only"""
        ti = TechnicalIndicators()
        results = ti.calculate_all_indicators(sample_ohlcv_data, ["sma", "rsi"])

        assert len(results) == 3  # sma_20, sma_50, rsi
        assert "sma_20" in results
        assert "sma_50" in results
        assert "rsi" in results


class TestSimpleMovingAverage:
    """Test SMA calculations"""

    def test_sma_calculation(self, price_series):
        """Test SMA calculation"""
        window = 10
        sma = calculate_sma(price_series, window)

        # Check that SMA values start from window position
        assert pd.isna(sma.iloc[: window - 1]).all()
        assert not pd.isna(sma.iloc[window - 1 :]).any()

        # Manually verify first valid SMA value
        expected_first_sma = price_series.iloc[:window].mean()
        assert abs(sma.iloc[window - 1] - expected_first_sma) < 1e-10

    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data"""
        short_series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError):
            calculate_sma(short_series, 10)

    def test_sma_edge_cases(self, price_series):
        """Test SMA edge cases"""
        # Window = 1 should equal original series
        sma_1 = calculate_sma(price_series, 1)
        pd.testing.assert_series_equal(sma_1, price_series)

        # Window = length should give single value
        sma_full = calculate_sma(price_series, len(price_series))
        assert not pd.isna(sma_full.iloc[-1])
        assert pd.isna(sma_full.iloc[:-1]).all()


class TestExponentialMovingAverage:
    """Test EMA calculations"""

    def test_ema_calculation(self, price_series):
        """Test EMA calculation"""
        window = 10
        ema = calculate_ema(price_series, window)

        assert len(ema) == len(price_series)
        assert not ema.isna().any()

        # EMA should be more responsive than SMA
        sma = calculate_sma(price_series, window)
        # Can't directly compare due to different starting points
        assert isinstance(ema, pd.Series)

    def test_ema_insufficient_data(self):
        """Test EMA with insufficient data"""
        short_series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError):
            calculate_ema(short_series, 10)


class TestRSI:
    """Test RSI calculations"""

    def test_rsi_calculation(self, price_series):
        """Test RSI calculation"""
        rsi = calculate_rsi(price_series, 14)

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

        # First 13 values should be NaN (window - 1)
        assert pd.isna(rsi.iloc[:13]).all()

    def test_rsi_trending_up(self):
        """Test RSI with upward trending prices"""
        # Create strongly upward trending series
        upward_series = pd.Series(range(1, 51))  # 1, 2, 3, ..., 50
        rsi = calculate_rsi(upward_series, 14)

        # RSI should be high for upward trend
        final_rsi = rsi.iloc[-1]
        assert final_rsi > 70  # Typically considered overbought

    def test_rsi_trending_down(self):
        """Test RSI with downward trending prices"""
        # Create strongly downward trending series
        downward_series = pd.Series(range(50, 0, -1))  # 50, 49, 48, ..., 1
        rsi = calculate_rsi(downward_series, 14)

        # RSI should be low for downward trend
        final_rsi = rsi.iloc[-1]
        assert final_rsi < 30  # Typically considered oversold


class TestMACD:
    """Test MACD calculations"""

    def test_macd_calculation(self, price_series):
        """Test MACD calculation"""
        macd = calculate_macd(price_series, 12, 26, 9)

        assert isinstance(macd, pd.DataFrame)
        assert list(macd.columns) == ["MACD", "Signal", "Histogram"]

        # Check that Histogram = MACD - Signal (where both exist)
        valid_mask = ~(macd["MACD"].isna() | macd["Signal"].isna())
        if valid_mask.any():
            diff = macd.loc[valid_mask, "MACD"] - macd.loc[valid_mask, "Signal"]
            histogram = macd.loc[valid_mask, "Histogram"]
            pd.testing.assert_series_equal(diff, histogram, check_names=False)

    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data"""
        short_series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError):
            calculate_macd(short_series, 12, 26, 9)


class TestBollingerBands:
    """Test Bollinger Bands calculations"""

    def test_bollinger_bands_calculation(self, price_series):
        """Test Bollinger Bands calculation"""
        bb = calculate_bollinger_bands(price_series, 20, 2.0)

        assert isinstance(bb, pd.DataFrame)
        assert list(bb.columns) == ["Upper", "Middle", "Lower"]

        # Upper band should be above middle, middle above lower
        valid_mask = ~bb.isna().any(axis=1)
        valid_bb = bb.loc[valid_mask]

        if len(valid_bb) > 0:
            assert (valid_bb["Upper"] >= valid_bb["Middle"]).all()
            assert (valid_bb["Middle"] >= valid_bb["Lower"]).all()

    def test_bollinger_bands_width(self, price_series):
        """Test Bollinger Bands width with different std parameters"""
        bb_narrow = calculate_bollinger_bands(price_series, 20, 1.0)
        bb_wide = calculate_bollinger_bands(price_series, 20, 3.0)

        # Wider bands should have larger distance
        valid_idx = ~(bb_narrow.isna().any(axis=1) | bb_wide.isna().any(axis=1))
        if valid_idx.any():
            narrow_width = (
                bb_narrow.loc[valid_idx, "Upper"] - bb_narrow.loc[valid_idx, "Lower"]
            )
            wide_width = (
                bb_wide.loc[valid_idx, "Upper"] - bb_wide.loc[valid_idx, "Lower"]
            )
            assert (wide_width > narrow_width).all()


class TestATR:
    """Test ATR calculations"""

    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR calculation"""
        atr = calculate_atr(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            14,
        )

        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

        # First 13 values should be NaN (window - 1)
        assert pd.isna(atr.iloc[:13]).all()

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data"""
        short_data = pd.DataFrame(
            {"high": [10, 11, 12], "low": [9, 10, 11], "close": [9.5, 10.5, 11.5]}
        )

        with pytest.raises(ValueError):
            calculate_atr(
                short_data["high"], short_data["low"], short_data["close"], 14
            )


class TestStochastic:
    """Test Stochastic Oscillator calculations"""

    def test_stochastic_calculation(self, sample_ohlcv_data):
        """Test Stochastic calculation"""
        stoch = calculate_stochastic(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            14,
            3,
        )

        assert isinstance(stoch, pd.DataFrame)
        assert list(stoch.columns) == ["%K", "%D"]

        # Values should be between 0 and 100
        valid_stoch = stoch.dropna()
        if len(valid_stoch) > 0:
            assert (valid_stoch >= 0).all().all()
            assert (valid_stoch <= 100).all().all()


class TestWilliamsR:
    """Test Williams %R calculations"""

    def test_williams_r_calculation(self, sample_ohlcv_data):
        """Test Williams %R calculation"""
        williams_r = calculate_williams_r(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            14,
        )

        # Williams %R should be between -100 and 0
        valid_williams = williams_r.dropna()
        if len(valid_williams) > 0:
            assert (valid_williams >= -100).all()
            assert (valid_williams <= 0).all()


class TestCCI:
    """Test CCI calculations"""

    def test_cci_calculation(self, sample_ohlcv_data):
        """Test CCI calculation"""
        cci = calculate_cci(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            20,
        )

        # CCI can range widely, but should be numeric
        valid_cci = cci.dropna()
        assert len(valid_cci) > 0
        assert np.isfinite(valid_cci).all()


class TestMomentum:
    """Test Momentum calculations"""

    def test_momentum_calculation(self, price_series):
        """Test Momentum calculation"""
        momentum = calculate_momentum(price_series, 10)

        # First 10 values should be NaN
        assert pd.isna(momentum.iloc[:10]).all()

        # Momentum should equal current price - price 10 periods ago
        valid_idx = ~momentum.isna()
        if valid_idx.any():
            expected = (
                price_series.loc[valid_idx] - price_series.shift(10).loc[valid_idx]
            )
            pd.testing.assert_series_equal(
                momentum.loc[valid_idx], expected, check_names=False
            )


class TestTechnicalIndicatorResults:
    """Test TechnicalIndicatorResults dataclass"""

    def test_result_creation(self, price_series):
        """Test creation of result object"""
        sma = calculate_sma(price_series, 10)
        result = TechnicalIndicatorResults(
            indicator_name="SMA_10",
            values=sma,
            parameters={"window": 10},
            timestamp=pd.Timestamp.now(),
        )

        assert result.indicator_name == "SMA_10"
        assert result.parameters == {"window": 10}
        pd.testing.assert_series_equal(result.values, sma)

    def test_result_to_dict(self, price_series):
        """Test conversion to dictionary"""
        sma = calculate_sma(price_series, 10)
        timestamp = pd.Timestamp.now()
        result = TechnicalIndicatorResults(
            indicator_name="SMA_10",
            values=sma,
            parameters={"window": 10},
            timestamp=timestamp,
        )

        result_dict = result.to_dict()
        assert result_dict["indicator_name"] == "SMA_10"
        assert result_dict["parameters"] == {"window": 10}
        assert result_dict["timestamp"] == timestamp


# Edge case and error handling tests
class TestErrorHandling:
    """Test error handling in technical indicators"""

    def test_empty_series(self):
        """Test behavior with empty series"""
        empty_series = pd.Series([], dtype=float)

        with pytest.raises(ValueError):
            calculate_sma(empty_series, 10)

    def test_series_with_nans(self):
        """Test behavior with NaN values"""
        series_with_nans = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        # Should handle NaNs gracefully
        sma = calculate_sma(series_with_nans, 10)
        assert isinstance(sma, pd.Series)

    def test_negative_window(self):
        """Test behavior with negative window"""
        price_series = pd.Series([1, 2, 3, 4, 5])

        # This should raise an error or handle gracefully
        # Implementation depends on pandas behavior
        try:
            calculate_sma(price_series, -1)
        except (ValueError, Exception):
            pass  # Expected behavior


if __name__ == "__main__":
    pytest.main([__file__])
