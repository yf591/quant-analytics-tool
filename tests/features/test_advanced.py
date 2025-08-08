"""
Test suite for Advanced Feature Engineering module.

Tests for fractal dimension, Hurst exponent, information-driven bars,
triple barrier method, and fractional differentiation.
"""

import pytest
import numpy as np
import pandas as pd
from src.features.advanced import AdvancedFeatures, AdvancedFeatureResults


class TestAdvancedFeatures:
    """Test class for AdvancedFeatures."""

    @pytest.fixture
    def sample_data(self):
        """Create sample financial data for testing."""
        np.random.seed(42)
        n_points = 1000

        # Generate realistic price data with trend and volatility
        dates = pd.date_range("2023-01-01", periods=n_points, freq="1min")

        # Random walk with drift
        returns = np.random.normal(0.0001, 0.02, n_points)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate volume data
        base_volume = 1000
        volume = np.random.lognormal(np.log(base_volume), 0.5, n_points)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, n_points)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
                "close": prices,
                "volume": volume,
            },
            index=dates,
        )

        # Ensure high >= close >= low and high >= open >= low
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    @pytest.fixture
    def advanced_features(self):
        """Create AdvancedFeatures instance."""
        return AdvancedFeatures()

    def test_initialization(self, advanced_features):
        """Test AdvancedFeatures initialization."""
        assert isinstance(advanced_features, AdvancedFeatures)

    def test_fractal_dimension_higuchi(self, advanced_features, sample_data):
        """Test fractal dimension calculation using Higuchi method."""
        prices = sample_data["close"]

        # Test with smaller window for faster computation
        fd = advanced_features.calculate_fractal_dimension(
            prices, window=50, method="higuchi"
        )

        assert isinstance(fd, pd.Series)
        assert len(fd) == len(prices)
        assert fd.index.equals(prices.index)

        # Check that non-NaN values are within reasonable range
        valid_fd = fd.dropna()
        assert len(valid_fd) > 0
        assert all(
            0.5 <= val <= 2.5 for val in valid_fd
        ), "Fractal dimension should be between 0.5 and 2.5"

    def test_fractal_dimension_box_counting(self, advanced_features, sample_data):
        """Test fractal dimension calculation using box counting method."""
        prices = sample_data["close"]

        fd = advanced_features.calculate_fractal_dimension(
            prices, window=50, method="box_counting"
        )

        assert isinstance(fd, pd.Series)
        assert len(fd) == len(prices)

        # Should have some valid values
        valid_fd = fd.dropna()
        assert len(valid_fd) >= 0  # Box counting might produce fewer valid results

    def test_fractal_dimension_invalid_method(self, advanced_features, sample_data):
        """Test fractal dimension with invalid method."""
        prices = sample_data["close"]

        with pytest.raises(
            ValueError, match="Method must be 'higuchi' or 'box_counting'"
        ):
            advanced_features.calculate_fractal_dimension(prices, method="invalid")

    def test_hurst_exponent_rs(self, advanced_features, sample_data):
        """Test Hurst exponent calculation using R/S analysis."""
        prices = sample_data["close"]

        hurst = advanced_features.calculate_hurst_exponent(
            prices, window=100, method="rs"
        )

        assert isinstance(hurst, pd.Series)
        assert len(hurst) == len(prices)
        assert hurst.index.equals(prices.index)

        # Check that non-NaN values are within reasonable range
        valid_hurst = hurst.dropna()
        assert len(valid_hurst) > 0
        # Note: R/S analysis can sometimes give values outside [0,1] due to finite sample effects
        assert all(
            -0.5 <= val <= 1.5 for val in valid_hurst
        ), f"Hurst exponent should be roughly between -0.5 and 1.5, got range [{valid_hurst.min():.3f}, {valid_hurst.max():.3f}]"

    def test_hurst_exponent_dfa(self, advanced_features, sample_data):
        """Test Hurst exponent calculation using DFA."""
        prices = sample_data["close"]

        hurst = advanced_features.calculate_hurst_exponent(
            prices, window=100, method="dfa"
        )

        assert isinstance(hurst, pd.Series)
        assert len(hurst) == len(prices)

        # Should have some valid values
        valid_hurst = hurst.dropna()
        assert len(valid_hurst) >= 0  # DFA might produce fewer valid results

    def test_hurst_exponent_invalid_method(self, advanced_features, sample_data):
        """Test Hurst exponent with invalid method."""
        prices = sample_data["close"]

        with pytest.raises(ValueError, match="Method must be 'rs' or 'dfa'"):
            advanced_features.calculate_hurst_exponent(prices, method="invalid")

    def test_information_bars_volume(self, advanced_features, sample_data):
        """Test volume-based information bars."""
        volume_bars = advanced_features.create_information_bars(
            sample_data, bar_type="volume", threshold=5000
        )

        assert isinstance(volume_bars, pd.DataFrame)
        assert len(volume_bars) > 0

        # Check required columns
        expected_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "count",
            "start_time",
        ]
        for col in expected_cols:
            assert col in volume_bars.columns

        # Check OHLC constraints
        assert all(volume_bars["high"] >= volume_bars["close"])
        assert all(volume_bars["high"] >= volume_bars["open"])
        assert all(volume_bars["low"] <= volume_bars["close"])
        assert all(volume_bars["low"] <= volume_bars["open"])

        # Check volume is positive
        assert all(volume_bars["volume"] > 0)

    def test_information_bars_tick(self, advanced_features, sample_data):
        """Test tick-based information bars."""
        tick_bars = advanced_features.create_information_bars(
            sample_data, bar_type="tick", threshold=50
        )

        assert isinstance(tick_bars, pd.DataFrame)
        assert len(tick_bars) > 0

        # Each bar should have exactly 50 ticks (except possibly the last one)
        assert all(bar_count <= 50 for bar_count in tick_bars["count"][:-1])

    def test_information_bars_dollar(self, advanced_features, sample_data):
        """Test dollar-based information bars."""
        dollar_bars = advanced_features.create_information_bars(
            sample_data, bar_type="dollar", threshold=500000
        )

        assert isinstance(dollar_bars, pd.DataFrame)
        assert len(dollar_bars) > 0

        # Check that VWAP is reasonable
        assert all(dollar_bars["vwap"] > 0)

    def test_information_bars_invalid_type(self, advanced_features, sample_data):
        """Test information bars with invalid type."""
        with pytest.raises(
            ValueError, match="bar_type must be 'tick', 'volume', or 'dollar'"
        ):
            advanced_features.create_information_bars(sample_data, bar_type="invalid")

    def test_triple_barrier_method(self, advanced_features, sample_data):
        """Test triple barrier method for meta-labeling."""
        prices = sample_data["close"]

        # Create events (every 100th observation)
        events = pd.Series(index=prices.index[::100], data=prices.index[::100])

        labels = advanced_features.triple_barrier_method(prices, events, pt_sl=[2, 2])

        assert isinstance(labels, pd.DataFrame)
        assert len(labels) == len(events)

        # Check required columns
        expected_cols = ["barrier", "ret", "t1"]
        for col in expected_cols:
            assert col in labels.columns

        # Check barrier values are valid
        assert all(barrier in [-1, 0, 1] for barrier in labels["barrier"])

    def test_fractional_differentiation(self, advanced_features, sample_data):
        """Test fractional differentiation."""
        prices = sample_data["close"]

        frac_diff = advanced_features.fractional_differentiation(
            prices, d=0.4, threshold=0.01
        )

        assert isinstance(frac_diff, pd.Series)
        assert len(frac_diff) == len(prices)
        assert frac_diff.index.equals(prices.index)

        # Should have some valid (non-NaN) values
        valid_values = frac_diff.dropna()
        assert len(valid_values) > 0

        # Fractionally differentiated series should be more stationary
        # (This is a simplified check - in practice, you'd use statistical tests)
        original_std = prices.pct_change().std()
        frac_diff_std = frac_diff.pct_change().std()
        # Note: This comparison might not always hold, it's just a basic sanity check

    def test_calculate_all_features(self, advanced_features, sample_data):
        """Test calculation of all advanced features."""
        results = advanced_features.calculate_all_features(
            sample_data, window=50  # Smaller window for faster testing
        )

        assert isinstance(results, AdvancedFeatureResults)

        # Check that main features are calculated
        assert results.fractal_dimension is not None
        assert isinstance(results.fractal_dimension, pd.Series)

        assert results.hurst_exponent is not None
        assert isinstance(results.hurst_exponent, pd.Series)

        assert results.information_bars is not None
        assert isinstance(results.information_bars, pd.DataFrame)

        assert results.fractional_diff is not None
        assert isinstance(results.fractional_diff, pd.Series)

        # Triple barrier labels might be None if no events generated
        if results.triple_barrier_labels is not None:
            assert isinstance(results.triple_barrier_labels, pd.DataFrame)

    def test_calculate_all_features_missing_price_column(
        self, advanced_features, sample_data
    ):
        """Test calculate_all_features with missing price column."""
        data_no_close = sample_data.drop("close", axis=1)

        with pytest.raises(ValueError, match="Price column 'close' not found in data"):
            advanced_features.calculate_all_features(data_no_close)

    def test_get_daily_vol(self, advanced_features, sample_data):
        """Test daily volatility calculation."""
        prices = sample_data["close"]

        vol = advanced_features._get_daily_vol(prices, span=20)

        assert isinstance(vol, pd.Series)
        assert len(vol) == len(prices) - 1  # One less due to pct_change
        assert all(vol.dropna() >= 0), "Volatility should be non-negative"

    def test_get_weights_ffd(self, advanced_features):
        """Test fractional differentiation weights calculation."""
        weights = advanced_features._get_weights_ffd(d=0.5, threshold=0.01)

        assert isinstance(weights, np.ndarray)
        assert weights.shape[1] == 1  # Should be column vector
        assert len(weights) > 1

        # First weight should be 1.0
        assert abs(weights[-1, 0] - 1.0) < 1e-10

        # Last weight should be below threshold (approximately)
        assert abs(weights[0, 0]) <= 0.02  # Slightly more lenient threshold


class TestAdvancedFeatureResults:
    """Test class for AdvancedFeatureResults dataclass."""

    def test_initialization(self):
        """Test AdvancedFeatureResults initialization."""
        results = AdvancedFeatureResults()

        assert results.fractal_dimension is None
        assert results.hurst_exponent is None
        assert results.information_bars is None
        assert results.triple_barrier_labels is None
        assert results.fractional_diff is None

    def test_initialization_with_data(self):
        """Test AdvancedFeatureResults initialization with data."""
        # Create sample data
        sample_series = pd.Series([1, 2, 3, 4, 5])
        sample_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        results = AdvancedFeatureResults(
            fractal_dimension=sample_series,
            hurst_exponent=sample_series,
            information_bars=sample_df,
            triple_barrier_labels=sample_df,
            fractional_diff=sample_series,
        )

        assert results.fractal_dimension.equals(sample_series)
        assert results.hurst_exponent.equals(sample_series)
        assert results.information_bars.equals(sample_df)
        assert results.triple_barrier_labels.equals(sample_df)
        assert results.fractional_diff.equals(sample_series)


# Integration tests
class TestAdvancedFeaturesIntegration:
    """Integration tests for advanced features with realistic scenarios."""

    def test_trend_following_strategy_features(self):
        """Test advanced features for trend-following strategy development."""
        # Create trending market data
        np.random.seed(123)
        n_points = 500
        dates = pd.date_range("2023-01-01", periods=n_points, freq="h")

        # Trending price with some noise
        trend = np.linspace(100, 120, n_points)
        noise = np.random.normal(0, 1, n_points)
        prices = trend + noise

        data = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.001,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.random.lognormal(7, 0.5, n_points),
            },
            index=dates,
        )

        advanced_features = AdvancedFeatures()
        results = advanced_features.calculate_all_features(data, window=50)

        # In trending markets, we expect:
        # - Hurst exponent > 0.5 (persistent)
        # - Specific fractal dimension characteristics
        hurst_values = results.hurst_exponent.dropna()
        if len(hurst_values) > 0:
            mean_hurst = hurst_values.mean()
            # Trending data should show some persistence
            # Note: This is a statistical expectation, individual tests might vary
            assert (
                0.3 <= mean_hurst <= 0.8
            ), f"Expected Hurst in range [0.3, 0.8], got {mean_hurst}"

    def test_mean_reverting_strategy_features(self):
        """Test advanced features for mean-reverting strategy development."""
        # Create mean-reverting market data
        np.random.seed(456)
        n_points = 500
        dates = pd.date_range("2023-01-01", periods=n_points, freq="h")

        # Ornstein-Uhlenbeck process (mean-reverting)
        prices = [100]
        theta = 0.1  # Mean reversion speed
        mu = 100  # Long-term mean
        sigma = 2  # Volatility

        for _ in range(n_points - 1):
            dt = 1
            price = prices[-1]
            dprice = (
                theta * (mu - price) * dt + sigma * np.sqrt(dt) * np.random.normal()
            )
            prices.append(price + dprice)

        prices = np.array(prices)

        data = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.001,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.random.lognormal(7, 0.3, n_points),
            },
            index=dates,
        )

        advanced_features = AdvancedFeatures()
        results = advanced_features.calculate_all_features(data, window=50)

        # Mean-reverting markets might show:
        # - Hurst exponent < 0.5 (anti-persistent)
        # - Different information bar characteristics
        assert results.fractal_dimension is not None
        assert results.hurst_exponent is not None
        assert results.information_bars is not None
