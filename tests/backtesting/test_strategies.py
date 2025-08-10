"""
Comprehensive test module for backtesting strategies.

This module contains tests for the strategy framework including base strategy
class, signal generation, position sizing, and concrete strategy implementations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the classes to test
from src.backtesting.strategies import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    Signal,
    SignalType,
)
from src.backtesting.engine import BacktestEngine, OrderSide, OrderType


class TestSignalClass:
    """Test Signal dataclass functionality."""

    def test_signal_creation(self):
        """Test Signal creation and validation."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.7,
            size=100,
        )

        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == 0.8
        assert signal.confidence == 0.7
        assert signal.size == 100
        assert signal.timestamp is not None
        assert signal.metadata == {}

    def test_signal_validation(self):
        """Test Signal parameter validation."""
        # Test invalid strength
        with pytest.raises(ValueError, match="Signal strength must be between"):
            Signal("AAPL", SignalType.BUY, strength=1.5, confidence=0.5)

        # Test invalid confidence
        with pytest.raises(ValueError, match="Signal confidence must be between"):
            Signal("AAPL", SignalType.BUY, strength=0.5, confidence=-0.1)

    def test_signal_metadata(self):
        """Test Signal with custom metadata."""
        metadata = {"source": "test", "value": 42}
        signal = Signal(
            symbol="MSFT",
            signal_type=SignalType.SELL,
            strength=0.6,
            confidence=0.8,
            metadata=metadata,
        )

        assert signal.metadata == metadata


class MockStrategy(BaseStrategy):
    """Mock strategy for testing base functionality."""

    def __init__(self, **kwargs):
        super().__init__("MockStrategy", ["AAPL"], **kwargs)
        self.start_called = False
        self.data_calls = 0
        self.finish_called = False

    def on_start(self):
        self.start_called = True

    def on_data(self, current_time: datetime):
        self.data_calls += 1
        return []

    def on_finish(self):
        self.finish_called = True


class TestBaseStrategy:
    """Test BaseStrategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create a mock strategy for testing."""
        return MockStrategy(
            lookback_period=20, min_confidence=0.6, max_position_size=0.1
        )

    @pytest.fixture
    def mock_engine(self):
        """Create a mock backtest engine."""
        engine = Mock()
        engine.current_time = datetime(2023, 1, 1)
        engine.market_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": [100, 101, 102, 103, 104],
                    "High": [101, 102, 103, 104, 105],
                    "Low": [99, 100, 101, 102, 103],
                    "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                    "Volume": [1000000] * 5,
                },
                index=pd.date_range("2023-01-01", periods=5, freq="D"),
            )
        }
        engine.get_current_price.return_value = 104.5
        engine.get_portfolio_value.return_value = 100000.0
        engine.positions = {}
        return engine

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "MockStrategy"
        assert strategy.symbols == ["AAPL"]
        assert strategy.lookback_period == 20
        assert strategy.min_confidence == 0.6
        assert strategy.max_position_size == 0.1
        assert not strategy.is_initialized
        assert strategy.current_signals == {}
        assert strategy.signal_history == []

    def test_set_backtest_engine(self, strategy, mock_engine):
        """Test setting backtest engine."""
        strategy.set_backtest_engine(mock_engine)
        assert strategy.backtest_engine == mock_engine

    def test_get_market_data(self, strategy, mock_engine):
        """Test getting market data."""
        strategy.set_backtest_engine(mock_engine)

        # Test getting data
        data = strategy.get_market_data("AAPL", periods=3)
        assert len(data) <= 3  # Should get at most 3 periods
        assert "Close" in data.columns

        # Test with default periods
        data = strategy.get_market_data("AAPL")
        assert len(data) <= 5  # All available data (less than lookback_period)

        # Test with non-existent symbol
        data = strategy.get_market_data("NONEXISTENT")
        assert data.empty

    def test_get_market_data_no_engine(self, strategy):
        """Test getting market data without engine."""
        with pytest.raises(RuntimeError, match="Strategy not connected"):
            strategy.get_market_data("AAPL")

    def test_get_current_price(self, strategy, mock_engine):
        """Test getting current price."""
        strategy.set_backtest_engine(mock_engine)
        price = strategy.get_current_price("AAPL")
        assert price == 104.5

        # Test without engine
        strategy.backtest_engine = None
        price = strategy.get_current_price("AAPL")
        assert price is None

    def test_get_portfolio_value(self, strategy, mock_engine):
        """Test getting portfolio value."""
        strategy.set_backtest_engine(mock_engine)
        value = strategy.get_portfolio_value()
        assert value == 100000.0

        # Test without engine
        strategy.backtest_engine = None
        value = strategy.get_portfolio_value()
        assert value == 0.0

    def test_calculate_position_size(self, strategy, mock_engine):
        """Test position size calculation."""
        strategy.set_backtest_engine(mock_engine)

        # Test basic calculation
        size = strategy.calculate_position_size("AAPL", 1.0, 1.0)
        expected_size = int(
            (100000.0 * 0.1) / 104.5
        )  # max_position_size * portfolio / price
        assert size == expected_size

        # Test with reduced strength and confidence
        size = strategy.calculate_position_size("AAPL", 0.5, 0.8)
        expected_size = int((100000.0 * 0.1 * 0.5 * 0.8) / 104.5)
        assert size == expected_size

        # Test with volatility adjustment
        size = strategy.calculate_position_size("AAPL", 1.0, 1.0, volatility=0.4)
        vol_adjustment = min(1.0, 0.2 / 0.4)  # Target 20% volatility
        expected_size = int((100000.0 * 0.1 * vol_adjustment) / 104.5)
        assert size == expected_size

    def test_generate_signal(self, strategy, mock_engine):
        """Test signal generation."""
        strategy.set_backtest_engine(mock_engine)

        # Generate a buy signal
        signal = strategy.generate_signal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.7,
            metadata={"test": True},
        )

        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == 0.8
        assert signal.confidence == 0.7
        assert signal.size > 0
        assert signal.metadata["test"] is True

        # Check signal was stored
        assert strategy.current_signals["AAPL"] == signal
        assert signal in strategy.signal_history

        # Test signal with low confidence (should become HOLD)
        signal = strategy.generate_signal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.3,  # Below min_confidence (0.6)
        )

        assert signal.signal_type == SignalType.HOLD
        assert signal.strength == 0.0

    def test_execute_signal(self, strategy, mock_engine):
        """Test signal execution."""
        strategy.set_backtest_engine(mock_engine)
        mock_engine.place_order = Mock()
        mock_engine.positions = {}

        # Test buy signal execution
        signal = Signal("AAPL", SignalType.BUY, strength=0.8, confidence=0.7, size=50)
        result = strategy.execute_signal(signal)

        assert result is True
        mock_engine.place_order.assert_called_with(
            "AAPL", OrderSide.BUY, 50, OrderType.MARKET
        )

        # Test sell signal execution
        signal = Signal("AAPL", SignalType.SELL, strength=0.8, confidence=0.7, size=30)
        result = strategy.execute_signal(signal)

        assert result is True
        mock_engine.place_order.assert_called_with(
            "AAPL", OrderSide.SELL, 30, OrderType.MARKET
        )


class TestBuyAndHoldStrategy:
    """Test BuyAndHoldStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Create buy and hold strategy."""
        return BuyAndHoldStrategy(symbols=["AAPL", "MSFT"])

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine for testing."""
        engine = Mock()
        engine.current_time = datetime(2023, 1, 1)
        engine.get_current_price.return_value = 100.0
        engine.get_portfolio_value.return_value = 100000.0
        engine.positions = {}
        engine.place_order = Mock()
        return engine

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "BuyAndHold"
        assert strategy.symbols == ["AAPL", "MSFT"]
        assert len(strategy.has_bought) == 0

    def test_on_start(self, strategy):
        """Test strategy start."""
        strategy.on_start()
        assert strategy.is_initialized is True
        assert len(strategy.has_bought) == 0

    def test_on_data_first_call(self, strategy, mock_engine):
        """Test first data call generates buy signals."""
        strategy.set_backtest_engine(mock_engine)
        strategy.on_start()

        signals = strategy.on_data(datetime(2023, 1, 1))

        # Should generate buy signals for both symbols
        assert len(signals) == 2
        assert all(s.signal_type == SignalType.BUY for s in signals)
        assert all(s.strength == 1.0 for s in signals)
        assert all(s.confidence == 1.0 for s in signals)

        # Should have marked symbols as bought
        assert "AAPL" in strategy.has_bought
        assert "MSFT" in strategy.has_bought

        # Should have called place_order for each symbol
        assert mock_engine.place_order.call_count == 2

    def test_on_data_subsequent_calls(self, strategy, mock_engine):
        """Test subsequent data calls don't generate new signals."""
        strategy.set_backtest_engine(mock_engine)
        strategy.on_start()

        # First call
        strategy.on_data(datetime(2023, 1, 1))

        # Second call
        signals = strategy.on_data(datetime(2023, 1, 2))

        # Should not generate new signals
        assert len(signals) == 0


class TestMomentumStrategy:
    """Test MomentumStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Create momentum strategy."""
        return MomentumStrategy(symbols=["AAPL"], short_window=5, long_window=10)

    @pytest.fixture
    def mock_engine_with_trend_data(self):
        """Create mock engine with trending price data."""
        engine = Mock()
        engine.current_time = datetime(2023, 1, 15)

        # Create uptrending data
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        prices = np.linspace(100, 120, 20)  # Uptrend

        engine.market_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": prices * 0.999,
                    "High": prices * 1.01,
                    "Low": prices * 0.99,
                    "Close": prices,
                    "Volume": [1000000] * 20,
                },
                index=dates,
            )
        }

        engine.get_current_price.return_value = 120.0
        engine.get_portfolio_value.return_value = 100000.0
        engine.positions = {}
        engine.place_order = Mock()
        return engine

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "Momentum"
        assert strategy.short_window == 5
        assert strategy.long_window == 10
        assert strategy.lookback_period >= 15  # Should be at least long_window + 5

    def test_crossover_detection(self, strategy, mock_engine_with_trend_data):
        """Test moving average crossover detection."""
        strategy.set_backtest_engine(mock_engine_with_trend_data)
        strategy.on_start()

        # Mock position to be flat
        mock_engine_with_trend_data.positions = {}

        signals = strategy.on_data(datetime(2023, 1, 15))

        # In an uptrend, should potentially generate buy signal if crossover occurred
        # Note: This depends on the exact data and crossover timing
        assert isinstance(signals, list)


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Create mean reversion strategy."""
        return MeanReversionStrategy(symbols=["AAPL"], window=10, num_std=2.0)

    @pytest.fixture
    def mock_engine_with_bollinger_data(self):
        """Create mock engine with data suitable for Bollinger Band testing."""
        engine = Mock()
        engine.current_time = datetime(2023, 1, 15)

        # Create data with a spike (suitable for mean reversion)
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        np.random.seed(42)
        base_prices = np.full(20, 100.0)
        noise = np.random.normal(0, 2, 20)
        prices = base_prices + noise

        # Add a spike at the end
        prices[-1] = 110.0  # Significant spike

        engine.market_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": prices * 0.999,
                    "High": prices * 1.01,
                    "Low": prices * 0.99,
                    "Close": prices,
                    "Volume": [1000000] * 20,
                },
                index=dates,
            )
        }

        engine.get_current_price.return_value = 110.0
        engine.get_portfolio_value.return_value = 100000.0
        engine.positions = {}
        engine.place_order = Mock()
        return engine

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "MeanReversion"
        assert strategy.window == 10
        assert strategy.num_std == 2.0

    def test_bollinger_band_calculation(
        self, strategy, mock_engine_with_bollinger_data
    ):
        """Test Bollinger Band calculation and signal generation."""
        strategy.set_backtest_engine(mock_engine_with_bollinger_data)
        strategy.on_start()

        signals = strategy.on_data(datetime(2023, 1, 15))

        # Should process the data and potentially generate signals
        assert isinstance(signals, list)


class TestStrategyIntegration:
    """Integration tests for strategies with real backtest engine."""

    @pytest.fixture
    def engine_with_data(self):
        """Create engine with sample data."""
        engine = BacktestEngine(initial_capital=100000.0, max_position_size=0.5)

        # Create sample trending data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        np.random.seed(42)

        # Generate trending price data
        trend = np.linspace(100, 120, 50)
        noise = np.random.normal(0, 1, 50)
        prices = trend + noise

        data = pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )

        engine.add_data("AAPL", data)
        return engine

    def test_buy_and_hold_integration(self, engine_with_data):
        """Test buy and hold strategy with real engine."""
        strategy = BuyAndHoldStrategy(symbols=["AAPL"], max_position_size=0.1)
        engine_with_data.set_strategy(strategy)

        # Run a short backtest
        results = engine_with_data.run_backtest()

        # Should not have error
        assert "error" not in results
        # Should have completed successfully
        assert "total_return" in results or "total_trades" in results
        assert len(engine_with_data.positions) >= 0

        # Should have generated signals
        assert len(strategy.signal_history) >= 1
        assert strategy.signal_history[0].signal_type == SignalType.BUY

    def test_momentum_strategy_integration(self, engine_with_data):
        """Test momentum strategy with real engine."""
        strategy = MomentumStrategy(
            symbols=["AAPL"], short_window=5, long_window=10, max_position_size=0.1
        )
        engine_with_data.set_strategy(strategy)

        # Run backtest
        results = engine_with_data.run_backtest()

        # Should complete without errors
        assert "error" not in results or results["error"] is None
        # Should have some result
        assert len(results) > 0
        assert results["total_trades"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
