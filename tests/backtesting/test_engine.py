"""
Comprehensive test module for backtesting engine functionality.

This module contains comprehensive tests for the BacktestEngine class,
covering all major functionality including order management, position tracking,
portfolio calculations, and performance metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the classes to test
from src.backtesting.engine import (
    BacktestEngine,
    Order,
    Trade,
    Position,
    OrderType,
    OrderSide,
    OrderStatus,
)


class TestOrderClasses:
    """Test Order, Trade, and Position data classes."""

    def test_order_creation(self):
        """Test Order creation and default values."""
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.timestamp is not None
        assert order.order_id is not None

    def test_trade_creation(self):
        """Test Trade creation."""
        timestamp = datetime.now()
        trade = Trade(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            price=300.0,
            timestamp=timestamp,
        )

        assert trade.symbol == "MSFT"
        assert trade.side == OrderSide.SELL
        assert trade.quantity == 50
        assert trade.price == 300.0
        assert trade.timestamp == timestamp
        assert trade.trade_id is not None

    def test_position_properties(self):
        """Test Position properties and calculations."""
        # Test flat position
        position = Position("GOOGL")
        assert position.is_flat
        assert not position.is_long
        assert not position.is_short
        assert position.market_value == 0.0

        # Test long position
        position = Position("GOOGL", quantity=10, avg_price=100.0)
        assert position.is_long
        assert not position.is_short
        assert not position.is_flat
        assert position.market_value == 1000.0

        # Test short position
        position = Position("GOOGL", quantity=-5, avg_price=100.0)
        assert position.is_short
        assert not position.is_long
        assert not position.is_flat
        assert position.market_value == -500.0


class TestBacktestEngine:
    """Test BacktestEngine functionality."""

    @pytest.fixture
    def engine(self):
        """Create a BacktestEngine instance for testing."""
        return BacktestEngine(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            min_commission=1.0,
            max_position_size=0.5,  # Allow larger positions for testing
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        np.random.seed(42)  # For reproducible tests

        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(30) * 0.02)
        open_prices = close_prices * (1 + np.random.randn(30) * 0.001)
        high_prices = np.maximum(open_prices, close_prices) * (
            1 + np.abs(np.random.randn(30)) * 0.005
        )
        low_prices = np.minimum(open_prices, close_prices) * (
            1 - np.abs(np.random.randn(30)) * 0.005
        )
        volumes = 1000000 + np.random.randint(-100000, 100000, 30)

        data = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volumes,
            },
            index=dates,
        )

        return data

    def test_engine_initialization(self, engine):
        """Test BacktestEngine initialization."""
        assert engine.initial_capital == 100000.0
        assert engine.current_capital == 100000.0
        assert engine.available_capital == 100000.0
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005
        assert engine.min_commission == 1.0
        assert len(engine.positions) == 0
        assert len(engine.orders) == 0
        assert len(engine.trades) == 0

    def test_add_data(self, engine, sample_data):
        """Test adding market data."""
        engine.add_data("AAPL", sample_data)

        assert "AAPL" in engine.market_data
        assert len(engine.market_data["AAPL"]) == 30
        assert all(
            col in engine.market_data["AAPL"].columns
            for col in ["Open", "High", "Low", "Close", "Volume"]
        )

    def test_add_data_invalid_columns(self, engine):
        """Test adding data with missing columns."""
        invalid_data = pd.DataFrame(
            {"Price": [100, 101, 102], "Vol": [1000, 1100, 1200]}
        )

        with pytest.raises(ValueError, match="Data must contain columns"):
            engine.add_data("INVALID", invalid_data)

    def test_get_current_price(self, engine, sample_data):
        """Test getting current price."""
        engine.add_data("AAPL", sample_data)

        # Test with no current time set
        assert engine.get_current_price("AAPL") is None

        # Test with valid current time
        engine.current_time = sample_data.index[10]
        price = engine.get_current_price("AAPL")
        assert price is not None
        assert price == sample_data.iloc[10]["Close"]

        # Test different price types
        open_price = engine.get_current_price("AAPL", "Open")
        assert open_price == sample_data.iloc[10]["Open"]

        # Test with future date
        engine.current_time = sample_data.index[-1] + timedelta(days=1)
        price = engine.get_current_price("AAPL")
        assert price == sample_data.iloc[-1]["Close"]  # Should get last available price

    def test_get_portfolio_value(self, engine, sample_data):
        """Test portfolio value calculation."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        # Initial portfolio value should equal cash
        assert engine.get_portfolio_value() == engine.available_capital

        # Add a position manually for testing
        engine.positions["AAPL"] = Position("AAPL", quantity=100, avg_price=100.0)
        engine.available_capital = 90000.0  # Assume we spent 10k on the position

        portfolio_value = engine.get_portfolio_value()
        current_price = engine.get_current_price("AAPL")
        expected_value = 90000.0 + (100 * current_price)

        assert abs(portfolio_value - expected_value) < 0.01

    def test_place_market_order_buy_small(self, engine, sample_data):
        """Test placing a small market buy order."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        order_id = engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

        assert order_id is not None
        assert len(engine.orders) == 1
        assert len(engine.trades) == 1

        order = engine.orders[0]
        trade = engine.trades[0]

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 50
        assert trade.symbol == "AAPL"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 50

        # Check position was created
        assert "AAPL" in engine.positions
        position = engine.positions["AAPL"]
        assert position.quantity == 50
        assert position.is_long

    def test_place_market_order_sell(self, engine, sample_data):
        """Test placing a market sell order."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        # First create a long position
        engine.place_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET)

        # Reset for clean test
        initial_trades = len(engine.trades)

        # Now sell half the position
        order_id = engine.place_order("AAPL", OrderSide.SELL, 50, OrderType.MARKET)

        assert order_id is not None
        assert len(engine.trades) == initial_trades + 1

        # Check position was updated
        position = engine.positions["AAPL"]
        assert position.quantity == 50  # Should have 50 shares left
        assert position.is_long

    def test_place_limit_order(self, engine, sample_data):
        """Test placing a limit order."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        current_price = engine.get_current_price("AAPL")
        limit_price = current_price * 0.95  # Limit buy 5% below current

        order_id = engine.place_order(
            "AAPL", OrderSide.BUY, 50, OrderType.LIMIT, price=limit_price
        )

        assert order_id is not None
        assert len(engine.orders) == 1
        assert len(engine.pending_orders) == 1
        assert len(engine.trades) == 0  # Should not execute immediately

        order = engine.orders[0]
        assert order.status == OrderStatus.PENDING

    def test_order_validation_insufficient_capital(self, engine, sample_data):
        """Test order validation with insufficient capital."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]
        engine.available_capital = 1000.0  # Very low capital

        # Try to buy expensive shares
        order_id = engine.place_order("AAPL", OrderSide.BUY, 1000, OrderType.MARKET)

        order = next(o for o in engine.orders if o.order_id == order_id)
        assert order.status == OrderStatus.REJECTED
        assert len(engine.trades) == 0

    def test_order_validation_position_size_limit(self, engine, sample_data):
        """Test order validation with position size limits."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]
        engine.max_position_size = 0.05  # 5% max position size

        current_price = engine.get_current_price("AAPL")
        max_shares = int((engine.get_portfolio_value() * 0.05) / current_price)

        # Try to buy more than allowed
        order_id = engine.place_order(
            "AAPL", OrderSide.BUY, max_shares * 2, OrderType.MARKET
        )

        order = next(o for o in engine.orders if o.order_id == order_id)
        assert order.status == OrderStatus.REJECTED

    def test_position_update_long_to_short(self, engine, sample_data):
        """Test position updates when going from long to short."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        # Buy 100 shares
        engine.place_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        position = engine.positions["AAPL"]
        assert position.quantity == 100
        assert position.is_long

        # Sell 150 shares (go short 50)
        engine.place_order("AAPL", OrderSide.SELL, 150, OrderType.MARKET)
        position = engine.positions["AAPL"]
        assert position.quantity == -50
        assert position.is_short
        assert position.realized_pnl != 0  # Should have realized P&L from closing long

    def test_commission_calculation(self, engine, sample_data):
        """Test commission calculation."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        initial_commission = engine.total_commission

        engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

        assert engine.total_commission > initial_commission

        # Check that minimum commission is applied
        trade = engine.trades[0]
        expected_commission = max(
            trade.price * trade.quantity * engine.commission_rate, engine.min_commission
        )
        assert abs(trade.commission - expected_commission) < 0.01

    def test_slippage_calculation(self, engine, sample_data):
        """Test slippage calculation."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        current_price = engine.get_current_price("AAPL")

        engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

        trade = engine.trades[0]
        expected_buy_price = current_price * (1 + engine.slippage_rate)

        assert abs(trade.price - expected_buy_price) < 0.01
        assert engine.total_slippage > 0

    def test_unrealized_pnl_calculation(self, engine, sample_data):
        """Test unrealized P&L calculation."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        # Buy shares
        engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)
        position = engine.positions["AAPL"]
        avg_price = position.avg_price

        # Move time forward and update unrealized P&L
        engine.current_time = sample_data.index[5]
        engine._update_unrealized_pnl()

        current_price = engine.get_current_price("AAPL")
        expected_unrealized = (current_price - avg_price) * 50

        assert abs(position.unrealized_pnl - expected_unrealized) < 0.01

    def test_reset_functionality(self, engine, sample_data):
        """Test engine reset functionality."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        # Execute some trades
        engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

        # Verify state has changed
        assert len(engine.positions) > 0
        assert len(engine.trades) > 0
        assert engine.available_capital < engine.initial_capital

        # Reset and verify
        engine.reset()

        assert engine.current_capital == engine.initial_capital
        assert engine.available_capital == engine.initial_capital
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
        assert len(engine.orders) == 0
        assert engine.current_time is None

    def test_get_positions_summary(self, engine, sample_data):
        """Test positions summary generation."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        # No positions initially
        summary = engine.get_positions_summary()
        assert len(summary) == 0

        # Add position
        engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

        summary = engine.get_positions_summary()
        assert len(summary) == 1
        assert summary.iloc[0]["Symbol"] == "AAPL"
        assert summary.iloc[0]["Quantity"] == 50

    def test_get_trades_summary(self, engine, sample_data):
        """Test trades summary generation."""
        engine.add_data("AAPL", sample_data)
        engine.current_time = sample_data.index[0]

        # No trades initially
        summary = engine.get_trades_summary()
        assert len(summary) == 0

        # Execute trade
        engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

        summary = engine.get_trades_summary()
        assert len(summary) == 1
        assert summary.iloc[0]["Symbol"] == "AAPL"
        assert summary.iloc[0]["Side"] == "buy"
        assert summary.iloc[0]["Quantity"] == 50


class TestBacktestEngineIntegration:
    """Integration tests for BacktestEngine with mock strategy."""

    @pytest.fixture
    def engine_with_data(self):
        """Create engine with sample data."""
        engine = BacktestEngine(initial_capital=100000.0, max_position_size=0.5)

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(30) * 0.02)

        data = pd.DataFrame(
            {
                "Open": close_prices * 0.999,
                "High": close_prices * 1.002,
                "Low": close_prices * 0.998,
                "Close": close_prices,
                "Volume": 1000000,
            },
            index=dates,
        )

        engine.add_data("AAPL", data)
        return engine

    def test_simple_buy_and_hold_strategy(self, engine_with_data):
        """Test a simple buy and hold strategy."""
        engine = engine_with_data

        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.on_start = Mock()
        mock_strategy.on_finish = Mock()

        # Strategy that buys on first day
        def on_data_impl(current_time):
            if len(engine.trades) == 0:  # Buy only once
                engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

        mock_strategy.on_data = Mock(side_effect=on_data_impl)

        engine.set_strategy(mock_strategy)

        # Run backtest
        results = engine.run_backtest()

        # Verify strategy was called
        mock_strategy.on_start.assert_called_once()
        mock_strategy.on_finish.assert_called_once()
        assert mock_strategy.on_data.call_count > 0

        # Verify results structure
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "total_trades" in results
        assert results["total_trades"] == 1

        # Verify we have a position
        assert len(engine.positions) == 1
        assert "AAPL" in engine.positions

    def test_backtest_without_strategy(self, engine_with_data):
        """Test backtest fails without strategy."""
        with pytest.raises(ValueError, match="Strategy must be set"):
            engine_with_data.run_backtest()

    def test_backtest_without_data(self):
        """Test backtest fails without market data."""
        engine = BacktestEngine()
        mock_strategy = Mock()
        engine.set_strategy(mock_strategy)

        with pytest.raises(ValueError, match="Market data must be added"):
            engine.run_backtest()


if __name__ == "__main__":
    pytest.main([__file__])
