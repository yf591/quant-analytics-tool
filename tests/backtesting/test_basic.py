"""Simple test to validate BacktestEngine basic functionality."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.backtesting.engine import BacktestEngine, OrderSide, OrderType, OrderStatus


def test_basic_engine_functionality():
    """Test basic BacktestEngine functionality."""
    # Create engine
    engine = BacktestEngine(initial_capital=100000.0, max_position_size=0.5)

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(10) * 0.02)
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

    # Add data and set time
    engine.add_data("AAPL", data)
    engine.current_time = dates[0]

    # Test order placement
    order_id = engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

    # Verify results
    assert order_id is not None
    assert len(engine.orders) == 1
    assert len(engine.trades) == 1
    assert len(engine.positions) == 1

    order = engine.orders[0]
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 50

    position = list(engine.positions.values())[0]
    assert position.quantity == 50
    assert position.is_long


def test_position_size_limits():
    """Test position size validation."""
    engine = BacktestEngine(
        initial_capital=100000.0, max_position_size=0.05
    )  # 5% limit

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "Open": [100] * 10,
            "High": [102] * 10,
            "Low": [98] * 10,
            "Close": [100] * 10,
            "Volume": [1000000] * 10,
        },
        index=dates,
    )

    engine.add_data("AAPL", data)
    engine.current_time = dates[0]

    # Try to buy more than allowed (should be rejected)
    order_id = engine.place_order(
        "AAPL", OrderSide.BUY, 100, OrderType.MARKET
    )  # $10,000 > 5% of $100,000

    # Should be rejected
    order = next(o for o in engine.orders if o.order_id == order_id)
    assert order.status == OrderStatus.REJECTED
    assert len(engine.trades) == 0


def test_commission_and_slippage():
    """Test commission and slippage calculations."""
    engine = BacktestEngine(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_position_size=0.5,
    )

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "Open": [100] * 10,
            "High": [102] * 10,
            "Low": [98] * 10,
            "Close": [100] * 10,
            "Volume": [1000000] * 10,
        },
        index=dates,
    )

    engine.add_data("AAPL", data)
    engine.current_time = dates[0]

    initial_commission = engine.total_commission
    initial_slippage = engine.total_slippage

    # Place order
    engine.place_order("AAPL", OrderSide.BUY, 50, OrderType.MARKET)

    # Check that commission and slippage were applied
    assert engine.total_commission > initial_commission
    assert engine.total_slippage > initial_slippage

    trade = engine.trades[0]
    assert trade.commission > 0
    assert trade.price > 100  # Should include slippage for buy order


if __name__ == "__main__":
    pytest.main([__file__])
