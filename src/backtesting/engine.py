"""
Comprehensive backtesting engine for quantitative trading strategies.

This module implements a sophisticated backtesting framework that supports:
- Event-driven simulation
- Multiple asset classes
- Realistic transaction costs
- Advanced order types
- Risk management integration
- Performance analysis

References:
- Advances in Financial Machine Learning by Marcos López de Prado
- Quantitative Trading strategies and methodologies
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the backtesting engine."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides (buy/sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Represents a trading order with all relevant information.

    Attributes:
        symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
        side: Buy or sell order
        order_type: Market, limit, stop, etc.
        quantity: Number of shares/units
        price: Order price (for limit orders)
        stop_price: Stop price (for stop orders)
        timestamp: Order creation time
        order_id: Unique order identifier
        status: Current order status
        filled_quantity: Quantity already filled
        avg_fill_price: Average execution price
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0

    def __post_init__(self):
        """Set default values after initialization."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.order_id is None:
            self.order_id = (
                f"{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
            )


@dataclass
class Trade:
    """
    Represents a completed trade transaction.

    Attributes:
        symbol: Trading symbol
        side: Buy or sell
        quantity: Number of shares traded
        price: Execution price
        timestamp: Trade execution time
        commission: Trading commission paid
        trade_id: Unique trade identifier
        order_id: Associated order ID
    """

    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    trade_id: Optional[str] = None
    order_id: Optional[str] = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.trade_id is None:
            self.trade_id = f"trade_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"


@dataclass
class Position:
    """
    Represents a position in a particular symbol.

    Attributes:
        symbol: Trading symbol
        quantity: Current position size (positive for long, negative for short)
        avg_price: Average cost basis
        unrealized_pnl: Current unrealized P&L
        realized_pnl: Cumulative realized P&L
    """

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.quantity * self.avg_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no holdings)."""
        return abs(self.quantity) < 1e-8


class BacktestEngine:
    """
    Comprehensive backtesting engine for quantitative trading strategies.

    This engine provides:
    - Event-driven simulation architecture
    - Realistic order execution with slippage and commissions
    - Portfolio tracking and risk management
    - Comprehensive performance metrics
    - Support for multiple asset classes
    - Advanced order types (market, limit, stop)

    Features:
    - Time-aware backtesting with proper data alignment
    - Transaction cost modeling
    - Position sizing and risk controls
    - Drawdown monitoring
    - Performance attribution analysis
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_rate: float = 0.0005,  # 0.05% slippage
        min_commission: float = 1.0,
        max_position_size: float = 0.1,  # 10% max position size
        benchmark_symbol: str = "SPY",
        data_frequency: str = "1D",
    ):
        """
        Initialize the backtesting engine.

        Args:
            initial_capital: Starting capital amount
            commission_rate: Commission rate as percentage of trade value
            slippage_rate: Slippage rate as percentage of trade value
            min_commission: Minimum commission per trade
            max_position_size: Maximum position size as fraction of portfolio
            benchmark_symbol: Benchmark symbol for comparison
            data_frequency: Data frequency (1D, 1H, etc.)
        """
        # Capital and portfolio tracking
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital

        # Trading costs
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission

        # Risk management
        self.max_position_size = max_position_size

        # Market data and timing
        self.benchmark_symbol = benchmark_symbol
        self.data_frequency = data_frequency
        self.current_time: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Trading state
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.pending_orders: List[Order] = []

        # Market data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}

        # Performance tracking
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.drawdowns: List[Tuple[datetime, float]] = []
        self.benchmark_values: List[Tuple[datetime, float]] = []

        # Statistics
        self.total_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration = timedelta(0)

        # Strategy reference
        self.strategy = None

        logger.info(
            f"BacktestEngine initialized with ${initial_capital:,.2f} initial capital"
        )

    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Add market data for a symbol.

        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        # Ensure data is sorted by date
        data = data.sort_index()

        self.market_data[symbol] = data
        logger.info(f"Added {len(data)} data points for {symbol}")

    def set_strategy(self, strategy) -> None:
        """
        Set the trading strategy for backtesting.

        Args:
            strategy: Strategy instance implementing BaseStrategy interface
        """
        self.strategy = strategy
        strategy.set_backtest_engine(self)
        logger.info(f"Strategy set: {strategy.__class__.__name__}")

    def get_current_price(
        self, symbol: str, price_type: str = "Close"
    ) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol
            price_type: Price type (Open, High, Low, Close)

        Returns:
            Current price or None if not available
        """
        if symbol not in self.market_data or self.current_time is None:
            return None

        data = self.market_data[symbol]

        # Find the current or most recent available price
        available_dates = data.index[data.index <= self.current_time]
        if len(available_dates) == 0:
            return None

        latest_date = available_dates[-1]
        return data.loc[latest_date, price_type]

    def get_portfolio_value(self) -> float:
        """
        Calculate current total portfolio value.

        Returns:
            Total portfolio value (cash + positions)
        """
        total_value = self.available_capital

        for symbol, position in self.positions.items():
            if not position.is_flat:
                current_price = self.get_current_price(symbol)
                if current_price is not None:
                    position_value = position.quantity * current_price
                    total_value += position_value

        return total_value

    def place_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: float,
        order_type: Union[OrderType, str] = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> str:
        """
        Place a trading order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type (market, limit, stop)
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Order ID
        """
        # Convert string enums to enum types
        if isinstance(side, str):
            side = OrderSide(side.lower())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.lower())

        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=abs(quantity),  # Ensure positive quantity
            price=price,
            stop_price=stop_price,
            timestamp=self.current_time,
        )

        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order.order_id}")
            # Still add rejected orders to orders list for tracking
            self.orders.append(order)
            return order.order_id

        # Add to orders list
        self.orders.append(order)

        # Try to execute immediately if market order
        if order_type == OrderType.MARKET:
            self._execute_order(order)
        else:
            self.pending_orders.append(order)

        return order.order_id

    def _validate_order(self, order: Order) -> bool:
        """
        Validate order before execution.

        Args:
            order: Order to validate

        Returns:
            True if order is valid
        """
        # Check if symbol has data
        if order.symbol not in self.market_data:
            logger.error(f"No market data for symbol: {order.symbol}")
            return False

        # Check position size limits
        current_price = self.get_current_price(order.symbol)
        if current_price is None:
            logger.error(f"No current price for symbol: {order.symbol}")
            return False

        order_value = order.quantity * current_price
        max_order_value = self.get_portfolio_value() * self.max_position_size

        if order_value > max_order_value:
            logger.warning(
                f"Order exceeds position size limit: {order_value:.2f} > {max_order_value:.2f}"
            )
            return False

        # Check available capital for buy orders
        if order.side == OrderSide.BUY:
            estimated_cost = order_value * (
                1 + self.commission_rate + self.slippage_rate
            )
            if estimated_cost > self.available_capital:
                logger.warning(
                    f"Insufficient capital: {estimated_cost:.2f} > {self.available_capital:.2f}"
                )
                return False

        return True

    def _execute_order(self, order: Order) -> None:
        """
        Execute a validated order.

        Args:
            order: Order to execute
        """
        current_price = self.get_current_price(order.symbol)
        if current_price is None:
            order.status = OrderStatus.REJECTED
            return

        # Calculate execution price with slippage
        if order.order_type == OrderType.MARKET:
            execution_price = current_price
            if order.side == OrderSide.BUY:
                execution_price *= 1 + self.slippage_rate
            else:
                execution_price *= 1 - self.slippage_rate
        else:
            execution_price = order.price or current_price

        # Calculate commission
        trade_value = order.quantity * execution_price
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Create trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=self.current_time,
            commission=commission,
            order_id=order.order_id,
        )

        # Update position
        self._update_position(trade)

        # Update capital
        if order.side == OrderSide.BUY:
            self.available_capital -= trade_value + commission
        else:
            self.available_capital += trade_value - commission

        # Update statistics
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += abs(execution_price - current_price) * order.quantity

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price

        # Add trade to history
        self.trades.append(trade)

        # Remove from pending orders if applicable
        if order in self.pending_orders:
            self.pending_orders.remove(order)

        logger.debug(
            f"Order executed: {order.symbol} {order.side.value} {order.quantity} @ ${execution_price:.2f}"
        )

    def _update_position(self, trade: Trade) -> None:
        """
        Update position based on executed trade.

        Args:
            trade: Executed trade
        """
        symbol = trade.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]

        if trade.side == OrderSide.BUY:
            # Calculate new average price
            if position.quantity >= 0:
                # Adding to long position or opening long
                total_cost = (position.quantity * position.avg_price) + (
                    trade.quantity * trade.price
                )
                total_quantity = position.quantity + trade.quantity
                position.avg_price = (
                    total_cost / total_quantity if total_quantity > 0 else 0
                )
                position.quantity = total_quantity
            else:
                # Covering short position
                if trade.quantity <= abs(position.quantity):
                    # Partial or full cover
                    realized_pnl = (position.avg_price - trade.price) * trade.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity += trade.quantity
                else:
                    # Cover and reverse to long
                    cover_quantity = abs(position.quantity)
                    realized_pnl = (position.avg_price - trade.price) * cover_quantity
                    position.realized_pnl += realized_pnl

                    remaining_quantity = trade.quantity - cover_quantity
                    position.quantity = remaining_quantity
                    position.avg_price = trade.price

        else:  # SELL
            if position.quantity <= 0:
                # Adding to short position or opening short
                total_cost = (abs(position.quantity) * position.avg_price) + (
                    trade.quantity * trade.price
                )
                total_quantity = abs(position.quantity) + trade.quantity
                position.avg_price = (
                    total_cost / total_quantity if total_quantity > 0 else 0
                )
                position.quantity = -total_quantity
            else:
                # Selling long position
                if trade.quantity <= position.quantity:
                    # Partial or full sale
                    realized_pnl = (trade.price - position.avg_price) * trade.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
                else:
                    # Sell and reverse to short
                    sell_quantity = position.quantity
                    realized_pnl = (trade.price - position.avg_price) * sell_quantity
                    position.realized_pnl += realized_pnl

                    remaining_quantity = trade.quantity - sell_quantity
                    position.quantity = -remaining_quantity
                    position.avg_price = trade.price

    def _process_pending_orders(self) -> None:
        """Process pending limit and stop orders."""
        executed_orders = []

        for order in self.pending_orders:
            current_price = self.get_current_price(order.symbol)
            if current_price is None:
                continue

            should_execute = False

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_execute = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_execute = True

            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_execute = True
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_execute = True

            if should_execute:
                self._execute_order(order)
                executed_orders.append(order)

        # Remove executed orders from pending
        for order in executed_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)

    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L for all positions."""
        for symbol, position in self.positions.items():
            if not position.is_flat:
                current_price = self.get_current_price(symbol)
                if current_price is not None:
                    if position.is_long:
                        position.unrealized_pnl = (
                            current_price - position.avg_price
                        ) * position.quantity
                    else:
                        position.unrealized_pnl = (
                            position.avg_price - current_price
                        ) * abs(position.quantity)

    def run_backtest(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete backtest.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dictionary containing backtest results
        """
        if self.strategy is None:
            raise ValueError("Strategy must be set before running backtest")

        if not self.market_data:
            raise ValueError("Market data must be added before running backtest")

        # Determine date range
        all_dates = set()
        for data in self.market_data.values():
            all_dates.update(data.index)
        all_dates = sorted(list(all_dates))

        if start_date is None:
            start_date = all_dates[0]
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = all_dates[-1]
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        self.start_time = start_date
        self.end_time = end_date

        # Filter dates within range
        backtest_dates = [d for d in all_dates if start_date <= d <= end_date]

        logger.info(
            f"Starting backtest from {start_date} to {end_date} ({len(backtest_dates)} periods)"
        )

        # Initialize strategy
        self.strategy.on_start()

        # Main backtest loop
        for i, current_date in enumerate(backtest_dates):
            self.current_time = current_date

            # Update current prices
            for symbol in self.market_data:
                price = self.get_current_price(symbol)
                if price is not None:
                    self.current_prices[symbol] = price

            # Process pending orders
            self._process_pending_orders()

            # Update unrealized P&L
            self._update_unrealized_pnl()

            # Call strategy
            try:
                self.strategy.on_data(current_date)
            except Exception as e:
                logger.error(f"Strategy error at {current_date}: {e}")
                break

            # Record portfolio value
            portfolio_value = self.get_portfolio_value()
            self.portfolio_values.append((current_date, portfolio_value))

            # Update drawdown tracking
            if len(self.portfolio_values) > 0:
                peak_value = max(pv[1] for pv in self.portfolio_values)
                current_drawdown = (peak_value - portfolio_value) / peak_value
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                self.drawdowns.append((current_date, current_drawdown))

            # Progress logging
            if i % 100 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(backtest_dates)} ({100*(i+1)/len(backtest_dates):.1f}%)"
                )

        # Finalize strategy
        self.strategy.on_finish()

        # Calculate final results
        results = self._calculate_results()

        logger.info("Backtest completed successfully")
        return results

    def _calculate_results(self) -> Dict[str, Any]:
        """
        Calculate comprehensive backtest results.

        Returns:
            Dictionary containing all performance metrics
        """
        if not self.portfolio_values:
            return {"error": "No portfolio values recorded"}

        # Basic metrics
        initial_value = self.initial_capital
        final_value = self.portfolio_values[-1][1]
        total_return = (final_value - initial_value) / initial_value

        # Convert to Series for calculations
        values_df = pd.DataFrame(self.portfolio_values, columns=["Date", "Value"])
        values_df.set_index("Date", inplace=True)
        values_series = values_df["Value"]

        # Calculate returns
        daily_returns = values_series.pct_change().dropna()

        # Performance metrics
        annualized_return = (1 + total_return) ** (252 / len(values_series)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (
            annualized_return / downside_deviation if downside_deviation > 0 else 0
        )

        # Win rate
        winning_trades = len([r for r in daily_returns if r > 0])
        win_rate = winning_trades / len(daily_returns) if len(daily_returns) > 0 else 0

        # Calmar ratio
        calmar_ratio = (
            annualized_return / self.max_drawdown if self.max_drawdown > 0 else 0
        )

        # Trade statistics
        total_trades = len(self.trades)
        avg_trade_pnl = (
            sum(
                t.price * t.quantity * (1 if t.side == OrderSide.SELL else -1)
                for t in self.trades
            )
            / total_trades
            if total_trades > 0
            else 0
        )

        results = {
            # Basic metrics
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            # Performance metrics
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "volatility": volatility,
            "volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            # Risk metrics
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown * 100,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            # Trading metrics
            "total_trades": total_trades,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "avg_trade_pnl": avg_trade_pnl,
            # Positions
            "final_positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for symbol, pos in self.positions.items()
                if not pos.is_flat
            },
            # Time series data
            "portfolio_values": self.portfolio_values,
            "drawdowns": self.drawdowns,
            "daily_returns": daily_returns.tolist(),
            # Period info
            "start_date": self.start_time,
            "end_date": self.end_time,
            "total_days": len(values_series),
        }

        return results

    def get_positions_summary(self) -> pd.DataFrame:
        """
        Get summary of current positions.

        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()

        positions_data = []
        for symbol, position in self.positions.items():
            if not position.is_flat:
                current_price = self.get_current_price(symbol)
                market_value = position.quantity * current_price if current_price else 0

                positions_data.append(
                    {
                        "Symbol": symbol,
                        "Quantity": position.quantity,
                        "Avg_Price": position.avg_price,
                        "Current_Price": current_price,
                        "Market_Value": market_value,
                        "Unrealized_PnL": position.unrealized_pnl,
                        "Realized_PnL": position.realized_pnl,
                        "Total_PnL": position.unrealized_pnl + position.realized_pnl,
                    }
                )

        return pd.DataFrame(positions_data)

    def get_trades_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades.

        Returns:
            DataFrame with trade history
        """
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append(
                {
                    "Trade_ID": trade.trade_id,
                    "Symbol": trade.symbol,
                    "Side": trade.side.value,
                    "Quantity": trade.quantity,
                    "Price": trade.price,
                    "Commission": trade.commission,
                    "Timestamp": trade.timestamp,
                    "Order_ID": trade.order_id,
                }
            )

        return pd.DataFrame(trades_data)

    def reset(self) -> None:
        """Reset the backtesting engine to initial state."""
        self.current_capital = self.initial_capital
        self.available_capital = self.initial_capital
        self.current_time = None
        self.start_time = None
        self.end_time = None

        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.pending_orders.clear()
        self.current_prices.clear()

        self.portfolio_values.clear()
        self.drawdowns.clear()
        self.benchmark_values.clear()

        self.total_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.max_drawdown = 0.0

        logger.info("BacktestEngine reset to initial state")
