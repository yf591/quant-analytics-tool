"""
Portfolio Management Module

This module provides comprehensive portfolio management capabilities for backtesting,
following methodologies from "Advances in Financial Machine Learning" (AFML).

Key Features:
- Portfolio construction and optimization
- Risk management and position sizing
- Portfolio analytics and attribution
- Dynamic rebalancing strategies
- Multi-asset portfolio support
- Risk budgeting and allocation

Author: Quantitative Analysis Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import logging
from scipy import optimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    CUSTOM = "custom"


class RiskModel(Enum):
    """Risk model types for portfolio optimization."""

    EQUAL_WEIGHT = "equal_weight"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"


@dataclass
class PortfolioPosition:
    """
    Individual position within a portfolio.

    Based on AFML position management concepts.
    """

    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    realized_pnl: float
    entry_date: datetime
    last_update: datetime

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        """Return percentage since entry."""
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


@dataclass
class PortfolioConstraints:
    """
    Portfolio construction constraints.
    """

    max_weight_per_asset: float = 0.1  # Maximum weight per asset
    max_sector_weight: float = 0.3  # Maximum sector concentration
    max_leverage: float = 1.0  # Maximum leverage ratio
    min_liquidity: float = 1000000  # Minimum daily volume requirement
    max_turnover: float = 0.5  # Maximum portfolio turnover
    max_tracking_error: float = 0.05  # Maximum tracking error vs benchmark

    # Risk limits
    max_var_95: float = 0.02  # Maximum 1-day VaR at 95% confidence
    max_drawdown_limit: float = 0.1  # Maximum allowed drawdown

    # Transaction constraints
    min_trade_size: float = 1000  # Minimum trade size
    max_trades_per_day: int = 50  # Maximum trades per day


@dataclass
class AllocationTarget:
    """Target allocation for portfolio optimization."""

    symbol: str
    target_weight: float
    min_weight: float = 0.0
    max_weight: float = 1.0
    sector: Optional[str] = None
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None


class Portfolio:
    """
    Advanced portfolio management system following AFML methodologies.

    This class provides comprehensive portfolio management including:
    - Dynamic position management
    - Risk monitoring and control
    - Portfolio optimization
    - Performance attribution
    - Rebalancing strategies
    """

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        base_currency: str = "USD",
        risk_free_rate: float = 0.02,
        constraints: Optional[PortfolioConstraints] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting portfolio value
            base_currency: Base currency for portfolio
            risk_free_rate: Risk-free rate for calculations
            constraints: Portfolio constraints
            benchmark_returns: Benchmark returns for tracking
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.base_currency = base_currency
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or PortfolioConstraints()
        self.benchmark_returns = benchmark_returns

        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash = initial_capital
        self.total_value = initial_capital

        # Performance tracking
        self.value_history: List[Tuple[datetime, float]] = []
        self.weights_history: List[Tuple[datetime, Dict[str, float]]] = []
        self.rebalance_history: List[Tuple[datetime, Dict[str, float]]] = []

        # Risk monitoring
        self.var_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[Tuple[datetime, float]] = []

        # Analytics
        self.sector_exposures: Dict[str, float] = {}
        self.factor_exposures: Dict[str, float] = {}

        logger.info(
            f"Portfolio initialized with {initial_capital:,.2f} {base_currency}"
        )

    def update_position(
        self,
        symbol: str,
        quantity_change: float,
        price: float,
        timestamp: datetime,
        sector: Optional[str] = None,
    ) -> None:
        """
        Update position in the portfolio.

        Args:
            symbol: Asset symbol
            quantity_change: Change in quantity (positive for buy, negative for sell)
            price: Execution price
            timestamp: Transaction timestamp
            sector: Asset sector classification
        """
        if symbol not in self.positions:
            # New position
            if quantity_change > 0:
                self.positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity_change,
                    avg_price=price,
                    current_price=price,
                    market_value=quantity_change * price,
                    weight=0.0,  # Will be calculated in _update_weights
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_date=timestamp,
                    last_update=timestamp,
                )

                # Update cash
                self.cash -= quantity_change * price

                logger.debug(f"New position: {symbol} {quantity_change} @ {price}")
            else:
                logger.warning(
                    f"Cannot create short position for {symbol} without existing long position"
                )
                return
        else:
            # Existing position
            position = self.positions[symbol]
            old_quantity = position.quantity

            if quantity_change > 0:
                # Adding to position
                total_cost = (
                    position.quantity * position.avg_price + quantity_change * price
                )
                position.quantity += quantity_change
                position.avg_price = total_cost / position.quantity
                position.current_price = price  # Update current price
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = position.quantity * (
                    position.current_price - position.avg_price
                )
                self.cash -= quantity_change * price

            elif quantity_change < 0:
                # Reducing position
                if abs(quantity_change) > position.quantity:
                    logger.warning(
                        f"Cannot sell {abs(quantity_change)} shares of {symbol}, only {position.quantity} available"
                    )
                    return

                # Calculate realized P&L
                realized_pnl = abs(quantity_change) * (price - position.avg_price)
                position.realized_pnl += realized_pnl
                position.quantity += quantity_change  # quantity_change is negative
                self.cash += abs(quantity_change) * price

                # Remove position if quantity becomes zero
                if abs(position.quantity) < 1e-6:  # Handle floating point precision
                    del self.positions[symbol]
                    logger.debug(f"Position closed: {symbol}")
                    return

            position.last_update = timestamp
            logger.debug(
                f"Updated position: {symbol} {old_quantity} -> {position.quantity}"
            )

        self._update_portfolio_value(timestamp)

    def update_prices(self, price_data: Dict[str, float], timestamp: datetime) -> None:
        """
        Update current prices for all positions.

        Args:
            price_data: Dictionary of symbol -> current price
            timestamp: Price update timestamp
        """
        for symbol, position in self.positions.items():
            if symbol in price_data:
                old_price = position.current_price
                position.current_price = price_data[symbol]
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = position.quantity * (
                    position.current_price - position.avg_price
                )
                position.last_update = timestamp

                logger.debug(
                    f"Price updated: {symbol} {old_price} -> {position.current_price}"
                )

        self._update_portfolio_value(timestamp)

    def _update_portfolio_value(self, timestamp: datetime) -> None:
        """Update total portfolio value and weights."""
        # Calculate total market value of positions
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = self.cash + total_market_value

        # Update position weights
        if self.total_value > 0:
            for position in self.positions.values():
                position.weight = position.market_value / self.total_value

        # Record portfolio value history
        self.value_history.append((timestamp, self.total_value))

        # Record weights history
        weights = {pos.symbol: pos.weight for pos in self.positions.values()}
        weights["CASH"] = self.cash / self.total_value if self.total_value > 0 else 1.0
        self.weights_history.append((timestamp, weights))

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.

        Returns:
            Dictionary containing portfolio analytics
        """
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())

        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "market_value": total_market_value,
            "total_pnl": total_unrealized_pnl + total_realized_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "total_return": (self.total_value - self.initial_capital)
            / self.initial_capital,
            "num_positions": len(self.positions),
            "cash_weight": (
                self.cash / self.total_value if self.total_value > 0 else 1.0
            ),
            "largest_position": max(
                (pos.weight for pos in self.positions.values()), default=0.0
            ),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "weight": pos.weight,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "return_pct": pos.return_pct,
                }
                for symbol, pos in self.positions.items()
            },
        }

    def calculate_portfolio_risk(
        self, returns_data: pd.DataFrame, lookback_days: int = 252
    ) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.

        Args:
            returns_data: DataFrame with returns data for portfolio assets
            lookback_days: Number of days to look back for calculations

        Returns:
            Dictionary containing risk metrics
        """
        if len(self.positions) == 0:
            return {
                "portfolio_volatility": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "beta": 0.0,
                "tracking_error": 0.0,
            }

        # Get portfolio weights
        weights = np.array([pos.weight for pos in self.positions.values()])
        symbols = list(self.positions.keys())

        # Filter returns data for portfolio assets
        portfolio_returns = returns_data[symbols].tail(lookback_days)

        if len(portfolio_returns) == 0:
            return {
                "portfolio_volatility": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "beta": 0.0,
                "tracking_error": 0.0,
            }

        # Calculate covariance matrix
        cov_matrix = portfolio_returns.cov().values

        # Portfolio volatility
        portfolio_volatility = np.sqrt(
            np.dot(weights, np.dot(cov_matrix, weights))
        ) * np.sqrt(252)

        # Portfolio returns for VaR calculation
        portfolio_returns_series = (portfolio_returns * weights).sum(axis=1)

        # VaR and CVaR
        var_95 = -np.percentile(portfolio_returns_series, 5)
        tail_returns = portfolio_returns_series[portfolio_returns_series <= -var_95]
        cvar_95 = -tail_returns.mean() if len(tail_returns) > 0 else 0.0

        # Beta and tracking error (if benchmark available)
        beta = 0.0
        tracking_error = 0.0

        if self.benchmark_returns is not None:
            aligned_data = pd.DataFrame(
                {
                    "portfolio": portfolio_returns_series,
                    "benchmark": self.benchmark_returns,
                }
            ).dropna()

            if len(aligned_data) > 1:
                beta = aligned_data.cov().iloc[0, 1] / aligned_data["benchmark"].var()
                excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]
                tracking_error = excess_returns.std() * np.sqrt(252)

        risk_metrics = {
            "portfolio_volatility": portfolio_volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "beta": beta,
            "tracking_error": tracking_error,
        }

        # Record VaR history
        self.var_history.append((datetime.now(), var_95))

        return risk_metrics

    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_model: RiskModel = RiskModel.MAXIMUM_SHARPE,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation using specified risk model.

        Args:
            expected_returns: Expected returns for assets
            cov_matrix: Covariance matrix of asset returns
            risk_model: Risk model to use for optimization
            target_return: Target return (for specific optimization types)
            target_volatility: Target volatility (for specific optimization types)

        Returns:
            Dictionary of optimized weights
        """
        n_assets = len(expected_returns)

        if n_assets == 0:
            return {}

        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]

        # Bounds (non-negative weights, respect max weight constraint)
        bounds = [(0.0, self.constraints.max_weight_per_asset) for _ in range(n_assets)]

        if risk_model == RiskModel.EQUAL_WEIGHT:
            # Equal weight portfolio
            weights = np.array([1.0 / n_assets] * n_assets)

        elif risk_model == RiskModel.MINIMUM_VARIANCE:
            # Minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix.values, weights))

            result = optimize.minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            weights = result.x if result.success else x0

        elif risk_model == RiskModel.MAXIMUM_SHARPE:
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns.values)
                portfolio_volatility = np.sqrt(
                    np.dot(weights, np.dot(cov_matrix.values, weights))
                )
                if portfolio_volatility == 0:
                    return -np.inf
                return -(portfolio_return - self.risk_free_rate) / portfolio_volatility

            result = optimize.minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            weights = result.x if result.success else x0

        elif risk_model == RiskModel.RISK_PARITY:
            # Risk parity (equal risk contribution)
            def risk_contribution(weights):
                portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
                marginal_contrib = np.dot(cov_matrix.values, weights)
                contrib = weights * marginal_contrib / portfolio_variance
                return contrib

            def objective(weights):
                contrib = risk_contribution(weights)
                target_contrib = 1.0 / n_assets
                return np.sum((contrib - target_contrib) ** 2)

            result = optimize.minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            weights = result.x if result.success else x0

        else:
            # Default to equal weight
            weights = np.array([1.0 / n_assets] * n_assets)

        # Create weights dictionary
        optimized_weights = {
            symbol: weight for symbol, weight in zip(expected_returns.index, weights)
        }

        logger.info(f"Portfolio optimized using {risk_model.value} model")

        return optimized_weights

    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        timestamp: datetime,
        min_trade_value: float = 1000.0,
    ) -> List[Dict[str, Any]]:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: Target allocation weights
            current_prices: Current asset prices
            timestamp: Rebalancing timestamp
            min_trade_value: Minimum trade value to execute

        Returns:
            List of trades to execute
        """
        trades = []

        # Calculate target dollar amounts
        target_values = {
            symbol: weight * self.total_value
            for symbol, weight in target_weights.items()
        }

        # Calculate current dollar amounts
        current_values = {
            pos.symbol: pos.market_value for pos in self.positions.values()
        }

        # Calculate required trades
        for symbol, target_value in target_values.items():
            current_value = current_values.get(symbol, 0.0)
            trade_value = target_value - current_value

            if abs(trade_value) > min_trade_value and symbol in current_prices:
                trade_quantity = trade_value / current_prices[symbol]

                trades.append(
                    {
                        "symbol": symbol,
                        "side": "BUY" if trade_quantity > 0 else "SELL",
                        "quantity": abs(trade_quantity),
                        "price": current_prices[symbol],
                        "value": trade_value,
                        "timestamp": timestamp,
                    }
                )

        # Record rebalancing
        self.rebalance_history.append((timestamp, target_weights))

        logger.info(f"Rebalancing generated {len(trades)} trades")

        return trades

    def check_risk_limits(self) -> List[str]:
        """
        Check portfolio against risk limits.

        Returns:
            List of risk limit violations
        """
        violations = []

        # Check position concentration
        for position in self.positions.values():
            if position.weight > self.constraints.max_weight_per_asset:
                violations.append(
                    f"Position {position.symbol} weight {position.weight:.2%} "
                    f"exceeds limit {self.constraints.max_weight_per_asset:.2%}"
                )

        # Check leverage
        total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        leverage = total_exposure / self.total_value if self.total_value > 0 else 0

        if leverage > self.constraints.max_leverage:
            violations.append(
                f"Portfolio leverage {leverage:.2f} exceeds limit {self.constraints.max_leverage:.2f}"
            )

        # Check VaR limit (if VaR history available)
        if self.var_history and len(self.var_history) > 0:
            current_var = self.var_history[-1][1]
            if current_var > self.constraints.max_var_95:
                violations.append(
                    f"Portfolio VaR {current_var:.2%} exceeds limit {self.constraints.max_var_95:.2%}"
                )

        return violations

    def get_performance_attribution(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate performance attribution analysis.

        Args:
            benchmark_returns: Benchmark returns for comparison
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Performance attribution breakdown
        """
        if len(self.value_history) < 2:
            return {
                "total_return": 0.0,
                "active_return": 0.0,
                "asset_allocation_effect": 0.0,
                "security_selection_effect": 0.0,
            }

        # Calculate portfolio returns
        values = pd.Series(
            [value for _, value in self.value_history],
            index=[date for date, _ in self.value_history],
        )

        portfolio_returns = values.pct_change().dropna()

        if start_date:
            portfolio_returns = portfolio_returns[portfolio_returns.index >= start_date]
        if end_date:
            portfolio_returns = portfolio_returns[portfolio_returns.index <= end_date]

        total_return = portfolio_returns.sum()

        attribution = {
            "total_return": total_return,
            "active_return": 0.0,
            "asset_allocation_effect": 0.0,
            "security_selection_effect": 0.0,
            "interaction_effect": 0.0,
        }

        # If benchmark available, calculate active return
        if benchmark_returns is not None:
            aligned_returns = pd.DataFrame(
                {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
            ).dropna()

            if len(aligned_returns) > 0:
                active_returns = (
                    aligned_returns["portfolio"] - aligned_returns["benchmark"]
                )
                attribution["active_return"] = active_returns.sum()

        return attribution

    def export_portfolio_data(self) -> Dict[str, Any]:
        """
        Export comprehensive portfolio data for analysis.

        Returns:
            Complete portfolio dataset
        """
        return {
            "summary": self.get_portfolio_summary(),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "weight": pos.weight,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "total_pnl": pos.total_pnl,
                    "return_pct": pos.return_pct,
                    "entry_date": pos.entry_date.isoformat(),
                    "last_update": pos.last_update.isoformat(),
                }
                for symbol, pos in self.positions.items()
            },
            "history": {
                "values": [
                    {"date": date.isoformat(), "value": value}
                    for date, value in self.value_history
                ],
                "weights": [
                    {"date": date.isoformat(), "weights": weights}
                    for date, weights in self.weights_history
                ],
                "rebalances": [
                    {"date": date.isoformat(), "weights": weights}
                    for date, weights in self.rebalance_history
                ],
            },
            "constraints": {
                "max_weight_per_asset": self.constraints.max_weight_per_asset,
                "max_leverage": self.constraints.max_leverage,
                "max_var_95": self.constraints.max_var_95,
                "max_drawdown_limit": self.constraints.max_drawdown_limit,
            },
        }
