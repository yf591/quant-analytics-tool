"""
Backtesting Framework

A comprehensive backtesting engine for quantitative trading strategies,
following methodologies from "Advances in Financial Machine Learning" (AFML).

Components:
- engine: Core backtesting engine with event-driven simulation
- strategies: Strategy framework with base classes and implementations
- metrics: Performance metrics calculation and analysis
- portfolio: Portfolio management and risk control
- execution: Advanced trade execution simulation and cost modeling
"""

from .engine import (
    BacktestEngine,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Trade,
    Position,
)

from .strategies import (
    BaseStrategy,
    Signal,
    SignalType,
    BuyAndHoldStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
)

from .advanced_strategies import (
    ModelBasedStrategy,
    MultiAssetStrategy,
)

from .metrics import (
    PerformanceMetrics,
    DrawdownMetrics,
    PerformanceCalculator,
    create_performance_report,
)

from .portfolio import (
    Portfolio,
    PortfolioPosition,
    PortfolioConstraints,
    AllocationTarget,
    RebalanceFrequency,
    RiskModel,
)

from .execution import (
    ExecutionAlgorithm,
    VenueType,
    MarketData,
    ExecutionInstruction,
    ExecutionReport,
    TransactionCostModel,
    ExecutionSimulator,
    create_execution_summary,
)

__all__ = [
    # Engine components
    "BacktestEngine",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Trade",
    "Position",
    # Strategy components
    "BaseStrategy",
    "Signal",
    "SignalType",
    "BuyAndHoldStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    # Advanced strategy components
    "ModelBasedStrategy",
    "MultiAssetStrategy",
    # Metrics components
    "PerformanceMetrics",
    "DrawdownMetrics",
    "PerformanceCalculator",
    "create_performance_report",
    # Portfolio components
    "Portfolio",
    "PortfolioPosition",
    "PortfolioConstraints",
    "AllocationTarget",
    "RebalanceFrequency",
    "RiskModel",
    # Execution components
    "ExecutionAlgorithm",
    "VenueType",
    "MarketData",
    "ExecutionInstruction",
    "ExecutionReport",
    "TransactionCostModel",
    "ExecutionSimulator",
    "create_execution_summary",
]

__version__ = "1.0.0"
__author__ = "Quant Analytics Tool"
