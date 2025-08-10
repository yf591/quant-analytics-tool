"""
Trade Execution Module

This module provides advanced trade execution capabilities for backtesting,
following methodologies from "Advances in Financial Machine Learning" (AFML),
particularly Chapter 19: Microstructure Features.

Key Features:
- Market impact modeling and transaction cost analysis
- Execution algorithms (TWAP, VWAP, Implementation Shortfall)
- Order book simulation and liquidity modeling
- Slippage and transaction cost optimization
- Multi-venue execution simulation
- Latency and timing effects modeling

Author: Quantitative Analysis Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import logging
from scipy import stats
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Trade execution algorithm types."""

    MARKET = "market"  # Immediate market order
    LIMIT = "limit"  # Limit order with price
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    IMPLEMENTATION_SHORTFALL = (
        "implementation_shortfall"  # Minimize implementation shortfall
    )
    PARTICIPATION_RATE = "participation_rate"  # Target participation rate
    ARRIVAL_PRICE = "arrival_price"  # Minimize distance from arrival price


class VenueType(Enum):
    """Trading venue types."""

    PRIMARY_EXCHANGE = "primary_exchange"
    DARK_POOL = "dark_pool"
    ELECTRONIC_ECN = "electronic_ecn"
    CROSSING_NETWORK = "crossing_network"
    RETAIL_BROKER = "retail_broker"


@dataclass
class MarketData:
    """
    Market microstructure data for execution modeling.

    Based on AFML Chapter 19 microstructure features.
    """

    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float

    # Microstructure features
    bid_ask_spread: float = field(init=False)
    mid_price: float = field(init=False)
    microstructure_noise: float = field(init=False)
    order_flow_imbalance: float = field(init=False)

    def __post_init__(self):
        """Calculate derived microstructure features."""
        self.bid_ask_spread = self.ask_price - self.bid_price
        self.mid_price = (self.bid_price + self.ask_price) / 2.0

        # Microstructure noise (price deviation from efficient price)
        self.microstructure_noise = (
            abs(self.last_price - self.mid_price) / self.mid_price
        )

        # Order flow imbalance
        total_size = self.bid_size + self.ask_size
        if total_size > 0:
            self.order_flow_imbalance = (self.bid_size - self.ask_size) / total_size
        else:
            self.order_flow_imbalance = 0.0


@dataclass
class ExecutionInstruction:
    """
    Trade execution instruction with execution parameters.
    """

    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    algorithm: ExecutionAlgorithm
    urgency: float = 0.5  # 0 = patient, 1 = urgent

    # Algorithm-specific parameters
    limit_price: Optional[float] = None
    time_horizon: Optional[timedelta] = None
    participation_rate: Optional[float] = None
    target_volume: Optional[float] = None

    # Risk parameters
    max_slippage_bps: Optional[float] = None
    max_market_impact_bps: Optional[float] = None

    # Venue preferences
    preferred_venues: List[VenueType] = field(default_factory=list)
    dark_pool_preference: float = 0.3  # Preference for dark pools

    def __post_init__(self):
        """Validate instruction parameters."""
        if self.quantity < 0:
            raise ValueError("Quantity must be non-negative")
        if self.side not in ["BUY", "SELL"]:
            raise ValueError("Side must be 'BUY' or 'SELL'")
        if not 0 <= self.urgency <= 1:
            raise ValueError("Urgency must be between 0 and 1")


@dataclass
class ExecutionReport:
    """
    Detailed execution report for trade analysis.
    """

    instruction_id: str
    symbol: str
    side: str
    requested_quantity: float
    executed_quantity: float
    average_price: float

    # Cost analysis
    arrival_price: float
    implementation_shortfall: float
    market_impact_bps: float
    timing_cost_bps: float
    commission: float
    total_cost_bps: float

    # Execution details
    num_fills: int
    execution_time: timedelta
    venue_breakdown: Dict[VenueType, float]

    # Performance metrics
    participation_rate_achieved: float
    price_improvement: float
    slippage_bps: float

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate as executed/requested quantity."""
        if self.requested_quantity == 0:
            return 0.0
        return self.executed_quantity / self.requested_quantity


class TransactionCostModel:
    """
    Advanced transaction cost model based on AFML methodologies.

    Models market impact, timing costs, and execution costs using
    empirical relationships and microstructure theory.
    """

    def __init__(
        self,
        permanent_impact_coeff: float = 0.1,
        temporary_impact_coeff: float = 0.5,
        participation_impact_coeff: float = 0.3,
        volatility_impact_coeff: float = 0.2,
    ):
        """
        Initialize transaction cost model.

        Args:
            permanent_impact_coeff: Coefficient for permanent market impact
            temporary_impact_coeff: Coefficient for temporary market impact
            participation_impact_coeff: Coefficient for participation rate impact
            volatility_impact_coeff: Coefficient for volatility impact
        """
        self.permanent_impact_coeff = permanent_impact_coeff
        self.temporary_impact_coeff = temporary_impact_coeff
        self.participation_impact_coeff = participation_impact_coeff
        self.volatility_impact_coeff = volatility_impact_coeff

    def calculate_market_impact(
        self,
        quantity: float,
        avg_daily_volume: float,
        volatility: float,
        participation_rate: float = 0.1,
    ) -> Dict[str, float]:
        """
        Calculate market impact components.

        Args:
            quantity: Trade quantity
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            participation_rate: Participation rate in market volume

        Returns:
            Dictionary with impact components in basis points
        """
        # Size ratio (trade size / daily volume)
        size_ratio = quantity / avg_daily_volume if avg_daily_volume > 0 else 0

        # Permanent impact (square root law)
        permanent_impact = (
            self.permanent_impact_coeff * np.sqrt(size_ratio) * volatility * 10000
        )

        # Temporary impact (linear in participation rate)
        temporary_impact = (
            self.temporary_impact_coeff * participation_rate * volatility * 10000
        )

        # Participation rate penalty
        participation_penalty = (
            self.participation_impact_coeff
            * (participation_rate**1.5)
            * volatility
            * 10000
        )

        # Volatility adjustment
        volatility_adjustment = (
            self.volatility_impact_coeff * volatility * np.sqrt(size_ratio) * 10000
        )

        return {
            "permanent_impact_bps": permanent_impact,
            "temporary_impact_bps": temporary_impact,
            "participation_penalty_bps": participation_penalty,
            "volatility_adjustment_bps": volatility_adjustment,
            "total_impact_bps": permanent_impact
            + temporary_impact
            + participation_penalty
            + volatility_adjustment,
        }

    def calculate_timing_cost(
        self,
        arrival_price: float,
        decision_price: float,
        market_drift: float,
        execution_time: timedelta,
    ) -> float:
        """
        Calculate timing cost due to market movement during execution.

        Args:
            arrival_price: Price when decision was made
            decision_price: Price when execution started
            market_drift: Expected market drift rate
            execution_time: Time taken for execution

        Returns:
            Timing cost in basis points
        """
        # Price drift during decision delay
        decision_delay_cost = (
            abs(decision_price - arrival_price) / arrival_price * 10000
        )

        # Market drift during execution
        execution_hours = execution_time.total_seconds() / 3600
        drift_cost = abs(market_drift) * execution_hours * 10000

        return decision_delay_cost + drift_cost

    def calculate_opportunity_cost(
        self, executed_quantity: float, target_quantity: float, price_move: float
    ) -> float:
        """
        Calculate opportunity cost from incomplete execution.

        Args:
            executed_quantity: Quantity actually executed
            target_quantity: Target quantity
            price_move: Price movement during execution period

        Returns:
            Opportunity cost in basis points
        """
        unfilled_quantity = target_quantity - executed_quantity
        if target_quantity == 0:
            return 0.0

        unfilled_ratio = unfilled_quantity / target_quantity
        return abs(unfilled_ratio * price_move) * 10000


class ExecutionSimulator:
    """
    Advanced execution simulator with microstructure modeling.

    Simulates realistic trade execution including:
    - Order book dynamics
    - Market impact
    - Venue routing
    - Latency effects
    - Liquidity constraints
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        latency_ms: float = 1.0,
        fill_probability: float = 0.95,
        dark_pool_fill_rate: float = 0.3,
    ):
        """
        Initialize execution simulator.

        Args:
            cost_model: Transaction cost model
            latency_ms: Network latency in milliseconds
            fill_probability: Probability of fill for limit orders
            dark_pool_fill_rate: Fill rate in dark pools
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.latency_ms = latency_ms
        self.fill_probability = fill_probability
        self.dark_pool_fill_rate = dark_pool_fill_rate

        # Execution state
        self.execution_reports: List[ExecutionReport] = []
        self.market_data_cache: Dict[str, MarketData] = {}

        logger.info("ExecutionSimulator initialized")

    def execute_instruction(
        self,
        instruction: ExecutionInstruction,
        market_data: MarketData,
        historical_data: Optional[pd.DataFrame] = None,
    ) -> ExecutionReport:
        """
        Execute a trade instruction and return detailed execution report.

        Args:
            instruction: Execution instruction
            market_data: Current market data
            historical_data: Historical data for modeling

        Returns:
            Execution report with detailed analysis
        """
        start_time = datetime.now()

        # Cache market data
        self.market_data_cache[instruction.symbol] = market_data

        # Choose execution strategy based on algorithm
        if instruction.algorithm == ExecutionAlgorithm.MARKET:
            report = self._execute_market_order(instruction, market_data)
        elif instruction.algorithm == ExecutionAlgorithm.LIMIT:
            report = self._execute_limit_order(instruction, market_data)
        elif instruction.algorithm == ExecutionAlgorithm.TWAP:
            report = self._execute_twap(instruction, market_data, historical_data)
        elif instruction.algorithm == ExecutionAlgorithm.VWAP:
            report = self._execute_vwap(instruction, market_data, historical_data)
        elif instruction.algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
            report = self._execute_implementation_shortfall(
                instruction, market_data, historical_data
            )
        else:
            # Default to market order
            report = self._execute_market_order(instruction, market_data)

        # Store execution report
        self.execution_reports.append(report)

        logger.info(
            f"Executed {instruction.symbol} {instruction.side} {instruction.quantity} "
            f"using {instruction.algorithm.value}: avg_price={report.average_price:.2f}, "
            f"impact={report.market_impact_bps:.1f}bps"
        )

        return report

    def _execute_market_order(
        self, instruction: ExecutionInstruction, market_data: MarketData
    ) -> ExecutionReport:
        """Execute immediate market order."""
        # Determine execution price based on side
        if instruction.side == "BUY":
            execution_price = market_data.ask_price
            available_liquidity = market_data.ask_size
        else:
            execution_price = market_data.bid_price
            available_liquidity = market_data.bid_size

        # Calculate market impact
        daily_volume = market_data.volume * 8  # Assume 8 hours trading day
        volatility = 0.02  # Default 2% daily volatility

        impact = self.cost_model.calculate_market_impact(
            instruction.quantity,
            daily_volume,
            volatility,
            participation_rate=min(instruction.quantity / daily_volume, 0.5),
        )

        # Apply market impact to price
        impact_direction = 1 if instruction.side == "BUY" else -1
        impacted_price = execution_price * (
            1 + impact_direction * impact["total_impact_bps"] / 10000
        )

        # Determine executed quantity (may be limited by liquidity)
        executed_quantity = min(instruction.quantity, available_liquidity)

        # Calculate costs
        arrival_price = market_data.mid_price
        implementation_shortfall = (
            abs(impacted_price - arrival_price) / arrival_price * 10000
        )
        slippage = abs(impacted_price - execution_price) / execution_price * 10000

        return ExecutionReport(
            instruction_id=f"{instruction.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=instruction.symbol,
            side=instruction.side,
            requested_quantity=instruction.quantity,
            executed_quantity=executed_quantity,
            average_price=impacted_price,
            arrival_price=arrival_price,
            implementation_shortfall=implementation_shortfall,
            market_impact_bps=impact["total_impact_bps"],
            timing_cost_bps=0.0,  # No timing cost for immediate execution
            commission=executed_quantity * 0.005,  # $0.005 per share
            total_cost_bps=implementation_shortfall + 5.0,  # Include commission
            num_fills=1,
            execution_time=timedelta(milliseconds=self.latency_ms),
            venue_breakdown={VenueType.PRIMARY_EXCHANGE: 1.0},
            participation_rate_achieved=instruction.quantity / daily_volume,
            price_improvement=0.0,
            slippage_bps=slippage,
        )

    def _execute_limit_order(
        self, instruction: ExecutionInstruction, market_data: MarketData
    ) -> ExecutionReport:
        """Execute limit order with fill probability."""
        if instruction.limit_price is None:
            raise ValueError("Limit price required for limit order")

        # Determine if order can be filled
        if instruction.side == "BUY":
            can_fill = instruction.limit_price >= market_data.ask_price
            reference_price = market_data.ask_price
        else:
            can_fill = instruction.limit_price <= market_data.bid_price
            reference_price = market_data.bid_price

        if can_fill:
            # Immediate fill at limit price
            executed_quantity = instruction.quantity
            execution_price = instruction.limit_price
            fill_probability = 1.0
        else:
            # Partial fill based on probability
            executed_quantity = instruction.quantity * self.fill_probability
            execution_price = instruction.limit_price
            fill_probability = self.fill_probability

        # Price improvement for limit orders
        price_improvement = (
            abs(execution_price - reference_price) / reference_price * 10000
        )
        if instruction.side == "BUY" and execution_price > reference_price:
            price_improvement = -price_improvement  # Negative improvement (worse price)
        elif instruction.side == "SELL" and execution_price < reference_price:
            price_improvement = -price_improvement

        arrival_price = market_data.mid_price
        implementation_shortfall = (
            abs(execution_price - arrival_price) / arrival_price * 10000
        )

        return ExecutionReport(
            instruction_id=f"{instruction.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=instruction.symbol,
            side=instruction.side,
            requested_quantity=instruction.quantity,
            executed_quantity=executed_quantity,
            average_price=execution_price,
            arrival_price=arrival_price,
            implementation_shortfall=implementation_shortfall,
            market_impact_bps=0.0,  # Minimal impact for limit orders
            timing_cost_bps=10.0 * (1 - fill_probability),  # Cost of delay
            commission=executed_quantity * 0.003,  # Lower commission for limit orders
            total_cost_bps=implementation_shortfall + 3.0,
            num_fills=1 if executed_quantity > 0 else 0,
            execution_time=timedelta(minutes=5),  # Average time for limit order fill
            venue_breakdown={
                VenueType.PRIMARY_EXCHANGE: 0.7,
                VenueType.ELECTRONIC_ECN: 0.3,
            },
            participation_rate_achieved=0.05,  # Conservative for limit orders
            price_improvement=price_improvement,
            slippage_bps=0.0,
        )

    def _execute_twap(
        self,
        instruction: ExecutionInstruction,
        market_data: MarketData,
        historical_data: Optional[pd.DataFrame],
    ) -> ExecutionReport:
        """Execute Time-Weighted Average Price algorithm."""
        time_horizon = instruction.time_horizon or timedelta(hours=1)
        num_slices = max(int(time_horizon.total_seconds() / 300), 1)  # 5-minute slices
        slice_size = instruction.quantity / num_slices

        # Simulate execution over time slices
        total_cost = 0.0
        total_quantity = 0.0
        weighted_price = 0.0

        for i in range(num_slices):
            # Simulate price evolution (random walk with drift)
            price_change = np.random.normal(0, 0.001)  # 0.1% std per slice
            current_price = market_data.mid_price * (1 + price_change * (i + 1))

            # Calculate slice impact
            daily_volume = market_data.volume * 8
            slice_impact = self.cost_model.calculate_market_impact(
                slice_size,
                daily_volume,
                0.02,  # 2% daily vol
                participation_rate=0.05,  # Conservative TWAP
            )

            slice_price = current_price * (1 + slice_impact["total_impact_bps"] / 10000)
            slice_cost = slice_impact["total_impact_bps"]

            weighted_price += slice_price * slice_size
            total_cost += slice_cost * slice_size
            total_quantity += slice_size

        average_price = (
            weighted_price / total_quantity
            if total_quantity > 0
            else market_data.mid_price
        )
        average_impact = total_cost / total_quantity if total_quantity > 0 else 0

        arrival_price = market_data.mid_price
        implementation_shortfall = (
            abs(average_price - arrival_price) / arrival_price * 10000
        )

        return ExecutionReport(
            instruction_id=f"{instruction.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=instruction.symbol,
            side=instruction.side,
            requested_quantity=instruction.quantity,
            executed_quantity=total_quantity,
            average_price=average_price,
            arrival_price=arrival_price,
            implementation_shortfall=implementation_shortfall,
            market_impact_bps=average_impact,
            timing_cost_bps=5.0,  # Timing cost due to delayed execution
            commission=total_quantity * 0.004,
            total_cost_bps=implementation_shortfall + average_impact + 5.0,
            num_fills=num_slices,
            execution_time=time_horizon,
            venue_breakdown={
                VenueType.PRIMARY_EXCHANGE: 0.6,
                VenueType.DARK_POOL: 0.3,
                VenueType.ELECTRONIC_ECN: 0.1,
            },
            participation_rate_achieved=0.05,
            price_improvement=2.0,  # TWAP often gets price improvement
            slippage_bps=average_impact,
        )

    def _execute_vwap(
        self,
        instruction: ExecutionInstruction,
        market_data: MarketData,
        historical_data: Optional[pd.DataFrame],
    ) -> ExecutionReport:
        """Execute Volume-Weighted Average Price algorithm."""
        # Use historical volume pattern or default pattern
        if historical_data is not None and "volume" in historical_data.columns:
            volume_pattern = historical_data["volume"].values[-10:]  # Last 10 periods
        else:
            # Default U-shaped volume pattern
            volume_pattern = np.array(
                [0.15, 0.12, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.12, 0.23]
            )

        volume_pattern = volume_pattern / volume_pattern.sum()  # Normalize

        # Simulate VWAP execution
        total_cost = 0.0
        total_quantity = 0.0
        weighted_price = 0.0

        for i, vol_weight in enumerate(volume_pattern):
            slice_size = instruction.quantity * vol_weight

            # Price evolution
            price_change = np.random.normal(0, 0.0008)
            current_price = market_data.mid_price * (1 + price_change * (i + 1))

            # VWAP gets better pricing due to volume matching
            participation_rate = vol_weight * 0.1  # Scale with volume

            slice_impact = self.cost_model.calculate_market_impact(
                slice_size,
                market_data.volume * 8,
                0.02,
                participation_rate=participation_rate,
            )

            # VWAP discount due to volume timing
            vwap_discount = vol_weight * 3.0  # Up to 3bps discount
            slice_price = current_price * (
                1 + (slice_impact["total_impact_bps"] - vwap_discount) / 10000
            )

            weighted_price += slice_price * slice_size
            total_cost += slice_impact["total_impact_bps"] * slice_size
            total_quantity += slice_size

        average_price = (
            weighted_price / total_quantity
            if total_quantity > 0
            else market_data.mid_price
        )
        average_impact = total_cost / total_quantity if total_quantity > 0 else 0

        arrival_price = market_data.mid_price
        implementation_shortfall = (
            abs(average_price - arrival_price) / arrival_price * 10000
        )

        return ExecutionReport(
            instruction_id=f"{instruction.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=instruction.symbol,
            side=instruction.side,
            requested_quantity=instruction.quantity,
            executed_quantity=total_quantity,
            average_price=average_price,
            arrival_price=arrival_price,
            implementation_shortfall=implementation_shortfall,
            market_impact_bps=average_impact,
            timing_cost_bps=3.0,
            commission=total_quantity * 0.0035,
            total_cost_bps=implementation_shortfall + average_impact + 3.0,
            num_fills=len(volume_pattern),
            execution_time=timedelta(hours=6),  # Full trading day
            venue_breakdown={
                VenueType.PRIMARY_EXCHANGE: 0.5,
                VenueType.DARK_POOL: 0.4,
                VenueType.ELECTRONIC_ECN: 0.1,
            },
            participation_rate_achieved=0.08,
            price_improvement=3.0,  # VWAP typically gets good pricing
            slippage_bps=max(0, average_impact - 2.0),  # Reduced slippage
        )

    def _execute_implementation_shortfall(
        self,
        instruction: ExecutionInstruction,
        market_data: MarketData,
        historical_data: Optional[pd.DataFrame],
    ) -> ExecutionReport:
        """Execute Implementation Shortfall algorithm (Almgren-Chriss model)."""
        # Implementation Shortfall optimizes trade-off between market impact and timing risk
        urgency = instruction.urgency
        time_horizon = instruction.time_horizon or timedelta(hours=2)

        # Almgren-Chriss optimal strategy parameters
        volatility = 0.02  # Daily volatility
        risk_aversion = 1e-6  # Risk aversion parameter
        temp_impact_coeff = 0.5
        perm_impact_coeff = 0.1

        # Calculate optimal execution rate
        kappa = np.sqrt(risk_aversion * volatility**2 / temp_impact_coeff)
        optimal_time = np.sqrt(instruction.quantity / (kappa * perm_impact_coeff))

        # Adjust based on urgency
        execution_time = optimal_time * (2 - urgency)  # More urgent = faster execution
        execution_time = min(execution_time, time_horizon.total_seconds() / 3600)

        # High urgency should result in much faster execution
        if urgency > 0.8:
            execution_time = min(
                execution_time, 0.5
            )  # Maximum 30 minutes for high urgency

        num_slices = max(int(execution_time * 4), 1)  # 15-minute slices
        slice_size = instruction.quantity / num_slices

        # Simulate optimal execution
        total_cost = 0.0
        total_quantity = 0.0
        weighted_price = 0.0

        for i in range(num_slices):
            # Adaptive execution rate
            remaining_time = execution_time - (i * execution_time / num_slices)
            adaptive_rate = max(0.02, urgency * 0.1 + 0.02)  # 2-12% participation

            # Price impact calculation
            price_change = np.random.normal(
                0, volatility / np.sqrt(252 * 24)
            )  # Hourly vol
            current_price = market_data.mid_price * (1 + price_change * (i + 1))

            slice_impact = self.cost_model.calculate_market_impact(
                slice_size,
                market_data.volume * 8,
                volatility,
                participation_rate=adaptive_rate,
            )

            # Implementation shortfall optimization reduces impact
            optimization_benefit = urgency * 5.0  # Up to 5bps benefit
            slice_price = current_price * (
                1 + (slice_impact["total_impact_bps"] - optimization_benefit) / 10000
            )

            weighted_price += slice_price * slice_size
            total_cost += slice_impact["total_impact_bps"] * slice_size
            total_quantity += slice_size

        average_price = (
            weighted_price / total_quantity
            if total_quantity > 0
            else market_data.mid_price
        )
        average_impact = total_cost / total_quantity if total_quantity > 0 else 0

        arrival_price = market_data.mid_price
        implementation_shortfall = (
            abs(average_price - arrival_price) / arrival_price * 10000
        )

        return ExecutionReport(
            instruction_id=f"{instruction.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=instruction.symbol,
            side=instruction.side,
            requested_quantity=instruction.quantity,
            executed_quantity=total_quantity,
            average_price=average_price,
            arrival_price=arrival_price,
            implementation_shortfall=implementation_shortfall,
            market_impact_bps=average_impact,
            timing_cost_bps=2.0
            * (1 - urgency),  # Lower timing cost with higher urgency
            commission=total_quantity * 0.0038,
            total_cost_bps=implementation_shortfall + average_impact + 2.0,
            num_fills=num_slices,
            execution_time=timedelta(hours=execution_time),
            venue_breakdown={
                VenueType.PRIMARY_EXCHANGE: 0.4,
                VenueType.DARK_POOL: 0.5,
                VenueType.ELECTRONIC_ECN: 0.1,
            },
            participation_rate_achieved=urgency * 0.08 + 0.02,
            price_improvement=urgency * 4.0,  # Better pricing with optimal execution
            slippage_bps=max(0, average_impact - urgency * 3.0),
        )

    def get_execution_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive execution analytics.

        Returns:
            Dictionary with execution performance metrics
        """
        if not self.execution_reports:
            return {
                "total_executions": 0,
                "average_impact_bps": 0.0,
                "average_slippage_bps": 0.0,
                "fill_rate": 0.0,
            }

        reports = self.execution_reports

        total_executions = len(reports)
        total_quantity_requested = sum(r.requested_quantity for r in reports)
        total_quantity_executed = sum(r.executed_quantity for r in reports)

        fill_rate = (
            total_quantity_executed / total_quantity_requested
            if total_quantity_requested > 0
            else 0
        )

        # Volume-weighted averages
        avg_impact = (
            sum(r.market_impact_bps * r.executed_quantity for r in reports)
            / total_quantity_executed
            if total_quantity_executed > 0
            else 0
        )
        avg_slippage = (
            sum(r.slippage_bps * r.executed_quantity for r in reports)
            / total_quantity_executed
            if total_quantity_executed > 0
            else 0
        )
        avg_implementation_shortfall = (
            sum(r.implementation_shortfall * r.executed_quantity for r in reports)
            / total_quantity_executed
            if total_quantity_executed > 0
            else 0
        )

        # Venue analysis
        venue_volumes = {}
        for report in reports:
            for venue, allocation in report.venue_breakdown.items():
                venue_volumes[venue] = (
                    venue_volumes.get(venue, 0) + allocation * report.executed_quantity
                )

        total_venue_volume = sum(venue_volumes.values())
        venue_breakdown = {
            venue: volume / total_venue_volume if total_venue_volume > 0 else 0
            for venue, volume in venue_volumes.items()
        }

        return {
            "total_executions": total_executions,
            "total_quantity_requested": total_quantity_requested,
            "total_quantity_executed": total_quantity_executed,
            "fill_rate": fill_rate,
            "average_impact_bps": avg_impact,
            "average_slippage_bps": avg_slippage,
            "average_implementation_shortfall_bps": avg_implementation_shortfall,
            "venue_breakdown": venue_breakdown,
            "algorithm_performance": self._analyze_algorithm_performance(),
        }

    def _analyze_algorithm_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by execution algorithm."""
        algorithm_stats = {}

        for report in self.execution_reports:
            # Extract algorithm from instruction (would need to store this)
            algorithm = "unknown"  # In practice, would store this in report

            if algorithm not in algorithm_stats:
                algorithm_stats[algorithm] = {
                    "count": 0,
                    "total_impact": 0.0,
                    "total_quantity": 0.0,
                    "total_shortfall": 0.0,
                }

            stats = algorithm_stats[algorithm]
            stats["count"] += 1
            stats["total_impact"] += report.market_impact_bps * report.executed_quantity
            stats["total_quantity"] += report.executed_quantity
            stats["total_shortfall"] += (
                report.implementation_shortfall * report.executed_quantity
            )

        # Calculate averages
        for algorithm, stats in algorithm_stats.items():
            if stats["total_quantity"] > 0:
                stats["avg_impact_bps"] = (
                    stats["total_impact"] / stats["total_quantity"]
                )
                stats["avg_shortfall_bps"] = (
                    stats["total_shortfall"] / stats["total_quantity"]
                )
            else:
                stats["avg_impact_bps"] = 0.0
                stats["avg_shortfall_bps"] = 0.0

        return algorithm_stats


def create_execution_summary(reports: List[ExecutionReport]) -> Dict[str, Any]:
    """
    Create comprehensive execution summary report.

    Args:
        reports: List of execution reports

    Returns:
        Summary with key metrics and analysis
    """
    if not reports:
        return {"message": "No execution reports available"}

    total_requested = sum(r.requested_quantity for r in reports)
    total_executed = sum(r.executed_quantity for r in reports)

    # Calculate weighted averages
    if total_executed > 0:
        avg_price = (
            sum(r.average_price * r.executed_quantity for r in reports) / total_executed
        )
        avg_impact = (
            sum(r.market_impact_bps * r.executed_quantity for r in reports)
            / total_executed
        )
        avg_slippage = (
            sum(r.slippage_bps * r.executed_quantity for r in reports) / total_executed
        )
        avg_commission_rate = sum(r.commission for r in reports) / total_executed
    else:
        avg_price = avg_impact = avg_slippage = avg_commission_rate = 0.0

    return {
        "summary": {
            "total_executions": len(reports),
            "total_requested_quantity": total_requested,
            "total_executed_quantity": total_executed,
            "fill_rate": (
                f"{(total_executed / total_requested * 100):.1f}%"
                if total_requested > 0
                else "0%"
            ),
            "average_execution_price": f"{avg_price:.2f}",
            "average_market_impact": f"{avg_impact:.1f} bps",
            "average_slippage": f"{avg_slippage:.1f} bps",
        },
        "cost_analysis": {
            "total_commission": sum(r.commission for r in reports),
            "average_commission_rate": f"{avg_commission_rate * 1000:.2f} mils",
            "total_market_impact_cost": avg_impact * total_executed / 10000 * avg_price,
            "total_implementation_shortfall": sum(
                r.implementation_shortfall
                * r.executed_quantity
                / 10000
                * r.average_price
                for r in reports
            ),
        },
        "execution_quality": {
            "price_improvement_instances": sum(
                1 for r in reports if r.price_improvement > 0
            ),
            "average_fill_time": f"{np.mean([r.execution_time.total_seconds() / 60 for r in reports]):.1f} minutes",
            "venue_diversification": len(
                set().union(*[r.venue_breakdown.keys() for r in reports])
            ),
        },
    }
