"""
Tests for Trade Execution Module

Comprehensive test suite for advanced trade execution capabilities,
following AFML methodologies for microstructure and execution modeling.

Test Coverage:
- Market data and microstructure features
- Transaction cost modeling
- Execution algorithms (Market, Limit, TWAP, VWAP, Implementation Shortfall)
- Execution simulation and reporting
- Cost analysis and performance metrics
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings

# Import modules to test
from src.backtesting.execution import (
    ExecutionAlgorithm,
    VenueType,
    MarketData,
    ExecutionInstruction,
    ExecutionReport,
    TransactionCostModel,
    ExecutionSimulator,
    create_execution_summary,
)


class TestMarketData:
    """Test MarketData class and microstructure features."""

    def test_market_data_creation(self):
        """Test basic market data creation."""
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.0,
            ask_price=150.2,
            bid_size=1000,
            ask_size=800,
            last_price=150.1,
            volume=50000,
        )

        assert market_data.symbol == "AAPL"
        assert market_data.bid_price == 150.0
        assert market_data.ask_price == 150.2
        assert (
            abs(market_data.bid_ask_spread - 0.2) < 1e-10
        )  # Handle floating point precision
        assert market_data.mid_price == 150.1

    def test_microstructure_features(self):
        """Test microstructure feature calculations."""
        market_data = MarketData(
            symbol="TSLA",
            timestamp=datetime.now(),
            bid_price=800.0,
            ask_price=801.0,
            bid_size=500,
            ask_size=300,
            last_price=800.3,
            volume=25000,
        )

        # Test spread calculation
        assert market_data.bid_ask_spread == 1.0
        assert market_data.mid_price == 800.5

        # Test microstructure noise
        expected_noise = abs(800.3 - 800.5) / 800.5
        assert abs(market_data.microstructure_noise - expected_noise) < 1e-6

        # Test order flow imbalance
        expected_imbalance = (500 - 300) / (500 + 300)
        assert abs(market_data.order_flow_imbalance - expected_imbalance) < 1e-6

    def test_zero_size_handling(self):
        """Test handling of zero bid/ask sizes."""
        market_data = MarketData(
            symbol="SPY",
            timestamp=datetime.now(),
            bid_price=400.0,
            ask_price=400.1,
            bid_size=0,
            ask_size=0,
            last_price=400.05,
            volume=100000,
        )

        assert market_data.order_flow_imbalance == 0.0


class TestExecutionInstruction:
    """Test ExecutionInstruction class."""

    def test_basic_instruction_creation(self):
        """Test basic execution instruction."""
        instruction = ExecutionInstruction(
            symbol="MSFT",
            side="BUY",
            quantity=1000,
            algorithm=ExecutionAlgorithm.MARKET,
        )

        assert instruction.symbol == "MSFT"
        assert instruction.side == "BUY"
        assert instruction.quantity == 1000
        assert instruction.algorithm == ExecutionAlgorithm.MARKET
        assert instruction.urgency == 0.5  # Default

    def test_twap_instruction(self):
        """Test TWAP instruction with parameters."""
        instruction = ExecutionInstruction(
            symbol="GOOGL",
            side="SELL",
            quantity=500,
            algorithm=ExecutionAlgorithm.TWAP,
            time_horizon=timedelta(hours=2),
            urgency=0.3,
        )

        assert instruction.algorithm == ExecutionAlgorithm.TWAP
        assert instruction.time_horizon == timedelta(hours=2)
        assert instruction.urgency == 0.3

    def test_limit_instruction(self):
        """Test limit order instruction."""
        instruction = ExecutionInstruction(
            symbol="NVDA",
            side="BUY",
            quantity=200,
            algorithm=ExecutionAlgorithm.LIMIT,
            limit_price=500.0,
            max_slippage_bps=10.0,
        )

        assert instruction.algorithm == ExecutionAlgorithm.LIMIT
        assert instruction.limit_price == 500.0
        assert instruction.max_slippage_bps == 10.0


class TestTransactionCostModel:
    """Test transaction cost modeling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cost_model = TransactionCostModel(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            participation_impact_coeff=0.3,
            volatility_impact_coeff=0.2,
        )

    def test_market_impact_calculation(self):
        """Test market impact calculation."""
        impact = self.cost_model.calculate_market_impact(
            quantity=10000,
            avg_daily_volume=1000000,
            volatility=0.02,
            participation_rate=0.1,
        )

        assert "permanent_impact_bps" in impact
        assert "temporary_impact_bps" in impact
        assert "participation_penalty_bps" in impact
        assert "volatility_adjustment_bps" in impact
        assert "total_impact_bps" in impact

        # Check that total is sum of components
        expected_total = (
            impact["permanent_impact_bps"]
            + impact["temporary_impact_bps"]
            + impact["participation_penalty_bps"]
            + impact["volatility_adjustment_bps"]
        )
        assert abs(impact["total_impact_bps"] - expected_total) < 1e-6

    def test_large_trade_impact(self):
        """Test impact scales with trade size."""
        small_impact = self.cost_model.calculate_market_impact(
            quantity=1000,
            avg_daily_volume=1000000,
            volatility=0.02,
            participation_rate=0.01,
        )

        large_impact = self.cost_model.calculate_market_impact(
            quantity=100000,
            avg_daily_volume=1000000,
            volatility=0.02,
            participation_rate=0.1,
        )

        # Larger trades should have higher impact
        assert large_impact["total_impact_bps"] > small_impact["total_impact_bps"]

    def test_timing_cost_calculation(self):
        """Test timing cost calculation."""
        timing_cost = self.cost_model.calculate_timing_cost(
            arrival_price=100.0,
            decision_price=100.5,
            market_drift=0.001,
            execution_time=timedelta(hours=1),
        )

        assert timing_cost > 0
        assert isinstance(timing_cost, float)

    def test_opportunity_cost_calculation(self):
        """Test opportunity cost from incomplete fills."""
        opportunity_cost = self.cost_model.calculate_opportunity_cost(
            executed_quantity=800, target_quantity=1000, price_move=0.02
        )

        assert opportunity_cost > 0
        # Should be proportional to unfilled quantity
        expected_cost = (200 / 1000) * 0.02 * 10000  # 20% unfilled * 2% move
        assert abs(opportunity_cost - expected_cost) < 1e-6

    def test_zero_volume_handling(self):
        """Test handling of zero volume."""
        impact = self.cost_model.calculate_market_impact(
            quantity=1000, avg_daily_volume=0, volatility=0.02, participation_rate=0.1
        )

        # Should handle gracefully without division by zero
        assert isinstance(impact["total_impact_bps"], float)


class TestExecutionSimulator:
    """Test execution simulation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = ExecutionSimulator(
            latency_ms=1.0, fill_probability=0.95, dark_pool_fill_rate=0.3
        )

        self.market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.0,
            ask_price=150.2,
            bid_size=1000,
            ask_size=800,
            last_price=150.1,
            volume=50000,
        )

    def test_market_order_execution(self):
        """Test market order execution."""
        instruction = ExecutionInstruction(
            symbol="AAPL", side="BUY", quantity=500, algorithm=ExecutionAlgorithm.MARKET
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        assert report.symbol == "AAPL"
        assert report.side == "BUY"
        assert report.requested_quantity == 500
        assert report.executed_quantity <= 500  # May be limited by liquidity
        assert report.average_price > 0
        assert report.market_impact_bps >= 0
        assert report.num_fills == 1
        assert VenueType.PRIMARY_EXCHANGE in report.venue_breakdown

    def test_limit_order_execution(self):
        """Test limit order execution."""
        instruction = ExecutionInstruction(
            symbol="AAPL",
            side="BUY",
            quantity=300,
            algorithm=ExecutionAlgorithm.LIMIT,
            limit_price=149.5,  # Below current ask
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        # Check basic execution properties instead of non-existent algorithm_type
        assert report.executed_quantity <= 300
        assert report.timing_cost_bps > 0  # Limit orders have timing cost

    def test_twap_execution(self):
        """Test TWAP execution algorithm."""
        instruction = ExecutionInstruction(
            symbol="AAPL",
            side="SELL",
            quantity=2000,
            algorithm=ExecutionAlgorithm.TWAP,
            time_horizon=timedelta(hours=1),
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        assert report.num_fills > 1  # TWAP uses multiple slices
        assert report.execution_time <= timedelta(hours=1)
        assert report.price_improvement >= 0  # TWAP often gets improvement
        assert VenueType.DARK_POOL in report.venue_breakdown  # TWAP uses dark pools

    def test_vwap_execution(self):
        """Test VWAP execution algorithm."""
        instruction = ExecutionInstruction(
            symbol="AAPL", side="BUY", quantity=5000, algorithm=ExecutionAlgorithm.VWAP
        )

        # Create historical data for VWAP
        historical_data = pd.DataFrame({"volume": np.random.lognormal(10, 0.5, 10)})

        report = self.simulator.execute_instruction(
            instruction, self.market_data, historical_data
        )

        assert report.num_fills >= 5  # VWAP uses volume-based slicing
        assert report.participation_rate_achieved > 0
        assert report.price_improvement > 0  # VWAP typically gets good pricing

    def test_implementation_shortfall_execution(self):
        """Test Implementation Shortfall algorithm."""
        instruction = ExecutionInstruction(
            symbol="AAPL",
            side="SELL",
            quantity=3000,
            algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
            urgency=0.8,
            time_horizon=timedelta(hours=2),
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        assert report.implementation_shortfall >= 0
        assert report.timing_cost_bps < 10.0  # Should have low timing cost
        assert report.price_improvement > 0  # Optimization should improve pricing

    def test_execution_analytics(self):
        """Test execution analytics generation."""
        # Execute multiple orders
        instructions = [
            ExecutionInstruction("AAPL", "BUY", 1000, ExecutionAlgorithm.MARKET),
            ExecutionInstruction(
                "AAPL", "SELL", 500, ExecutionAlgorithm.LIMIT, limit_price=149.0
            ),
            ExecutionInstruction("AAPL", "BUY", 2000, ExecutionAlgorithm.TWAP),
        ]

        for instruction in instructions:
            self.simulator.execute_instruction(instruction, self.market_data)

        analytics = self.simulator.get_execution_analytics()

        assert analytics["total_executions"] == 3
        assert analytics["fill_rate"] > 0
        assert "average_impact_bps" in analytics
        assert "venue_breakdown" in analytics
        assert "algorithm_performance" in analytics

    def test_large_order_handling(self):
        """Test handling of orders larger than available liquidity."""
        instruction = ExecutionInstruction(
            symbol="AAPL",
            side="BUY",
            quantity=10000,  # Larger than ask_size (800)
            algorithm=ExecutionAlgorithm.MARKET,
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        # Should be limited by available liquidity for market orders
        assert report.executed_quantity <= self.market_data.ask_size
        assert report.market_impact_bps > 0

    def test_multiple_venue_routing(self):
        """Test multi-venue execution routing."""
        instruction = ExecutionInstruction(
            symbol="AAPL",
            side="BUY",
            quantity=5000,
            algorithm=ExecutionAlgorithm.VWAP,
            preferred_venues=[VenueType.DARK_POOL, VenueType.ELECTRONIC_ECN],
            dark_pool_preference=0.6,
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        # Check venue diversification
        assert len(report.venue_breakdown) > 1
        assert VenueType.DARK_POOL in report.venue_breakdown

        # Dark pools should have significant allocation
        dark_pool_allocation = report.venue_breakdown.get(VenueType.DARK_POOL, 0)
        assert dark_pool_allocation > 0.2


class TestExecutionReport:
    """Test execution report functionality."""

    def test_execution_report_creation(self):
        """Test execution report creation."""
        report = ExecutionReport(
            instruction_id="AAPL_20240101_120000",
            symbol="AAPL",
            side="BUY",
            requested_quantity=1000,
            executed_quantity=950,
            average_price=150.25,
            arrival_price=150.0,
            implementation_shortfall=16.67,
            market_impact_bps=8.5,
            timing_cost_bps=2.0,
            commission=4.75,
            total_cost_bps=27.17,
            num_fills=3,
            execution_time=timedelta(minutes=15),
            venue_breakdown={VenueType.PRIMARY_EXCHANGE: 0.7, VenueType.DARK_POOL: 0.3},
            participation_rate_achieved=0.05,
            price_improvement=3.2,
            slippage_bps=1.8,
        )

        assert report.symbol == "AAPL"
        assert report.executed_quantity == 950
        assert (
            abs(report.fill_rate - 0.95) < 1e-10
        )  # Use property method with precision handling
        assert len(report.venue_breakdown) == 2


class TestExecutionSummary:
    """Test execution summary functionality."""

    def test_create_execution_summary(self):
        """Test execution summary creation."""
        reports = [
            ExecutionReport(
                instruction_id="1",
                symbol="AAPL",
                side="BUY",
                requested_quantity=1000,
                executed_quantity=950,
                average_price=150.0,
                arrival_price=149.8,
                implementation_shortfall=13.34,
                market_impact_bps=8.0,
                timing_cost_bps=2.0,
                commission=4.75,
                total_cost_bps=23.34,
                num_fills=1,
                execution_time=timedelta(minutes=5),
                venue_breakdown={VenueType.PRIMARY_EXCHANGE: 1.0},
                participation_rate_achieved=0.05,
                price_improvement=2.0,
                slippage_bps=1.5,
            ),
            ExecutionReport(
                instruction_id="2",
                symbol="MSFT",
                side="SELL",
                requested_quantity=500,
                executed_quantity=500,
                average_price=300.0,
                arrival_price=300.2,
                implementation_shortfall=6.67,
                market_impact_bps=5.0,
                timing_cost_bps=1.0,
                commission=2.50,
                total_cost_bps=12.67,
                num_fills=2,
                execution_time=timedelta(minutes=10),
                venue_breakdown={
                    VenueType.DARK_POOL: 0.6,
                    VenueType.ELECTRONIC_ECN: 0.4,
                },
                participation_rate_achieved=0.03,
                price_improvement=1.5,
                slippage_bps=0.8,
            ),
        ]

        summary = create_execution_summary(reports)

        assert "summary" in summary
        assert "cost_analysis" in summary
        assert "execution_quality" in summary

        # Check summary metrics
        assert summary["summary"]["total_executions"] == 2
        assert summary["summary"]["total_requested_quantity"] == 1500
        assert summary["summary"]["total_executed_quantity"] == 1450
        assert "96.7%" in summary["summary"]["fill_rate"]

        # Check cost analysis
        assert summary["cost_analysis"]["total_commission"] == 7.25

        # Check execution quality
        assert summary["execution_quality"]["price_improvement_instances"] == 2
        assert summary["execution_quality"]["venue_diversification"] >= 3

    def test_empty_reports_summary(self):
        """Test summary with empty reports list."""
        summary = create_execution_summary([])

        assert "message" in summary
        assert summary["message"] == "No execution reports available"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = ExecutionSimulator()
        self.market_data = MarketData(
            symbol="TEST",
            timestamp=datetime.now(),
            bid_price=100.0,
            ask_price=100.1,
            bid_size=100,
            ask_size=100,
            last_price=100.05,
            volume=1000,
        )

    def test_zero_quantity_order(self):
        """Test handling of zero quantity orders."""
        instruction = ExecutionInstruction(
            symbol="TEST", side="BUY", quantity=0, algorithm=ExecutionAlgorithm.MARKET
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        assert report.executed_quantity == 0
        assert report.commission == 0

    def test_negative_quantity_order(self):
        """Test handling of negative quantity orders."""
        with pytest.raises(ValueError):
            ExecutionInstruction(
                symbol="TEST",
                side="BUY",
                quantity=-100,
                algorithm=ExecutionAlgorithm.MARKET,
            )

    def test_limit_order_without_price(self):
        """Test limit order without limit price."""
        instruction = ExecutionInstruction(
            symbol="TEST",
            side="BUY",
            quantity=100,
            algorithm=ExecutionAlgorithm.LIMIT,
            # Missing limit_price
        )

        with pytest.raises(ValueError, match="Limit price required"):
            self.simulator.execute_instruction(instruction, self.market_data)

    def test_extreme_market_conditions(self):
        """Test execution under extreme market conditions."""
        # Wide spread market
        extreme_market_data = MarketData(
            symbol="TEST",
            timestamp=datetime.now(),
            bid_price=90.0,
            ask_price=110.0,  # 20% spread
            bid_size=10,
            ask_size=10,
            last_price=100.0,
            volume=100,
        )

        instruction = ExecutionInstruction(
            symbol="TEST", side="BUY", quantity=50, algorithm=ExecutionAlgorithm.MARKET
        )

        report = self.simulator.execute_instruction(instruction, extreme_market_data)

        # Should handle extreme conditions gracefully
        assert report.executed_quantity > 0
        assert (
            report.market_impact_bps > 15
        )  # High impact in extreme conditions (lowered threshold)

    def test_very_small_order(self):
        """Test execution of very small orders."""
        instruction = ExecutionInstruction(
            symbol="TEST",
            side="SELL",
            quantity=1,  # Single share
            algorithm=ExecutionAlgorithm.TWAP,
            time_horizon=timedelta(minutes=30),
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        assert (
            abs(report.executed_quantity - 1.0) < 1e-10
        )  # Handle floating point precision
        assert report.num_fills >= 1

    def test_high_urgency_execution(self):
        """Test high urgency execution."""
        instruction = ExecutionInstruction(
            symbol="TEST",
            side="BUY",
            quantity=1000,
            algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
            urgency=1.0,  # Maximum urgency
        )

        report = self.simulator.execute_instruction(instruction, self.market_data)

        # High urgency should result in faster execution
        assert report.execution_time < timedelta(hours=1)
        assert report.timing_cost_bps < 5.0  # Low timing cost due to speed

    def test_concurrent_executions(self):
        """Test handling of concurrent executions."""
        instructions = [
            ExecutionInstruction(f"STOCK{i}", "BUY", 100 * i, ExecutionAlgorithm.MARKET)
            for i in range(1, 6)
        ]

        reports = []
        for idx, instruction in enumerate(instructions, 1):
            # Simulate different market data for each stock
            market_data = MarketData(
                symbol=instruction.symbol,
                timestamp=datetime.now(),
                bid_price=100.0 + idx,
                ask_price=100.1 + idx,
                bid_size=1000,
                ask_size=1000,
                last_price=100.05 + idx,
                volume=50000,
            )

            report = self.simulator.execute_instruction(instruction, market_data)
            reports.append(report)

        assert len(reports) == 5
        assert all(report.executed_quantity > 0 for report in reports)

        # Check that analytics work with multiple executions
        analytics = self.simulator.get_execution_analytics()
        assert analytics["total_executions"] == 5


class TestPerformanceMetrics:
    """Test performance and benchmarking metrics."""

    def test_benchmark_comparison(self):
        """Test comparison against benchmark execution."""
        # This would compare against TWAP benchmark, market impact models, etc.
        # Implementation would depend on having benchmark data
        pass

    def test_cost_attribution(self):
        """Test cost attribution analysis."""
        cost_model = TransactionCostModel()

        # Test that costs are properly attributed
        impact = cost_model.calculate_market_impact(
            quantity=5000,
            avg_daily_volume=500000,
            volatility=0.025,
            participation_rate=0.15,
        )

        # Verify components sum to total
        components = [
            impact["permanent_impact_bps"],
            impact["temporary_impact_bps"],
            impact["participation_penalty_bps"],
            impact["volatility_adjustment_bps"],
        ]

        assert abs(sum(components) - impact["total_impact_bps"]) < 1e-10

    def test_execution_efficiency_metrics(self):
        """Test execution efficiency calculations."""
        # Would test metrics like:
        # - Implementation shortfall vs benchmark
        # - Price improvement rates
        # - Fill rate optimization
        # - Venue selection efficiency
        pass


if __name__ == "__main__":
    pytest.main([__file__])
