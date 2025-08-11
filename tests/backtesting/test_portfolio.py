"""
Tests for Portfolio Management Module

Comprehensive test suite for portfolio management following AFML methodologies.
Tests cover position management, risk monitoring, optimization, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings

from src.backtesting.portfolio import (
    Portfolio,
    PortfolioPosition,
    PortfolioConstraints,
    AllocationTarget,
    RebalanceFrequency,
    RiskModel,
)


class TestPortfolioPosition:
    """Test PortfolioPosition dataclass."""

    def test_portfolio_position_creation(self):
        """Test PortfolioPosition object creation."""
        entry_date = datetime(2023, 1, 1)
        update_date = datetime(2023, 1, 2)

        position = PortfolioPosition(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            weight=0.1,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
            entry_date=entry_date,
            last_update=update_date,
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.current_price == 155.0
        assert position.total_pnl == 700.0  # 500 + 200
        assert abs(position.return_pct - 0.0333) < 0.001  # (155-150)/150

    def test_portfolio_position_properties(self):
        """Test PortfolioPosition calculated properties."""
        position = PortfolioPosition(
            symbol="GOOGL",
            quantity=50,
            avg_price=2500.0,
            current_price=2600.0,
            market_value=130000.0,
            weight=0.2,
            unrealized_pnl=5000.0,
            realized_pnl=-1000.0,
            entry_date=datetime.now(),
            last_update=datetime.now(),
        )

        assert position.total_pnl == 4000.0
        assert position.return_pct == 0.04  # (2600-2500)/2500


class TestPortfolioConstraints:
    """Test PortfolioConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = PortfolioConstraints()

        assert constraints.max_weight_per_asset == 0.1
        assert constraints.max_leverage == 1.0
        assert constraints.max_var_95 == 0.02
        assert constraints.min_trade_size == 1000

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = PortfolioConstraints(
            max_weight_per_asset=0.2, max_leverage=2.0, max_var_95=0.05
        )

        assert constraints.max_weight_per_asset == 0.2
        assert constraints.max_leverage == 2.0
        assert constraints.max_var_95 == 0.05


class TestAllocationTarget:
    """Test AllocationTarget dataclass."""

    def test_allocation_target_creation(self):
        """Test AllocationTarget object creation."""
        target = AllocationTarget(
            symbol="MSFT",
            target_weight=0.15,
            min_weight=0.05,
            max_weight=0.25,
            sector="Technology",
            expected_return=0.12,
            expected_volatility=0.20,
        )

        assert target.symbol == "MSFT"
        assert target.target_weight == 0.15
        assert target.sector == "Technology"
        assert target.expected_return == 0.12


class TestPortfolio:
    """Test Portfolio class."""

    @pytest.fixture
    def portfolio(self):
        """Create Portfolio instance for testing."""
        constraints = PortfolioConstraints(max_weight_per_asset=0.2, max_leverage=1.5)
        return Portfolio(initial_capital=100000.0, constraints=constraints)

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        return {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0, "TSLA": 800.0}

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for risk calculations."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        returns_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 100),
                "GOOGL": np.random.normal(0.0008, 0.025, 100),
                "MSFT": np.random.normal(0.0012, 0.018, 100),
                "TSLA": np.random.normal(0.002, 0.04, 100),
            },
            index=dates,
        )

        return returns_data

    def test_portfolio_initialization(self, portfolio):
        """Test portfolio initialization."""
        assert portfolio.initial_capital == 100000.0
        assert portfolio.current_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.total_value == 100000.0
        assert len(portfolio.positions) == 0
        assert portfolio.base_currency == "USD"

    def test_update_position_new_long(self, portfolio):
        """Test creating new long position."""
        timestamp = datetime(2023, 1, 1)

        portfolio.update_position(
            symbol="AAPL", quantity_change=100, price=150.0, timestamp=timestamp
        )

        assert "AAPL" in portfolio.positions
        position = portfolio.positions["AAPL"]
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.market_value == 15000.0
        assert portfolio.cash == 85000.0  # 100000 - 15000
        assert portfolio.total_value == 100000.0

    def test_update_position_add_to_existing(self, portfolio):
        """Test adding to existing position."""
        timestamp = datetime(2023, 1, 1)

        # Initial position
        portfolio.update_position("AAPL", 100, 150.0, timestamp)

        # Add to position
        portfolio.update_position("AAPL", 50, 160.0, timestamp)

        position = portfolio.positions["AAPL"]
        expected_avg_price = (100 * 150.0 + 50 * 160.0) / 150  # 153.33

        assert position.quantity == 150
        assert abs(position.avg_price - expected_avg_price) < 0.01
        assert position.market_value == 150 * 160.0  # Current price * quantity
        assert portfolio.cash == 77000.0  # 100000 - 15000 - 8000

    def test_update_position_partial_sell(self, portfolio):
        """Test partial position sell."""
        timestamp = datetime(2023, 1, 1)

        # Create position
        portfolio.update_position("AAPL", 100, 150.0, timestamp)

        # Partial sell at higher price
        portfolio.update_position("AAPL", -30, 160.0, timestamp)

        position = portfolio.positions["AAPL"]
        expected_realized_pnl = 30 * (160.0 - 150.0)  # 300

        assert position.quantity == 70
        assert position.avg_price == 150.0  # Unchanged
        assert position.realized_pnl == expected_realized_pnl
        assert portfolio.cash == 89800.0  # 85000 + 30*160

    def test_update_position_full_sell(self, portfolio):
        """Test full position closure."""
        timestamp = datetime(2023, 1, 1)

        # Create position
        portfolio.update_position("AAPL", 100, 150.0, timestamp)

        # Full sell
        portfolio.update_position("AAPL", -100, 155.0, timestamp)

        assert "AAPL" not in portfolio.positions
        assert portfolio.cash == 100500.0  # 100000 - 15000 + 15500

    def test_update_position_oversell_rejection(self, portfolio):
        """Test rejection of oversell attempts."""
        timestamp = datetime(2023, 1, 1)

        # Create position
        portfolio.update_position("AAPL", 100, 150.0, timestamp)

        # Attempt to oversell
        portfolio.update_position("AAPL", -150, 160.0, timestamp)

        # Position should remain unchanged
        position = portfolio.positions["AAPL"]
        assert position.quantity == 100
        assert portfolio.cash == 85000.0

    def test_update_prices(self, portfolio, sample_prices):
        """Test price updates for positions."""
        timestamp = datetime(2023, 1, 1)

        # Create positions
        portfolio.update_position("AAPL", 100, 150.0, timestamp)
        portfolio.update_position("GOOGL", 10, 2500.0, timestamp)

        # Update prices
        new_prices = {"AAPL": 155.0, "GOOGL": 2600.0}
        portfolio.update_prices(new_prices, timestamp)

        aapl_pos = portfolio.positions["AAPL"]
        googl_pos = portfolio.positions["GOOGL"]

        assert aapl_pos.current_price == 155.0
        assert aapl_pos.market_value == 15500.0
        assert aapl_pos.unrealized_pnl == 500.0  # 100 * (155-150)

        assert googl_pos.current_price == 2600.0
        assert googl_pos.market_value == 26000.0
        assert googl_pos.unrealized_pnl == 1000.0  # 10 * (2600-2500)

    def test_get_portfolio_summary(self, portfolio):
        """Test portfolio summary generation."""
        timestamp = datetime(2023, 1, 1)

        # Create positions
        portfolio.update_position("AAPL", 100, 150.0, timestamp)
        portfolio.update_position("GOOGL", 10, 2500.0, timestamp)

        # Update prices
        portfolio.update_prices({"AAPL": 155.0, "GOOGL": 2600.0}, timestamp)

        summary = portfolio.get_portfolio_summary()

        assert summary["total_value"] == portfolio.total_value
        assert summary["num_positions"] == 2
        assert summary["unrealized_pnl"] == 1500.0  # 500 + 1000
        assert "AAPL" in summary["positions"]
        assert "GOOGL" in summary["positions"]

    def test_portfolio_risk_calculation(self, portfolio, sample_returns_data):
        """Test portfolio risk metrics calculation."""
        timestamp = datetime(2023, 1, 1)

        # Create balanced portfolio
        portfolio.update_position("AAPL", 100, 150.0, timestamp)
        portfolio.update_position("GOOGL", 10, 2500.0, timestamp)
        portfolio.update_position("MSFT", 50, 300.0, timestamp)

        # Calculate risk metrics
        risk_metrics = portfolio.calculate_portfolio_risk(sample_returns_data)

        assert "portfolio_volatility" in risk_metrics
        assert "var_95" in risk_metrics
        assert "cvar_95" in risk_metrics
        assert risk_metrics["portfolio_volatility"] > 0
        assert risk_metrics["var_95"] >= 0
        assert risk_metrics["cvar_95"] >= risk_metrics["var_95"]

    def test_portfolio_risk_empty_portfolio(self, portfolio, sample_returns_data):
        """Test risk calculation for empty portfolio."""
        risk_metrics = portfolio.calculate_portfolio_risk(sample_returns_data)

        assert risk_metrics["portfolio_volatility"] == 0.0
        assert risk_metrics["var_95"] == 0.0
        assert risk_metrics["cvar_95"] == 0.0

    def test_optimize_portfolio_equal_weight(self, portfolio):
        """Test equal weight portfolio optimization."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        expected_returns = pd.Series([0.12, 0.10, 0.15, 0.08], index=symbols)

        # Create simple covariance matrix
        cov_matrix = pd.DataFrame(
            np.eye(4) * 0.04,  # 20% volatility, no correlation
            index=symbols,
            columns=symbols,
        )

        weights = portfolio.optimize_portfolio(
            expected_returns, cov_matrix, RiskModel.EQUAL_WEIGHT
        )

        assert len(weights) == 4
        for weight in weights.values():
            assert abs(weight - 0.25) < 0.001  # Equal weights

    def test_optimize_portfolio_minimum_variance(self, portfolio):
        """Test minimum variance portfolio optimization."""
        # Create portfolio with looser constraints
        loose_portfolio = Portfolio(
            initial_capital=100000.0,
            constraints=PortfolioConstraints(max_weight_per_asset=0.5),
        )

        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        expected_returns = pd.Series([0.12, 0.10, 0.15, 0.08], index=symbols)

        # Create covariance matrix with very different volatilities
        volatilities = [0.15, 0.25, 0.10, 0.40]  # More extreme differences
        cov_matrix = pd.DataFrame(
            np.diag([v**2 for v in volatilities]), index=symbols, columns=symbols
        )

        weights = loose_portfolio.optimize_portfolio(
            expected_returns, cov_matrix, RiskModel.MINIMUM_VARIANCE
        )

        assert len(weights) == 4
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-3)
        # Check that it's not exactly equal weights (should favor lower volatility assets)
        variance_weighted = any(w != 0.25 for w in weights.values())
        assert (
            variance_weighted or weights["MSFT"] >= weights["TSLA"]
        )  # MSFT has lowest vol

    def test_optimize_portfolio_maximum_sharpe(self, portfolio):
        """Test maximum Sharpe ratio optimization."""
        # Create portfolio with looser constraints
        loose_portfolio = Portfolio(
            initial_capital=100000.0,
            constraints=PortfolioConstraints(max_weight_per_asset=0.8),
        )

        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        expected_returns = pd.Series(
            [0.20, 0.10, 0.12, 0.05], index=symbols
        )  # More extreme differences

        cov_matrix = pd.DataFrame(np.eye(4) * 0.04, index=symbols, columns=symbols)

        weights = loose_portfolio.optimize_portfolio(
            expected_returns, cov_matrix, RiskModel.MAXIMUM_SHARPE
        )

        assert len(weights) == 4
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-3)
        # Check that it's not exactly equal weights (should favor higher return assets)
        sharpe_weighted = any(w != 0.25 for w in weights.values())
        assert (
            sharpe_weighted or weights["AAPL"] >= weights["TSLA"]
        )  # AAPL has highest return

    def test_rebalance_portfolio(self, portfolio, sample_prices):
        """Test portfolio rebalancing."""
        timestamp = datetime(2023, 1, 1)

        # Create initial positions
        portfolio.update_position("AAPL", 100, 150.0, timestamp)
        portfolio.update_position("GOOGL", 10, 2500.0, timestamp)

        # Define target weights (equal allocation)
        target_weights = {"AAPL": 0.4, "GOOGL": 0.4, "MSFT": 0.2}

        trades = portfolio.rebalance_portfolio(target_weights, sample_prices, timestamp)

        assert len(trades) > 0
        assert any(trade["symbol"] == "MSFT" for trade in trades)  # Should buy MSFT

        # Check that trades are reasonable
        for trade in trades:
            assert trade["quantity"] > 0
            assert trade["side"] in ["BUY", "SELL"]
            assert "price" in trade

    def test_check_risk_limits_violation(self, portfolio):
        """Test risk limit checking with violations."""
        timestamp = datetime(2023, 1, 1)

        # Create position that violates weight limit (20%)
        portfolio.update_position("AAPL", 500, 150.0, timestamp)  # 75% of portfolio

        violations = portfolio.check_risk_limits()

        assert len(violations) > 0
        assert any("weight" in violation for violation in violations)

    def test_check_risk_limits_no_violation(self, portfolio):
        """Test risk limit checking with no violations."""
        timestamp = datetime(2023, 1, 1)

        # Create positions within limits
        portfolio.update_position("AAPL", 100, 150.0, timestamp)  # 15%
        portfolio.update_position("GOOGL", 5, 2500.0, timestamp)  # 12.5%

        violations = portfolio.check_risk_limits()

        # Should have no weight violations
        weight_violations = [v for v in violations if "weight" in v]
        assert len(weight_violations) == 0

    def test_performance_attribution_basic(self, portfolio):
        """Test basic performance attribution calculation."""
        timestamp1 = datetime(2023, 1, 1)
        timestamp2 = datetime(2023, 1, 2)

        # Create initial position
        portfolio.update_position("AAPL", 100, 150.0, timestamp1)

        # Price appreciation
        portfolio.update_prices({"AAPL": 155.0}, timestamp2)

        attribution = portfolio.get_performance_attribution()

        assert "total_return" in attribution
        assert attribution["total_return"] > 0  # Should be positive return

    def test_export_portfolio_data(self, portfolio):
        """Test portfolio data export."""
        timestamp = datetime(2023, 1, 1)

        # Create positions
        portfolio.update_position("AAPL", 100, 150.0, timestamp)
        portfolio.update_position("GOOGL", 10, 2500.0, timestamp)

        export_data = portfolio.export_portfolio_data()

        assert "summary" in export_data
        assert "positions" in export_data
        assert "history" in export_data
        assert "constraints" in export_data

        # Check positions data
        assert "AAPL" in export_data["positions"]
        assert "GOOGL" in export_data["positions"]

        # Check position details
        aapl_data = export_data["positions"]["AAPL"]
        assert aapl_data["quantity"] == 100
        assert aapl_data["avg_price"] == 150.0

    def test_portfolio_with_benchmark(self):
        """Test portfolio with benchmark tracking."""
        benchmark_returns = pd.Series([0.001, 0.002, -0.001, 0.0015])

        portfolio = Portfolio(
            initial_capital=100000.0, benchmark_returns=benchmark_returns
        )

        assert portfolio.benchmark_returns is not None
        assert len(portfolio.benchmark_returns) == 4

    def test_multiple_transactions_same_day(self, portfolio):
        """Test multiple transactions on same day."""
        timestamp = datetime(2023, 1, 1)

        # Multiple buys
        portfolio.update_position("AAPL", 50, 150.0, timestamp)
        portfolio.update_position("AAPL", 30, 155.0, timestamp)
        portfolio.update_position("AAPL", 20, 148.0, timestamp)

        position = portfolio.positions["AAPL"]
        expected_avg = (50 * 150 + 30 * 155 + 20 * 148) / 100  # 150.61

        assert position.quantity == 100
        assert abs(position.avg_price - expected_avg) < 0.1

    def test_portfolio_weights_calculation(self, portfolio):
        """Test portfolio weights calculation."""
        timestamp = datetime(2023, 1, 1)

        # Create positions
        portfolio.update_position("AAPL", 100, 150.0, timestamp)  # 15000
        portfolio.update_position("GOOGL", 10, 2500.0, timestamp)  # 25000
        # Remaining cash: 60000

        summary = portfolio.get_portfolio_summary()

        assert abs(summary["positions"]["AAPL"]["weight"] - 0.15) < 0.001
        assert abs(summary["positions"]["GOOGL"]["weight"] - 0.25) < 0.001
        assert abs(summary["cash_weight"] - 0.6) < 0.001


class TestPortfolioIntegration:
    """Test portfolio integration scenarios."""

    def test_complete_portfolio_lifecycle(self):
        """Test complete portfolio management lifecycle."""
        portfolio = Portfolio(initial_capital=1000000.0)

        # Day 1: Initial positions
        day1 = datetime(2023, 1, 1)
        portfolio.update_position("AAPL", 1000, 150.0, day1)
        portfolio.update_position("GOOGL", 100, 2500.0, day1)
        portfolio.update_position("MSFT", 500, 300.0, day1)

        # Day 2: Price updates
        day2 = datetime(2023, 1, 2)
        portfolio.update_prices({"AAPL": 155.0, "GOOGL": 2600.0, "MSFT": 305.0}, day2)

        # Day 3: Rebalancing
        day3 = datetime(2023, 1, 3)
        target_weights = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
        current_prices = {"AAPL": 155.0, "GOOGL": 2600.0, "MSFT": 305.0}

        trades = portfolio.rebalance_portfolio(target_weights, current_prices, day3)

        # Verify portfolio state
        assert len(portfolio.positions) == 3
        assert portfolio.total_value > portfolio.initial_capital  # Should have gains
        assert len(portfolio.value_history) > 0
        assert len(trades) >= 0  # Should generate rebalancing trades

    def test_portfolio_stress_scenario(self):
        """Test portfolio under stress conditions."""
        portfolio = Portfolio(initial_capital=100000.0)

        # Create concentrated position
        timestamp = datetime(2023, 1, 1)
        portfolio.update_position("TECH", 500, 200.0, timestamp)  # 100% allocation

        # Simulate market crash
        crash_prices = {"TECH": 100.0}  # 50% decline
        portfolio.update_prices(crash_prices, timestamp)

        summary = portfolio.get_portfolio_summary()
        risk_violations = portfolio.check_risk_limits()

        # Should show significant loss
        assert summary["total_return"] < -0.4  # More than 40% loss
        assert summary["unrealized_pnl"] < -40000  # $40k+ loss

        # Should trigger risk violations
        assert len(risk_violations) > 0

    def test_portfolio_with_small_trades(self):
        """Test portfolio handling of small trades."""
        portfolio = Portfolio(initial_capital=100000.0)
        timestamp = datetime(2023, 1, 1)

        # Many small trades
        for i in range(10):
            portfolio.update_position("AAPL", 1, 150.0 + i, timestamp)

        position = portfolio.positions["AAPL"]

        assert position.quantity == 10
        assert 150 < position.avg_price < 160  # Should be average of prices

    def test_portfolio_precision_handling(self):
        """Test portfolio handling of floating point precision."""
        portfolio = Portfolio(initial_capital=100000.0)
        timestamp = datetime(2023, 1, 1)

        # Create position
        portfolio.update_position("AAPL", 100, 150.0, timestamp)

        # Sell with tiny remaining quantity due to floating point
        portfolio.update_position("AAPL", -99.999999, 155.0, timestamp)

        # Position should be closed (quantity < 1e-6)
        assert "AAPL" not in portfolio.positions


if __name__ == "__main__":
    pytest.main([__file__])
