"""
Tests for Backtesting Performance Metrics Module

Comprehensive test suite for performance metrics calculation following AFML methodologies.
Tests cover edge cases, mathematical accuracy, and integration scenarios.

Author: Quantitative Analysis Team
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings

from src.backtesting.metrics import (
    PerformanceMetrics,
    DrawdownMetrics,
    PerformanceCalculator,
    create_performance_report,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics object creation."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.12,
            cumulative_return=0.15,
            volatility=0.02,
            annualized_volatility=0.18,
            max_drawdown=0.05,
            avg_drawdown=0.02,
            drawdown_duration=10,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=2.4,
            information_ratio=0.8,
            probabilistic_sharpe_ratio=0.85,
            deflated_sharpe_ratio=0.75,
            minimum_track_record_length=200.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            avg_win=150.0,
            avg_loss=-80.0,
            profit_factor=2.5,
            skewness=0.1,
            kurtosis=0.5,
            var_95=0.03,
            cvar_95=0.045,
        )

        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_trades == 50
        assert metrics.win_rate == 0.6

    def test_performance_metrics_optional_fields(self):
        """Test PerformanceMetrics with optional benchmark fields."""
        metrics = PerformanceMetrics(
            total_return=0.10,
            annualized_return=0.08,
            cumulative_return=0.10,
            volatility=0.015,
            annualized_volatility=0.16,
            max_drawdown=0.03,
            avg_drawdown=0.01,
            drawdown_duration=5,
            sharpe_ratio=1.2,
            sortino_ratio=1.4,
            calmar_ratio=2.7,
            information_ratio=0.6,
            probabilistic_sharpe_ratio=0.80,
            deflated_sharpe_ratio=0.70,
            minimum_track_record_length=150.0,
            total_trades=25,
            winning_trades=15,
            losing_trades=10,
            win_rate=0.6,
            avg_win=120.0,
            avg_loss=-70.0,
            profit_factor=2.0,
            skewness=0.05,
            kurtosis=0.3,
            var_95=0.025,
            cvar_95=0.035,
            beta=1.1,
            alpha=0.02,
            tracking_error=0.05,
        )

        assert metrics.beta == 1.1
        assert metrics.alpha == 0.02
        assert metrics.tracking_error == 0.05


class TestDrawdownMetrics:
    """Test DrawdownMetrics dataclass."""

    def test_drawdown_metrics_creation(self):
        """Test DrawdownMetrics object creation."""
        sample_series = pd.Series([0.0, -0.05, -0.03, -0.08, 0.0])

        metrics = DrawdownMetrics(
            max_drawdown=0.08,
            max_drawdown_duration=15,
            avg_drawdown=0.04,
            avg_drawdown_duration=8.5,
            recovery_factor=1.5,
            drawdown_series=sample_series,
            underwater_curve=sample_series,
        )

        assert metrics.max_drawdown == 0.08
        assert metrics.max_drawdown_duration == 15
        assert len(metrics.drawdown_series) == 5


class TestPerformanceCalculator:
    """Test PerformanceCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create PerformanceCalculator instance."""
        return PerformanceCalculator(
            risk_free_rate=0.02, trading_days_per_year=252, confidence_level=0.05
        )

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        return returns

    @pytest.fixture
    def sample_portfolio_values(self):
        """Create sample portfolio values."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        values = [100000]
        for i in range(99):
            change = np.random.normal(0.001, 0.02)
            values.append(values[-1] * (1 + change))
        return pd.Series(values, index=dates)

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data."""
        return [
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100,
                "price": 150.0,
                "pnl": 500.0,
            },
            {
                "symbol": "AAPL",
                "side": "SELL",
                "quantity": 100,
                "price": 155.0,
                "pnl": 500.0,
            },
            {
                "symbol": "GOOGL",
                "side": "BUY",
                "quantity": 50,
                "price": 2500.0,
                "pnl": -200.0,
            },
            {
                "symbol": "GOOGL",
                "side": "SELL",
                "quantity": 50,
                "price": 2496.0,
                "pnl": -200.0,
            },
            {
                "symbol": "MSFT",
                "side": "BUY",
                "quantity": 200,
                "price": 300.0,
                "pnl": 1000.0,
            },
        ]

    def test_calculator_initialization(self, calculator):
        """Test PerformanceCalculator initialization."""
        assert calculator.risk_free_rate == 0.02
        assert calculator.trading_days_per_year == 252
        assert calculator.confidence_level == 0.05

    def test_calculate_total_return(self, calculator):
        """Test total return calculation."""
        portfolio_values = pd.Series([100000, 105000, 110000, 108000, 112000])
        initial_capital = 100000

        total_return = calculator._calculate_total_return(
            portfolio_values, initial_capital
        )
        expected_return = (112000 - 100000) / 100000

        assert abs(total_return - expected_return) < 1e-6

    def test_calculate_total_return_empty_series(self, calculator):
        """Test total return calculation with empty series."""
        portfolio_values = pd.Series(dtype=float)
        total_return = calculator._calculate_total_return(portfolio_values, 100000)
        assert total_return == 0.0

    def test_calculate_annualized_return(self, calculator):
        """Test annualized return calculation."""
        # 1 year of daily returns with 10% total return
        returns = pd.Series([0.0001] * 252)  # Small daily returns
        returns.iloc[-1] = 0.095  # Adjust last return to get ~10% total

        ann_return = calculator._calculate_annualized_return(returns)
        assert abs(ann_return - 0.1) < 0.05  # Allow some tolerance

    def test_calculate_cumulative_return(self, calculator):
        """Test cumulative return calculation."""
        returns = pd.Series([0.01, 0.02, -0.005, 0.015])

        cum_return = calculator._calculate_cumulative_return(returns)
        expected = (1.01 * 1.02 * 0.995 * 1.015) - 1

        assert abs(cum_return - expected) < 1e-6

    def test_calculate_volatility(self, calculator):
        """Test volatility calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.015])

        volatility = calculator._calculate_volatility(returns)
        expected = returns.std()

        assert abs(volatility - expected) < 1e-6

    def test_calculate_volatility_single_value(self, calculator):
        """Test volatility calculation with single value."""
        returns = pd.Series([0.01])
        volatility = calculator._calculate_volatility(returns)
        assert volatility == 0.0

    def test_calculate_drawdown_metrics(self, calculator):
        """Test comprehensive drawdown metrics calculation."""
        # Portfolio values with clear drawdown pattern
        portfolio_values = pd.Series(
            [
                100000,
                105000,
                110000,
                108000,
                106000,  # Peak at 110000, drawdown to 106000
                107000,
                109000,
                111000,
                115000,
                113000,  # Recovery and new peak
            ]
        )

        drawdown_metrics = calculator._calculate_drawdown_metrics(portfolio_values)

        # Maximum drawdown should be (110000 - 106000) / 110000 ≈ 0.0364
        assert abs(drawdown_metrics.max_drawdown - 0.0364) < 0.001
        assert drawdown_metrics.max_drawdown_duration > 0
        assert drawdown_metrics.avg_drawdown > 0
        assert len(drawdown_metrics.drawdown_series) == len(portfolio_values)

    def test_calculate_drawdown_metrics_empty_series(self, calculator):
        """Test drawdown metrics with empty portfolio values."""
        portfolio_values = pd.Series(dtype=float)

        drawdown_metrics = calculator._calculate_drawdown_metrics(portfolio_values)

        assert drawdown_metrics.max_drawdown == 0.0
        assert drawdown_metrics.max_drawdown_duration == 0
        assert drawdown_metrics.avg_drawdown == 0.0

    def test_calculate_sharpe_ratio(self, calculator):
        """Test Sharpe ratio calculation."""
        # Returns with known statistics
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 252)
        )  # Daily returns for 1 year

        sharpe = calculator._calculate_sharpe_ratio(returns)

        # Should be a reasonable value for the given parameters
        assert -5 < sharpe < 5  # Reasonable range for Sharpe ratio

    def test_calculate_sharpe_ratio_edge_cases(self, calculator):
        """Test Sharpe ratio edge cases."""
        # Empty series
        empty_returns = pd.Series(dtype=float)
        assert calculator._calculate_sharpe_ratio(empty_returns) == 0.0

        # Single value
        single_return = pd.Series([0.01])
        assert calculator._calculate_sharpe_ratio(single_return) == 0.0

    def test_calculate_sortino_ratio(self, calculator):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01, -0.02, 0.025])

        sortino = calculator._calculate_sortino_ratio(returns)

        # Should be finite and reasonable (expanded range for small samples)
        assert np.isfinite(sortino)
        assert -20 < sortino < 20

    def test_calculate_calmar_ratio(self, calculator):
        """Test Calmar ratio calculation."""
        annualized_return = 0.12
        max_drawdown = 0.05

        calmar = calculator._calculate_calmar_ratio(annualized_return, max_drawdown)
        expected = 0.12 / 0.05

        assert abs(calmar - expected) < 1e-6

    def test_calculate_calmar_ratio_zero_drawdown(self, calculator):
        """Test Calmar ratio with zero drawdown."""
        calmar = calculator._calculate_calmar_ratio(0.12, 0.0)
        assert calmar == np.inf

        calmar = calculator._calculate_calmar_ratio(-0.05, 0.0)
        assert calmar == 0.0

    def test_calculate_probabilistic_sharpe_ratio(self, calculator):
        """Test Probabilistic Sharpe Ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        sharpe_ratio = 1.5

        psr = calculator._calculate_probabilistic_sharpe_ratio(returns, sharpe_ratio)

        # Should be between 0 and 1
        assert 0 <= psr <= 1

    def test_calculate_trade_metrics(self, calculator, sample_trades):
        """Test trade-based metrics calculation."""
        metrics = calculator._calculate_trade_metrics(sample_trades)

        assert metrics["total_trades"] == 5
        assert metrics["winning_trades"] == 3  # 2 profitable + 1 with positive PnL
        assert metrics["losing_trades"] == 2  # 2 with negative PnL
        assert metrics["win_rate"] == 0.6
        assert metrics["profit_factor"] > 0

    def test_calculate_trade_metrics_empty(self, calculator):
        """Test trade metrics with empty trades list."""
        metrics = calculator._calculate_trade_metrics([])

        assert metrics["total_trades"] == 0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0

    def test_calculate_var(self, calculator):
        """Test Value at Risk calculation."""
        returns = pd.Series([-0.02, -0.01, 0.01, 0.005, -0.015, 0.02, -0.03])

        var_95 = calculator._calculate_var(returns, 0.05)

        # VaR should be positive (magnitude of loss)
        assert var_95 > 0
        assert var_95 <= 0.03  # Should not exceed maximum loss

    def test_calculate_cvar(self, calculator):
        """Test Conditional Value at Risk calculation."""
        returns = pd.Series([-0.02, -0.01, 0.01, 0.005, -0.015, 0.02, -0.03])

        cvar_95 = calculator._calculate_cvar(returns, 0.05)

        # CVaR should be positive and >= VaR
        var_95 = calculator._calculate_var(returns, 0.05)
        assert cvar_95 > 0
        assert cvar_95 >= var_95

    def test_calculate_beta(self, calculator):
        """Test beta calculation."""
        portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        benchmark_returns = pd.Series([0.008, 0.015, -0.005, 0.012])

        beta = calculator._calculate_beta(portfolio_returns, benchmark_returns)

        assert beta is not None
        assert isinstance(beta, float)
        assert 0 < beta < 3  # Reasonable range for beta

    def test_calculate_beta_edge_cases(self, calculator):
        """Test beta calculation edge cases."""
        # Different lengths
        portfolio_returns = pd.Series([0.01, 0.02])
        benchmark_returns = pd.Series([0.008])

        beta = calculator._calculate_beta(portfolio_returns, benchmark_returns)
        assert beta is None

        # Empty series
        empty_returns = pd.Series(dtype=float)
        beta = calculator._calculate_beta(empty_returns, empty_returns)
        assert beta is None

    def test_calculate_alpha(self, calculator):
        """Test alpha calculation."""
        portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        benchmark_returns = pd.Series([0.008, 0.015, -0.005, 0.012])
        beta = 1.2

        alpha = calculator._calculate_alpha(portfolio_returns, benchmark_returns, beta)

        assert alpha is not None
        assert isinstance(alpha, float)

    def test_calculate_information_ratio(self, calculator):
        """Test information ratio calculation."""
        portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        benchmark_returns = pd.Series([0.008, 0.015, -0.005, 0.012])

        ir = calculator._calculate_information_ratio(
            portfolio_returns, benchmark_returns
        )

        assert ir is not None
        assert isinstance(ir, float)

    def test_calculate_tracking_error(self, calculator):
        """Test tracking error calculation."""
        portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        benchmark_returns = pd.Series([0.008, 0.015, -0.005, 0.012])

        te = calculator._calculate_tracking_error(portfolio_returns, benchmark_returns)

        assert te is not None
        assert te > 0
        assert isinstance(te, float)

    def test_comprehensive_metrics_calculation(
        self, calculator, sample_returns, sample_portfolio_values, sample_trades
    ):
        """Test comprehensive metrics calculation."""
        metrics = calculator.calculate_comprehensive_metrics(
            returns=sample_returns,
            portfolio_values=sample_portfolio_values,
            trades=sample_trades,
            initial_capital=100000.0,
        )

        # Verify all required fields are present and reasonable
        assert isinstance(metrics, PerformanceMetrics)
        assert -1 <= metrics.total_return <= 2  # Reasonable range
        assert metrics.volatility >= 0
        assert metrics.max_drawdown >= 0
        assert metrics.total_trades == len(sample_trades)
        assert 0 <= metrics.win_rate <= 1

    def test_comprehensive_metrics_with_benchmark(
        self, calculator, sample_returns, sample_portfolio_values, sample_trades
    ):
        """Test comprehensive metrics with benchmark."""
        # Create benchmark returns
        benchmark_returns = sample_returns * 0.8  # Correlated but different

        metrics = calculator.calculate_comprehensive_metrics(
            returns=sample_returns,
            portfolio_values=sample_portfolio_values,
            trades=sample_trades,
            benchmark_returns=benchmark_returns,
            initial_capital=100000.0,
        )

        # Should have benchmark-relative metrics
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.information_ratio is not None
        assert metrics.tracking_error is not None

    def test_comprehensive_metrics_empty_returns(self, calculator):
        """Test comprehensive metrics with empty returns."""
        empty_returns = pd.Series(dtype=float)
        empty_portfolio = pd.Series(dtype=float)

        metrics = calculator.calculate_comprehensive_metrics(
            returns=empty_returns,
            portfolio_values=empty_portfolio,
            trades=[],
            initial_capital=100000.0,
        )

        # Should return empty metrics without errors
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
        assert metrics.total_trades == 0

    def test_get_consecutive_periods(self, calculator):
        """Test consecutive periods calculation."""
        boolean_series = pd.Series([True, True, False, True, True, True, False, False])

        periods = calculator._get_consecutive_periods(boolean_series)

        assert periods == [2, 3]  # Two consecutive True periods of length 2 and 3

    def test_get_consecutive_periods_edge_cases(self, calculator):
        """Test consecutive periods edge cases."""
        # All False
        all_false = pd.Series([False, False, False])
        periods = calculator._get_consecutive_periods(all_false)
        assert periods == []

        # All True
        all_true = pd.Series([True, True, True])
        periods = calculator._get_consecutive_periods(all_true)
        assert periods == [3]

        # Empty series
        empty = pd.Series(dtype=bool)
        periods = calculator._get_consecutive_periods(empty)
        assert periods == []


class TestCreatePerformanceReport:
    """Test performance report creation."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample PerformanceMetrics for testing."""
        return PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.12,
            cumulative_return=0.15,
            volatility=0.02,
            annualized_volatility=0.18,
            max_drawdown=0.05,
            avg_drawdown=0.02,
            drawdown_duration=10,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=2.4,
            information_ratio=0.8,
            probabilistic_sharpe_ratio=0.85,
            deflated_sharpe_ratio=0.75,
            minimum_track_record_length=200.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            avg_win=150.0,
            avg_loss=-80.0,
            profit_factor=2.5,
            skewness=0.1,
            kurtosis=0.5,
            var_95=0.03,
            cvar_95=0.045,
        )

    def test_create_performance_report(self, sample_metrics):
        """Test performance report creation."""
        report = create_performance_report(sample_metrics)

        # Check report structure
        assert "summary" in report
        assert "returns" in report
        assert "risk" in report
        assert "risk_adjusted" in report
        assert "advanced" in report
        assert "trades" in report
        assert "distribution" in report

        # Check summary section
        assert "Total Return" in report["summary"]
        assert "Sharpe Ratio" in report["summary"]
        assert report["summary"]["Total Return"] == "15.00%"

        # Check formatted values
        assert report["trades"]["Win Rate"] == "60.00%"
        assert isinstance(report["returns"]["Total Return"], float)

    def test_performance_report_content(self, sample_metrics):
        """Test performance report content accuracy."""
        report = create_performance_report(sample_metrics)

        # Verify specific values
        assert report["returns"]["Total Return"] == 0.15
        assert report["risk"]["Max Drawdown"] == 0.05
        assert report["risk_adjusted"]["Sharpe Ratio"] == 1.5
        assert report["trades"]["Total Trades"] == 50
        assert report["distribution"]["Skewness"] == 0.1


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_realistic_performance_scenario(self):
        """Test with realistic market data scenario."""
        calculator = PerformanceCalculator()

        # Create realistic returns data (S&P 500-like)
        np.random.seed(123)
        n_days = 252  # 1 year
        daily_returns = np.random.normal(0.0008, 0.015, n_days)  # ~8% annual, 15% vol
        returns = pd.Series(daily_returns)

        # Create corresponding portfolio values
        portfolio_values = pd.Series([100000])
        for ret in returns:
            new_value = portfolio_values.iloc[-1] * (1 + ret)
            portfolio_values = pd.concat([portfolio_values, pd.Series([new_value])])
        portfolio_values = portfolio_values[:-1]  # Remove extra value

        # Create realistic trades
        trades = [{"pnl": np.random.normal(50, 200)} for _ in range(20)]

        metrics = calculator.calculate_comprehensive_metrics(
            returns=returns,
            portfolio_values=portfolio_values,
            trades=trades,
            initial_capital=100000.0,
        )

        # Verify realistic ranges
        assert -0.5 <= metrics.total_return <= 0.5  # -50% to 50% annual return
        assert 0.05 <= metrics.annualized_volatility <= 0.4  # 5% to 40% volatility
        assert metrics.max_drawdown >= 0
        assert -3 <= metrics.sharpe_ratio <= 3  # Reasonable Sharpe range

    def test_extreme_market_conditions(self):
        """Test performance under extreme market conditions."""
        calculator = PerformanceCalculator()

        # Create extreme negative returns (market crash scenario)
        crash_returns = pd.Series([-0.1, -0.05, -0.08, -0.03, -0.12, 0.02, 0.01])
        portfolio_values = pd.Series(
            [100000, 90000, 85500, 78660, 76301, 67628, 69001, 69691]
        )

        metrics = calculator.calculate_comprehensive_metrics(
            returns=crash_returns,
            portfolio_values=portfolio_values,
            trades=[],
            initial_capital=100000.0,
        )

        # Should handle extreme conditions gracefully
        assert metrics.total_return < 0  # Negative return
        assert metrics.max_drawdown > 0.2  # Significant drawdown
        assert metrics.sharpe_ratio < 0  # Negative Sharpe ratio

    def test_perfect_strategy_scenario(self):
        """Test with perfect strategy (all positive returns)."""
        calculator = PerformanceCalculator()

        # Perfect strategy with only positive returns
        perfect_returns = pd.Series([0.01, 0.005, 0.02, 0.008, 0.015])
        portfolio_values = pd.Series([100000, 101000, 101505, 103535, 104363, 105928])

        winning_trades = [{"pnl": 100 + i * 50} for i in range(10)]

        metrics = calculator.calculate_comprehensive_metrics(
            returns=perfect_returns,
            portfolio_values=portfolio_values,
            trades=winning_trades,
            initial_capital=100000.0,
        )

        # Perfect strategy characteristics
        assert metrics.total_return > 0
        assert metrics.max_drawdown == 0  # No drawdowns
        assert metrics.win_rate == 1.0  # 100% win rate
        assert metrics.sharpe_ratio > 0
        assert (
            metrics.sortino_ratio >= metrics.sharpe_ratio
        )  # Should be higher with no negative returns


if __name__ == "__main__":
    pytest.main([__file__])
