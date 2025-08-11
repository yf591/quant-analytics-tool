"""
Backtesting Performance Metrics Module

This module provides comprehensive performance metrics calculation for backtesting results,
following methodologies from "Advances in Financial Machine Learning" (AFML).

Key Features:
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis and risk metrics
- Trade-based performance analysis
- Information ratio and tracking error
- Probabilistic Sharpe ratio (PSR) calculation
- Time-weighted and money-weighted returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics container.

    Based on AFML Chapter 14: Backtest Statistics
    """

    # Returns metrics
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk metrics
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Advanced metrics
    probabilistic_sharpe_ratio: float
    deflated_sharpe_ratio: float
    minimum_track_record_length: float

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Additional metrics
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None


@dataclass
class DrawdownMetrics:
    """
    Detailed drawdown analysis metrics.
    """

    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    avg_drawdown_duration: float
    recovery_factor: float
    drawdown_series: pd.Series
    underwater_curve: pd.Series


class PerformanceCalculator:
    """
    Advanced performance metrics calculator following AFML methodologies.

    This calculator provides comprehensive analysis of backtesting results with
    focus on statistical significance and practical interpretation of results.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        confidence_level: float = 0.05,
    ):
        """
        Initialize performance calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            trading_days_per_year: Number of trading days per year for annualization
            confidence_level: Confidence level for VaR and CVaR calculations
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.confidence_level = confidence_level

    def calculate_comprehensive_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        trades: List[Dict],
        benchmark_returns: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Series of portfolio returns
            portfolio_values: Series of portfolio values over time
            trades: List of trade dictionaries
            benchmark_returns: Optional benchmark returns for relative metrics
            initial_capital: Initial portfolio capital

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(returns) == 0:
            logger.warning("Empty returns series provided")
            return self._create_empty_metrics()

        # Basic return metrics
        total_return = self._calculate_total_return(portfolio_values, initial_capital)
        annualized_return = self._calculate_annualized_return(returns)
        cumulative_return = self._calculate_cumulative_return(returns)

        # Risk metrics
        volatility = self._calculate_volatility(returns)
        annualized_volatility = volatility * np.sqrt(self.trading_days_per_year)

        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_values)

        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(
            annualized_return, drawdown_metrics.max_drawdown
        )

        # Advanced metrics
        psr = self._calculate_probabilistic_sharpe_ratio(returns, sharpe_ratio)
        dsr = self._calculate_deflated_sharpe_ratio(returns, sharpe_ratio)
        mtrl = self._calculate_minimum_track_record_length(returns, sharpe_ratio)

        # Trade-based metrics
        trade_metrics = self._calculate_trade_metrics(trades)

        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        var_95 = self._calculate_var(returns, self.confidence_level)
        cvar_95 = self._calculate_cvar(returns, self.confidence_level)

        # Benchmark-relative metrics
        beta = None
        alpha = None
        information_ratio = None
        tracking_error = None

        if benchmark_returns is not None:
            beta = self._calculate_beta(returns, benchmark_returns)
            alpha = self._calculate_alpha(returns, benchmark_returns, beta)
            information_ratio = self._calculate_information_ratio(
                returns, benchmark_returns
            )
            tracking_error = self._calculate_tracking_error(returns, benchmark_returns)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            max_drawdown=drawdown_metrics.max_drawdown,
            avg_drawdown=drawdown_metrics.avg_drawdown,
            drawdown_duration=drawdown_metrics.max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio or 0.0,
            probabilistic_sharpe_ratio=psr,
            deflated_sharpe_ratio=dsr,
            minimum_track_record_length=mtrl,
            total_trades=trade_metrics["total_trades"],
            winning_trades=trade_metrics["winning_trades"],
            losing_trades=trade_metrics["losing_trades"],
            win_rate=trade_metrics["win_rate"],
            avg_win=trade_metrics["avg_win"],
            avg_loss=trade_metrics["avg_loss"],
            profit_factor=trade_metrics["profit_factor"],
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
        )

    def _calculate_total_return(
        self, portfolio_values: pd.Series, initial_capital: float
    ) -> float:
        """Calculate total return over the entire period."""
        if len(portfolio_values) == 0:
            return 0.0
        final_value = portfolio_values.iloc[-1]
        return (final_value - initial_capital) / initial_capital

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        total_periods = len(returns)
        if total_periods == 0:
            return 0.0

        # Calculate compound annual growth rate (CAGR)
        cumulative_return = (1 + returns).prod()
        periods_per_year = self.trading_days_per_year
        years = total_periods / periods_per_year

        if years <= 0 or cumulative_return <= 0:
            return 0.0

        return (cumulative_return ** (1 / years)) - 1

    def _calculate_cumulative_return(self, returns: pd.Series) -> float:
        """Calculate cumulative return."""
        if len(returns) == 0:
            return 0.0
        return (1 + returns).prod() - 1

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate return volatility (standard deviation)."""
        if len(returns) <= 1:
            return 0.0
        return returns.std()

    def _calculate_drawdown_metrics(
        self, portfolio_values: pd.Series
    ) -> DrawdownMetrics:
        """
        Calculate comprehensive drawdown metrics.

        Args:
            portfolio_values: Time series of portfolio values

        Returns:
            DrawdownMetrics object with detailed drawdown analysis
        """
        if len(portfolio_values) == 0:
            return DrawdownMetrics(
                max_drawdown=0.0,
                max_drawdown_duration=0,
                avg_drawdown=0.0,
                avg_drawdown_duration=0.0,
                recovery_factor=0.0,
                drawdown_series=pd.Series(dtype=float),
                underwater_curve=pd.Series(dtype=float),
            )

        # Calculate running maximum (peak)
        running_max = portfolio_values.expanding().max()

        # Calculate drawdown as percentage from peak
        drawdown_series = (portfolio_values - running_max) / running_max

        # Calculate underwater curve (time below previous high)
        underwater_curve = drawdown_series.copy()

        # Maximum drawdown
        max_drawdown = abs(drawdown_series.min())

        # Calculate drawdown durations
        is_drawdown = drawdown_series < 0
        drawdown_periods = self._get_consecutive_periods(is_drawdown)

        if len(drawdown_periods) > 0:
            max_drawdown_duration = max(drawdown_periods)
            avg_drawdown_duration = np.mean(drawdown_periods)
        else:
            max_drawdown_duration = 0
            avg_drawdown_duration = 0.0

        # Average drawdown
        drawdown_values = drawdown_series[drawdown_series < 0]
        avg_drawdown = abs(drawdown_values.mean()) if len(drawdown_values) > 0 else 0.0

        # Recovery factor (total return / max drawdown)
        total_return = (
            portfolio_values.iloc[-1] - portfolio_values.iloc[0]
        ) / portfolio_values.iloc[0]
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0.0

        return DrawdownMetrics(
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            avg_drawdown=avg_drawdown,
            avg_drawdown_duration=avg_drawdown_duration,
            recovery_factor=recovery_factor,
            drawdown_series=drawdown_series,
            underwater_curve=underwater_curve,
        )

    def _get_consecutive_periods(self, boolean_series: pd.Series) -> List[int]:
        """Get lengths of consecutive True periods."""
        periods = []
        current_length = 0

        for value in boolean_series:
            if value:
                current_length += 1
            else:
                if current_length > 0:
                    periods.append(current_length)
                    current_length = 0

        # Add final period if series ends with True
        if current_length > 0:
            periods.append(current_length)

        return periods

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
        """
        if len(returns) <= 1:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.trading_days_per_year)
        return (
            excess_returns.mean()
            / excess_returns.std()
            * np.sqrt(self.trading_days_per_year)
        )

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio (downside deviation version of Sharpe ratio).

        Sortino Ratio = (Portfolio Return - Risk Free Rate) / Downside Deviation
        """
        if len(returns) <= 1:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.trading_days_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0

        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0

        return (
            excess_returns.mean()
            / downside_deviation
            * np.sqrt(self.trading_days_per_year)
        )

    def _calculate_calmar_ratio(
        self, annualized_return: float, max_drawdown: float
    ) -> float:
        """
        Calculate Calmar ratio.

        Calmar Ratio = Annualized Return / Maximum Drawdown
        """
        if max_drawdown == 0:
            return np.inf if annualized_return > 0 else 0.0
        return annualized_return / max_drawdown

    def _calculate_probabilistic_sharpe_ratio(
        self, returns: pd.Series, sharpe_ratio: float
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio (PSR) from AFML.

        PSR indicates the probability that the estimated Sharpe ratio is greater
        than a given threshold, accounting for estimation uncertainty.
        """
        if len(returns) <= 2:
            return 0.0

        n = len(returns)
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Handle NaN values
        if np.isnan(skew):
            skew = 0.0
        if np.isnan(kurt):
            kurt = 0.0

        # Standard error of Sharpe ratio estimate
        sr_variance = (
            1
            + 0.5 * sharpe_ratio**2
            - skew * sharpe_ratio
            + (kurt - 3) / 4 * sharpe_ratio**2
        ) / (n - 1)

        # Ensure variance is positive
        if sr_variance <= 0:
            return 0.5  # Neutral probability

        sr_std = np.sqrt(sr_variance)

        # PSR with threshold of 0 (probability that true SR > 0)
        if sr_std == 0:
            return 1.0 if sharpe_ratio > 0 else 0.0

        psr = stats.norm.cdf(sharpe_ratio / sr_std)

        # Ensure result is valid
        if np.isnan(psr) or np.isinf(psr):
            return 0.5

        return max(0.0, min(1.0, psr))

    def _calculate_deflated_sharpe_ratio(
        self, returns: pd.Series, sharpe_ratio: float
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio (DSR) from AFML.

        DSR adjusts for multiple testing and selection bias in strategy development.
        """
        if len(returns) <= 2:
            return 0.0

        n = len(returns)
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Handle NaN values
        if np.isnan(skew):
            skew = 0.0
        if np.isnan(kurt):
            kurt = 0.0

        # Assume moderate multiple testing (can be adjusted based on strategy development process)
        trials = 10  # Number of strategies tested

        # Standard error adjustment
        sr_variance = (
            1
            + 0.5 * sharpe_ratio**2
            - skew * sharpe_ratio
            + (kurt - 3) / 4 * sharpe_ratio**2
        ) / (n - 1)

        # Ensure variance is positive
        if sr_variance <= 0:
            return 0.5  # Neutral probability

        sr_std = np.sqrt(sr_variance)

        # Expected maximum Sharpe ratio under null hypothesis
        expected_max_sr = (1 - np.euler_gamma) * stats.norm.ppf(
            1 - 1.0 / trials
        ) + np.euler_gamma * stats.norm.ppf(1 - 1.0 / (trials * np.e))

        # Deflated Sharpe ratio
        if sr_std == 0:
            return 1.0 if sharpe_ratio > expected_max_sr else 0.0

        dsr = stats.norm.cdf((sharpe_ratio - expected_max_sr) / sr_std)

        # Ensure result is valid
        if np.isnan(dsr) or np.isinf(dsr):
            return 0.5

        return max(0.0, min(1.0, dsr))

    def _calculate_minimum_track_record_length(
        self, returns: pd.Series, sharpe_ratio: float
    ) -> float:
        """
        Calculate Minimum Track Record Length (MTRL) from AFML.

        MTRL is the minimum track record length required for the Sharpe ratio
        to be statistically significant.
        """
        if len(returns) <= 2 or sharpe_ratio <= 0:
            return np.inf

        skew = returns.skew()
        kurt = returns.kurtosis()

        # Target Sharpe ratio threshold (e.g., 0.5)
        target_sr = 0.5

        # Calculate minimum track record length
        mtrl = (
            1
            + (
                1
                + 0.5 * sharpe_ratio**2
                - skew * sharpe_ratio
                + (kurt - 3) / 4 * sharpe_ratio**2
            )
            * (stats.norm.ppf(0.95) / (sharpe_ratio - target_sr)) ** 2
        )

        return max(0, mtrl)

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate trade-based performance metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
            }

        total_trades = len(trades)
        profits = [trade.get("pnl", 0.0) for trade in trades]

        winning_trades = sum(1 for pnl in profits if pnl > 0)
        losing_trades = sum(1 for pnl in profits if pnl < 0)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        winning_profits = [pnl for pnl in profits if pnl > 0]
        losing_profits = [pnl for pnl in profits if pnl < 0]

        avg_win = np.mean(winning_profits) if winning_profits else 0.0
        avg_loss = np.mean(losing_profits) if losing_profits else 0.0

        gross_profit = sum(winning_profits)
        gross_loss = abs(sum(losing_profits))

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf if gross_profit > 0 else 0.0
        )

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0
        return -np.percentile(returns, confidence_level * 100)

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall."""
        if len(returns) == 0:
            return 0.0

        var_threshold = -self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var_threshold]

        return -tail_returns.mean() if len(tail_returns) > 0 else 0.0

    def _calculate_beta(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Optional[float]:
        """Calculate beta relative to benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) <= 1:
            return None

        aligned_returns = pd.DataFrame(
            {"portfolio": returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_returns) <= 1:
            return None

        covariance = aligned_returns.cov().iloc[0, 1]
        benchmark_variance = aligned_returns["benchmark"].var()

        return covariance / benchmark_variance if benchmark_variance != 0 else None

    def _calculate_alpha(
        self, returns: pd.Series, benchmark_returns: pd.Series, beta: Optional[float]
    ) -> Optional[float]:
        """Calculate alpha (Jensen's alpha) relative to benchmark."""
        if beta is None or len(returns) != len(benchmark_returns):
            return None

        portfolio_return = returns.mean() * self.trading_days_per_year
        benchmark_return = benchmark_returns.mean() * self.trading_days_per_year

        expected_return = self.risk_free_rate + beta * (
            benchmark_return - self.risk_free_rate
        )

        return portfolio_return - expected_return

    def _calculate_information_ratio(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Optional[float]:
        """Calculate information ratio."""
        if len(returns) != len(benchmark_returns) or len(returns) <= 1:
            return None

        aligned_data = pd.DataFrame(
            {"portfolio": returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_data) <= 1:
            return None

        excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]
        tracking_error = excess_returns.std()

        if tracking_error == 0:
            return None

        return (
            excess_returns.mean() / tracking_error * np.sqrt(self.trading_days_per_year)
        )

    def _calculate_tracking_error(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Optional[float]:
        """Calculate tracking error (standard deviation of excess returns)."""
        if len(returns) != len(benchmark_returns) or len(returns) <= 1:
            return None

        aligned_data = pd.DataFrame(
            {"portfolio": returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_data) <= 1:
            return None

        excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]
        return excess_returns.std() * np.sqrt(self.trading_days_per_year)

    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics object for edge cases."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            volatility=0.0,
            annualized_volatility=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            probabilistic_sharpe_ratio=0.0,
            deflated_sharpe_ratio=0.0,
            minimum_track_record_length=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            skewness=0.0,
            kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
        )


def create_performance_report(metrics: PerformanceMetrics) -> Dict[str, Any]:
    """
    Create a comprehensive performance report.

    Args:
        metrics: PerformanceMetrics object

    Returns:
        Dictionary containing formatted performance report
    """
    return {
        "summary": {
            "Total Return": f"{metrics.total_return:.2%}",
            "Annualized Return": f"{metrics.annualized_return:.2%}",
            "Volatility": f"{metrics.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{metrics.sharpe_ratio:.3f}",
            "Max Drawdown": f"{metrics.max_drawdown:.2%}",
        },
        "returns": {
            "Total Return": metrics.total_return,
            "Annualized Return": metrics.annualized_return,
            "Cumulative Return": metrics.cumulative_return,
        },
        "risk": {
            "Volatility": metrics.volatility,
            "Annualized Volatility": metrics.annualized_volatility,
            "Max Drawdown": metrics.max_drawdown,
            "Average Drawdown": metrics.avg_drawdown,
            "VaR 95%": metrics.var_95,
            "CVaR 95%": metrics.cvar_95,
        },
        "risk_adjusted": {
            "Sharpe Ratio": metrics.sharpe_ratio,
            "Sortino Ratio": metrics.sortino_ratio,
            "Calmar Ratio": metrics.calmar_ratio,
            "Information Ratio": metrics.information_ratio,
        },
        "advanced": {
            "Probabilistic Sharpe Ratio": metrics.probabilistic_sharpe_ratio,
            "Deflated Sharpe Ratio": metrics.deflated_sharpe_ratio,
            "Minimum Track Record Length": metrics.minimum_track_record_length,
        },
        "trades": {
            "Total Trades": metrics.total_trades,
            "Win Rate": f"{metrics.win_rate:.2%}",
            "Average Win": f"{metrics.avg_win:.2f}",
            "Average Loss": f"{metrics.avg_loss:.2f}",
            "Profit Factor": f"{metrics.profit_factor:.3f}",
        },
        "distribution": {"Skewness": metrics.skewness, "Kurtosis": metrics.kurtosis},
    }
