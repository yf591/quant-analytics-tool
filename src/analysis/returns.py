"""
Return Analysis Module

AFML-compliant return calculation and analysis functions.
Implements various return metrics and statistical analysis methods
based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReturnStatistics:
    """Container for return analysis results."""

    mean: float
    std: float
    skewness: float
    kurtosis: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    annualized_return: float
    annualized_volatility: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
        }


class ReturnAnalyzer:
    """
    AFML-compliant return analysis class.

    Provides comprehensive return calculation and analysis capabilities
    following financial machine learning best practices.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize return analyzer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        logger.info("ReturnAnalyzer initialized")

    def calculate_simple_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}

        Args:
            prices: Price series with datetime index

        Returns:
            Simple returns series
        """
        try:
            returns = prices.pct_change().dropna()
            logger.debug(f"Calculated simple returns for {len(returns)} periods")
            return returns
        except Exception as e:
            logger.error(f"Error calculating simple returns: {e}")
            raise

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate log returns: R_t = ln(P_t / P_{t-1})

        Preferred for AFML applications due to time-additivity.

        Args:
            prices: Price series with datetime index

        Returns:
            Log returns series
        """
        try:
            log_returns = np.log(prices / prices.shift(1)).dropna()
            logger.debug(f"Calculated log returns for {len(log_returns)} periods")
            return log_returns
        except Exception as e:
            logger.error(f"Error calculating log returns: {e}")
            raise

    def calculate_cumulative_returns(
        self, returns: pd.Series, method: str = "simple"
    ) -> pd.Series:
        """
        Calculate cumulative returns.

        Args:
            returns: Returns series
            method: 'simple' or 'log' returns method

        Returns:
            Cumulative returns series
        """
        try:
            if method == "simple":
                cum_returns = (1 + returns).cumprod() - 1
            elif method == "log":
                cum_returns = np.exp(returns.cumsum()) - 1
            else:
                raise ValueError("Method must be 'simple' or 'log'")

            logger.debug(f"Calculated cumulative returns using {method} method")
            return cum_returns
        except Exception as e:
            logger.error(f"Error calculating cumulative returns: {e}")
            raise

    def calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.

        Args:
            prices: Price series

        Returns:
            Drawdown series (negative values)
        """
        try:
            # Calculate running maximum (peak)
            peak = prices.cummax()
            # Calculate drawdown
            drawdown = (prices - peak) / peak
            logger.debug(f"Calculated drawdown series")
            return drawdown
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            raise

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            prices: Price series

        Returns:
            Maximum drawdown (negative value)
        """
        try:
            drawdown = self.calculate_drawdown(prices)
            max_dd = drawdown.min()
            logger.debug(f"Maximum drawdown: {max_dd:.4f}")
            return max_dd
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            raise

    def calculate_sharpe_ratio(
        self, returns: pd.Series, annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Returns series
            annualize: Whether to annualize the ratio

        Returns:
            Sharpe ratio
        """
        try:
            excess_returns = returns - self.risk_free_rate
            sharpe = excess_returns.mean() / excess_returns.std()

            if annualize:
                # Assume daily data, annualize with sqrt(252)
                sharpe *= np.sqrt(252)

            logger.debug(f"Sharpe ratio: {sharpe:.4f}")
            return sharpe
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            raise

    def analyze_returns(
        self, prices: pd.Series, return_type: str = "simple"
    ) -> ReturnStatistics:
        """
        Comprehensive return analysis.

        Args:
            prices: Price series
            return_type: 'simple' or 'log' returns

        Returns:
            ReturnStatistics object with analysis results
        """
        try:
            # Calculate returns
            if return_type == "simple":
                returns = self.calculate_simple_returns(prices)
            elif return_type == "log":
                returns = self.calculate_log_returns(prices)
            else:
                raise ValueError("return_type must be 'simple' or 'log'")

            # Calculate statistics
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            # Calculate performance metrics
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(prices)

            # Calculate total and annualized returns
            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

            # Annualized metrics (assuming daily data)
            trading_days = 252
            periods = len(returns)
            years = periods / trading_days

            annualized_return = (1 + total_return) ** (1 / years) - 1
            annualized_volatility = std_return * np.sqrt(trading_days)

            result = ReturnStatistics(
                mean=mean_return,
                std=std_return,
                skewness=skewness,
                kurtosis=kurtosis,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_return=total_return,
                annualized_return=annualized_return,
                annualized_volatility=annualized_volatility,
            )

            logger.info(f"Return analysis completed for {len(returns)} periods")
            return result

        except Exception as e:
            logger.error(f"Error in return analysis: {e}")
            raise


# Standalone functions for backward compatibility and convenience
def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns."""
    analyzer = ReturnAnalyzer()
    return analyzer.calculate_simple_returns(prices)


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns."""
    analyzer = ReturnAnalyzer()
    return analyzer.calculate_log_returns(prices)


def calculate_cumulative_returns(
    returns: pd.Series, method: str = "simple"
) -> pd.Series:
    """Calculate cumulative returns."""
    analyzer = ReturnAnalyzer()
    return analyzer.calculate_cumulative_returns(returns, method)
