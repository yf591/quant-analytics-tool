"""
Volatility Analysis Module

AFML-compliant volatility estimation and analysis functions.
Implements various volatility models and estimation methods
based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolatilityStatistics:
    """Container for volatility analysis results."""

    current_volatility: float
    average_volatility: float
    volatility_std: float
    volatility_skewness: float
    volatility_kurtosis: float
    garch_persistence: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "current_volatility": self.current_volatility,
            "average_volatility": self.average_volatility,
            "volatility_std": self.volatility_std,
            "volatility_skewness": self.volatility_skewness,
            "volatility_kurtosis": self.volatility_kurtosis,
            "garch_persistence": self.garch_persistence,
        }


class VolatilityAnalyzer:
    """
    AFML-compliant volatility analysis class.

    Provides comprehensive volatility estimation and analysis capabilities
    following financial machine learning best practices.
    """

    def __init__(self, window: int = 30):
        """
        Initialize volatility analyzer.

        Args:
            window: Default window size for rolling calculations
        """
        self.window = window
        logger.info(f"VolatilityAnalyzer initialized with window={window}")

    def calculate_simple_volatility(
        self, returns: pd.Series, window: Optional[int] = None, annualize: bool = True
    ) -> pd.Series:
        """
        Calculate simple historical volatility.

        Args:
            returns: Returns series
            window: Rolling window size (if None, uses self.window)
            annualize: Whether to annualize volatility

        Returns:
            Volatility series
        """
        try:
            window = window or self.window
            vol = returns.rolling(window=window).std()

            if annualize:
                # Annualize assuming daily data
                vol *= np.sqrt(252)

            logger.debug(f"Calculated simple volatility with window={window}")
            return vol.dropna()
        except Exception as e:
            logger.error(f"Error calculating simple volatility: {e}")
            raise

    def calculate_ewma_volatility(
        self, returns: pd.Series, lambda_param: float = 0.94, annualize: bool = True
    ) -> pd.Series:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility.

        AFML-preferred method for volatility estimation due to adaptive nature.

        Args:
            returns: Returns series
            lambda_param: Decay parameter (RiskMetrics uses 0.94 for daily data)
            annualize: Whether to annualize volatility

        Returns:
            EWMA volatility series
        """
        try:
            # Calculate squared returns
            squared_returns = returns**2

            # Apply EWMA
            ewma_var = squared_returns.ewm(alpha=1 - lambda_param, adjust=False).mean()
            ewma_vol = np.sqrt(ewma_var)

            if annualize:
                ewma_vol *= np.sqrt(252)

            logger.debug(f"Calculated EWMA volatility with lambda={lambda_param}")
            return ewma_vol
        except Exception as e:
            logger.error(f"Error calculating EWMA volatility: {e}")
            raise

    def calculate_garman_klass_volatility(
        self,
        ohlc_data: pd.DataFrame,
        window: Optional[int] = None,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator.

        More efficient than close-to-close volatility when OHLC data is available.

        Args:
            ohlc_data: DataFrame with columns ['Open', 'High', 'Low', 'Close']
            window: Rolling window size
            annualize: Whether to annualize volatility

        Returns:
            Garman-Klass volatility series
        """
        try:
            window = window or self.window

            # Ensure required columns exist
            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in ohlc_data.columns for col in required_cols):
                raise ValueError(f"OHLC data must contain columns: {required_cols}")

            # Calculate log prices
            log_high = np.log(ohlc_data["High"])
            log_low = np.log(ohlc_data["Low"])
            log_close = np.log(ohlc_data["Close"])
            log_open = np.log(ohlc_data["Open"])

            # Garman-Klass estimator
            gk_var = (
                0.5 * (log_high - log_low) ** 2
                - (2 * np.log(2) - 1) * (log_close - log_open) ** 2
            )

            # Rolling average
            gk_vol = np.sqrt(gk_var.rolling(window=window).mean())

            if annualize:
                gk_vol *= np.sqrt(252)

            logger.debug(f"Calculated Garman-Klass volatility with window={window}")
            return gk_vol.dropna()
        except Exception as e:
            logger.error(f"Error calculating Garman-Klass volatility: {e}")
            raise

    def calculate_parkinson_volatility(
        self,
        ohlc_data: pd.DataFrame,
        window: Optional[int] = None,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Calculate Parkinson volatility estimator.

        Uses only high and low prices, more efficient than close-to-close.

        Args:
            ohlc_data: DataFrame with columns ['High', 'Low']
            window: Rolling window size
            annualize: Whether to annualize volatility

        Returns:
            Parkinson volatility series
        """
        try:
            window = window or self.window

            # Calculate log high/low ratio
            log_hl = np.log(ohlc_data["High"] / ohlc_data["Low"])

            # Parkinson estimator
            park_var = (1 / (4 * np.log(2))) * (log_hl**2)
            park_vol = np.sqrt(park_var.rolling(window=window).mean())

            if annualize:
                park_vol *= np.sqrt(252)

            logger.debug(f"Calculated Parkinson volatility with window={window}")
            return park_vol.dropna()
        except Exception as e:
            logger.error(f"Error calculating Parkinson volatility: {e}")
            raise

    def calculate_realized_volatility(
        self, returns: pd.Series, frequency: str = "D"
    ) -> pd.Series:
        """
        Calculate realized volatility by aggregating high-frequency returns.

        Args:
            returns: High-frequency returns series
            frequency: Aggregation frequency ('D', 'W', 'M')

        Returns:
            Realized volatility series
        """
        try:
            # Square the returns
            squared_returns = returns**2

            # Aggregate by frequency
            realized_var = squared_returns.resample(frequency).sum()
            realized_vol = np.sqrt(realized_var)

            logger.debug(f"Calculated realized volatility with frequency={frequency}")
            return realized_vol
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            raise

    def calculate_volatility_clustering(
        self, returns: pd.Series, window: int = 252
    ) -> pd.Series:
        """
        Measure volatility clustering using autocorrelation of squared returns.

        Args:
            returns: Returns series
            window: Window for rolling correlation

        Returns:
            Volatility clustering measure
        """
        try:
            squared_returns = returns**2

            # Calculate rolling autocorrelation at lag 1
            clustering = squared_returns.rolling(window=window).apply(
                lambda x: pd.Series(x).autocorr(lag=1), raw=False
            )

            logger.debug(f"Calculated volatility clustering with window={window}")
            return clustering
        except Exception as e:
            logger.error(f"Error calculating volatility clustering: {e}")
            raise

    def analyze_volatility(
        self, returns: pd.Series, ohlc_data: Optional[pd.DataFrame] = None
    ) -> VolatilityStatistics:
        """
        Comprehensive volatility analysis.

        Args:
            returns: Returns series
            ohlc_data: Optional OHLC data for advanced estimators

        Returns:
            VolatilityStatistics object with analysis results
        """
        try:
            # Calculate simple volatility
            simple_vol = self.calculate_simple_volatility(returns)

            # Calculate EWMA volatility
            ewma_vol = self.calculate_ewma_volatility(returns)

            # Use the most recent and average values
            current_volatility = ewma_vol.iloc[-1]
            average_volatility = simple_vol.mean()

            # Volatility of volatility (using simple volatility)
            vol_std = simple_vol.std()
            vol_skew = simple_vol.skew()
            vol_kurt = simple_vol.kurtosis()

            result = VolatilityStatistics(
                current_volatility=current_volatility,
                average_volatility=average_volatility,
                volatility_std=vol_std,
                volatility_skewness=vol_skew,
                volatility_kurtosis=vol_kurt,
            )

            logger.info(f"Volatility analysis completed")
            return result

        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            raise


# Standalone functions for backward compatibility and convenience
def calculate_simple_volatility(
    returns: pd.Series, window: int = 30, annualize: bool = True
) -> pd.Series:
    """Calculate simple historical volatility."""
    analyzer = VolatilityAnalyzer(window=window)
    return analyzer.calculate_simple_volatility(returns, annualize=annualize)


def calculate_ewma_volatility(
    returns: pd.Series, lambda_param: float = 0.94, annualize: bool = True
) -> pd.Series:
    """Calculate EWMA volatility."""
    analyzer = VolatilityAnalyzer()
    return analyzer.calculate_ewma_volatility(returns, lambda_param, annualize)


def calculate_garman_klass_volatility(
    ohlc_data: pd.DataFrame, window: int = 30, annualize: bool = True
) -> pd.Series:
    """Calculate Garman-Klass volatility."""
    analyzer = VolatilityAnalyzer(window=window)
    return analyzer.calculate_garman_klass_volatility(ohlc_data, annualize=annualize)
