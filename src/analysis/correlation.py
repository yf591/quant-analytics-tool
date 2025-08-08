"""
Correlation Analysis Module

AFML-compliant correlation analysis functions.
Implements various correlation measures and dynamic correlation analysis
based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationStatistics:
    """Container for correlation analysis results."""

    correlation_matrix: pd.DataFrame
    average_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_stability: float
    eigenvalues: np.ndarray
    condition_number: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "correlation_matrix": self.correlation_matrix.to_dict(),
            "average_correlation": self.average_correlation,
            "max_correlation": self.max_correlation,
            "min_correlation": self.min_correlation,
            "correlation_stability": self.correlation_stability,
            "eigenvalues": self.eigenvalues.tolist(),
            "condition_number": self.condition_number,
        }


class CorrelationAnalyzer:
    """
    AFML-compliant correlation analysis class.

    Provides comprehensive correlation analysis capabilities
    following financial machine learning best practices.
    """

    def __init__(self, method: str = "pearson"):
        """
        Initialize correlation analyzer.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        self.method = method
        logger.info(f"CorrelationAnalyzer initialized with method={method}")

    def calculate_correlation_matrix(
        self, data: pd.DataFrame, method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Args:
            data: DataFrame with financial time series
            method: Correlation method (if None, uses self.method)

        Returns:
            Correlation matrix
        """
        try:
            method = method or self.method
            corr_matrix = data.corr(method=method)

            logger.debug(
                f"Calculated {method} correlation matrix for {len(data.columns)} assets"
            )
            return corr_matrix
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            raise

    def calculate_rolling_correlation(
        self,
        data1: pd.Series,
        data2: pd.Series,
        window: int = 60,
        method: Optional[str] = None,
    ) -> pd.Series:
        """
        Calculate rolling correlation between two series.

        Args:
            data1: First time series
            data2: Second time series
            window: Rolling window size
            method: Correlation method

        Returns:
            Rolling correlation series
        """
        try:
            method = method or self.method

            if method == "pearson":
                rolling_corr = data1.rolling(window=window).corr(data2)
            else:
                # For non-Pearson methods, use apply with corr function
                rolling_corr = data1.rolling(window=window).apply(
                    lambda x: x.corr(data2.iloc[x.index], method=method), raw=False
                )

            logger.debug(
                f"Calculated rolling {method} correlation with window={window}"
            )
            return rolling_corr
        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}")
            raise

    def calculate_dynamic_correlation(
        self, data: pd.DataFrame, window: int = 60, method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate dynamic (rolling) correlation matrix.

        Args:
            data: DataFrame with financial time series
            window: Rolling window size
            method: Correlation method

        Returns:
            DataFrame with dynamic correlations (MultiIndex columns)
        """
        try:
            method = method or self.method

            # Get all unique pairs
            assets = data.columns.tolist()
            pairs = [
                (assets[i], assets[j])
                for i in range(len(assets))
                for j in range(i + 1, len(assets))
            ]

            dynamic_corrs = []

            for asset1, asset2 in pairs:
                corr_series = self.calculate_rolling_correlation(
                    data[asset1], data[asset2], window, method
                )
                corr_series.name = f"{asset1}_{asset2}"
                dynamic_corrs.append(corr_series)

            result = pd.concat(dynamic_corrs, axis=1)

            logger.debug(f"Calculated dynamic correlations for {len(pairs)} pairs")
            return result
        except Exception as e:
            logger.error(f"Error calculating dynamic correlation: {e}")
            raise

    def calculate_ewma_correlation(
        self, data1: pd.Series, data2: pd.Series, lambda_param: float = 0.94
    ) -> pd.Series:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) correlation.

        Args:
            data1: First time series
            data2: Second time series
            lambda_param: Decay parameter

        Returns:
            EWMA correlation series
        """
        try:
            # Calculate EWMA covariance and variances
            mean1 = data1.ewm(alpha=1 - lambda_param).mean()
            mean2 = data2.ewm(alpha=1 - lambda_param).mean()

            # Centered data
            centered1 = data1 - mean1
            centered2 = data2 - mean2

            # EWMA covariance
            cov = (centered1 * centered2).ewm(alpha=1 - lambda_param).mean()

            # EWMA variances
            var1 = (centered1**2).ewm(alpha=1 - lambda_param).mean()
            var2 = (centered2**2).ewm(alpha=1 - lambda_param).mean()

            # EWMA correlation
            ewma_corr = cov / np.sqrt(var1 * var2)

            logger.debug(f"Calculated EWMA correlation with lambda={lambda_param}")
            return ewma_corr
        except Exception as e:
            logger.error(f"Error calculating EWMA correlation: {e}")
            raise

    def calculate_correlation_stability(
        self, data: pd.DataFrame, window: int = 60, method: Optional[str] = None
    ) -> float:
        """
        Calculate correlation stability measure.

        Measures how stable correlations are over time by calculating
        the standard deviation of rolling correlations.

        Args:
            data: DataFrame with financial time series
            window: Rolling window size
            method: Correlation method

        Returns:
            Average correlation stability (lower is more stable)
        """
        try:
            dynamic_corr = self.calculate_dynamic_correlation(data, window, method)

            # Calculate standard deviation of each correlation series
            stability_measures = dynamic_corr.std()

            # Return average stability
            avg_stability = stability_measures.mean()

            logger.debug(f"Correlation stability: {avg_stability:.4f}")
            return avg_stability
        except Exception as e:
            logger.error(f"Error calculating correlation stability: {e}")
            raise

    def calculate_portfolio_correlation(
        self, returns: pd.DataFrame, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate portfolio-level correlation measure.

        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights (if None, uses equal weights)

        Returns:
            Portfolio correlation measure
        """
        try:
            if weights is None:
                weights = np.ones(len(returns.columns)) / len(returns.columns)

            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(returns)

            # Weighted average correlation
            portfolio_corr = np.sum(
                weights.reshape(-1, 1) * weights.reshape(1, -1) * corr_matrix.values
            )

            logger.debug(f"Portfolio correlation: {portfolio_corr:.4f}")
            return portfolio_corr
        except Exception as e:
            logger.error(f"Error calculating portfolio correlation: {e}")
            raise

    def detect_correlation_regimes(
        self, data: pd.DataFrame, window: int = 60, threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Detect correlation regime changes.

        Args:
            data: DataFrame with financial time series
            window: Rolling window size
            threshold: Threshold for regime change detection

        Returns:
            DataFrame with regime indicators
        """
        try:
            # Calculate average rolling correlation
            dynamic_corr = self.calculate_dynamic_correlation(data, window)
            avg_corr = dynamic_corr.mean(axis=1)

            # Detect regime changes using rolling standard deviation
            corr_vol = avg_corr.rolling(window=window // 2).std()

            # High correlation regime when volatility is low and correlation is high
            high_corr_regime = (avg_corr > avg_corr.median()) & (
                corr_vol < corr_vol.median()
            )

            # Create regime DataFrame
            regimes = pd.DataFrame(index=avg_corr.index)
            regimes["average_correlation"] = avg_corr
            regimes["correlation_volatility"] = corr_vol
            regimes["high_correlation_regime"] = high_corr_regime

            logger.debug(f"Detected correlation regimes")
            return regimes
        except Exception as e:
            logger.error(f"Error detecting correlation regimes: {e}")
            raise

    def analyze_correlation_structure(
        self, data: pd.DataFrame
    ) -> CorrelationStatistics:
        """
        Comprehensive correlation structure analysis.

        Args:
            data: DataFrame with financial time series

        Returns:
            CorrelationStatistics object with analysis results
        """
        try:
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(data)

            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix.values[mask]

            # Basic statistics
            avg_corr = correlations.mean()
            max_corr = correlations.max()
            min_corr = correlations.min()

            # Correlation stability
            stability = self.calculate_correlation_stability(data)

            # Eigenvalue analysis
            eigenvalues = np.linalg.eigvals(corr_matrix.values)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            condition_number = (
                eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] != 0 else np.inf
            )

            result = CorrelationStatistics(
                correlation_matrix=corr_matrix,
                average_correlation=avg_corr,
                max_correlation=max_corr,
                min_correlation=min_corr,
                correlation_stability=stability,
                eigenvalues=eigenvalues,
                condition_number=condition_number,
            )

            logger.info(f"Correlation structure analysis completed")
            return result

        except Exception as e:
            logger.error(f"Error in correlation structure analysis: {e}")
            raise

    def calculate_tail_correlation(
        self, data1: pd.Series, data2: pd.Series, quantile: float = 0.05
    ) -> float:
        """
        Calculate tail correlation (correlation in extreme events).

        Args:
            data1: First time series
            data2: Second time series
            quantile: Quantile for defining tail events

        Returns:
            Tail correlation coefficient
        """
        try:
            # Define tail events for both series
            threshold1_low = data1.quantile(quantile)
            threshold1_high = data1.quantile(1 - quantile)
            threshold2_low = data2.quantile(quantile)
            threshold2_high = data2.quantile(1 - quantile)

            # Create tail event indicators
            tail1 = (data1 <= threshold1_low) | (data1 >= threshold1_high)
            tail2 = (data2 <= threshold2_low) | (data2 >= threshold2_high)

            # Calculate correlation during tail events
            tail_mask = tail1 | tail2
            if tail_mask.sum() < 10:  # Need minimum observations
                return np.nan

            tail_corr = data1[tail_mask].corr(data2[tail_mask])

            logger.debug(f"Tail correlation at {quantile} quantile: {tail_corr:.4f}")
            return tail_corr
        except Exception as e:
            logger.error(f"Error calculating tail correlation: {e}")
            raise


# Standalone functions for backward compatibility and convenience
def calculate_correlation_matrix(
    data: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Calculate correlation matrix."""
    analyzer = CorrelationAnalyzer(method=method)
    return analyzer.calculate_correlation_matrix(data)


def calculate_rolling_correlation(
    data1: pd.Series, data2: pd.Series, window: int = 60, method: str = "pearson"
) -> pd.Series:
    """Calculate rolling correlation."""
    analyzer = CorrelationAnalyzer(method=method)
    return analyzer.calculate_rolling_correlation(data1, data2, window)
