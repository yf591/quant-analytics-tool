"""
Statistics Analysis Module

AFML-compliant statistical analysis functions.
Implements comprehensive statistical measures and distribution analysis
based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BasicStatistics:
    """Container for basic statistical measures."""

    count: int
    mean: float
    std: float
    min: float
    percentile_25: float
    median: float
    percentile_75: float
    max: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "25%": self.percentile_25,
            "median": self.median,
            "75%": self.percentile_75,
            "max": self.max,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


@dataclass
class RiskMetrics:
    """Container for risk analysis results."""

    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional Value at Risk at 95%
    cvar_99: float  # Conditional Value at Risk at 99%
    downside_deviation: float
    sortino_ratio: float
    calmar_ratio: float
    maximum_drawdown: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "downside_deviation": self.downside_deviation,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "maximum_drawdown": self.maximum_drawdown,
        }


@dataclass
class DistributionAnalysis:
    """Container for distribution analysis results."""

    normality_test_statistic: float
    normality_p_value: float
    is_normal: bool
    autocorrelation_lag1: float
    ljung_box_statistic: float
    ljung_box_p_value: float
    has_autocorrelation: bool
    arch_test_statistic: float
    arch_test_p_value: float
    has_arch_effects: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "normality_test_statistic": self.normality_test_statistic,
            "normality_p_value": self.normality_p_value,
            "is_normal": self.is_normal,
            "autocorrelation_lag1": self.autocorrelation_lag1,
            "ljung_box_statistic": self.ljung_box_statistic,
            "ljung_box_p_value": self.ljung_box_p_value,
            "has_autocorrelation": self.has_autocorrelation,
            "arch_test_statistic": self.arch_test_statistic,
            "arch_test_p_value": self.arch_test_p_value,
            "has_arch_effects": self.has_arch_effects,
        }


class StatisticsAnalyzer:
    """
    AFML-compliant statistical analysis class.

    Provides comprehensive statistical analysis capabilities
    following financial machine learning best practices.
    """

    def __init__(self, confidence_level: float = 0.05):
        """
        Initialize statistics analyzer.

        Args:
            confidence_level: Significance level for statistical tests
        """
        self.confidence_level = confidence_level
        logger.info(
            f"StatisticsAnalyzer initialized with confidence_level={confidence_level}"
        )

    def calculate_basic_statistics(self, data: pd.Series) -> BasicStatistics:
        """
        Calculate comprehensive basic statistics.

        Args:
            data: Data series for analysis

        Returns:
            BasicStatistics object with results
        """
        try:
            # Basic descriptive statistics
            desc = data.describe()

            result = BasicStatistics(
                count=int(desc["count"]),
                mean=desc["mean"],
                std=desc["std"],
                min=desc["min"],
                percentile_25=desc["25%"],
                median=desc["50%"],
                percentile_75=desc["75%"],
                max=desc["max"],
                skewness=data.skew(),
                kurtosis=data.kurtosis(),
            )

            logger.debug(f"Calculated basic statistics for {len(data)} observations")
            return result

        except Exception as e:
            logger.error(f"Error calculating basic statistics: {e}")
            raise

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Returns series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)

        Returns:
            VaR value (positive number represents loss)
        """
        try:
            var = -np.percentile(returns, confidence_level * 100)
            logger.debug(f"VaR at {(1-confidence_level)*100}% confidence: {var:.4f}")
            return var
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            raise

    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

        Args:
            returns: Returns series
            confidence_level: Confidence level

        Returns:
            CVaR value (positive number represents loss)
        """
        try:
            var = self.calculate_var(returns, confidence_level)
            # CVaR is the expected value of losses beyond VaR
            cvar = -returns[returns <= -var].mean()
            logger.debug(f"CVaR at {(1-confidence_level)*100}% confidence: {cvar:.4f}")
            return cvar
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            raise

    def calculate_downside_deviation(
        self, returns: pd.Series, target_return: float = 0.0
    ) -> float:
        """
        Calculate downside deviation.

        Args:
            returns: Returns series
            target_return: Minimum acceptable return

        Returns:
            Downside deviation
        """
        try:
            downside_returns = returns[returns < target_return]
            downside_dev = np.sqrt(((downside_returns - target_return) ** 2).mean())
            logger.debug(f"Downside deviation: {downside_dev:.4f}")
            return downside_dev
        except Exception as e:
            logger.error(f"Error calculating downside deviation: {e}")
            raise

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
    ) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate
            target_return: Minimum acceptable return

        Returns:
            Sortino ratio
        """
        try:
            excess_return = returns.mean() - risk_free_rate
            downside_dev = self.calculate_downside_deviation(returns, target_return)

            if downside_dev == 0:
                return np.inf if excess_return > 0 else 0

            sortino = excess_return / downside_dev
            # Annualize
            sortino *= np.sqrt(252)

            logger.debug(f"Sortino ratio: {sortino:.4f}")
            return sortino
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            raise

    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """
        Calculate Calmar ratio (Annual return / Maximum drawdown).

        Args:
            returns: Returns series
            prices: Price series for drawdown calculation

        Returns:
            Calmar ratio
        """
        try:
            # Calculate annualized return
            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
            periods = len(returns)
            years = periods / 252  # Assuming daily data
            annualized_return = (1 + total_return) ** (1 / years) - 1

            # Calculate maximum drawdown
            peak = prices.cummax()
            drawdown = (prices - peak) / peak
            max_drawdown = abs(drawdown.min())

            if max_drawdown == 0:
                return np.inf if annualized_return > 0 else 0

            calmar = annualized_return / max_drawdown
            logger.debug(f"Calmar ratio: {calmar:.4f}")
            return calmar
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            raise

    def test_normality(self, data: pd.Series) -> Tuple[float, float]:
        """
        Test for normality using Jarque-Bera test.

        Args:
            data: Data series to test

        Returns:
            Tuple of (test_statistic, p_value)
        """
        try:
            statistic, p_value = stats.jarque_bera(data.dropna())
            logger.debug(
                f"Jarque-Bera test: statistic={statistic:.4f}, p-value={p_value:.4f}"
            )
            return statistic, p_value
        except Exception as e:
            logger.error(f"Error in normality test: {e}")
            raise

    def test_autocorrelation(
        self, data: pd.Series, lags: int = 10
    ) -> Tuple[float, float]:
        """
        Test for autocorrelation using Ljung-Box test.

        Args:
            data: Data series to test
            lags: Number of lags to test

        Returns:
            Tuple of (test_statistic, p_value)
        """
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            result = acorr_ljungbox(data.dropna(), lags=lags, return_df=True)
            # Use the overall test statistic and p-value
            statistic = result["lb_stat"].iloc[-1]
            p_value = result["lb_pvalue"].iloc[-1]

            logger.debug(
                f"Ljung-Box test: statistic={statistic:.4f}, p-value={p_value:.4f}"
            )
            return statistic, p_value
        except ImportError:
            logger.warning(
                "statsmodels not available, using simplified autocorrelation test"
            )
            # Simplified autocorrelation test
            autocorr = data.autocorr(lag=1)
            # Approximate test under null of no autocorrelation
            n = len(data)
            statistic = autocorr * np.sqrt(n)
            p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
            return statistic, p_value
        except Exception as e:
            logger.error(f"Error in autocorrelation test: {e}")
            raise

    def test_arch_effects(
        self, returns: pd.Series, lags: int = 5
    ) -> Tuple[float, float]:
        """
        Test for ARCH effects (heteroscedasticity).

        Args:
            returns: Returns series
            lags: Number of lags for the test

        Returns:
            Tuple of (test_statistic, p_value)
        """
        try:
            # ARCH test using squared returns
            squared_returns = returns**2
            squared_returns = squared_returns.dropna()

            if len(squared_returns) < lags + 10:  # Need sufficient data
                logger.warning("Insufficient data for ARCH test")
                return 0.0, 1.0

            # Create lagged variables with proper alignment
            from sklearn.linear_model import LinearRegression

            # Build the regression matrices with proper indexing
            X_list = []

            # Create lagged variables
            for i in range(1, lags + 1):
                lagged_series = squared_returns.shift(i)
                X_list.append(lagged_series)

            # Combine all lagged variables
            X_df = pd.concat(X_list, axis=1)
            X_df.columns = [f"lag_{i}" for i in range(1, lags + 1)]

            # Align X and y by dropping NaN values
            aligned_data = pd.concat([squared_returns, X_df], axis=1).dropna()

            if len(aligned_data) < lags + 5:  # Need minimum observations
                logger.warning("Insufficient aligned data for ARCH test")
                return 0.0, 1.0

            y = aligned_data.iloc[:, 0].values  # Current squared returns
            X = aligned_data.iloc[:, 1:].values  # Lagged squared returns

            if len(X) == 0 or len(y) == 0 or X.shape[0] != y.shape[0]:
                logger.warning("Data alignment issue in ARCH test")
                return 0.0, 1.0

            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # ARCH test statistic
            statistic = len(y) * r_squared
            p_value = 1 - stats.chi2.cdf(statistic, lags)

            logger.debug(f"ARCH test: statistic={statistic:.4f}, p-value={p_value:.4f}")
            return statistic, p_value

            logger.debug(f"ARCH test: statistic={statistic:.4f}, p-value={p_value:.4f}")
            return statistic, p_value
        except Exception as e:
            logger.error(f"Error in ARCH test: {e}")
            # Return neutral result if test fails
            return 0.0, 1.0

    def analyze_distribution(self, data: pd.Series) -> DistributionAnalysis:
        """
        Comprehensive distribution analysis.

        Args:
            data: Data series for analysis

        Returns:
            DistributionAnalysis object with results
        """
        try:
            # Normality test
            norm_stat, norm_p = self.test_normality(data)
            is_normal = norm_p > self.confidence_level

            # Autocorrelation
            autocorr_lag1 = data.autocorr(lag=1)
            ljung_stat, ljung_p = self.test_autocorrelation(data)
            has_autocorr = ljung_p <= self.confidence_level

            # ARCH effects test
            arch_stat, arch_p = self.test_arch_effects(data)
            has_arch = arch_p <= self.confidence_level

            result = DistributionAnalysis(
                normality_test_statistic=norm_stat,
                normality_p_value=norm_p,
                is_normal=is_normal,
                autocorrelation_lag1=autocorr_lag1,
                ljung_box_statistic=ljung_stat,
                ljung_box_p_value=ljung_p,
                has_autocorrelation=has_autocorr,
                arch_test_statistic=arch_stat,
                arch_test_p_value=arch_p,
                has_arch_effects=has_arch,
            )

            logger.info(f"Distribution analysis completed")
            return result

        except Exception as e:
            logger.error(f"Error in distribution analysis: {e}")
            raise

    def calculate_risk_metrics(
        self, returns: pd.Series, prices: pd.Series
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: Returns series
            prices: Price series

        Returns:
            RiskMetrics object with results
        """
        try:
            # VaR calculations
            var_95 = self.calculate_var(returns, 0.05)
            var_99 = self.calculate_var(returns, 0.01)

            # CVaR calculations
            cvar_95 = self.calculate_cvar(returns, 0.05)
            cvar_99 = self.calculate_cvar(returns, 0.01)

            # Other risk metrics
            downside_dev = self.calculate_downside_deviation(returns)
            sortino = self.calculate_sortino_ratio(returns)
            calmar = self.calculate_calmar_ratio(returns, prices)

            # Maximum drawdown
            peak = prices.cummax()
            drawdown = (prices - peak) / peak
            max_dd = abs(drawdown.min())

            result = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                downside_deviation=downside_dev,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                maximum_drawdown=max_dd,
            )

            logger.info(f"Risk metrics calculation completed")
            return result

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise


# Standalone functions for backward compatibility and convenience
def calculate_basic_statistics(data: pd.Series) -> BasicStatistics:
    """Calculate basic statistics."""
    analyzer = StatisticsAnalyzer()
    return analyzer.calculate_basic_statistics(data)


def calculate_risk_metrics(returns: pd.Series, prices: pd.Series) -> RiskMetrics:
    """Calculate risk metrics."""
    analyzer = StatisticsAnalyzer()
    return analyzer.calculate_risk_metrics(returns, prices)


def analyze_distribution(data: pd.Series) -> DistributionAnalysis:
    """Analyze data distribution."""
    analyzer = StatisticsAnalyzer()
    return analyzer.analyze_distribution(data)
