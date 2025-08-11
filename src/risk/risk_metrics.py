"""
Risk Metrics Module - Week 12 Risk Management

Advanced risk measurement and analysis tools including VaR, CVaR, drawdown analysis,
and AFML-based risk metrics following Chapter 15 methodologies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Advanced risk metrics calculator with AFML methodologies.

    Implements Value at Risk (VaR), Conditional Value at Risk (CVaR),
    drawdown analysis, and other risk measures following AFML best practices.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        rolling_window: int = 252,
        min_periods: int = 30,
    ):
        """
        Initialize RiskMetrics calculator.

        Args:
            confidence_level: Confidence level for VaR/CVaR calculations
            rolling_window: Rolling window for time-varying risk metrics
            min_periods: Minimum periods required for calculation
        """
        self.confidence_level = confidence_level
        self.rolling_window = rolling_window
        self.min_periods = min_periods

        # Risk-free rate (annualized)
        self.risk_free_rate = 0.02

        logger.info(
            f"RiskMetrics initialized with confidence={confidence_level:.2%}, "
            f"window={rolling_window}, min_periods={min_periods}"
        )

    def value_at_risk(
        self,
        returns: pd.Series,
        method: str = "parametric",
        confidence_level: Optional[float] = None,
    ) -> float:
        """
        Calculate Value at Risk (VaR) using various methods.

        Args:
            returns: Return series
            method: VaR method ('parametric', 'historical', 'cornish_fisher')
            confidence_level: Override default confidence level

        Returns:
            VaR value (positive number represents potential loss)
        """
        try:
            if len(returns) < self.min_periods:
                logger.warning("Insufficient data for VaR calculation")
                return np.nan

            confidence = confidence_level or self.confidence_level
            alpha = 1 - confidence

            if method == "parametric":
                return self._parametric_var(returns, alpha)
            elif method == "historical":
                return self._historical_var(returns, alpha)
            elif method == "cornish_fisher":
                return self._cornish_fisher_var(returns, alpha)
            else:
                raise ValueError(f"Unknown VaR method: {method}")

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return np.nan

    def _parametric_var(self, returns: pd.Series, alpha: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        mean_return = returns.mean()
        std_return = returns.std()

        # Z-score for given confidence level
        z_score = stats.norm.ppf(alpha)

        # VaR calculation (negative return = loss)
        var = -(mean_return + z_score * std_return)

        logger.debug(f"Parametric VaR: {var:.4f} (alpha={alpha:.3f})")
        return var

    def _historical_var(self, returns: pd.Series, alpha: float) -> float:
        """Calculate historical VaR using empirical quantiles."""
        # Sort returns and find quantile
        sorted_returns = returns.sort_values()
        quantile_index = int(alpha * len(sorted_returns))

        # Handle edge cases
        if quantile_index == 0:
            var = -sorted_returns.iloc[0]
        else:
            var = -sorted_returns.iloc[quantile_index - 1]

        logger.debug(f"Historical VaR: {var:.4f} (alpha={alpha:.3f})")
        return var

    def _cornish_fisher_var(self, returns: pd.Series, alpha: float) -> float:
        """Calculate VaR using Cornish-Fisher expansion for non-normal distributions."""
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Z-score for normal distribution
        z_score = stats.norm.ppf(alpha)

        # Cornish-Fisher adjustment
        cf_adjustment = (
            z_score
            + (z_score**2 - 1) * skewness / 6
            + (z_score**3 - 3 * z_score) * kurtosis / 24
            - (2 * z_score**3 - 5 * z_score) * skewness**2 / 36
        )

        # VaR calculation
        var = -(mean_return + cf_adjustment * std_return)

        logger.debug(f"Cornish-Fisher VaR: {var:.4f} (alpha={alpha:.3f})")
        return var

    def conditional_var(
        self,
        returns: pd.Series,
        method: str = "parametric",
        confidence_level: Optional[float] = None,
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Return series
            method: CVaR method ('parametric', 'historical')
            confidence_level: Override default confidence level

        Returns:
            CVaR value (positive number represents expected loss beyond VaR)
        """
        try:
            if len(returns) < self.min_periods:
                logger.warning("Insufficient data for CVaR calculation")
                return np.nan

            confidence = confidence_level or self.confidence_level
            alpha = 1 - confidence

            if method == "parametric":
                return self._parametric_cvar(returns, alpha)
            elif method == "historical":
                return self._historical_cvar(returns, alpha)
            else:
                raise ValueError(f"Unknown CVaR method: {method}")

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return np.nan

    def _parametric_cvar(self, returns: pd.Series, alpha: float) -> float:
        """Calculate parametric CVaR assuming normal distribution."""
        mean_return = returns.mean()
        std_return = returns.std()

        # Z-score for given confidence level
        z_score = stats.norm.ppf(alpha)

        # CVaR calculation for normal distribution
        cvar = -(mean_return - std_return * stats.norm.pdf(z_score) / alpha)

        logger.debug(f"Parametric CVaR: {cvar:.4f} (alpha={alpha:.3f})")
        return cvar

    def _historical_cvar(self, returns: pd.Series, alpha: float) -> float:
        """Calculate historical CVaR using empirical tail expectation."""
        # Sort returns and find tail
        sorted_returns = returns.sort_values()
        tail_index = int(alpha * len(sorted_returns))

        if tail_index == 0:
            # Edge case: use worst return
            cvar = -sorted_returns.iloc[0]
        else:
            # Average of worst returns
            tail_returns = sorted_returns.iloc[:tail_index]
            cvar = -tail_returns.mean()

        logger.debug(f"Historical CVaR: {cvar:.4f} (alpha={alpha:.3f})")
        return cvar

    def maximum_drawdown(
        self, returns: pd.Series, method: str = "simple"
    ) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            returns: Return series
            method: Calculation method ('simple', 'log')

        Returns:
            Dictionary with drawdown metrics
        """
        try:
            if len(returns) < 2:
                logger.warning("Insufficient data for drawdown calculation")
                return {"max_drawdown": np.nan, "max_duration": np.nan}

            if method == "log":
                # Convert to log returns
                cumulative = (1 + returns).cumprod().apply(np.log)
            else:
                # Simple cumulative returns
                cumulative = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cumulative.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max

            # Maximum drawdown
            max_dd = drawdown.min()

            # Drawdown duration analysis
            is_drawdown = drawdown < 0
            dd_durations = []

            if is_drawdown.any():
                # Find consecutive drawdown periods
                dd_periods = (is_drawdown != is_drawdown.shift(1)).cumsum()[is_drawdown]

                for period in dd_periods.unique():
                    duration = (dd_periods == period).sum()
                    dd_durations.append(duration)

            max_duration = max(dd_durations) if dd_durations else 0

            # Time to recovery (for latest drawdown)
            if is_drawdown.iloc[-1]:
                # Currently in drawdown
                recovery_time = np.nan
            else:
                # Find last recovery
                last_peak_idx = running_max.index[running_max == cumulative.iloc[-1]][
                    -1
                ]
                last_trough_idx = drawdown.index[
                    drawdown == drawdown[last_peak_idx:].min()
                ][0]
                recovery_time = returns.index.get_loc(
                    last_peak_idx
                ) - returns.index.get_loc(last_trough_idx)

            metrics = {
                "max_drawdown": abs(max_dd),
                "max_duration": max_duration,
                "current_drawdown": abs(drawdown.iloc[-1]),
                "recovery_time": recovery_time,
            }

            logger.info(f"Drawdown analysis: MDD={abs(max_dd):.2%}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return {"max_drawdown": np.nan, "max_duration": np.nan}

    def risk_adjusted_returns(
        self, returns: pd.Series, benchmark: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted return metrics.

        Args:
            returns: Return series
            benchmark: Benchmark return series (optional)

        Returns:
            Dictionary with risk-adjusted metrics
        """
        try:
            if len(returns) < self.min_periods:
                logger.warning("Insufficient data for risk-adjusted metrics")
                return {}

            # Basic statistics
            mean_return = returns.mean()
            std_return = returns.std()
            total_return = (1 + returns).prod() - 1

            # Annualization factors
            periods_per_year = self._infer_frequency(returns)
            annual_return = (1 + mean_return) ** periods_per_year - 1
            annual_volatility = std_return * np.sqrt(periods_per_year)

            # Sharpe ratio
            excess_return = annual_return - self.risk_free_rate
            sharpe_ratio = (
                excess_return / annual_volatility if annual_volatility > 0 else 0
            )

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(periods_per_year)
            sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

            # Calmar ratio (return/max drawdown)
            dd_metrics = self.maximum_drawdown(returns)
            max_dd = dd_metrics["max_drawdown"]
            calmar_ratio = annual_return / max_dd if max_dd > 0 else 0

            metrics = {
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "total_return": total_return,
            }

            # Add benchmark-relative metrics if provided
            if benchmark is not None and len(benchmark) >= self.min_periods:
                # Align series
                aligned_returns, aligned_benchmark = returns.align(
                    benchmark, join="inner"
                )

                if len(aligned_returns) >= self.min_periods:
                    # Tracking error
                    excess_returns = aligned_returns - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

                    # Information ratio
                    mean_excess = excess_returns.mean() * periods_per_year
                    information_ratio = (
                        mean_excess / tracking_error if tracking_error > 0 else 0
                    )

                    # Beta
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = aligned_benchmark.var()
                    beta = (
                        covariance / benchmark_variance if benchmark_variance > 0 else 0
                    )

                    # Alpha
                    benchmark_annual = (
                        1 + aligned_benchmark.mean()
                    ) ** periods_per_year - 1
                    alpha = annual_return - (
                        self.risk_free_rate
                        + beta * (benchmark_annual - self.risk_free_rate)
                    )

                    metrics.update(
                        {
                            "tracking_error": tracking_error,
                            "information_ratio": information_ratio,
                            "beta": beta,
                            "alpha": alpha,
                        }
                    )

            logger.info(f"Risk-adjusted metrics: Sharpe={sharpe_ratio:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {e}")
            return {}

    def _infer_frequency(self, returns: pd.Series) -> int:
        """Infer the frequency of return series for annualization."""
        if isinstance(returns.index, pd.DatetimeIndex):
            # Calculate typical time delta
            time_diffs = returns.index[1:] - returns.index[:-1]
            median_diff = time_diffs.median()

            if median_diff <= pd.Timedelta(days=1):
                return 252  # Daily
            elif median_diff <= pd.Timedelta(days=7):
                return 52  # Weekly
            elif median_diff <= pd.Timedelta(days=31):
                return 12  # Monthly
            else:
                return 1  # Annual
        else:
            # Default to daily if no datetime index
            return 252

    def rolling_risk_metrics(
        self, returns: pd.Series, metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics over time.

        Args:
            returns: Return series
            metrics: List of metrics to calculate ('var', 'cvar', 'volatility', 'sharpe')

        Returns:
            DataFrame with rolling metrics
        """
        try:
            if metrics is None:
                metrics = ["var", "cvar", "volatility", "sharpe"]

            if len(returns) < self.rolling_window + self.min_periods:
                logger.warning("Insufficient data for rolling metrics")
                return pd.DataFrame()

            results = {}

            # Calculate rolling metrics
            for i in range(self.rolling_window, len(returns) + 1):
                window_returns = returns.iloc[i - self.rolling_window : i]
                date = returns.index[i - 1]

                row_metrics = {}

                if "var" in metrics:
                    row_metrics["var"] = self.value_at_risk(window_returns)

                if "cvar" in metrics:
                    row_metrics["cvar"] = self.conditional_var(window_returns)

                if "volatility" in metrics:
                    periods_per_year = self._infer_frequency(returns)
                    row_metrics["volatility"] = window_returns.std() * np.sqrt(
                        periods_per_year
                    )

                if "sharpe" in metrics:
                    risk_adj = self.risk_adjusted_returns(window_returns)
                    row_metrics["sharpe"] = risk_adj.get("sharpe_ratio", np.nan)

                results[date] = row_metrics

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(results, orient="index")

            logger.info(f"Rolling metrics calculated for {len(df)} periods")
            return df

        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()

    def stress_test_scenarios(
        self, returns: pd.Series, scenarios: Dict[str, Dict] = None
    ) -> Dict[str, Dict]:
        """
        Perform stress testing under various market scenarios.

        Args:
            returns: Return series
            scenarios: Custom scenarios dict

        Returns:
            Dictionary with stress test results
        """
        try:
            if scenarios is None:
                # Default stress scenarios
                scenarios = {
                    "market_crash": {"mean_shock": -0.05, "vol_multiplier": 2.0},
                    "volatility_spike": {"mean_shock": 0.0, "vol_multiplier": 3.0},
                    "recession": {"mean_shock": -0.02, "vol_multiplier": 1.5},
                    "black_swan": {"mean_shock": -0.10, "vol_multiplier": 4.0},
                }

            base_mean = returns.mean()
            base_std = returns.std()

            stress_results = {}

            for scenario_name, params in scenarios.items():
                # Apply shock to returns
                shocked_mean = base_mean + params.get("mean_shock", 0)
                shocked_std = base_std * params.get("vol_multiplier", 1)

                # Generate stressed returns
                shocked_returns = pd.Series(
                    np.random.normal(shocked_mean, shocked_std, len(returns)),
                    index=returns.index,
                )

                # Calculate metrics under stress
                stress_var = self.value_at_risk(shocked_returns)
                stress_cvar = self.conditional_var(shocked_returns)
                stress_dd = self.maximum_drawdown(shocked_returns)

                stress_results[scenario_name] = {
                    "var": stress_var,
                    "cvar": stress_cvar,
                    "max_drawdown": stress_dd["max_drawdown"],
                    "mean_shock": params.get("mean_shock", 0),
                    "vol_multiplier": params.get("vol_multiplier", 1),
                }

            logger.info(f"Stress testing completed for {len(scenarios)} scenarios")
            return stress_results

        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}


class PortfolioRiskAnalyzer:
    """
    Portfolio-level risk analysis with correlation and concentration metrics.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize portfolio risk analyzer.

        Args:
            confidence_level: Confidence level for risk metrics
        """
        self.confidence_level = confidence_level
        self.risk_metrics = RiskMetrics(confidence_level=confidence_level)

        logger.info("PortfolioRiskAnalyzer initialized")

    def portfolio_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        method: str = "parametric",
    ) -> float:
        """
        Calculate portfolio Value at Risk.

        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights
            method: VaR calculation method

        Returns:
            Portfolio VaR
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)

            # Calculate VaR
            portfolio_var = self.risk_metrics.value_at_risk(
                portfolio_returns, method=method
            )

            logger.info(f"Portfolio VaR: {portfolio_var:.4f}")
            return portfolio_var

        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return np.nan

    def component_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        method: str = "parametric",
    ) -> pd.Series:
        """
        Calculate component VaR (contribution of each asset to portfolio VaR).

        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights
            method: VaR calculation method

        Returns:
            Series with component VaR for each asset
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            portfolio_var = self.risk_metrics.value_at_risk(
                portfolio_returns, method=method
            )

            if np.isnan(portfolio_var):
                return pd.Series(np.nan, index=returns.columns)

            # For parametric VaR, we can calculate exact component VaR
            if method == "parametric":
                # Calculate covariance matrix
                cov_matrix = returns.cov()

                # Portfolio variance
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_vol = np.sqrt(portfolio_variance)

                # Marginal contribution to risk
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol

                # Component VaR = weight * marginal contribution * portfolio VaR / portfolio vol
                component_vars = (
                    weights * marginal_contrib * portfolio_var / portfolio_vol
                )

            else:
                # For non-parametric methods, use finite differences with smaller epsilon
                epsilon = 0.0001
                component_vars = []

                for i, asset in enumerate(returns.columns):
                    # Create perturbed weights
                    perturbed_weights = weights.copy()
                    perturbed_weights[i] += epsilon

                    # Renormalize
                    perturbed_weights = perturbed_weights / perturbed_weights.sum()

                    # Calculate perturbed portfolio VaR
                    perturbed_portfolio_returns = (returns * perturbed_weights).sum(
                        axis=1
                    )
                    perturbed_var = self.risk_metrics.value_at_risk(
                        perturbed_portfolio_returns, method=method
                    )

                    # Marginal VaR
                    marginal_var = (perturbed_var - portfolio_var) / epsilon

                    # Component VaR
                    component_var = marginal_var * weights[i]
                    component_vars.append(component_var)

            result = pd.Series(component_vars, index=returns.columns)

            logger.info("Component VaR calculated")
            return result

        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return pd.Series(np.nan, index=returns.columns)

    def concentration_risk(
        self, weights: np.ndarray, method: str = "herfindahl"
    ) -> float:
        """
        Calculate portfolio concentration risk.

        Args:
            weights: Portfolio weights
            method: Concentration measure ('herfindahl', 'entropy')

        Returns:
            Concentration risk measure
        """
        try:
            if method == "herfindahl":
                # Herfindahl-Hirschman Index
                concentration = np.sum(weights**2)
            elif method == "entropy":
                # Shannon entropy (negative for concentration)
                weights_positive = weights[weights > 0]
                concentration = -np.sum(weights_positive * np.log(weights_positive))
            else:
                raise ValueError(f"Unknown concentration method: {method}")

            logger.info(f"Concentration risk ({method}): {concentration:.4f}")
            return concentration

        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return np.nan
