"""
Position Sizing Algorithms Implementation

This module implements position sizing algorithms following AFML methodologies:
- Kelly Criterion for optimal position sizing
- Risk Parity for risk-based allocation
- Volatility Targeting for consistent risk exposure
- Fixed Fractional for simple risk management
- AFML Bet Sizing from prediction probabilities

References:
- Advances in Financial Machine Learning, Chapter 10: Bet Sizing
- Modern Portfolio Theory
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Position sizing algorithms for risk management and portfolio optimization.

    Implements various position sizing techniques from AFML and modern portfolio theory.
    """

    def __init__(
        self,
        max_position_size: float = 1.0,
        min_position_size: float = 0.01,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize Position Sizer.

        Args:
            max_position_size: Maximum allowed position size (as fraction of portfolio)
            min_position_size: Minimum position size threshold
            risk_free_rate: Risk-free rate for calculations
        """
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.risk_free_rate = risk_free_rate

    def kelly_criterion(
        self, win_prob: float, win_loss_ratio: float, leverage: float = 1.0
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            win_prob: Probability of winning trade
            win_loss_ratio: Average win / average loss ratio
            leverage: Leverage factor (default 1.0)

        Returns:
            Optimal position size as fraction of capital
        """
        if not (0 <= win_prob <= 1):
            logger.error("Win probability must be between 0 and 1")
            raise ValueError("Win probability must be between 0 and 1")

        if win_loss_ratio <= 0:
            logger.error("Win/loss ratio must be positive")
            raise ValueError("Win/loss ratio must be positive")

        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds received on the wager
            # p = probability of winning
            # q = probability of losing = 1 - p
            kelly_f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

            # Apply leverage
            kelly_f *= leverage

            # Apply constraints
            kelly_f = max(0, min(kelly_f, self.max_position_size))

            if kelly_f < self.min_position_size:
                kelly_f = 0

            logger.info(
                f"Kelly position size: {kelly_f:.4f} (win_prob={win_prob:.3f}, ratio={win_loss_ratio:.3f})"
            )

            return kelly_f

        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            return 0.0

    def kelly_from_returns(
        self, returns: pd.Series, kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly position size from historical returns.

        Args:
            returns: Historical returns series
            kelly_fraction: Safety fraction of Kelly to use

        Returns:
            Optimal position size
        """
        try:
            if len(returns) < 10:
                logger.warning("Insufficient data for Kelly calculation")
                return 0.0

            # Calculate win probability and average win/loss
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            if len(positive_returns) == 0 or len(negative_returns) == 0:
                logger.warning("No winning or losing trades found")
                return 0.0

            win_prob = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())

            if avg_loss == 0:
                return 0.0

            win_loss_ratio = avg_win / avg_loss

            return self.kelly_criterion(win_prob, win_loss_ratio, kelly_fraction)

        except Exception as e:
            logger.error(f"Error calculating Kelly from returns: {e}")
            return 0.0

    def risk_parity_weights(
        self, cov_matrix: np.ndarray, risk_budget: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate risk parity portfolio weights.

        Risk parity allocates capital such that each asset contributes
        equally to portfolio risk.

        Args:
            cov_matrix: Covariance matrix of assets
            risk_budget: Target risk contribution for each asset (equal if None)

        Returns:
            Portfolio weights array
        """
        try:
            n_assets = cov_matrix.shape[0]

            if risk_budget is None:
                risk_budget = np.ones(n_assets) / n_assets

            # Objective function: minimize sum of squared deviations from risk budget
            def objective(weights):
                weights = np.array(weights)
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

                # Risk contributions
                marginal_contrib = np.dot(cov_matrix, weights)
                risk_contrib = weights * marginal_contrib / portfolio_var

                # Squared deviations from target risk budget
                deviation = risk_contrib - risk_budget
                return np.sum(deviation**2)

            # Constraints: weights sum to 1, all weights positive
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            ]

            bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Min 1%, max 50%

            # Initial guess: equal weights
            x0 = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )

            if result.success:
                weights = result.x
                logger.info(f"Risk parity optimization successful")
                return weights
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}")
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]

    def volatility_targeting(
        self,
        current_volatility: float,
        target_volatility: float,
        base_position: float = 1.0,
    ) -> float:
        """
        Scale position size to target volatility level.

        Args:
            current_volatility: Current asset volatility
            target_volatility: Target portfolio volatility
            base_position: Base position size

        Returns:
            Scaled position size
        """
        try:
            if current_volatility <= 0:
                logger.warning("Invalid current volatility")
                return 0.0

            volatility_scalar = target_volatility / current_volatility
            position_size = base_position * volatility_scalar

            # Don't apply max_position_size constraint for volatility targeting
            position_size = max(0, position_size)

            if position_size < self.min_position_size:
                position_size = 0

            logger.info(
                f"Volatility targeting: {position_size:.4f} (target={target_volatility:.3f}, current={current_volatility:.3f})"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error in volatility targeting: {e}")
            return 0.0

    def fixed_fractional(self, risk_per_trade: float, stop_loss_pct: float) -> float:
        """
        Calculate position size using fixed fractional method.

        Args:
            risk_per_trade: Maximum risk per trade as fraction of capital
            stop_loss_pct: Stop loss percentage

        Returns:
            Position size as fraction of capital
        """
        try:
            if stop_loss_pct <= 0:
                logger.warning("Invalid stop loss percentage")
                return 0.0

            position_size = risk_per_trade / stop_loss_pct

            # Apply constraints
            position_size = max(0, min(position_size, self.max_position_size))

            if position_size < self.min_position_size:
                position_size = 0

            logger.info(f"Fixed fractional position size: {position_size:.4f}")

            return position_size

        except Exception as e:
            logger.error(f"Error in fixed fractional sizing: {e}")
            return 0.0

    def afml_bet_sizing(
        self, prediction_prob: float, num_classes: int = 3, step_size: float = 0.1
    ) -> float:
        """
        Calculate bet size using AFML methodology from Chapter 10.

        Args:
            prediction_prob: Predicted probability of the target class
            num_classes: Number of classes in the prediction
            step_size: Step size for signal discretization

        Returns:
            Bet size as fraction of capital
        """
        if not (0 <= prediction_prob <= 1):
            logger.error("Prediction probability must be between 0 and 1")
            raise ValueError("Prediction probability must be between 0 and 1")

        try:
            # Calculate confidence as deviation from random chance
            random_chance = 1.0 / num_classes

            # Calculate signal strength based on prediction confidence
            if prediction_prob > random_chance:
                # Above random chance - positive signal
                confidence = (prediction_prob - random_chance) / (1.0 - random_chance)
            elif prediction_prob < random_chance:
                # Below random chance - negative signal (avoid bet)
                return 0.0
            else:
                # Exactly random chance - no confidence
                return 0.0

            # Simple signal conversion for AFML-style bet sizing
            signal_strength = confidence

            # Discretize signal to prevent overtrading
            discretized_signal = np.round(signal_strength / step_size) * step_size

            # Cap between 0 and 1 for position sizing
            discretized_signal = max(0, min(discretized_signal, 1))  # Apply constraints
            position_size = min(discretized_signal, self.max_position_size)

            if position_size < self.min_position_size:
                position_size = 0

            logger.info(
                f"AFML bet size: {position_size:.4f} (prob={prediction_prob:.3f}, signal={discretized_signal:.3f})"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error in AFML bet sizing: {e}")
            return 0.0

    def dynamic_position_sizing(
        self,
        prediction_prob: float,
        current_volatility: float,
        target_volatility: float = 0.15,
        kelly_fraction: float = 0.25,
        method: str = "combined",
    ) -> Dict[str, float]:
        """
        Combine multiple position sizing methods for robust sizing.

        Args:
            prediction_prob: ML prediction probability
            current_volatility: Current asset volatility
            target_volatility: Target portfolio volatility
            kelly_fraction: Kelly fraction for safety
            method: Sizing method ('kelly', 'volatility', 'afml', 'combined')

        Returns:
            Dictionary with position sizes from different methods
        """
        try:
            results = {}

            # AFML bet sizing
            afml_size = self.afml_bet_sizing(prediction_prob)
            results["afml"] = afml_size

            # Volatility targeting
            vol_size = self.volatility_targeting(
                current_volatility, target_volatility, afml_size
            )
            results["volatility"] = vol_size

            # Combined approach
            if method == "combined":
                # Weight AFML signal with volatility adjustment
                combined_size = afml_size * (
                    target_volatility / max(current_volatility, 0.01)
                )
                combined_size = max(0, min(combined_size, self.max_position_size))
                results["combined"] = combined_size
            elif method in results:
                results["combined"] = results[method]
            else:
                results["combined"] = afml_size

            logger.info(f"Dynamic position sizing results: {results}")

            return results

        except Exception as e:
            logger.error(f"Error in dynamic position sizing: {e}")
            return {"combined": 0.0}


class PortfolioSizer:
    """
    Portfolio-level position sizing and risk allocation.
    """

    def __init__(self, max_portfolio_risk: float = 0.02):
        """
        Initialize Portfolio Sizer.

        Args:
            max_portfolio_risk: Maximum portfolio risk per day
        """
        self.max_portfolio_risk = max_portfolio_risk

    def allocate_risk_budget(
        self,
        correlations: np.ndarray,
        individual_risks: np.ndarray,
        target_risk: float = None,
    ) -> np.ndarray:
        """
        Allocate risk budget across portfolio positions.

        Args:
            correlations: Asset correlation matrix
            individual_risks: Individual asset risk levels
            target_risk: Target portfolio risk level

        Returns:
            Risk-adjusted position sizes
        """
        try:
            if target_risk is None:
                target_risk = self.max_portfolio_risk

            n_assets = len(individual_risks)

            # Simple risk budgeting: inverse volatility weighting
            inv_vol_weights = (1 / individual_risks) / np.sum(1 / individual_risks)

            # Adjust for correlations
            correlation_adj = np.diag(correlations)
            adjusted_weights = inv_vol_weights / correlation_adj
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

            # Scale to target risk
            portfolio_risk = np.sqrt(
                np.dot(
                    adjusted_weights,
                    np.dot(
                        correlations * np.outer(individual_risks, individual_risks),
                        adjusted_weights,
                    ),
                )
            )

            if portfolio_risk > 0:
                risk_scalar = target_risk / portfolio_risk
                adjusted_weights *= risk_scalar

            logger.info(
                f"Portfolio risk allocation completed, target risk: {target_risk:.4f}"
            )

            return adjusted_weights

        except Exception as e:
            logger.error(f"Error in portfolio risk allocation: {e}")
            return np.ones(len(individual_risks)) / len(individual_risks)
