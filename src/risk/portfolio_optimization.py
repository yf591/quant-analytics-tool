"""
Portfolio Optimization Module - Week 12 Risk Management

Advanced portfolio optimization techniques including Modern Portfolio Theory,
Black-Litterman, Risk Parity, and AFML-based optimization methods.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy import optimize
from scipy.linalg import sqrtm, inv
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple objective functions and constraints.

    Implements Modern Portfolio Theory, Black-Litterman, Risk Parity,
    and other AFML-based optimization techniques.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        allow_short: bool = False,
    ):
        """
        Initialize PortfolioOptimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset
            allow_short: Whether to allow short positions
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_short else -max_weight
        self.allow_short = allow_short

        logger.info(
            f"PortfolioOptimizer initialized: risk_free_rate={risk_free_rate:.3f}, "
            f"weight_bounds=[{self.min_weight:.2f}, {max_weight:.2f}]"
        )

    def mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        objective: str = "sharpe",
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform mean-variance optimization (Markowitz).

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            target_return: Target portfolio return (if specified)
            target_volatility: Target portfolio volatility (if specified)
            objective: Optimization objective ('sharpe', 'min_var', 'max_return')

        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        try:
            n_assets = len(expected_returns)

            # Define constraints
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            # Add target return constraint if specified
            if target_return is not None:
                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda x: np.dot(x, expected_returns) - target_return,
                    }
                )

            # Add target volatility constraint if specified
            if target_volatility is not None:
                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda x: np.sqrt(
                            np.dot(x, np.dot(covariance_matrix, x))
                        )
                        - target_volatility,
                    }
                )

            # Define bounds
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

            # Define objective function
            if objective == "sharpe":

                def objective_func(weights):
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
                    portfolio_vol = np.sqrt(portfolio_var)
                    return -(portfolio_return - self.risk_free_rate) / portfolio_vol

            elif objective == "min_var":

                def objective_func(weights):
                    return np.dot(weights, np.dot(covariance_matrix, weights))

            elif objective == "max_return":

                def objective_func(weights):
                    return -np.dot(weights, expected_returns)

            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Initial guess (equal weights)
            initial_guess = np.ones(n_assets) / n_assets

            # Optimize
            result = optimize.minimize(
                objective_func,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                return {"weights": initial_guess, "success": False}

            optimal_weights = result.x

            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_var = np.dot(
                optimal_weights, np.dot(covariance_matrix, optimal_weights)
            )
            portfolio_vol = np.sqrt(portfolio_var)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

            optimization_result = {
                "weights": optimal_weights,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
                "success": True,
                "objective": objective,
            }

            logger.info(
                f"Mean-variance optimization completed: return={portfolio_return:.4f}, "
                f"vol={portfolio_vol:.4f}, sharpe={sharpe_ratio:.3f}"
            )

            return optimization_result

        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return {
                "weights": np.ones(len(expected_returns)) / len(expected_returns),
                "success": False,
            }

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        num_points: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Generate the efficient frontier.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            num_points: Number of points on the frontier

        Returns:
            Dictionary with returns, volatilities, and weights along the frontier
        """
        try:
            min_return = np.min(expected_returns)
            max_return = np.max(expected_returns)

            # Generate target returns
            target_returns = np.linspace(min_return, max_return, num_points)

            frontier_returns = []
            frontier_volatilities = []
            frontier_weights = []

            for target_return in target_returns:
                result = self.mean_variance_optimization(
                    expected_returns,
                    covariance_matrix,
                    target_return=target_return,
                    objective="min_var",
                )

                if result["success"]:
                    frontier_returns.append(result["expected_return"])
                    frontier_volatilities.append(result["volatility"])
                    frontier_weights.append(result["weights"])

            frontier_data = {
                "returns": np.array(frontier_returns),
                "volatilities": np.array(frontier_volatilities),
                "weights": np.array(frontier_weights),
            }

            logger.info(
                f"Efficient frontier generated with {len(frontier_returns)} points"
            )
            return frontier_data

        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return {
                "returns": np.array([]),
                "volatilities": np.array([]),
                "weights": np.array([]),
            }

    def risk_parity_optimization(
        self,
        covariance_matrix: np.ndarray,
        risk_budget: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform risk parity optimization.

        Args:
            covariance_matrix: Covariance matrix of returns
            risk_budget: Target risk budget for each asset (optional)

        Returns:
            Dictionary with optimal weights and risk contributions
        """
        try:
            n_assets = covariance_matrix.shape[0]

            if risk_budget is None:
                risk_budget = np.ones(n_assets) / n_assets

            def risk_contribution(weights, cov_matrix):
                """Calculate risk contribution of each asset."""
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib / portfolio_vol
                return risk_contrib

            def objective_function(weights):
                """Objective function for risk parity optimization."""
                risk_contrib = risk_contribution(weights, covariance_matrix)
                # Minimize sum of squared deviations from target risk budget
                return np.sum((risk_contrib - risk_budget) ** 2)

            # Constraints
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            # Bounds
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

            # Initial guess
            initial_guess = np.ones(n_assets) / n_assets

            # Optimize
            result = optimize.minimize(
                objective_function,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result.success:
                logger.warning(f"Risk parity optimization failed: {result.message}")
                return {"weights": initial_guess, "success": False}

            optimal_weights = result.x
            risk_contributions = risk_contribution(optimal_weights, covariance_matrix)

            optimization_result = {
                "weights": optimal_weights,
                "risk_contributions": risk_contributions,
                "success": True,
            }

            logger.info("Risk parity optimization completed")
            return optimization_result

        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return {
                "weights": np.ones(covariance_matrix.shape[0])
                / covariance_matrix.shape[0],
                "success": False,
            }

    def black_litterman_optimization(
        self,
        market_caps: np.ndarray,
        covariance_matrix: np.ndarray,
        views_matrix: Optional[np.ndarray] = None,
        views_returns: Optional[np.ndarray] = None,
        views_uncertainty: Optional[np.ndarray] = None,
        risk_aversion: float = 3.0,
        tau: float = 0.05,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform Black-Litterman optimization.

        Args:
            market_caps: Market capitalizations for each asset
            covariance_matrix: Covariance matrix of returns
            views_matrix: Matrix specifying investor views (optional)
            views_returns: Expected returns from investor views (optional)
            views_uncertainty: Uncertainty matrix for views (optional)
            risk_aversion: Risk aversion parameter
            tau: Scaling factor for uncertainty

        Returns:
            Dictionary with optimal weights and expected returns
        """
        try:
            n_assets = len(market_caps)

            # Market weights (proportional to market cap)
            market_weights = market_caps / np.sum(market_caps)

            # Implied equilibrium returns
            pi = risk_aversion * np.dot(covariance_matrix, market_weights)

            # If no views are provided, use equilibrium returns
            if views_matrix is None or views_returns is None:
                new_expected_returns = pi
                new_covariance = (1 + tau) * covariance_matrix
            else:
                # Black-Litterman with views
                if views_uncertainty is None:
                    # Default uncertainty based on view confidence
                    views_uncertainty = tau * np.dot(
                        views_matrix, np.dot(covariance_matrix, views_matrix.T)
                    )

                # Precision matrices
                tau_sigma = tau * covariance_matrix
                omega_inv = inv(views_uncertainty)

                # New expected returns
                sigma_inv = inv(tau_sigma)
                p_omega_p = np.dot(views_matrix.T, np.dot(omega_inv, views_matrix))

                new_sigma_inv = sigma_inv + p_omega_p
                new_covariance = inv(new_sigma_inv)

                mu_part1 = np.dot(sigma_inv, pi)
                mu_part2 = np.dot(views_matrix.T, np.dot(omega_inv, views_returns))

                new_expected_returns = np.dot(new_covariance, mu_part1 + mu_part2)

            # Optimize portfolio using new expected returns and covariance
            result = self.mean_variance_optimization(
                new_expected_returns, new_covariance, objective="sharpe"
            )

            result.update(
                {
                    "equilibrium_returns": pi,
                    "adjusted_returns": new_expected_returns,
                    "adjusted_covariance": new_covariance,
                    "market_weights": market_weights,
                }
            )

            logger.info("Black-Litterman optimization completed")
            return result

        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return {"weights": market_caps / np.sum(market_caps), "success": False}

    def minimum_variance_optimization(
        self, covariance_matrix: np.ndarray
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform minimum variance optimization.

        Args:
            covariance_matrix: Covariance matrix of returns

        Returns:
            Dictionary with optimal weights and portfolio variance
        """
        try:
            n_assets = covariance_matrix.shape[0]

            # Objective function: minimize portfolio variance
            def objective_function(weights):
                return np.dot(weights, np.dot(covariance_matrix, weights))

            # Constraints
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            # Bounds
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

            # Initial guess
            initial_guess = np.ones(n_assets) / n_assets

            # Optimize
            result = optimize.minimize(
                objective_function,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result.success:
                logger.warning(
                    f"Minimum variance optimization failed: {result.message}"
                )
                return {"weights": initial_guess, "success": False}

            optimal_weights = result.x
            portfolio_variance = objective_function(optimal_weights)

            optimization_result = {
                "weights": optimal_weights,
                "portfolio_variance": portfolio_variance,
                "portfolio_volatility": np.sqrt(portfolio_variance),
                "success": True,
            }

            logger.info(
                f"Minimum variance optimization completed: vol={np.sqrt(portfolio_variance):.4f}"
            )
            return optimization_result

        except Exception as e:
            logger.error(f"Error in minimum variance optimization: {e}")
            return {
                "weights": np.ones(covariance_matrix.shape[0])
                / covariance_matrix.shape[0],
                "success": False,
            }

    def hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        method: str = "single",
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform Hierarchical Risk Parity (HRP) optimization.

        Args:
            returns: DataFrame with asset returns
            method: Linkage method for clustering ('single', 'complete', 'average')

        Returns:
            Dictionary with optimal weights and clustering information
        """
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform

            # Calculate correlation matrix
            correlation_matrix = returns.corr()

            # Convert correlation to distance
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))

            # Hierarchical clustering
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method=method)

            # Get quasi-diagonalization order
            def _get_quasi_diag(linkage_matrix):
                link = linkage_matrix.astype(int)
                sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
                num_items = link[-1, 3]  # Number of original observations

                while sort_ix.max() >= num_items:
                    sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                    df0 = sort_ix[sort_ix >= num_items]
                    i = df0.index
                    j = df0.values - num_items
                    sort_ix.loc[i] = link[j, 0]  # Left child
                    df0 = pd.Series(link[j, 1], index=i + 1)
                    sort_ix = sort_ix.append(df0).sort_index()
                    sort_ix.index = range(sort_ix.shape[0])

                return sort_ix.tolist()

            quasi_diag_order = _get_quasi_diag(linkage_matrix)

            # Reorder correlation matrix
            ordered_corr = correlation_matrix.iloc[quasi_diag_order, quasi_diag_order]

            # Calculate HRP weights
            def _get_rec_bipart(cov, sort_ix):
                w = pd.Series(1, index=sort_ix)
                c_items = [sort_ix]

                while len(c_items) > 0:
                    c_items = [
                        i[j:k]
                        for i in c_items
                        for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                        if len(i) > 1
                    ]

                    for i in range(0, len(c_items), 2):
                        c_items0 = c_items[i]
                        c_items1 = c_items[i + 1]

                        # Calculate inverse variance weights for each cluster
                        cov0 = cov.loc[c_items0, c_items0]
                        inv_diag = 1 / np.diag(cov0.values)
                        inv_diag /= inv_diag.sum()
                        w0 = inv_diag.sum()

                        cov1 = cov.loc[c_items1, c_items1]
                        inv_diag = 1 / np.diag(cov1.values)
                        inv_diag /= inv_diag.sum()
                        w1 = inv_diag.sum()

                        # Calculate cluster weights
                        alpha = 1 - w0 / (w0 + w1)

                        # Update weights
                        w[c_items0] *= alpha
                        w[c_items1] *= 1 - alpha

                return w

            # Calculate covariance matrix for HRP
            covariance_matrix = returns.cov()
            ordered_assets = correlation_matrix.index[quasi_diag_order].tolist()

            # Get HRP weights
            hrp_weights = _get_rec_bipart(covariance_matrix, ordered_assets)

            # Reorder weights to original asset order
            weights = np.array([hrp_weights[asset] for asset in returns.columns])

            optimization_result = {
                "weights": weights,
                "ordered_assets": ordered_assets,
                "linkage_matrix": linkage_matrix,
                "success": True,
            }

            logger.info("Hierarchical Risk Parity optimization completed")
            return optimization_result

        except Exception as e:
            logger.error(f"Error in HRP optimization: {e}")
            n_assets = len(returns.columns)
            return {"weights": np.ones(n_assets) / n_assets, "success": False}

    def robust_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        uncertainty_set: str = "box",
        uncertainty_level: float = 0.1,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform robust portfolio optimization.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            uncertainty_set: Type of uncertainty set ('box', 'ellipsoidal')
            uncertainty_level: Level of uncertainty (e.g., 0.1 for 10% uncertainty)

        Returns:
            Dictionary with robust optimal weights
        """
        try:
            n_assets = len(expected_returns)

            if uncertainty_set == "box":
                # Box uncertainty: returns can deviate by ±uncertainty_level
                def robust_objective(weights):
                    # Worst-case expected return
                    worst_case_return = np.dot(
                        weights, expected_returns
                    ) - uncertainty_level * np.sum(np.abs(weights))
                    portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
                    portfolio_vol = np.sqrt(portfolio_var)
                    return -(worst_case_return - self.risk_free_rate) / portfolio_vol

            elif uncertainty_set == "ellipsoidal":
                # Ellipsoidal uncertainty
                uncertainty_matrix = uncertainty_level**2 * covariance_matrix

                def robust_objective(weights):
                    nominal_return = np.dot(weights, expected_returns)
                    uncertainty_penalty = np.sqrt(
                        np.dot(weights, np.dot(uncertainty_matrix, weights))
                    )
                    worst_case_return = nominal_return - uncertainty_penalty
                    portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
                    portfolio_vol = np.sqrt(portfolio_var)
                    return -(worst_case_return - self.risk_free_rate) / portfolio_vol

            else:
                raise ValueError(f"Unknown uncertainty set: {uncertainty_set}")

            # Constraints
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            # Bounds
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

            # Initial guess
            initial_guess = np.ones(n_assets) / n_assets

            # Optimize
            result = optimize.minimize(
                robust_objective,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result.success:
                logger.warning(f"Robust optimization failed: {result.message}")
                return {"weights": initial_guess, "success": False}

            optimal_weights = result.x

            optimization_result = {
                "weights": optimal_weights,
                "uncertainty_set": uncertainty_set,
                "uncertainty_level": uncertainty_level,
                "success": True,
            }

            logger.info(
                f"Robust optimization completed with {uncertainty_set} uncertainty"
            )
            return optimization_result

        except Exception as e:
            logger.error(f"Error in robust optimization: {e}")
            return {
                "weights": np.ones(len(expected_returns)) / len(expected_returns),
                "success": False,
            }


class AFMLPortfolioOptimizer:
    """
    AFML-specific portfolio optimization techniques.

    Implements advanced methods from 'Advances in Financial Machine Learning'
    including purged cross-validation aware optimization and metalabeling.
    """

    def __init__(self, base_optimizer: PortfolioOptimizer):
        """
        Initialize AFML portfolio optimizer.

        Args:
            base_optimizer: Base PortfolioOptimizer instance
        """
        self.base_optimizer = base_optimizer
        logger.info("AFML Portfolio Optimizer initialized")

    def purged_cross_validation_optimization(
        self,
        returns: pd.DataFrame,
        labels: pd.Series,
        cv_folds: int = 5,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio using purged cross-validation.

        Args:
            returns: DataFrame with asset returns
            labels: Series with trading signals/labels
            cv_folds: Number of cross-validation folds
            purge_pct: Percentage of data to purge around test set
            embargo_pct: Percentage of data to embargo after test set

        Returns:
            Dictionary with optimal weights from CV
        """
        try:
            n_samples = len(returns)
            fold_size = n_samples // cv_folds

            cv_results = []

            for fold in range(cv_folds):
                # Define test set
                test_start = fold * fold_size
                test_end = min((fold + 1) * fold_size, n_samples)

                # Define purge and embargo zones
                purge_size = int(purge_pct * n_samples)
                embargo_size = int(embargo_pct * n_samples)

                # Training set (excluding purge and embargo)
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[
                    max(0, test_start - purge_size) : min(
                        n_samples, test_end + embargo_size
                    )
                ] = False

                train_returns = returns.iloc[train_mask]
                train_labels = labels.iloc[train_mask]

                # Calculate expected returns based on labels
                expected_returns = []
                for asset in returns.columns:
                    asset_returns = train_returns[asset]
                    asset_labels = train_labels

                    # Weighted average return by signal strength
                    signal_weighted_return = np.average(
                        asset_returns, weights=np.abs(asset_labels)
                    )
                    expected_returns.append(signal_weighted_return)

                expected_returns = np.array(expected_returns)
                covariance_matrix = train_returns.cov().values

                # Optimize for this fold
                fold_result = self.base_optimizer.mean_variance_optimization(
                    expected_returns, covariance_matrix, objective="sharpe"
                )

                if fold_result["success"]:
                    cv_results.append(fold_result["weights"])

            if not cv_results:
                logger.warning("All CV folds failed")
                n_assets = len(returns.columns)
                return {"weights": np.ones(n_assets) / n_assets, "success": False}

            # Average weights across folds
            avg_weights = np.mean(cv_results, axis=0)

            # Renormalize to ensure sum = 1
            avg_weights = avg_weights / np.sum(avg_weights)

            optimization_result = {
                "weights": avg_weights,
                "cv_folds": cv_folds,
                "successful_folds": len(cv_results),
                "success": True,
            }

            logger.info(
                f"Purged CV optimization completed: {len(cv_results)}/{cv_folds} folds successful"
            )
            return optimization_result

        except Exception as e:
            logger.error(f"Error in purged CV optimization: {e}")
            n_assets = len(returns.columns)
            return {"weights": np.ones(n_assets) / n_assets, "success": False}

    def meta_labeling_optimization(
        self,
        returns: pd.DataFrame,
        primary_signals: pd.Series,
        secondary_features: pd.DataFrame,
        meta_model_predictions: pd.Series,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio using meta-labeling predictions.

        Args:
            returns: DataFrame with asset returns
            primary_signals: Primary trading signals
            secondary_features: Secondary features for meta-labeling
            meta_model_predictions: Meta-model predictions (probability of success)

        Returns:
            Dictionary with optimal weights incorporating meta-predictions
        """
        try:
            # Adjust signals by meta-model predictions
            adjusted_signals = primary_signals * meta_model_predictions

            # Calculate expected returns weighted by adjusted signals
            expected_returns = []
            for asset in returns.columns:
                asset_returns = returns[asset]

                # Weight returns by adjusted signal strength
                signal_weighted_return = np.average(
                    asset_returns, weights=np.abs(adjusted_signals)
                )
                expected_returns.append(signal_weighted_return)

            expected_returns = np.array(expected_returns)
            covariance_matrix = returns.cov().values

            # Optimize portfolio
            result = self.base_optimizer.mean_variance_optimization(
                expected_returns, covariance_matrix, objective="sharpe"
            )

            result.update(
                {
                    "meta_labeling": True,
                    "avg_meta_prediction": meta_model_predictions.mean(),
                }
            )

            logger.info("Meta-labeling optimization completed")
            return result

        except Exception as e:
            logger.error(f"Error in meta-labeling optimization: {e}")
            n_assets = len(returns.columns)
            return {"weights": np.ones(n_assets) / n_assets, "success": False}

    def ensemble_optimization(
        self,
        returns: pd.DataFrame,
        optimization_methods: List[str] = None,
        ensemble_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Ensemble multiple optimization methods.

        Args:
            returns: DataFrame with asset returns
            optimization_methods: List of optimization methods to ensemble
            ensemble_weights: Weights for combining different methods

        Returns:
            Dictionary with ensemble optimal weights
        """
        try:
            if optimization_methods is None:
                optimization_methods = ["mean_variance", "risk_parity", "min_variance"]

            expected_returns = returns.mean().values
            covariance_matrix = returns.cov().values

            method_results = []
            successful_methods = []

            for method in optimization_methods:
                if method == "mean_variance":
                    result = self.base_optimizer.mean_variance_optimization(
                        expected_returns, covariance_matrix, objective="sharpe"
                    )
                elif method == "risk_parity":
                    result = self.base_optimizer.risk_parity_optimization(
                        covariance_matrix
                    )
                elif method == "min_variance":
                    result = self.base_optimizer.minimum_variance_optimization(
                        covariance_matrix
                    )
                else:
                    logger.warning(f"Unknown optimization method: {method}")
                    continue

                if result["success"]:
                    method_results.append(result["weights"])
                    successful_methods.append(method)

            if not method_results:
                logger.warning("All optimization methods failed")
                n_assets = len(returns.columns)
                return {"weights": np.ones(n_assets) / n_assets, "success": False}

            # Ensemble weights
            if ensemble_weights is None:
                # Equal weighting of successful methods
                ensemble_weights = np.ones(len(method_results)) / len(method_results)
            else:
                # Use only weights for successful methods
                ensemble_weights = ensemble_weights[: len(method_results)]
                ensemble_weights = ensemble_weights / np.sum(ensemble_weights)

            # Combine results
            ensemble_portfolio = np.average(
                method_results, axis=0, weights=ensemble_weights
            )

            # Renormalize
            ensemble_portfolio = ensemble_portfolio / np.sum(ensemble_portfolio)

            optimization_result = {
                "weights": ensemble_portfolio,
                "methods_used": successful_methods,
                "ensemble_weights": ensemble_weights,
                "success": True,
            }

            logger.info(
                f"Ensemble optimization completed using {len(successful_methods)} methods"
            )
            return optimization_result

        except Exception as e:
            logger.error(f"Error in ensemble optimization: {e}")
            n_assets = len(returns.columns)
            return {"weights": np.ones(n_assets) / n_assets, "success": False}
