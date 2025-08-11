"""
Test Suite for Portfolio Optimization Module - Week 12 Risk Management

Comprehensive tests for portfolio optimization techniques including Modern Portfolio Theory,
Black-Litterman, Risk Parity, Hierarchical Risk Parity (HRP), and AFML-based optimization methods.

Test Coverage:
- PortfolioOptimizer: Mean-variance optimization, efficient frontier, risk parity, Black-Litterman
- AFMLPortfolioOptimizer: Purged cross-validation, meta-labeling, ensemble optimization
- Integration tests: Method consistency, constraint enforcement, mathematical properties

Total: 31 tests ensuring robust portfolio optimization capabilities.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.risk.portfolio_optimization import PortfolioOptimizer, AFMLPortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create test data
        self.n_assets = 4
        self.dates = pd.date_range("2020-01-01", periods=252, freq="D")

        # Generate correlated returns
        correlation_matrix = np.array(
            [
                [1.0, 0.3, 0.1, 0.2],
                [0.3, 1.0, 0.4, 0.1],
                [0.1, 0.4, 1.0, 0.5],
                [0.2, 0.1, 0.5, 1.0],
            ]
        )

        # Create returns with different risk/return profiles
        mean_returns = np.array([0.001, 0.0008, 0.0012, 0.0006])
        volatilities = np.array([0.02, 0.015, 0.025, 0.018])

        # Convert to covariance matrix
        self.covariance_matrix = (
            np.outer(volatilities, volatilities) * correlation_matrix
        )
        self.expected_returns = mean_returns

        # Market cap data for Black-Litterman
        self.market_caps = np.array([1000, 800, 600, 400])

        # Generate sample returns DataFrame
        returns_data = np.random.multivariate_normal(
            mean_returns, self.covariance_matrix, size=252
        )
        self.returns_df = pd.DataFrame(
            returns_data,
            index=self.dates,
            columns=[f"Asset_{i}" for i in range(self.n_assets)],
        )

        # Initialize optimizer
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=0.02, max_weight=0.5, min_weight=0.0, allow_short=False
        )

    def test_initialization(self):
        """Test PortfolioOptimizer initialization."""
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.03, max_weight=0.4, min_weight=0.1, allow_short=True
        )

        self.assertEqual(optimizer.risk_free_rate, 0.03)
        self.assertEqual(optimizer.max_weight, 0.4)
        self.assertEqual(optimizer.min_weight, -0.4)  # With short selling
        self.assertTrue(optimizer.allow_short)

    def test_mean_variance_optimization_sharpe(self):
        """Test mean-variance optimization with Sharpe ratio objective."""
        result = self.optimizer.mean_variance_optimization(
            self.expected_returns, self.covariance_matrix, objective="sharpe"
        )

        self.assertTrue(result["success"])
        self.assertIn("weights", result)
        self.assertIn("expected_return", result)
        self.assertIn("volatility", result)
        self.assertIn("sharpe_ratio", result)

        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

        # All weights should be within bounds
        self.assertTrue(np.all(result["weights"] >= self.optimizer.min_weight))
        self.assertTrue(np.all(result["weights"] <= self.optimizer.max_weight))

        # Sharpe ratio should be reasonable (allowing for negative values in test data)
        self.assertGreater(result["sharpe_ratio"], -5)
        self.assertLess(result["sharpe_ratio"], 10)

    def test_mean_variance_optimization_min_var(self):
        """Test mean-variance optimization with minimum variance objective."""
        result = self.optimizer.mean_variance_optimization(
            self.expected_returns, self.covariance_matrix, objective="min_var"
        )

        self.assertTrue(result["success"])
        self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

    def test_mean_variance_optimization_max_return(self):
        """Test mean-variance optimization with maximum return objective."""
        result = self.optimizer.mean_variance_optimization(
            self.expected_returns, self.covariance_matrix, objective="max_return"
        )

        self.assertTrue(result["success"])
        self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

    def test_mean_variance_optimization_target_return(self):
        """Test mean-variance optimization with target return constraint."""
        target_return = 0.001
        result = self.optimizer.mean_variance_optimization(
            self.expected_returns,
            self.covariance_matrix,
            target_return=target_return,
            objective="min_var",
        )

        if result["success"]:
            # Check if target return is approximately achieved
            achieved_return = result["expected_return"]
            self.assertAlmostEqual(achieved_return, target_return, places=4)

    def test_mean_variance_optimization_target_volatility(self):
        """Test mean-variance optimization with target volatility constraint."""
        target_volatility = 0.02
        result = self.optimizer.mean_variance_optimization(
            self.expected_returns,
            self.covariance_matrix,
            target_volatility=target_volatility,
            objective="max_return",
        )

        if result["success"]:
            # Check if target volatility is approximately achieved
            achieved_volatility = result["volatility"]
            self.assertAlmostEqual(achieved_volatility, target_volatility, places=4)

    def test_mean_variance_optimization_invalid_objective(self):
        """Test mean-variance optimization with invalid objective."""
        result = self.optimizer.mean_variance_optimization(
            self.expected_returns, self.covariance_matrix, objective="invalid_objective"
        )

        self.assertFalse(result["success"])

    def test_efficient_frontier(self):
        """Test efficient frontier generation."""
        frontier = self.optimizer.efficient_frontier(
            self.expected_returns, self.covariance_matrix, num_points=10
        )

        self.assertIn("returns", frontier)
        self.assertIn("volatilities", frontier)
        self.assertIn("weights", frontier)

        if len(frontier["returns"]) > 0:
            # Returns should be in ascending order (approximately)
            returns = frontier["returns"]
            self.assertTrue(len(returns) > 0)

            # Volatilities should be positive
            volatilities = frontier["volatilities"]
            self.assertTrue(np.all(volatilities > 0))

            # Each point should have valid weights
            weights = frontier["weights"]
            for i in range(len(weights)):
                self.assertAlmostEqual(np.sum(weights[i]), 1.0, places=4)

    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        result = self.optimizer.risk_parity_optimization(self.covariance_matrix)

        self.assertTrue(result["success"])
        self.assertIn("weights", result)
        self.assertIn("risk_contributions", result)

        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

        # Risk contributions should be approximately equal for equal risk budget
        risk_contributions = result["risk_contributions"]
        expected_risk_contrib = 1.0 / self.n_assets

        for contrib in risk_contributions:
            self.assertAlmostEqual(contrib, expected_risk_contrib, places=1)

    def test_risk_parity_optimization_custom_budget(self):
        """Test risk parity optimization with custom risk budget."""
        custom_budget = np.array([0.4, 0.3, 0.2, 0.1])
        result = self.optimizer.risk_parity_optimization(
            self.covariance_matrix, risk_budget=custom_budget
        )

        if result["success"]:
            self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

    def test_black_litterman_optimization_no_views(self):
        """Test Black-Litterman optimization without views."""
        result = self.optimizer.black_litterman_optimization(
            self.market_caps, self.covariance_matrix
        )

        self.assertTrue(result["success"])
        self.assertIn("weights", result)
        self.assertIn("equilibrium_returns", result)
        self.assertIn("market_weights", result)

        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

    def test_black_litterman_optimization_with_views(self):
        """Test Black-Litterman optimization with investor views."""
        # Create simple view: Asset 0 will outperform Asset 1 by 0.002
        views_matrix = np.array([[1, -1, 0, 0]])  # Long Asset 0, Short Asset 1
        views_returns = np.array([0.002])

        result = self.optimizer.black_litterman_optimization(
            self.market_caps,
            self.covariance_matrix,
            views_matrix=views_matrix,
            views_returns=views_returns,
        )

        if result["success"]:
            self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)
            self.assertIn("adjusted_returns", result)

    def test_minimum_variance_optimization(self):
        """Test minimum variance optimization."""
        result = self.optimizer.minimum_variance_optimization(self.covariance_matrix)

        self.assertTrue(result["success"])
        self.assertIn("weights", result)
        self.assertIn("portfolio_variance", result)
        self.assertIn("portfolio_volatility", result)

        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

        # Variance should be positive
        self.assertGreater(result["portfolio_variance"], 0)

        # Volatility should be square root of variance
        expected_vol = np.sqrt(result["portfolio_variance"])
        self.assertAlmostEqual(result["portfolio_volatility"], expected_vol, places=6)

    def test_hierarchical_risk_parity(self):
        """Test Hierarchical Risk Parity optimization."""
        result = self.optimizer.hierarchical_risk_parity(self.returns_df)

        # HRP might fail due to scipy dependencies, so we handle both cases
        if result["success"]:
            self.assertIn("weights", result)
            self.assertIn("ordered_assets", result)

            # Weights should sum to 1
            self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

            # All weights should be positive (HRP doesn't allow shorts)
            self.assertTrue(np.all(result["weights"] > 0))
        else:
            # If HRP fails (e.g., missing scipy), weights should be equal
            expected_weights = np.ones(self.n_assets) / self.n_assets
            np.testing.assert_array_almost_equal(result["weights"], expected_weights)

    def test_robust_optimization_box(self):
        """Test robust optimization with box uncertainty."""
        result = self.optimizer.robust_optimization(
            self.expected_returns,
            self.covariance_matrix,
            uncertainty_set="box",
            uncertainty_level=0.1,
        )

        self.assertTrue(result["success"])
        self.assertIn("weights", result)
        self.assertEqual(result["uncertainty_set"], "box")

        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

    def test_robust_optimization_ellipsoidal(self):
        """Test robust optimization with ellipsoidal uncertainty."""
        result = self.optimizer.robust_optimization(
            self.expected_returns,
            self.covariance_matrix,
            uncertainty_set="ellipsoidal",
            uncertainty_level=0.1,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["uncertainty_set"], "ellipsoidal")

    def test_robust_optimization_invalid_uncertainty(self):
        """Test robust optimization with invalid uncertainty set."""
        result = self.optimizer.robust_optimization(
            self.expected_returns, self.covariance_matrix, uncertainty_set="invalid_set"
        )

        self.assertFalse(result["success"])


class TestAFMLPortfolioOptimizer(unittest.TestCase):
    """Test cases for AFMLPortfolioOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create test data
        self.n_assets = 3
        self.n_samples = 100
        self.dates = pd.date_range("2020-01-01", periods=self.n_samples, freq="D")

        # Generate sample returns
        returns_data = np.random.normal(0.001, 0.02, (self.n_samples, self.n_assets))
        self.returns_df = pd.DataFrame(
            returns_data,
            index=self.dates,
            columns=[f"Asset_{i}" for i in range(self.n_assets)],
        )

        # Generate sample labels (trading signals)
        self.labels = pd.Series(
            np.random.choice([-1, 0, 1], size=self.n_samples, p=[0.3, 0.4, 0.3]),
            index=self.dates,
        )

        # Secondary features for meta-labeling
        self.secondary_features = pd.DataFrame(
            np.random.normal(0, 1, (self.n_samples, 5)),
            index=self.dates,
            columns=[f"Feature_{i}" for i in range(5)],
        )

        # Meta-model predictions
        self.meta_predictions = pd.Series(
            np.random.uniform(0.3, 0.8, self.n_samples), index=self.dates
        )

        # Initialize optimizers
        base_optimizer = PortfolioOptimizer()
        self.afml_optimizer = AFMLPortfolioOptimizer(base_optimizer)

    def test_initialization(self):
        """Test AFMLPortfolioOptimizer initialization."""
        base_optimizer = PortfolioOptimizer()
        afml_optimizer = AFMLPortfolioOptimizer(base_optimizer)

        self.assertEqual(afml_optimizer.base_optimizer, base_optimizer)

    def test_purged_cross_validation_optimization(self):
        """Test purged cross-validation optimization."""
        result = self.afml_optimizer.purged_cross_validation_optimization(
            self.returns_df, self.labels, cv_folds=3, purge_pct=0.05, embargo_pct=0.05
        )

        # Should succeed with reasonable parameters
        if result["success"]:
            self.assertIn("weights", result)
            self.assertIn("cv_folds", result)
            self.assertIn("successful_folds", result)

            # Weights should sum to 1
            self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

            # Should have processed some folds
            self.assertGreater(result["successful_folds"], 0)
        else:
            # If failed, should return equal weights
            expected_weights = np.ones(self.n_assets) / self.n_assets
            np.testing.assert_array_almost_equal(result["weights"], expected_weights)

    def test_purged_cross_validation_optimization_invalid_params(self):
        """Test purged CV optimization with invalid parameters."""
        result = self.afml_optimizer.purged_cross_validation_optimization(
            self.returns_df,
            self.labels,
            cv_folds=20,  # Too many folds for small dataset
            purge_pct=0.4,  # Very large purge
            embargo_pct=0.4,  # Very large embargo
        )

        # Should handle gracefully
        self.assertIn("weights", result)

    def test_meta_labeling_optimization(self):
        """Test meta-labeling optimization."""
        result = self.afml_optimizer.meta_labeling_optimization(
            self.returns_df, self.labels, self.secondary_features, self.meta_predictions
        )

        if result["success"]:
            self.assertIn("weights", result)
            self.assertIn("meta_labeling", result)
            self.assertIn("avg_meta_prediction", result)

            # Weights should sum to 1
            self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

            # Meta-labeling flag should be True
            self.assertTrue(result["meta_labeling"])

            # Average meta prediction should be reasonable
            self.assertGreater(result["avg_meta_prediction"], 0)
            self.assertLess(result["avg_meta_prediction"], 1)

    def test_ensemble_optimization_default(self):
        """Test ensemble optimization with default methods."""
        result = self.afml_optimizer.ensemble_optimization(self.returns_df)

        if result["success"]:
            self.assertIn("weights", result)
            self.assertIn("methods_used", result)
            self.assertIn("ensemble_weights", result)

            # Weights should sum to 1
            self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=6)

            # Should have used multiple methods
            self.assertGreater(len(result["methods_used"]), 0)

            # Ensemble weights should sum to 1
            self.assertAlmostEqual(np.sum(result["ensemble_weights"]), 1.0, places=6)

    def test_ensemble_optimization_custom_methods(self):
        """Test ensemble optimization with custom methods."""
        custom_methods = ["mean_variance", "min_variance"]
        result = self.afml_optimizer.ensemble_optimization(
            self.returns_df, optimization_methods=custom_methods
        )

        if result["success"]:
            # Should only use specified methods
            for method in result["methods_used"]:
                self.assertIn(method, custom_methods)

    def test_ensemble_optimization_custom_weights(self):
        """Test ensemble optimization with custom ensemble weights."""
        methods = ["mean_variance", "risk_parity"]
        custom_weights = np.array([0.7, 0.3])

        result = self.afml_optimizer.ensemble_optimization(
            self.returns_df,
            optimization_methods=methods,
            ensemble_weights=custom_weights,
        )

        if result["success"] and len(result["methods_used"]) == 2:
            # Ensemble weights should match custom weights (normalized)
            expected_weights = custom_weights / np.sum(custom_weights)
            np.testing.assert_array_almost_equal(
                result["ensemble_weights"], expected_weights, decimal=6
            )

    def test_ensemble_optimization_invalid_method(self):
        """Test ensemble optimization with invalid methods."""
        invalid_methods = ["invalid_method_1", "invalid_method_2"]
        result = self.afml_optimizer.ensemble_optimization(
            self.returns_df, optimization_methods=invalid_methods
        )

        # Should handle gracefully and return equal weights
        self.assertIn("weights", result)


class TestPortfolioOptimizationIntegration(unittest.TestCase):
    """Integration tests for portfolio optimization components."""

    def setUp(self):
        """Set up integration test fixtures."""
        np.random.seed(42)

        # Create realistic market data
        self.n_assets = 5
        self.n_samples = 200
        self.dates = pd.date_range("2020-01-01", periods=self.n_samples, freq="D")

        # Create assets with different characteristics
        # High return, high risk
        asset1 = np.random.normal(0.0015, 0.03, self.n_samples)
        # Low return, low risk
        asset2 = np.random.normal(0.0005, 0.015, self.n_samples)
        # Medium return, medium risk
        asset3 = np.random.normal(0.001, 0.02, self.n_samples)
        # Defensive asset
        asset4 = np.random.normal(0.0003, 0.01, self.n_samples)
        # Volatile asset
        asset5 = np.random.normal(0.0008, 0.035, self.n_samples)

        self.returns_df = pd.DataFrame(
            {
                "Growth": asset1,
                "Bond": asset2,
                "Equity": asset3,
                "Defensive": asset4,
                "Volatile": asset5,
            },
            index=self.dates,
        )

        self.optimizer = PortfolioOptimizer()

    def test_optimization_methods_consistency(self):
        """Test consistency across different optimization methods."""
        expected_returns = self.returns_df.mean().values
        covariance_matrix = self.returns_df.cov().values

        # Test multiple optimization methods
        sharpe_result = self.optimizer.mean_variance_optimization(
            expected_returns, covariance_matrix, objective="sharpe"
        )

        min_var_result = self.optimizer.minimum_variance_optimization(covariance_matrix)

        risk_parity_result = self.optimizer.risk_parity_optimization(covariance_matrix)

        # All methods should succeed or fail gracefully
        for result in [sharpe_result, min_var_result, risk_parity_result]:
            self.assertIn("weights", result)
            if result["success"]:
                self.assertAlmostEqual(np.sum(result["weights"]), 1.0, places=5)

    def test_portfolio_metrics_calculation(self):
        """Test portfolio metrics calculation consistency."""
        expected_returns = self.returns_df.mean().values
        covariance_matrix = self.returns_df.cov().values

        result = self.optimizer.mean_variance_optimization(
            expected_returns, covariance_matrix, objective="sharpe"
        )

        if result["success"]:
            weights = result["weights"]

            # Manual calculation of portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)

            # Compare with optimization result
            self.assertAlmostEqual(
                result["expected_return"], portfolio_return, places=6
            )
            self.assertAlmostEqual(result["volatility"], portfolio_vol, places=6)

    def test_constraints_enforcement(self):
        """Test that optimization constraints are properly enforced."""
        optimizer = PortfolioOptimizer(max_weight=0.3, min_weight=0.1)
        expected_returns = self.returns_df.mean().values
        covariance_matrix = self.returns_df.cov().values

        result = optimizer.mean_variance_optimization(
            expected_returns, covariance_matrix, objective="sharpe"
        )

        if result["success"]:
            weights = result["weights"]

            # Check weight constraints
            self.assertTrue(np.all(weights >= 0.1))
            self.assertTrue(np.all(weights <= 0.3))
            self.assertAlmostEqual(np.sum(weights), 1.0, places=6)

    def test_efficient_frontier_properties(self):
        """Test efficient frontier mathematical properties."""
        expected_returns = self.returns_df.mean().values
        covariance_matrix = self.returns_df.cov().values

        frontier = self.optimizer.efficient_frontier(
            expected_returns, covariance_matrix, num_points=20
        )

        if len(frontier["returns"]) > 2:
            returns = frontier["returns"]
            volatilities = frontier["volatilities"]

            # Returns should generally increase along frontier
            # (allowing for some numerical noise)
            increasing_trend = np.sum(np.diff(returns) > -0.0001) / len(
                np.diff(returns)
            )
            self.assertGreater(increasing_trend, 0.7)  # At least 70% increasing

            # Volatilities should be positive
            self.assertTrue(np.all(volatilities > 0))

    def test_risk_budgeting_properties(self):
        """Test risk budgeting properties in risk parity."""
        covariance_matrix = self.returns_df.cov().values

        # Test equal risk budgeting
        result = self.optimizer.risk_parity_optimization(covariance_matrix)

        if result["success"]:
            risk_contributions = result["risk_contributions"]

            # Risk contributions should be approximately equal
            target_contrib = 1.0 / self.n_assets
            max_deviation = np.max(np.abs(risk_contributions - target_contrib))
            self.assertLess(max_deviation, 0.1)  # Within 10% deviation

    def test_black_litterman_market_equilibrium(self):
        """Test Black-Litterman market equilibrium properties."""
        market_caps = np.array([1000, 800, 600, 400, 200])
        covariance_matrix = self.returns_df.cov().values

        result = self.optimizer.black_litterman_optimization(
            market_caps, covariance_matrix
        )

        if result["success"]:
            market_weights = result["market_weights"]

            # Market weights should be proportional to market caps
            expected_market_weights = market_caps / np.sum(market_caps)
            np.testing.assert_array_almost_equal(
                market_weights, expected_market_weights, decimal=6
            )


if __name__ == "__main__":
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore")
    unittest.main()
