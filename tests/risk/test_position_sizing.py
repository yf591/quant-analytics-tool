"""
Tests for Position Sizing Algorithms

Test cases for all position sizing methods including:
- Kelly Criterion
- Risk Parity
- Volatility Targeting
- Fixed Fractional
- AFML Bet Sizing
"""

import unittest
import numpy as np
import pandas as pd
from src.risk.position_sizing import PositionSizer, PortfolioSizer


class TestPositionSizer(unittest.TestCase):
    """Test cases for PositionSizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer(
            max_position_size=0.5, min_position_size=0.01, risk_free_rate=0.02
        )

    def test_kelly_criterion_basic(self):
        """Test basic Kelly Criterion calculation."""
        # Test case: 60% win rate, 2:1 win/loss ratio
        win_prob = 0.6
        win_loss_ratio = 2.0
        kelly_fraction = 1.0

        expected = (2.0 * 0.6 - 0.4) / 2.0  # (bp - q) / b = 0.4
        result = self.sizer.kelly_criterion(win_prob, win_loss_ratio, kelly_fraction)

        self.assertAlmostEqual(result, expected, places=4)

    def test_kelly_criterion_constraints(self):
        """Test Kelly Criterion with position size constraints."""
        # Test maximum position constraint
        result = self.sizer.kelly_criterion(0.9, 10.0, 1.0)
        self.assertLessEqual(result, self.sizer.max_position_size)

        # Test minimum position constraint (should return 0 if below minimum)
        result = self.sizer.kelly_criterion(0.51, 1.01, 0.01)
        self.assertTrue(result == 0 or result >= self.sizer.min_position_size)

    def test_kelly_criterion_edge_cases(self):
        """Test Kelly Criterion edge cases."""
        # Invalid probability
        with self.assertRaises(ValueError):
            self.sizer.kelly_criterion(-0.1, 2.0)

        with self.assertRaises(ValueError):
            self.sizer.kelly_criterion(1.1, 2.0)

        # Invalid win/loss ratio
        with self.assertRaises(ValueError):
            self.sizer.kelly_criterion(0.6, -1.0)

        # Negative Kelly (should return 0)
        result = self.sizer.kelly_criterion(0.3, 1.0)  # Negative expectation
        self.assertEqual(result, 0.0)

    def test_kelly_from_returns(self):
        """Test Kelly calculation from returns series."""
        # Create sample returns with positive expectation
        np.random.seed(42)
        returns = pd.Series([0.02, -0.01, 0.03, -0.015, 0.025, 0.01, -0.02, 0.035])

        result = self.sizer.kelly_from_returns(returns)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, self.sizer.max_position_size)

    def test_kelly_from_returns_insufficient_data(self):
        """Test Kelly with insufficient data."""
        short_returns = pd.Series([0.01, 0.02])
        result = self.sizer.kelly_from_returns(short_returns)
        self.assertEqual(result, 0.0)

    def test_risk_parity_weights(self):
        """Test risk parity portfolio weights calculation."""
        # Create sample covariance matrix
        cov_matrix = np.array(
            [[0.04, 0.01, 0.02], [0.01, 0.09, 0.015], [0.02, 0.015, 0.16]]
        )

        weights = self.sizer.risk_parity_weights(cov_matrix)

        # Check weights sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)

        # Check all weights are positive
        self.assertTrue(np.all(weights > 0))

        # Check no weight exceeds bounds
        self.assertTrue(np.all(weights <= 0.5))
        self.assertTrue(np.all(weights >= 0.01))

    def test_risk_parity_with_risk_budget(self):
        """Test risk parity with custom risk budget."""
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])

        # Custom risk budget: 70% first asset, 30% second asset
        risk_budget = np.array([0.7, 0.3])

        weights = self.sizer.risk_parity_weights(cov_matrix, risk_budget)

        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue(np.all(weights > 0))

    def test_volatility_targeting(self):
        """Test volatility targeting position sizing."""
        current_volatility = 0.20
        target_volatility = 0.15
        base_position = 1.0

        result = self.sizer.volatility_targeting(
            current_volatility, target_volatility, base_position
        )

        expected = target_volatility / current_volatility  # 0.75
        self.assertAlmostEqual(result, expected, places=4)

    def test_volatility_targeting_constraints(self):
        """Test volatility targeting with constraints."""
        # Test maximum position constraint
        result = self.sizer.volatility_targeting(0.05, 0.15, 1.0)  # High leverage case
        # Since we removed max constraint for vol targeting, this should be 3.0
        self.assertGreater(result, 1.0)

        # Test minimum position constraint
        result = self.sizer.volatility_targeting(1.0, 0.005, 1.0)  # Very low target
        self.assertTrue(result == 0 or result >= self.sizer.min_position_size)

    def test_volatility_targeting_invalid_input(self):
        """Test volatility targeting with invalid inputs."""
        result = self.sizer.volatility_targeting(0, 0.15, 1.0)
        self.assertEqual(result, 0.0)

        result = self.sizer.volatility_targeting(-0.1, 0.15, 1.0)
        self.assertEqual(result, 0.0)

    def test_fixed_fractional(self):
        """Test fixed fractional position sizing."""
        risk_per_trade = 0.02  # 2% risk
        stop_loss_pct = 0.05  # 5% stop loss

        result = self.sizer.fixed_fractional(risk_per_trade, stop_loss_pct)

        expected = risk_per_trade / stop_loss_pct  # 0.4
        self.assertAlmostEqual(result, expected, places=4)

    def test_fixed_fractional_constraints(self):
        """Test fixed fractional with constraints."""
        # Test maximum position constraint
        result = self.sizer.fixed_fractional(0.1, 0.01)  # Very high leverage
        self.assertLessEqual(result, self.sizer.max_position_size)

        # Test minimum position constraint
        result = self.sizer.fixed_fractional(0.001, 0.5)  # Very small position
        self.assertTrue(result == 0 or result >= self.sizer.min_position_size)

    def test_fixed_fractional_invalid_input(self):
        """Test fixed fractional with invalid inputs."""
        result = self.sizer.fixed_fractional(0.02, 0)
        self.assertEqual(result, 0.0)

        result = self.sizer.fixed_fractional(0.02, -0.1)
        self.assertEqual(result, 0.0)

    def test_afml_bet_sizing_basic(self):
        """Test AFML bet sizing with basic inputs."""
        # High confidence prediction
        result = self.sizer.afml_bet_sizing(0.8, num_classes=3)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, self.sizer.max_position_size)

        # Low confidence prediction (around baseline)
        result = self.sizer.afml_bet_sizing(0.33, num_classes=3)
        self.assertGreaterEqual(result, 0)

    def test_afml_bet_sizing_extreme_cases(self):
        """Test AFML bet sizing with extreme cases."""
        # Perfect prediction
        result = self.sizer.afml_bet_sizing(1.0, num_classes=3)
        self.assertGreater(result, 0)

        # No prediction (baseline)
        result = self.sizer.afml_bet_sizing(0.33333, num_classes=3)
        self.assertGreaterEqual(result, 0)

        # Worst prediction
        result = self.sizer.afml_bet_sizing(0.0, num_classes=3)
        self.assertEqual(result, 0.0)

    def test_afml_bet_sizing_invalid_input(self):
        """Test AFML bet sizing with invalid inputs."""
        with self.assertRaises(ValueError):
            self.sizer.afml_bet_sizing(-0.1)

        with self.assertRaises(ValueError):
            self.sizer.afml_bet_sizing(1.1)

    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing combination."""
        prediction_prob = 0.7
        current_vol = 0.20
        target_vol = 0.15

        results = self.sizer.dynamic_position_sizing(
            prediction_prob, current_vol, target_vol
        )

        # Check all required keys are present
        required_keys = ["afml", "volatility", "combined"]
        for key in required_keys:
            self.assertIn(key, results)
            self.assertGreaterEqual(results[key], 0)
            self.assertLessEqual(results[key], self.sizer.max_position_size)

    def test_dynamic_position_sizing_methods(self):
        """Test dynamic position sizing with different methods."""
        prediction_prob = 0.65
        current_vol = 0.18

        for method in ["afml", "volatility", "combined"]:
            results = self.sizer.dynamic_position_sizing(
                prediction_prob, current_vol, method=method
            )
            self.assertIn("combined", results)
            self.assertGreaterEqual(results["combined"], 0)


class TestPortfolioSizer(unittest.TestCase):
    """Test cases for PortfolioSizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.portfolio_sizer = PortfolioSizer(max_portfolio_risk=0.02)

    def test_allocate_risk_budget_basic(self):
        """Test basic risk budget allocation."""
        # Create sample correlation matrix and risks
        correlations = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])

        individual_risks = np.array([0.15, 0.20, 0.25])

        weights = self.portfolio_sizer.allocate_risk_budget(
            correlations, individual_risks
        )

        # Check weights sum to approximately 1 (may be scaled)
        self.assertGreater(np.sum(weights), 0)
        self.assertTrue(np.all(weights >= 0))

    def test_allocate_risk_budget_with_target(self):
        """Test risk budget allocation with specific target."""
        correlations = np.array([[1.0, 0.1], [0.1, 1.0]])

        individual_risks = np.array([0.10, 0.15])
        target_risk = 0.05

        weights = self.portfolio_sizer.allocate_risk_budget(
            correlations, individual_risks, target_risk
        )

        self.assertTrue(np.all(weights >= 0))
        self.assertGreater(np.sum(weights), 0)

    def test_allocate_risk_budget_edge_cases(self):
        """Test risk budget allocation edge cases."""
        # Single asset
        correlations = np.array([[1.0]])
        individual_risks = np.array([0.20])

        weights = self.portfolio_sizer.allocate_risk_budget(
            correlations, individual_risks
        )

        self.assertEqual(len(weights), 1)
        self.assertGreater(weights[0], 0)


class TestPositionSizingIntegration(unittest.TestCase):
    """Integration tests for position sizing components."""

    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer()
        self.portfolio_sizer = PortfolioSizer()

    def test_end_to_end_position_sizing(self):
        """Test complete position sizing workflow."""
        # Sample data
        prediction_prob = 0.72
        current_vol = 0.18
        target_vol = 0.15

        # Get position size
        position_results = self.sizer.dynamic_position_sizing(
            prediction_prob, current_vol, target_vol
        )

        self.assertIn("combined", position_results)
        final_position = position_results["combined"]

        # Should be reasonable size
        self.assertGreater(final_position, 0)
        self.assertLess(final_position, 1.0)

    def test_multiple_assets_portfolio_sizing(self):
        """Test portfolio-level position sizing for multiple assets."""
        n_assets = 4

        # Generate sample data
        np.random.seed(42)
        correlations = np.random.rand(n_assets, n_assets)
        correlations = (correlations + correlations.T) / 2  # Make symmetric
        np.fill_diagonal(correlations, 1.0)

        individual_risks = np.random.uniform(0.10, 0.30, n_assets)

        # Get portfolio allocation
        portfolio_weights = self.portfolio_sizer.allocate_risk_budget(
            correlations, individual_risks
        )

        self.assertEqual(len(portfolio_weights), n_assets)
        self.assertTrue(np.all(portfolio_weights >= 0))

    def test_risk_management_consistency(self):
        """Test consistency across different position sizing methods."""
        # Test that all methods produce reasonable results
        sizer = PositionSizer(max_position_size=0.3, min_position_size=0.005)

        # Kelly criterion
        kelly_size = sizer.kelly_criterion(0.6, 1.5, 0.5)

        # Volatility targeting
        vol_size = sizer.volatility_targeting(0.20, 0.15, kelly_size)

        # AFML bet sizing
        afml_size = sizer.afml_bet_sizing(0.65)

        # All should be within reasonable bounds
        for size in [kelly_size, vol_size, afml_size]:
            self.assertGreaterEqual(size, 0)
            self.assertLessEqual(size, sizer.max_position_size)


if __name__ == "__main__":
    unittest.main()
