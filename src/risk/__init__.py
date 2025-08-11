"""
Risk Management Module

This module provides comprehensive risk management capabilities for quantitative trading:
- Position sizing algorithms (Kelly criterion, Risk Parity, AFML bet sizing)
- Risk metrics calculation (VaR, CVaR, drawdown analysis)
- Portfolio optimization (MPT, Black-Litterman, HRP)
- Stress testing framework (Monte Carlo, scenario analysis)

All implementations follow AFML (Advances in Financial Machine Learning) methodologies.
"""

from .position_sizing import PositionSizer, PortfolioSizer
from .risk_metrics import RiskMetrics, PortfolioRiskAnalyzer
from .portfolio_optimization import PortfolioOptimizer, AFMLPortfolioOptimizer
from .stress_testing import (
    ScenarioGenerator,
    MonteCarloEngine,
    SensitivityAnalyzer,
    TailRiskAnalyzer,
    StressTesting,
)

__all__ = [
    # Position Sizing
    "PositionSizer",
    "PortfolioSizer",
    # Risk Metrics
    "RiskMetrics",
    "PortfolioRiskAnalyzer",
    # Portfolio Optimization
    "PortfolioOptimizer",
    "AFMLPortfolioOptimizer",
    # Stress Testing
    "ScenarioGenerator",
    "MonteCarloEngine",
    "SensitivityAnalyzer",
    "TailRiskAnalyzer",
    "StressTesting",
]

__version__ = "1.0.0"
__author__ = "Quant Analytics Tool"
__description__ = "AFML-compliant Risk Management System"
