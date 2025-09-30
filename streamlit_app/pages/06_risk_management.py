"""
Streamlit Page: Risk Management (Clean Architecture)
Week 14 UI Integration - Clean Frontend with Backend Integration Only
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Backend imports ONLY - NO local implementations
    from src.risk.risk_metrics import RiskMetrics, PortfolioRiskAnalyzer
    from src.risk.portfolio_optimization import (
        PortfolioOptimizer,
        AFMLPortfolioOptimizer,
    )
    from src.risk.position_sizing import PositionSizer, PortfolioSizer
    from src.risk.stress_testing import StressTesting, ScenarioGenerator

    # UI utilities for display only
    from utils.risk_management_utils import (
        RiskManagementProcessor,
        RiskVisualizationManager,
    )

except ImportError as e:
    st.error(f"Backend import error: {e}")
    st.error("Please ensure risk management backend modules are configured.")
    st.stop()


def main():
    """Clean Risk Management Interface - UI Only"""

    st.set_page_config(
        page_title="Risk Management",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Advanced Risk Management")
    st.markdown("**Comprehensive risk analysis powered by AFML backend**")

    # Check for backtest data
    if "backtest_cache" not in st.session_state or not st.session_state.backtest_cache:
        st.warning("No backtest results available. Please run a backtest first.")
        if st.button("Go to Backtesting"):
            st.switch_page("pages/05_backtesting.py")
        return

    # Risk management workflow tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Portfolio Risk Analysis", "🎯 Portfolio Optimization", "🚨 Stress Testing"]
    )

    with tab1:
        portfolio_risk_workflow()

    with tab2:
        portfolio_optimization_workflow()

    with tab3:
        stress_testing_workflow()


def portfolio_risk_workflow():
    """Portfolio risk analysis workflow - UI only"""

    st.subheader("📊 Portfolio Risk Analysis")

    # Select backtest result
    backtest_keys = list(st.session_state.backtest_cache.keys())
    selected_key = st.selectbox("Select Backtest Result", backtest_keys)

    if not selected_key:
        return

    st.subheader("🔧 Risk Configuration")

    # Risk parameters in columns for input
    col1, col2, col3 = st.columns(3)

    with col1:
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)

    with col2:
        lookback_days = st.slider("Lookback Days", 30, 500, 252)

    with col3:
        var_method = st.selectbox(
            "VaR Method", ["Historical", "Parametric", "Cornish-Fisher"]
        )

    # Calculate button centered
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("📊 Calculate Risk Metrics", type="primary"):
        display_risk_analysis_results(
            selected_key, confidence_level, lookback_days, var_method
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Display results below (full width)
    display_risk_results(selected_key)


def portfolio_optimization_workflow():
    """Portfolio optimization workflow - UI only"""

    st.subheader("🎯 Portfolio Optimization")

    # Select backtest result
    backtest_keys = list(st.session_state.backtest_cache.keys())
    selected_key = st.selectbox(
        "Select Backtest Result", backtest_keys, key="opt_select"
    )

    if not selected_key:
        return

    st.subheader("⚙️ Optimization Settings")

    # Optimization parameters in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox(
            "Optimization Method",
            ["Mean Variance", "Risk Parity", "Maximum Sharpe", "Minimum Variance"],
        )

    with col2:
        max_weight = st.slider("Max Asset Weight", 0.1, 1.0, 0.3, 0.05)

    with col3:
        target_volatility = st.slider("Target Volatility", 0.01, 0.50, 0.15, 0.01)

    # Optimize button centered
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🎯 Optimize Portfolio", type="primary"):
        display_portfolio_optimization(
            selected_key, method, max_weight, target_volatility
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Display optimization results below (full width)
    display_optimization_results(selected_key)


def stress_testing_workflow():
    """Stress testing workflow - UI only"""

    st.subheader("🚨 Stress Testing")

    # Select backtest result
    backtest_keys = list(st.session_state.backtest_cache.keys())
    selected_key = st.selectbox(
        "Select Backtest Result", backtest_keys, key="stress_select"
    )

    if not selected_key:
        return

    st.subheader("🔧 Stress Test Configuration")

    # Stress test parameters in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        stress_type = st.selectbox(
            "Stress Test Type",
            [
                "Market Crash",
                "Interest Rate Shock",
                "High Volatility",
                "Custom Scenario",
            ],
        )

    with col2:
        severity = st.selectbox("Severity Level", ["Mild", "Moderate", "Severe"])

    with col3:
        num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000, 500)

    # Run stress test button centered
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🚨 Run Stress Test", type="primary"):
        run_stress_test(selected_key, stress_type, severity, num_simulations)
    st.markdown("</div>", unsafe_allow_html=True)

    # Display stress test results below (full width)
    display_stress_test_results(selected_key)


def display_risk_analysis_results(
    backtest_key: str, confidence_level: float, lookback_days: int, var_method: str
):
    """Display risk analysis results from backend"""

    try:
        # Use RiskManagementProcessor from utils for proper backend integration
        from utils.risk_management_utils import RiskManagementProcessor

        processor = RiskManagementProcessor()

        # Debug: Show available backtest data
        if (
            "backtest_cache" in st.session_state
            and backtest_key in st.session_state.backtest_cache
        ):
            backtest_data = st.session_state.backtest_cache[backtest_key]
            st.info(
                f"Debug: Available data keys in backtest '{backtest_key}': {list(backtest_data.keys())}"
            )

            # Show sample of data structure
            with st.expander("🔍 Debug: Backtest Data Structure"):
                for key, value in backtest_data.items():
                    if isinstance(value, (list, pd.Series, np.ndarray)):
                        st.write(
                            f"- **{key}**: {type(value).__name__} with {len(value)} items"
                        )
                        if len(value) > 0:
                            st.write(
                                f"  - Sample: {value[:3] if hasattr(value, '__getitem__') else str(value)[:100]}"
                            )
                    elif isinstance(value, dict):
                        st.write(f"- **{key}**: dict with keys: {list(value.keys())}")
                    else:
                        st.write(
                            f"- **{key}**: {type(value).__name__} = {str(value)[:100]}"
                        )

        # Extract returns data from backtest
        returns_data = processor.extract_backtest_data(backtest_key)

        if returns_data is None or len(returns_data) == 0:
            st.error("No valid returns data found for analysis")
            st.info(
                "Please ensure your backtest has generated valid portfolio returns data."
            )
            return

        # Show returns data info
        st.success(f"✅ Found returns data: {len(returns_data)} data points")
        st.info(f"Returns range: {returns_data.min():.4f} to {returns_data.max():.4f}")

        # Get risk metrics using backend
        results = processor.get_risk_metrics_from_backend(
            returns_data, confidence_level
        )

        # Store for display
        if "risk_analysis" not in st.session_state:
            st.session_state.risk_analysis = {}
        st.session_state.risk_analysis[backtest_key] = results

        st.success("🎉 Risk analysis completed!")

    except Exception as e:
        st.error(f"Risk analysis failed: {e}")
        import traceback

        st.error(f"Full error: {traceback.format_exc()}")


def display_portfolio_optimization(
    backtest_key: str, method: str, max_weight: float, target_volatility: float
):
    """Display portfolio optimization from backend"""

    try:
        # Use RiskManagementProcessor for proper backend integration
        from utils.risk_management_utils import RiskManagementProcessor

        processor = RiskManagementProcessor()

        # Extract returns data from backtest (same as risk analysis)
        returns_data = processor.extract_backtest_data(backtest_key)

        if returns_data is None or len(returns_data) == 0:
            st.error("No valid returns data found for portfolio optimization")
            st.info(
                "Please ensure your backtest has generated valid portfolio returns data."
            )
            return

        st.info(f"Using returns data: {len(returns_data)} data points for optimization")

        # Get optimization results from backend
        results = processor.get_portfolio_optimization_from_backend(
            returns_data, method, max_weight, target_volatility
        )

        # Store for display
        if "optimization_results" not in st.session_state:
            st.session_state.optimization_results = {}
        st.session_state.optimization_results[backtest_key] = results

        if "error" not in results:
            st.success("🎉 Portfolio optimization completed!")
        else:
            st.warning(
                f"Optimization completed with issues: {results.get('error', 'Unknown error')}"
            )

    except Exception as e:
        st.error(f"Portfolio optimization failed: {e}")
        import traceback

        st.error(f"Full error: {traceback.format_exc()}")


def run_stress_test(
    backtest_key: str, stress_type: str, severity: str, num_simulations: int
):
    """Run stress test using backend ONLY"""

    try:
        # Use RiskManagementProcessor for proper backend integration
        from utils.risk_management_utils import RiskManagementProcessor

        processor = RiskManagementProcessor()

        # Extract returns data from backtest (same as risk analysis)
        returns_data = processor.extract_backtest_data(backtest_key)

        if returns_data is None or len(returns_data) == 0:
            st.error("No valid returns data found for stress testing")
            st.info(
                "Please ensure your backtest has generated valid portfolio returns data."
            )
            return

        st.info(
            f"Using returns data: {len(returns_data)} data points for stress testing"
        )

        # Get stress test results from backend
        results = processor.get_stress_test_from_backend(
            returns_data, stress_type, severity, num_simulations
        )

        # Store stress test results
        if "stress_test_results" not in st.session_state:
            st.session_state.stress_test_results = {}
        st.session_state.stress_test_results[backtest_key] = results

        if "error" not in results:
            st.success("🎉 Stress test completed!")
        else:
            st.warning(
                f"Stress test completed with issues: {results.get('error', 'Unknown error')}"
            )

    except Exception as e:
        st.error(f"Stress test failed: {e}")
        import traceback

        st.error(f"Full error: {traceback.format_exc()}")


def display_risk_results(backtest_key: str):
    """Display risk analysis results - UI only"""

    if (
        "risk_analysis" in st.session_state
        and backtest_key in st.session_state.risk_analysis
    ):

        results = st.session_state.risk_analysis[backtest_key]

        st.subheader("📊 Risk Analysis Results")
        st.markdown("---")

        # Display metrics in columns (full width layout)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("VaR (95%)", f"{results.get('var_95', 0):.2%}")
            st.metric("CVaR (95%)", f"{results.get('cvar_95', 0):.2%}")

        with col2:
            st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
            st.metric("Volatility (Annual)", f"{results.get('volatility', 0):.2%}")

        with col3:
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.3f}")
            st.metric("Sortino Ratio", f"{results.get('sortino_ratio', 0):.3f}")

        # Additional metrics row
        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("Calmar Ratio", f"{results.get('calmar_ratio', 0):.3f}")

        with col5:
            if "analysis_date" in results:
                st.write(
                    f"**Analysis Date:** {results['analysis_date'].strftime('%Y-%m-%d %H:%M')}"
                )

    else:
        st.info(
            "No risk analysis results available. Click 'Calculate Risk Metrics' to analyze."
        )


def display_optimization_results(backtest_key: str):
    """Display optimization results - UI only"""

    if (
        "optimization_results" in st.session_state
        and backtest_key in st.session_state.optimization_results
    ):

        results = st.session_state.optimization_results[backtest_key]

        st.subheader("🎯 Portfolio Optimization Results")
        st.markdown("---")

        # Display optimal weights if available
        if "weights" in results:
            st.write("**Optimal Asset Weights:**")
            weights_df = pd.DataFrame(
                {
                    "Asset": list(results["weights"].keys()),
                    "Weight": [f"{w:.1%}" for w in results["weights"].values()],
                }
            )
            st.dataframe(weights_df, use_container_width=True)

        # Display performance metrics if available
        if "expected_return" in results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Return", f"{results.get('expected_return', 0):.2%}")
            with col2:
                st.metric("Expected Volatility", f"{results.get('volatility', 0):.2%}")
            with col3:
                sharpe = results.get("expected_return", 0) / max(
                    results.get("volatility", 0.01), 0.01
                )
                st.metric("Expected Sharpe", f"{sharpe:.3f}")

    else:
        st.info(
            "No optimization results available. Click 'Optimize Portfolio' to optimize."
        )


def display_stress_test_results(backtest_key: str):
    """Display stress test results - UI only"""

    if (
        "stress_test_results" in st.session_state
        and backtest_key in st.session_state.stress_test_results
    ):

        results = st.session_state.stress_test_results[backtest_key]

        st.subheader("🚨 Stress Test Results")
        st.markdown("---")

        # Display scenario info
        scenario = results.get("scenario", {})
        st.write("**Applied Stress Scenario:**")

        # Format scenario information better
        if scenario:
            scenario_df = pd.DataFrame(
                [
                    (key.replace("_", " ").title(), str(value))
                    for key, value in scenario.items()
                ],
                columns=["Parameter", "Value"],
            )
            st.dataframe(scenario_df, use_container_width=True)

        # Display stress test metrics
        num_sims = results.get("num_simulations", 0)

        col1, col2, col3 = st.columns(3)

        with col1:
            worst_case = results.get("worst_case_loss", 0)
            st.metric("Worst Case Loss", f"{worst_case:.2%}")

        with col2:
            prob_loss = results.get("probability_of_loss", 0)
            st.metric("Probability of Loss", f"{prob_loss:.1%}")

        with col3:
            st.metric("Simulations", f"{num_sims:,}")

        # Additional metrics if available
        if "average_loss" in results:
            st.metric(
                "Average Loss (when loss occurs)", f"{results['average_loss']:.2%}"
            )

    else:
        st.info("No stress test results available. Click 'Run Stress Test' to analyze.")


if __name__ == "__main__":
    main()
