"""
Streamlit Page: Risk Management
Week 14 UI Integration - Professional Risk Management Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Week 12: Risk Management Framework Integration
    from src.risk.portfolio_optimization import (
        PortfolioOptimizer,
        AFMLPortfolioOptimizer,
    )
    from src.risk.risk_metrics import RiskMetrics, PortfolioRiskAnalyzer
    from src.backtesting.portfolio import Portfolio, RiskModel
    from src.config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Professional Risk Management Interface"""

    st.title("🛡️ Risk Management")
    st.markdown("**Professional Portfolio Risk Management Platform**")

    # Initialize session state
    if "risk_cache" not in st.session_state:
        st.session_state.risk_cache = {}

    # Check for available backtests
    if "backtest_cache" not in st.session_state or not st.session_state.backtest_cache:
        st.warning("⚡ Please run backtests first from Backtesting page")
        return

    # Professional UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        risk_control_panel()

    with col2:
        risk_display_panel()


def risk_control_panel():
    """Risk Management Control Panel"""

    st.subheader("🎯 Risk Analysis Configuration")

    # Backtest selection
    available_backtests = list(st.session_state.backtest_cache.keys())
    selected_backtest = st.selectbox("Select Backtest", available_backtests)

    if not selected_backtest:
        return

    # Risk analysis type
    st.subheader("🛡️ Risk Analysis Types")

    tab1, tab2, tab3 = st.tabs(["Portfolio Risk", "Optimization", "Stress Testing"])

    with tab1:
        portfolio_risk_config(selected_backtest)

    with tab2:
        portfolio_optimization_config(selected_backtest)

    with tab3:
        stress_testing_config(selected_backtest)


def portfolio_risk_config(backtest_key: str):
    """Portfolio Risk Metrics Configuration"""

    st.markdown("**Portfolio Risk Metrics**")

    # Risk calculation parameters
    confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
    lookback_days = st.slider("Lookback Days", 30, 252, 126)

    # VaR method selection
    var_method = st.selectbox(
        "VaR Method", ["Historical", "Parametric", "Cornish-Fisher"], key="var_method"
    )

    # Risk decomposition
    enable_decomposition = st.checkbox("Enable Risk Decomposition", value=True)

    # Calculate button
    if st.button("📊 Calculate Risk Metrics", type="primary", use_container_width=True):
        calculate_risk_metrics(
            backtest_key,
            confidence_level,
            lookback_days,
            var_method,
            enable_decomposition,
        )


def portfolio_optimization_config(backtest_key: str):
    """Portfolio Optimization Configuration"""

    st.markdown("**Portfolio Optimization**")

    # Optimization method
    optimization_method = st.selectbox(
        "Optimization Method",
        [
            "Mean Variance",
            "Risk Parity",
            "Minimum Variance",
            "Black-Litterman",
            "AFML Ensemble",
        ],
        key="opt_method",
    )

    # Risk constraints
    max_weight = st.slider("Max Weight per Asset", 0.1, 1.0, 0.3, key="max_weight")
    target_volatility = (
        st.slider("Target Volatility (%)", 5.0, 30.0, 15.0, key="target_vol") / 100
    )

    # Rebalancing frequency
    rebalance_freq = st.selectbox(
        "Rebalancing Frequency",
        ["Daily", "Weekly", "Monthly", "Quarterly"],
        index=2,
        key="rebalance_freq",
    )

    # Optimize button
    if st.button("⚖️ Optimize Portfolio", type="primary", use_container_width=True):
        optimize_portfolio(
            backtest_key,
            optimization_method,
            max_weight,
            target_volatility,
            rebalance_freq,
        )


def stress_testing_config(backtest_key: str):
    """Stress Testing Configuration"""

    st.markdown("**Stress Testing & Scenario Analysis**")

    # Stress test type
    stress_type = st.selectbox(
        "Stress Test Type",
        ["Market Crash", "Interest Rate Shock", "Volatility Spike", "Custom Scenario"],
        key="stress_type",
    )

    # Scenario parameters
    if stress_type == "Market Crash":
        market_drop = st.slider("Market Drop (%)", 10, 50, 20, key="market_drop")
        scenario_params = {"market_drop": market_drop / 100}
    elif stress_type == "Interest Rate Shock":
        rate_change = st.slider("Rate Change (bps)", 50, 500, 100, key="rate_change")
        scenario_params = {"rate_change": rate_change}
    elif stress_type == "Volatility Spike":
        vol_multiplier = st.slider(
            "Volatility Multiplier", 1.5, 5.0, 2.0, key="vol_mult"
        )
        scenario_params = {"vol_multiplier": vol_multiplier}
    else:  # Custom
        custom_shock = st.slider("Custom Shock (%)", -50, 50, -10, key="custom_shock")
        scenario_params = {"custom_shock": custom_shock / 100}

    # Monte Carlo simulations
    num_simulations = st.slider(
        "Number of Simulations", 100, 10000, 1000, key="num_sims"
    )

    # Run stress test button
    if st.button("🔥 Run Stress Test", type="primary", use_container_width=True):
        run_stress_test(
            backtest_key,
            stress_type,
            scenario_params,
            num_simulations,
        )


def risk_display_panel():
    """Risk Display and Analysis Panel"""

    if not st.session_state.risk_cache:
        st.info("🛡️ Configure and run risk analysis to see results")
        return

    # Risk analysis selection
    selected_analysis = st.selectbox(
        "Select Risk Analysis", list(st.session_state.risk_cache.keys())
    )

    if selected_analysis:
        display_risk_overview(selected_analysis)
        display_risk_charts(selected_analysis)


def calculate_risk_metrics(
    backtest_key: str,
    confidence_level: float,
    lookback_days: int,
    var_method: str,
    enable_decomposition: bool,
):
    """Calculate portfolio risk metrics using Week 12 modules"""

    try:
        cached_backtest = st.session_state.backtest_cache[backtest_key]
        returns = cached_backtest["returns"]
        portfolio_values = cached_backtest["portfolio_values"]

        with st.spinner("Calculating risk metrics..."):
            # Use existing Week 12 risk modules
            risk_metrics = RiskMetrics(
                confidence_level=confidence_level, min_periods=30
            )

            # Calculate basic risk metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            # Calculate VaR and CVaR
            if var_method == "Historical":
                var_result = risk_metrics.value_at_risk(returns, method="historical")
            elif var_method == "Parametric":
                var_result = risk_metrics.value_at_risk(returns, method="parametric")
            else:  # Cornish-Fisher
                var_result = risk_metrics.value_at_risk(
                    returns, method="cornish_fisher"
                )

            cvar = risk_metrics.conditional_var(returns, method=var_method.lower())

            # Portfolio-specific risk metrics
            portfolio = Portfolio(
                initial_capital=cached_backtest["engine_config"]["initial_capital"]
            )

            # Create mock returns data for portfolio risk calculation
            returns_df = pd.DataFrame({"ASSET": returns})
            portfolio_risk = portfolio.calculate_portfolio_risk(
                returns_df, lookback_days
            )

            # Maximum drawdown
            dd_metrics = risk_metrics.maximum_drawdown(returns)

            # Compile risk metrics
            risk_analysis = {
                "volatility": volatility,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "var_95": var_result,
                "cvar_95": cvar,
                "max_drawdown": dd_metrics["max_drawdown"],
                "drawdown_duration": dd_metrics["max_duration"],
                "portfolio_risk": portfolio_risk,
                "confidence_level": confidence_level,
                "lookback_days": lookback_days,
                "var_method": var_method,
                "returns": returns,
                "portfolio_values": portfolio_values,
                "analysis_type": "risk_metrics",
                "calculated_at": datetime.now(),
            }

            # Store results
            analysis_key = (
                f"risk_metrics_{backtest_key}_{datetime.now().strftime('%H%M%S')}"
            )
            st.session_state.risk_cache[analysis_key] = risk_analysis

        st.success("✅ Risk metrics calculated successfully")
        st.rerun()

    except Exception as e:
        st.error(f"Risk metrics calculation failed: {str(e)}")


def optimize_portfolio(
    backtest_key: str,
    optimization_method: str,
    max_weight: float,
    target_volatility: float,
    rebalance_freq: str,
):
    """Optimize portfolio using Week 12 modules"""

    try:
        cached_backtest = st.session_state.backtest_cache[backtest_key]
        returns = cached_backtest["returns"]

        with st.spinner(f"Running {optimization_method} optimization..."):
            # Use existing Week 12 optimization modules
            optimizer = PortfolioOptimizer()

            # Create mock multi-asset returns for optimization
            num_assets = 4
            mock_returns = np.random.multivariate_normal(
                mean=[returns.mean()] * num_assets,
                cov=np.eye(num_assets) * returns.var(),
                size=len(returns),
            )
            returns_df = pd.DataFrame(
                mock_returns,
                columns=[f"Asset_{i+1}" for i in range(num_assets)],
                index=returns.index[: len(mock_returns)],
            )

            expected_returns = returns_df.mean().values * 252  # Annualized
            covariance_matrix = returns_df.cov().values * 252  # Annualized

            # Run optimization based on method
            if optimization_method == "Mean Variance":
                result = optimizer.mean_variance_optimization(
                    expected_returns, covariance_matrix, objective="sharpe"
                )
            elif optimization_method == "Risk Parity":
                result = optimizer.risk_parity_optimization(covariance_matrix)
            elif optimization_method == "Minimum Variance":
                result = optimizer.minimum_variance_optimization(covariance_matrix)
            elif optimization_method == "Black-Litterman":
                market_caps = np.ones(num_assets) / num_assets  # Equal market caps
                result = optimizer.black_litterman_optimization(
                    market_caps, covariance_matrix
                )
            else:  # AFML Ensemble
                afml_optimizer = AFMLPortfolioOptimizer()
                result = afml_optimizer.ensemble_optimization(
                    returns_df,
                    optimization_methods=[
                        "mean_variance",
                        "risk_parity",
                        "min_variance",
                    ],
                )

            # Compile optimization results
            optimization_analysis = {
                "optimization_method": optimization_method,
                "optimization_result": result,
                "expected_returns": expected_returns,
                "covariance_matrix": covariance_matrix,
                "returns_data": returns_df,
                "constraints": {
                    "max_weight": max_weight,
                    "target_volatility": target_volatility,
                },
                "rebalance_freq": rebalance_freq,
                "analysis_type": "optimization",
                "calculated_at": datetime.now(),
            }

            # Store results
            analysis_key = f"optimization_{optimization_method}_{backtest_key}_{datetime.now().strftime('%H%M%S')}"
            st.session_state.risk_cache[analysis_key] = optimization_analysis

        st.success(f"✅ {optimization_method} optimization completed")
        st.rerun()

    except Exception as e:
        st.error(f"Portfolio optimization failed: {str(e)}")


def run_stress_test(
    backtest_key: str,
    stress_type: str,
    scenario_params: dict,
    num_simulations: int,
):
    """Run stress test and scenario analysis"""

    try:
        cached_backtest = st.session_state.backtest_cache[backtest_key]
        returns = cached_backtest["returns"]
        portfolio_values = cached_backtest["portfolio_values"]

        with st.spinner(f"Running {stress_type} stress test..."):
            # Generate stress scenarios
            np.random.seed(42)  # For reproducibility

            stressed_returns = []
            stressed_portfolio_values = []

            for i in range(num_simulations):
                # Apply stress scenario to returns
                if stress_type == "Market Crash":
                    shock = scenario_params["market_drop"]
                    stressed_return = returns + np.random.normal(
                        -shock, 0.01, len(returns)
                    )
                elif stress_type == "Interest Rate Shock":
                    # Simplified interest rate impact
                    rate_impact = (
                        scenario_params["rate_change"] / 10000
                    )  # bps to decimal
                    stressed_return = returns - rate_impact
                elif stress_type == "Volatility Spike":
                    vol_mult = scenario_params["vol_multiplier"]
                    stressed_return = returns * vol_mult
                else:  # Custom
                    shock = scenario_params["custom_shock"]
                    stressed_return = returns + shock

                stressed_returns.append(stressed_return)

                # Calculate stressed portfolio values
                initial_value = portfolio_values[0]
                stressed_values = [initial_value]
                for ret in stressed_return:
                    stressed_values.append(stressed_values[-1] * (1 + ret))

                stressed_portfolio_values.append(stressed_values[1:])

            # Calculate stress test statistics
            final_values = [pv[-1] for pv in stressed_portfolio_values]
            total_returns = [
                (fv / portfolio_values[0] - 1) * 100 for fv in final_values
            ]

            stress_statistics = {
                "mean_return": np.mean(total_returns),
                "median_return": np.median(total_returns),
                "worst_case": np.min(total_returns),
                "best_case": np.max(total_returns),
                "var_95": np.percentile(total_returns, 5),
                "var_99": np.percentile(total_returns, 1),
                "probability_of_loss": np.mean(np.array(total_returns) < 0) * 100,
            }

            # Compile stress test results
            stress_analysis = {
                "stress_type": stress_type,
                "scenario_params": scenario_params,
                "num_simulations": num_simulations,
                "stress_statistics": stress_statistics,
                "total_returns": total_returns,
                "stressed_portfolio_values": stressed_portfolio_values,
                "original_returns": returns,
                "original_portfolio_values": portfolio_values,
                "analysis_type": "stress_test",
                "calculated_at": datetime.now(),
            }

            # Store results
            analysis_key = f"stress_{stress_type}_{backtest_key}_{datetime.now().strftime('%H%M%S')}"
            st.session_state.risk_cache[analysis_key] = stress_analysis

        st.success(f"✅ {stress_type} stress test completed")
        st.rerun()

    except Exception as e:
        st.error(f"Stress test failed: {str(e)}")


def display_risk_overview(analysis_key: str):
    """Display risk analysis overview"""

    cached_analysis = st.session_state.risk_cache[analysis_key]
    analysis_type = cached_analysis["analysis_type"]

    st.subheader(f"🛡️ Risk Analysis: {analysis_key}")

    if analysis_type == "risk_metrics":
        display_risk_metrics_overview(cached_analysis)
    elif analysis_type == "optimization":
        display_optimization_overview(cached_analysis)
    elif analysis_type == "stress_test":
        display_stress_test_overview(cached_analysis)


def display_risk_metrics_overview(analysis: dict):
    """Display risk metrics overview"""

    # Key risk metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Volatility", f"{analysis['volatility'] * 100:.2f}%")

    with col2:
        st.metric("VaR 95%", f"{analysis['var_95']:.4f}")

    with col3:
        st.metric("CVaR 95%", f"{analysis['cvar_95']:.4f}")

    with col4:
        st.metric("Max Drawdown", f"{analysis['max_drawdown']:.3f}")

    # Risk metrics table
    st.subheader("📊 Detailed Risk Metrics")

    risk_data = {
        "Metric": [
            "Annualized Volatility",
            "Skewness",
            "Kurtosis",
            f"VaR {analysis['confidence_level']:.0%}",
            f"CVaR {analysis['confidence_level']:.0%}",
            "Maximum Drawdown",
            "Drawdown Duration (days)",
            "Portfolio Beta",
        ],
        "Value": [
            f"{analysis['volatility'] * 100:.2f}%",
            f"{analysis['skewness']:.3f}",
            f"{analysis['kurtosis']:.3f}",
            f"{analysis['var_95']:.4f}",
            f"{analysis['cvar_95']:.4f}",
            f"{analysis['max_drawdown']:.3f}",
            f"{analysis['drawdown_duration']:.0f}",
            f"{analysis['portfolio_risk'].get('beta', 0):.3f}",
        ],
    }

    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)


def display_optimization_overview(analysis: dict):
    """Display optimization overview"""

    optimization_result = analysis["optimization_result"]
    method = analysis["optimization_method"]

    # Optimization metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Method", method)

    with col2:
        if "expected_return" in optimization_result:
            st.metric(
                "Expected Return",
                f"{optimization_result['expected_return'] * 100:.2f}%",
            )

    with col3:
        if "volatility" in optimization_result:
            st.metric("Volatility", f"{optimization_result['volatility'] * 100:.2f}%")

    with col4:
        if "sharpe_ratio" in optimization_result:
            st.metric("Sharpe Ratio", f"{optimization_result['sharpe_ratio']:.3f}")

    # Optimal weights
    st.subheader("⚖️ Optimal Asset Allocation")

    if "weights" in optimization_result:
        weights = optimization_result["weights"]
        returns_data = analysis["returns_data"]

        weights_data = {
            "Asset": returns_data.columns,
            "Weight": [f"{w:.1%}" for w in weights],
            "Weight Value": weights,
        }

        weights_df = pd.DataFrame(weights_data)
        st.dataframe(
            weights_df[["Asset", "Weight"]], use_container_width=True, hide_index=True
        )

        # Weights pie chart
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=weights_data["Asset"],
                    values=weights_data["Weight Value"],
                    textinfo="label+percent",
                )
            ]
        )

        fig_pie.update_layout(
            title="Optimal Asset Allocation",
            height=400,
        )

        st.plotly_chart(fig_pie, use_container_width=True)


def display_stress_test_overview(analysis: dict):
    """Display stress test overview"""

    stress_stats = analysis["stress_statistics"]
    stress_type = analysis["stress_type"]
    num_sims = analysis["num_simulations"]

    # Stress test summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Worst Case", f"{stress_stats['worst_case']:.2f}%")

    with col2:
        st.metric("Best Case", f"{stress_stats['best_case']:.2f}%")

    with col3:
        st.metric("VaR 95%", f"{stress_stats['var_95']:.2f}%")

    with col4:
        st.metric("Loss Probability", f"{stress_stats['probability_of_loss']:.1f}%")

    # Stress test details
    st.subheader(f"🔥 {stress_type} Stress Test Results")

    stress_data = {
        "Statistic": [
            "Number of Simulations",
            "Mean Return",
            "Median Return",
            "Worst Case Return",
            "Best Case Return",
            "VaR 95%",
            "VaR 99%",
            "Probability of Loss",
        ],
        "Value": [
            f"{num_sims:,}",
            f"{stress_stats['mean_return']:.2f}%",
            f"{stress_stats['median_return']:.2f}%",
            f"{stress_stats['worst_case']:.2f}%",
            f"{stress_stats['best_case']:.2f}%",
            f"{stress_stats['var_95']:.2f}%",
            f"{stress_stats['var_99']:.2f}%",
            f"{stress_stats['probability_of_loss']:.1f}%",
        ],
    }

    stress_df = pd.DataFrame(stress_data)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)


def display_risk_charts(analysis_key: str):
    """Display risk analysis charts"""

    cached_analysis = st.session_state.risk_cache[analysis_key]
    analysis_type = cached_analysis["analysis_type"]

    st.subheader("📈 Risk Visualization")

    if analysis_type == "risk_metrics":
        display_risk_metrics_charts(cached_analysis)
    elif analysis_type == "optimization":
        display_optimization_charts(cached_analysis)
    elif analysis_type == "stress_test":
        display_stress_test_charts(cached_analysis)


def display_risk_metrics_charts(analysis: dict):
    """Display risk metrics charts"""

    returns = analysis["returns"]

    # Create risk metrics charts
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Returns Distribution",
            "Rolling Volatility",
            "VaR Evolution",
            "Drawdown Analysis",
        ),
    )

    # Returns histogram
    fig.add_trace(
        go.Histogram(x=returns, nbinsx=50, name="Returns", opacity=0.7), row=1, col=1
    )

    # Rolling volatility
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    fig.add_trace(
        go.Scatter(x=returns.index, y=rolling_vol, mode="lines", name="Rolling Vol"),
        row=1,
        col=2,
    )

    # Rolling VaR
    rolling_var = returns.rolling(30).quantile(0.05)
    fig.add_trace(
        go.Scatter(x=returns.index, y=rolling_var, mode="lines", name="Rolling VaR"),
        row=2,
        col=1,
    )

    # Drawdown
    portfolio_values = analysis["portfolio_values"]
    portfolio_series = pd.Series(
        portfolio_values, index=returns.index[: len(portfolio_values)]
    )
    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - rolling_max) / rolling_max

    fig.add_trace(
        go.Scatter(
            x=returns.index[: len(drawdown)],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, title="Risk Metrics Analysis")
    st.plotly_chart(fig, use_container_width=True)


def display_optimization_charts(analysis: dict):
    """Display optimization charts"""

    returns_data = analysis["returns_data"]
    optimization_result = analysis["optimization_result"]

    # Efficient frontier (simplified)
    st.write("**Efficient Frontier Visualization**")

    num_portfolios = 100
    results = np.zeros((3, num_portfolios))

    np.random.seed(42)
    weights_array = np.random.dirichlet(
        np.ones(len(returns_data.columns)), num_portfolios
    )

    for i, weights in enumerate(weights_array):
        portfolio_return = np.sum(returns_data.mean() * weights) * 252
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(returns_data.cov() * 252, weights))
        )

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = portfolio_return / portfolio_std if portfolio_std > 0 else 0

    # Plot efficient frontier
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results[1],
            y=results[0],
            mode="markers",
            marker=dict(
                color=results[2],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
            ),
            name="Random Portfolios",
        )
    )

    # Add optimal portfolio if available
    if "volatility" in optimization_result and "expected_return" in optimization_result:
        fig.add_trace(
            go.Scatter(
                x=[optimization_result["volatility"]],
                y=[optimization_result["expected_return"]],
                mode="markers",
                marker=dict(color="red", size=15, symbol="star"),
                name="Optimal Portfolio",
            )
        )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def display_stress_test_charts(analysis: dict):
    """Display stress test charts"""

    total_returns = analysis["total_returns"]
    stress_type = analysis["stress_type"]

    # Stress test results distribution
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Returns Distribution", "Cumulative Distribution"),
    )

    # Histogram of stressed returns
    fig.add_trace(
        go.Histogram(x=total_returns, nbinsx=50, name="Stressed Returns", opacity=0.7),
        row=1,
        col=1,
    )

    # Cumulative distribution
    sorted_returns = np.sort(total_returns)
    cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)

    fig.add_trace(
        go.Scatter(x=sorted_returns, y=cumulative_prob, mode="lines", name="CDF"),
        row=1,
        col=2,
    )

    fig.update_layout(title=f"{stress_type} Stress Test Results", height=400)

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
