"""
Demo: Phase 1 Week 3 - Basic Analysis Functions

Demonstrates the usage of the newly implemented basic analysis functions
following AFML principles.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis import (
    ReturnAnalyzer,
    VolatilityAnalyzer,
    StatisticsAnalyzer,
    CorrelationAnalyzer,
)

from data.collectors import YFinanceCollector
from data.validators import DataValidator
from data.storage import SQLiteStorage


def create_sample_data():
    """Create sample financial data for demonstration."""
    print("Creating sample financial data...")

    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")

    # Generate multiple assets
    assets = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    price_data = {}

    for asset in assets:
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [100]  # Starting price

        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        price_data[asset] = pd.Series(prices[1:], index=dates, name=asset)

    return pd.DataFrame(price_data)


def demo_return_analysis():
    """Demonstrate return analysis functionality."""
    print("\n" + "=" * 60)
    print("RETURN ANALYSIS DEMO")
    print("=" * 60)

    # Create sample data
    prices_df = create_sample_data()
    aapl_prices = prices_df["AAPL"]

    # Initialize return analyzer
    analyzer = ReturnAnalyzer(risk_free_rate=0.02)

    print(f"Analyzing {len(aapl_prices)} days of AAPL price data...")
    print(f"Price range: ${aapl_prices.min():.2f} - ${aapl_prices.max():.2f}")

    # Calculate different types of returns
    simple_returns = analyzer.calculate_simple_returns(aapl_prices)
    log_returns = analyzer.calculate_log_returns(aapl_prices)

    print(f"\nSimple Returns Statistics:")
    print(f"  Mean: {simple_returns.mean():.6f}")
    print(f"  Std:  {simple_returns.std():.6f}")

    print(f"\nLog Returns Statistics:")
    print(f"  Mean: {log_returns.mean():.6f}")
    print(f"  Std:  {log_returns.std():.6f}")

    # Comprehensive analysis
    print(f"\nComprehensive Return Analysis:")
    stats = analyzer.analyze_returns(aapl_prices, return_type="simple")

    print(f"  Total Return:        {stats.total_return:.2%}")
    print(f"  Annualized Return:   {stats.annualized_return:.2%}")
    print(f"  Annualized Vol:      {stats.annualized_volatility:.2%}")
    print(f"  Sharpe Ratio:        {stats.sharpe_ratio:.3f}")
    print(f"  Maximum Drawdown:    {stats.max_drawdown:.2%}")
    print(f"  Skewness:           {stats.skewness:.3f}")
    print(f"  Kurtosis:           {stats.kurtosis:.3f}")


def demo_volatility_analysis():
    """Demonstrate volatility analysis functionality."""
    print("\n" + "=" * 60)
    print("VOLATILITY ANALYSIS DEMO")
    print("=" * 60)

    # Create sample data
    prices_df = create_sample_data()
    aapl_prices = prices_df["AAPL"]

    # Initialize volatility analyzer
    analyzer = VolatilityAnalyzer(window=30)

    # Calculate returns
    returns = aapl_prices.pct_change().dropna()

    print(f"Analyzing volatility for {len(returns)} return observations...")

    # Simple volatility
    simple_vol = analyzer.calculate_simple_volatility(returns, window=30)
    print(f"\nSimple Volatility (30-day):")
    print(f"  Current: {simple_vol.iloc[-1]:.2%}")
    print(f"  Average: {simple_vol.mean():.2%}")
    print(f"  Range:   {simple_vol.min():.2%} - {simple_vol.max():.2%}")

    # EWMA volatility
    ewma_vol = analyzer.calculate_ewma_volatility(returns, lambda_param=0.94)
    print(f"\nEWMA Volatility (λ=0.94):")
    print(f"  Current: {ewma_vol.iloc[-1]:.2%}")
    print(f"  Average: {ewma_vol.mean():.2%}")

    # Create OHLC data for advanced estimators
    ohlc_data = pd.DataFrame(
        {
            "Open": aapl_prices,
            "High": aapl_prices * (1 + np.random.uniform(0, 0.02, len(aapl_prices))),
            "Low": aapl_prices * (1 - np.random.uniform(0, 0.02, len(aapl_prices))),
            "Close": aapl_prices,
        }
    )

    # Garman-Klass volatility
    gk_vol = analyzer.calculate_garman_klass_volatility(ohlc_data, window=30)
    print(f"\nGarman-Klass Volatility:")
    print(f"  Current: {gk_vol.iloc[-1]:.2%}")
    print(f"  Average: {gk_vol.mean():.2%}")

    # Comprehensive analysis
    vol_stats = analyzer.analyze_volatility(returns)
    print(f"\nVolatility Statistics:")
    print(f"  Current Volatility:     {vol_stats.current_volatility:.2%}")
    print(f"  Average Volatility:     {vol_stats.average_volatility:.2%}")
    print(f"  Volatility of Vol:      {vol_stats.volatility_std:.4f}")
    print(f"  Vol Skewness:          {vol_stats.volatility_skewness:.3f}")


def demo_statistics_analysis():
    """Demonstrate statistical analysis functionality."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS DEMO")
    print("=" * 60)

    # Create sample data
    prices_df = create_sample_data()
    aapl_prices = prices_df["AAPL"]
    returns = aapl_prices.pct_change().dropna()

    # Initialize statistics analyzer
    analyzer = StatisticsAnalyzer()

    print(f"Statistical analysis of {len(returns)} return observations...")

    # Basic statistics
    basic_stats = analyzer.calculate_basic_statistics(returns)
    print(f"\nBasic Statistics:")
    print(f"  Count:       {basic_stats.count}")
    print(f"  Mean:        {basic_stats.mean:.6f}")
    print(f"  Std Dev:     {basic_stats.std:.6f}")
    print(f"  Min:         {basic_stats.min:.6f}")
    print(f"  25%:         {basic_stats.percentile_25:.6f}")
    print(f"  Median:      {basic_stats.median:.6f}")
    print(f"  75%:         {basic_stats.percentile_75:.6f}")
    print(f"  Max:         {basic_stats.max:.6f}")
    print(f"  Skewness:    {basic_stats.skewness:.3f}")
    print(f"  Kurtosis:    {basic_stats.kurtosis:.3f}")

    # Risk metrics
    risk_metrics = analyzer.calculate_risk_metrics(returns, aapl_prices)
    print(f"\nRisk Metrics:")
    print(f"  VaR (95%):           {risk_metrics.var_95:.4f}")
    print(f"  VaR (99%):           {risk_metrics.var_99:.4f}")
    print(f"  CVaR (95%):          {risk_metrics.cvar_95:.4f}")
    print(f"  CVaR (99%):          {risk_metrics.cvar_99:.4f}")
    print(f"  Downside Deviation:  {risk_metrics.downside_deviation:.4f}")
    print(f"  Sortino Ratio:       {risk_metrics.sortino_ratio:.3f}")
    print(f"  Calmar Ratio:        {risk_metrics.calmar_ratio:.3f}")
    print(f"  Max Drawdown:        {risk_metrics.maximum_drawdown:.2%}")

    # Distribution analysis
    dist_analysis = analyzer.analyze_distribution(returns)
    print(f"\nDistribution Analysis:")
    print(f"  Normality Test:")
    print(f"    Statistic:         {dist_analysis.normality_test_statistic:.3f}")
    print(f"    P-value:           {dist_analysis.normality_p_value:.6f}")
    print(f"    Is Normal:         {dist_analysis.is_normal}")
    print(f"  Autocorrelation:")
    print(f"    Lag-1 Corr:        {dist_analysis.autocorrelation_lag1:.3f}")
    print(f"    Ljung-Box p-val:   {dist_analysis.ljung_box_p_value:.6f}")
    print(f"    Has Autocorr:      {dist_analysis.has_autocorrelation}")
    print(f"  ARCH Effects:")
    print(f"    Test Statistic:    {dist_analysis.arch_test_statistic:.3f}")
    print(f"    P-value:           {dist_analysis.arch_test_p_value:.6f}")
    print(f"    Has ARCH:          {dist_analysis.has_arch_effects}")


def demo_correlation_analysis():
    """Demonstrate correlation analysis functionality."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS DEMO")
    print("=" * 60)

    # Create sample data
    prices_df = create_sample_data()
    returns_df = prices_df.pct_change().dropna()

    # Initialize correlation analyzer
    analyzer = CorrelationAnalyzer(method="pearson")

    print(
        f"Correlation analysis of {len(returns_df.columns)} assets over {len(returns_df)} days..."
    )

    # Static correlation matrix
    corr_matrix = analyzer.calculate_correlation_matrix(returns_df)
    print(f"\nStatic Correlation Matrix:")
    print(corr_matrix.round(3))

    # Rolling correlation example
    aapl_googl_rolling = analyzer.calculate_rolling_correlation(
        returns_df["AAPL"], returns_df["GOOGL"], window=60
    )
    print(f"\nAAPL-GOOGL Rolling Correlation (60-day):")
    print(f"  Current:     {aapl_googl_rolling.iloc[-1]:.3f}")
    print(f"  Average:     {aapl_googl_rolling.mean():.3f}")
    print(
        f"  Range:       {aapl_googl_rolling.min():.3f} - {aapl_googl_rolling.max():.3f}"
    )

    # EWMA correlation
    ewma_corr = analyzer.calculate_ewma_correlation(
        returns_df["AAPL"], returns_df["MSFT"], lambda_param=0.94
    )
    print(f"\nAAPL-MSFT EWMA Correlation:")
    print(f"  Current:     {ewma_corr.iloc[-1]:.3f}")
    print(f"  Average:     {ewma_corr.mean():.3f}")

    # Comprehensive correlation analysis
    corr_stats = analyzer.analyze_correlation_structure(returns_df)
    print(f"\nCorrelation Structure Analysis:")
    print(f"  Average Correlation:     {corr_stats.average_correlation:.3f}")
    print(f"  Max Correlation:         {corr_stats.max_correlation:.3f}")
    print(f"  Min Correlation:         {corr_stats.min_correlation:.3f}")
    print(f"  Correlation Stability:   {corr_stats.correlation_stability:.3f}")
    print(f"  Condition Number:        {corr_stats.condition_number:.1f}")
    print(f"  Eigenvalues:            {[f'{x:.3f}' for x in corr_stats.eigenvalues]}")

    # Tail correlation
    tail_corr = analyzer.calculate_tail_correlation(
        returns_df["AAPL"], returns_df["TSLA"], quantile=0.05
    )
    print(f"\nAAPL-TSLA Tail Correlation (5%):")
    print(f"  Tail Correlation:        {tail_corr:.3f}")


def demo_integration_with_data_collection():
    """Demonstrate integration with data collection system."""
    print("\n" + "=" * 60)
    print("INTEGRATION WITH DATA COLLECTION DEMO")
    print("=" * 60)

    print(
        "This demo shows how the analysis functions integrate with the data collection system."
    )
    print("In a real scenario, you would:")
    print("1. Use YFinanceCollector to fetch real market data")
    print("2. Validate data using DataValidator")
    print("3. Store data using SQLiteStorage")
    print("4. Apply analysis functions to the validated data")

    print(f"\nExample workflow:")
    print(f"```python")
    print(f"# Collect real data")
    print(f"collector = YFinanceCollector()")
    print(f"data = collector.collect_data('AAPL', '2023-01-01', '2024-01-01')")
    print(f"")
    print(f"# Validate data")
    print(f"validator = DataValidator()")
    print(f"validation_result = validator.validate_data(data)")
    print(f"")
    print(f"# Store data")
    print(f"storage = SQLiteStorage('market_data.db')")
    print(f"storage.store_data('AAPL', data)")
    print(f"")
    print(f"# Analyze returns")
    print(f"analyzer = ReturnAnalyzer()")
    print(f"stats = analyzer.analyze_returns(data['Close'])")
    print(f"print(f'Sharpe Ratio: {{stats.sharpe_ratio:.3f}}')")
    print(f"```")


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("QUANT ANALYTICS TOOL - PHASE 1 WEEK 3 DEMO")
    print("Basic Analysis Functions Implementation")
    print("Based on 'Advances in Financial Machine Learning' by Marcos López de Prado")
    print("=" * 80)

    try:
        # Run all demos
        demo_return_analysis()
        demo_volatility_analysis()
        demo_statistics_analysis()
        demo_correlation_analysis()
        demo_integration_with_data_collection()

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ Phase 1 Week 3: Basic Analysis Functions - COMPLETE")
        print()
        print("Implemented modules:")
        print("  📊 analysis.returns     - Return calculation and analysis")
        print("  📈 analysis.volatility  - Volatility estimation and analysis")
        print(
            "  📉 analysis.statistics  - Statistical measures and distribution analysis"
        )
        print("  🔗 analysis.correlation - Correlation analysis and structure")
        print()
        print("Key features:")
        print("  • AFML-compliant implementations")
        print("  • Comprehensive error handling and logging")
        print("  • Vectorized operations for performance")
        print("  • Multiple volatility estimators (Simple, EWMA, Garman-Klass)")
        print("  • Advanced risk metrics (VaR, CVaR, Sortino, Calmar)")
        print("  • Distribution testing (Normality, Autocorrelation, ARCH)")
        print("  • Dynamic correlation analysis")
        print("  • Full test coverage")
        print()
        print("Ready for Phase 1 Week 4: Advanced Feature Engineering! 🚀")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
