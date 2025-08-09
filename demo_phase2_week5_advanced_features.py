#!/usr/bin/env python3
"""
Advanced Feature Engineering Demo

This demo showcases the advanced feature engineering capabilities implemented
in Phase 2 Week 5, including:
- Fractal Dimension calculation
- Hurst Exponent estimation
- Information-driven Bars (Tick, Volume, Dollar bars)
- Triple Barrier Method for meta-labeling
- Fractional Differentiation

Based on Advances in Financial Machine Learning (AFML) methodology.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

from src.features.advanced import AdvancedFeatures, AdvancedFeatureResults

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
warnings.filterwarnings("ignore")


def generate_sample_market_data(n_points=2000, regime="mixed"):
    """
    Generate realistic market data with different regimes.

    Args:
        n_points: Number of data points
        regime: 'trending', 'mean_reverting', or 'mixed'
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_points, freq="5min")

    if regime == "trending":
        # Geometric Brownian Motion with positive drift
        drift = 0.0002
        volatility = 0.015
        returns = np.random.normal(drift, volatility, n_points)

    elif regime == "mean_reverting":
        # Ornstein-Uhlenbeck process
        prices = [100]
        theta = 0.1  # Mean reversion speed
        mu = 100  # Long-term mean
        sigma = 1.5  # Volatility

        for _ in range(n_points - 1):
            dt = 1 / 288  # 5-minute intervals in trading day
            price = prices[-1]
            dprice = (
                theta * (mu - price) * dt + sigma * np.sqrt(dt) * np.random.normal()
            )
            prices.append(price + dprice)

        prices = np.array(prices)
        returns = np.diff(np.log(prices))
        returns = np.concatenate([[0], returns])

    else:  # mixed regime
        # Switch between trending and mean-reverting
        returns = []
        current_regime = "trending"
        regime_length = 200

        for i in range(n_points):
            if i % regime_length == 0:
                current_regime = (
                    "mean_reverting" if current_regime == "trending" else "trending"
                )

            if current_regime == "trending":
                ret = np.random.normal(0.0001, 0.012)
            else:
                ret = np.random.normal(-0.00005, 0.008)  # Slight mean reversion

            returns.append(ret)

        returns = np.array(returns)

    # Generate prices from returns
    if regime != "mean_reverting":
        prices = 100 * np.exp(np.cumsum(returns))

    # Generate volume with realistic patterns
    base_volume = 1000
    volume_trend = np.sin(np.arange(n_points) * 2 * np.pi / 100) * 200 + base_volume
    volume_noise = np.random.lognormal(0, 0.3, n_points)
    volume = volume_trend * volume_noise

    # Generate OHLC data
    price_noise = 0.001
    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, price_noise, n_points)),
            "high": prices
            * (1 + np.abs(np.random.normal(0, price_noise * 1.5, n_points))),
            "low": prices
            * (1 - np.abs(np.random.normal(0, price_noise * 1.5, n_points))),
            "close": prices,
            "volume": volume.astype(int),
        },
        index=dates,
    )

    # Ensure OHLC constraints
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    return data


def demonstrate_fractal_analysis(advanced_features, data):
    """Demonstrate fractal dimension analysis."""
    print("\\n" + "=" * 60)
    print("FRACTAL DIMENSION ANALYSIS")
    print("=" * 60)

    prices = data["close"]

    # Calculate fractal dimension using both methods
    print("Calculating fractal dimension using Higuchi method...")
    fd_higuchi = advanced_features.calculate_fractal_dimension(
        prices, window=100, method="higuchi"
    )

    print("Calculating fractal dimension using Box-counting method...")
    fd_box = advanced_features.calculate_fractal_dimension(
        prices, window=100, method="box_counting"
    )

    # Display statistics
    fd_higuchi_clean = fd_higuchi.dropna()
    fd_box_clean = fd_box.dropna()

    print(f"\\nFractal Dimension (Higuchi Method):")
    print(f"  Valid calculations: {len(fd_higuchi_clean)}")
    if len(fd_higuchi_clean) > 0:
        print(f"  Mean: {fd_higuchi_clean.mean():.4f}")
        print(f"  Std:  {fd_higuchi_clean.std():.4f}")
        print(f"  Min:  {fd_higuchi_clean.min():.4f}")
        print(f"  Max:  {fd_higuchi_clean.max():.4f}")

    print(f"\\nFractal Dimension (Box-counting Method):")
    print(f"  Valid calculations: {len(fd_box_clean)}")
    if len(fd_box_clean) > 0:
        print(f"  Mean: {fd_box_clean.mean():.4f}")
        print(f"  Std:  {fd_box_clean.std():.4f}")
        print(f"  Min:  {fd_box_clean.min():.4f}")
        print(f"  Max:  {fd_box_clean.max():.4f}")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Price series
    axes[0].plot(prices.index, prices.values, "b-", alpha=0.7, linewidth=1)
    axes[0].set_title("Price Series", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    # Fractal dimension (Higuchi)
    if len(fd_higuchi_clean) > 0:
        axes[1].plot(fd_higuchi.index, fd_higuchi.values, "r-", alpha=0.8, linewidth=1)
        axes[1].axhline(y=fd_higuchi_clean.mean(), color="r", linestyle="--", alpha=0.7)
        axes[1].set_title(
            "Fractal Dimension (Higuchi Method)", fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("Fractal Dimension")
        axes[1].grid(True, alpha=0.3)

    # Fractal dimension (Box-counting)
    if len(fd_box_clean) > 0:
        axes[2].plot(fd_box.index, fd_box.values, "g-", alpha=0.8, linewidth=1)
        axes[2].axhline(y=fd_box_clean.mean(), color="g", linestyle="--", alpha=0.7)
        axes[2].set_title(
            "Fractal Dimension (Box-counting Method)", fontsize=12, fontweight="bold"
        )
        axes[2].set_ylabel("Fractal Dimension")
        axes[2].set_xlabel("Time")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fd_higuchi, fd_box


def demonstrate_hurst_analysis(advanced_features, data):
    """Demonstrate Hurst exponent analysis."""
    print("\\n" + "=" * 60)
    print("HURST EXPONENT ANALYSIS")
    print("=" * 60)

    prices = data["close"]

    # Calculate Hurst exponent using both methods
    print("Calculating Hurst exponent using R/S analysis...")
    hurst_rs = advanced_features.calculate_hurst_exponent(
        prices, window=150, method="rs"
    )

    print("Calculating Hurst exponent using DFA...")
    hurst_dfa = advanced_features.calculate_hurst_exponent(
        prices, window=150, method="dfa"
    )

    # Display statistics
    hurst_rs_clean = hurst_rs.dropna()
    hurst_dfa_clean = hurst_dfa.dropna()

    print(f"\\nHurst Exponent (R/S Analysis):")
    print(f"  Valid calculations: {len(hurst_rs_clean)}")
    if len(hurst_rs_clean) > 0:
        print(f"  Mean: {hurst_rs_clean.mean():.4f}")
        print(f"  Std:  {hurst_rs_clean.std():.4f}")
        print(f"  Min:  {hurst_rs_clean.min():.4f}")
        print(f"  Max:  {hurst_rs_clean.max():.4f}")

        # Interpret results
        mean_hurst = hurst_rs_clean.mean()
        if mean_hurst > 0.55:
            print(f"  Interpretation: PERSISTENT/TRENDING (H > 0.5)")
        elif mean_hurst < 0.45:
            print(f"  Interpretation: ANTI-PERSISTENT/MEAN-REVERTING (H < 0.5)")
        else:
            print(f"  Interpretation: RANDOM WALK-LIKE (H ≈ 0.5)")

    print(f"\\nHurst Exponent (DFA):")
    print(f"  Valid calculations: {len(hurst_dfa_clean)}")
    if len(hurst_dfa_clean) > 0:
        print(f"  Mean: {hurst_dfa_clean.mean():.4f}")
        print(f"  Std:  {hurst_dfa_clean.std():.4f}")
        print(f"  Min:  {hurst_dfa_clean.min():.4f}")
        print(f"  Max:  {hurst_dfa_clean.max():.4f}")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Price series
    axes[0].plot(prices.index, prices.values, "b-", alpha=0.7, linewidth=1)
    axes[0].set_title("Price Series", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    # Hurst exponent (R/S)
    if len(hurst_rs_clean) > 0:
        axes[1].plot(hurst_rs.index, hurst_rs.values, "purple", alpha=0.8, linewidth=1)
        axes[1].axhline(
            y=0.5, color="black", linestyle="-", alpha=0.5, label="Random Walk (H=0.5)"
        )
        axes[1].axhline(
            y=hurst_rs_clean.mean(),
            color="purple",
            linestyle="--",
            alpha=0.7,
            label=f"Mean (H={hurst_rs_clean.mean():.3f})",
        )
        axes[1].fill_between(
            hurst_rs.index,
            0.45,
            0.55,
            alpha=0.2,
            color="gray",
            label="Random Walk Zone",
        )
        axes[1].set_title(
            "Hurst Exponent (R/S Analysis)", fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("Hurst Exponent")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Hurst exponent (DFA)
    if len(hurst_dfa_clean) > 0:
        axes[2].plot(
            hurst_dfa.index, hurst_dfa.values, "orange", alpha=0.8, linewidth=1
        )
        axes[2].axhline(
            y=0.5, color="black", linestyle="-", alpha=0.5, label="Random Walk (H=0.5)"
        )
        axes[2].axhline(
            y=hurst_dfa_clean.mean(),
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Mean (H={hurst_dfa_clean.mean():.3f})",
        )
        axes[2].fill_between(
            hurst_dfa.index,
            0.45,
            0.55,
            alpha=0.2,
            color="gray",
            label="Random Walk Zone",
        )
        axes[2].set_title("Hurst Exponent (DFA)", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("Hurst Exponent")
        axes[2].set_xlabel("Time")
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return hurst_rs, hurst_dfa


def demonstrate_information_bars(advanced_features, data):
    """Demonstrate information-driven bars."""
    print("\\n" + "=" * 60)
    print("INFORMATION-DRIVEN BARS ANALYSIS")
    print("=" * 60)

    # Create different types of information bars
    print("Creating tick bars...")
    tick_bars = advanced_features.create_information_bars(
        data, bar_type="tick", threshold=50
    )

    print("Creating volume bars...")
    volume_bars = advanced_features.create_information_bars(
        data, bar_type="volume", threshold=data["volume"].quantile(0.7)
    )

    print("Creating dollar bars...")
    avg_dollar_volume = (data["close"] * data["volume"]).mean()
    dollar_bars = advanced_features.create_information_bars(
        data, bar_type="dollar", threshold=avg_dollar_volume * 0.8
    )

    # Display statistics
    print(f"\\nOriginal time-based data: {len(data)} bars")
    print(f"Tick bars: {len(tick_bars)} bars")
    print(f"Volume bars: {len(volume_bars)} bars")
    print(f"Dollar bars: {len(dollar_bars)} bars")

    print(f"\\nTick bars statistics:")
    print(f"  Avg ticks per bar: {tick_bars['count'].mean():.1f}")
    print(f"  Avg volume per bar: {tick_bars['volume'].mean():.0f}")

    print(f"\\nVolume bars statistics:")
    print(f"  Avg ticks per bar: {volume_bars['count'].mean():.1f}")
    print(f"  Avg volume per bar: {volume_bars['volume'].mean():.0f}")

    print(f"\\nDollar bars statistics:")
    print(f"  Avg ticks per bar: {dollar_bars['count'].mean():.1f}")
    print(f"  Avg volume per bar: {dollar_bars['volume'].mean():.0f}")
    print(
        f"  Avg dollar volume per bar: {(dollar_bars['close'] * dollar_bars['volume']).mean():.0f}"
    )

    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    # Original time bars
    axes[0].plot(data.index, data["close"], "b-", alpha=0.8, linewidth=1)
    axes[0].set_title(
        f"Original Time Bars (n={len(data)})", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    # Tick bars
    axes[1].plot(
        tick_bars.index,
        tick_bars["close"],
        "r-",
        alpha=0.8,
        linewidth=1,
        marker="o",
        markersize=2,
    )
    axes[1].set_title(
        f'Tick Bars (n={len(tick_bars)}, ~{tick_bars["count"].iloc[0]:.0f} ticks/bar)',
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_ylabel("Price")
    axes[1].grid(True, alpha=0.3)

    # Volume bars
    axes[2].plot(
        volume_bars.index,
        volume_bars["close"],
        "g-",
        alpha=0.8,
        linewidth=1,
        marker="o",
        markersize=2,
    )
    axes[2].set_title(
        f'Volume Bars (n={len(volume_bars)}, ~{volume_bars["volume"].mean():.0f} volume/bar)',
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_ylabel("Price")
    axes[2].grid(True, alpha=0.3)

    # Dollar bars
    axes[3].plot(
        dollar_bars.index,
        dollar_bars["close"],
        "orange",
        alpha=0.8,
        linewidth=1,
        marker="o",
        markersize=2,
    )
    axes[3].set_title(
        f'Dollar Bars (n={len(dollar_bars)}, ~${(dollar_bars["close"] * dollar_bars["volume"]).mean():.0f}/bar)',
        fontsize=12,
        fontweight="bold",
    )
    axes[3].set_ylabel("Price")
    axes[3].set_xlabel("Time")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return tick_bars, volume_bars, dollar_bars


def demonstrate_triple_barrier_method(advanced_features, data):
    """Demonstrate triple barrier method for meta-labeling."""
    print("\\n" + "=" * 60)
    print("TRIPLE BARRIER METHOD ANALYSIS")
    print("=" * 60)

    prices = data["close"]

    # Generate events based on volatility breakouts
    returns = prices.pct_change().dropna()
    vol_window = 20
    vol = returns.rolling(window=vol_window).std()
    vol_threshold = vol.quantile(0.8)

    # Events occur when volatility exceeds threshold
    high_vol_events = vol[vol > vol_threshold].index

    print(f"Detected {len(high_vol_events)} high volatility events")
    print(f"Volatility threshold: {vol_threshold:.6f}")

    if len(high_vol_events) > 10:  # Ensure enough events for analysis
        # Subsample events for demonstration
        events = pd.Series(index=high_vol_events[::3], data=high_vol_events[::3])

        print(f"Using {len(events)} events for triple barrier analysis...")

        # Apply triple barrier method
        labels = advanced_features.triple_barrier_method(
            prices, events, pt_sl=[2.0, 2.0]
        )

        # Display results
        print(f"\\nTriple Barrier Results:")
        print(f"  Total events labeled: {len(labels)}")

        barrier_counts = labels["barrier"].value_counts()
        print(f"  Profit-taking hits (barrier=1): {barrier_counts.get(1, 0)}")
        print(f"  Stop-loss hits (barrier=-1): {barrier_counts.get(-1, 0)}")
        print(f"  Vertical barrier hits (barrier=0): {barrier_counts.get(0, 0)}")

        # Calculate statistics
        positive_returns = labels[labels["ret"] > 0]
        negative_returns = labels[labels["ret"] < 0]

        print(f"\\nReturn Statistics:")
        print(
            f"  Positive returns: {len(positive_returns)} ({len(positive_returns)/len(labels)*100:.1f}%)"
        )
        print(
            f"  Negative returns: {len(negative_returns)} ({len(negative_returns)/len(labels)*100:.1f}%)"
        )
        print(f"  Mean return: {labels['ret'].mean():.6f}")
        print(f"  Std return: {labels['ret'].std():.6f}")

        # Plot results
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))

        # Price series with events
        axes[0].plot(prices.index, prices.values, "b-", alpha=0.7, linewidth=1)
        axes[0].scatter(
            events.index,
            prices.loc[events.index],
            color="red",
            s=30,
            alpha=0.8,
            label="Events",
        )
        axes[0].set_title(
            "Price Series with Volatility Breakout Events",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Volatility with threshold
        axes[1].plot(vol.index, vol.values, "purple", alpha=0.7, linewidth=1)
        axes[1].axhline(
            y=vol_threshold,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Threshold ({vol_threshold:.6f})",
        )
        axes[1].scatter(
            events.index, vol.loc[events.index], color="red", s=30, alpha=0.8
        )
        axes[1].set_title(
            "Rolling Volatility with Events", fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("Volatility")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Barrier outcomes
        colors = {1: "green", -1: "red", 0: "orange"}
        barrier_colors = [colors[b] for b in labels["barrier"]]

        axes[2].scatter(labels.index, labels["ret"], c=barrier_colors, alpha=0.7, s=40)
        axes[2].axhline(y=0, color="black", linestyle="-", alpha=0.5)
        axes[2].set_title("Triple Barrier Outcomes", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("Return")
        axes[2].set_xlabel("Time")

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", label="Profit-taking"),
            Patch(facecolor="red", label="Stop-loss"),
            Patch(facecolor="orange", label="Vertical barrier"),
        ]
        axes[2].legend(handles=legend_elements)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return labels

    else:
        print("Not enough high volatility events for meaningful analysis")
        return None


def demonstrate_fractional_differentiation(advanced_features, data):
    """Demonstrate fractional differentiation."""
    print("\\n" + "=" * 60)
    print("FRACTIONAL DIFFERENTIATION ANALYSIS")
    print("=" * 60)

    prices = data["close"]

    # Apply different orders of fractional differentiation
    d_values = [0.2, 0.4, 0.6, 0.8]
    frac_diff_series = {}

    for d in d_values:
        print(f"Applying fractional differentiation with d = {d}...")
        frac_diff = advanced_features.fractional_differentiation(
            prices, d=d, threshold=0.01
        )
        frac_diff_series[d] = frac_diff

    # Also calculate regular first difference for comparison
    first_diff = prices.diff()

    # Display statistics
    print(f"\\nStationarity Analysis (ADF Test concept):")
    print(f"Original prices:")
    print(f"  Std: {prices.std():.4f}")
    print(f"  Mean: {prices.mean():.2f}")

    for d in d_values:
        series = frac_diff_series[d].dropna()
        if len(series) > 0:
            print(f"\\nFractional diff (d={d}):")
            print(f"  Valid observations: {len(series)}")
            print(f"  Std: {series.std():.6f}")
            print(f"  Mean: {series.mean():.6f}")

    first_diff_clean = first_diff.dropna()
    print(f"\\nFirst difference:")
    print(f"  Std: {first_diff_clean.std():.6f}")
    print(f"  Mean: {first_diff_clean.mean():.6f}")

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Original prices
    axes[0, 0].plot(prices.index, prices.values, "b-", alpha=0.8, linewidth=1)
    axes[0, 0].set_title("Original Price Series", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].grid(True, alpha=0.3)

    # First difference
    axes[0, 1].plot(first_diff.index, first_diff.values, "r-", alpha=0.8, linewidth=1)
    axes[0, 1].set_title("First Difference", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("First Diff")
    axes[0, 1].grid(True, alpha=0.3)

    # Fractional differences
    colors = ["green", "orange", "purple", "brown"]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

    for i, d in enumerate(d_values):
        row, col = positions[i]
        series = frac_diff_series[d]
        axes[row, col].plot(
            series.index, series.values, color=colors[i], alpha=0.8, linewidth=1
        )
        axes[row, col].set_title(
            f"Fractional Diff (d={d})", fontsize=12, fontweight="bold"
        )
        axes[row, col].set_ylabel(f"Frac Diff (d={d})")
        if row == 2:
            axes[row, col].set_xlabel("Time")
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return frac_diff_series


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ADVANCED FEATURE ENGINEERING DEMONSTRATION")
    print(
        "Phase 2 Week 5: Fractal Analysis, Hurst Exponent, Information Bars, Triple Barrier"
    )
    print("Based on Advances in Financial Machine Learning (AFML)")
    print("=" * 80)

    # Initialize advanced features
    advanced_features = AdvancedFeatures()

    # Generate sample data with mixed regime
    print("\\nGenerating sample market data with mixed regime...")
    data = generate_sample_market_data(n_points=1500, regime="mixed")

    print(f"Generated {len(data)} data points from {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"Average volume: {data['volume'].mean():.0f}")

    # 1. Fractal Dimension Analysis
    fd_higuchi, fd_box = demonstrate_fractal_analysis(advanced_features, data)

    # 2. Hurst Exponent Analysis
    hurst_rs, hurst_dfa = demonstrate_hurst_analysis(advanced_features, data)

    # 3. Information-driven Bars
    tick_bars, volume_bars, dollar_bars = demonstrate_information_bars(
        advanced_features, data
    )

    # 4. Triple Barrier Method
    labels = demonstrate_triple_barrier_method(advanced_features, data)

    # 5. Fractional Differentiation
    frac_diff_series = demonstrate_fractional_differentiation(advanced_features, data)

    # 6. Comprehensive Analysis
    print("\\n" + "=" * 60)
    print("COMPREHENSIVE ADVANCED FEATURES ANALYSIS")
    print("=" * 60)

    print("Calculating all advanced features...")
    results = advanced_features.calculate_all_features(data, window=100)

    print(f"\\nAdvanced Features Summary:")
    print(
        f"  Fractal Dimension: {len(results.fractal_dimension.dropna())} valid values"
    )
    print(f"  Hurst Exponent: {len(results.hurst_exponent.dropna())} valid values")
    print(f"  Information Bars: {len(results.information_bars)} bars")
    print(f"  Fractional Diff: {len(results.fractional_diff.dropna())} valid values")

    if results.triple_barrier_labels is not None:
        print(f"  Triple Barrier Labels: {len(results.triple_barrier_labels)} events")
    else:
        print(f"  Triple Barrier Labels: No events generated")

    # Feature correlation analysis
    print("\\nFeature Correlation Analysis:")
    feature_df = pd.DataFrame(
        {
            "price": data["close"],
            "returns": data["close"].pct_change(),
            "volume": data["volume"],
            "fractal_dim": results.fractal_dimension,
            "hurst_exp": results.hurst_exponent,
            "frac_diff": results.fractional_diff,
        }
    ).dropna()

    if len(feature_df) > 0:
        correlations = feature_df.corr()
        print(correlations)

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlations,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title(
            "Advanced Features Correlation Matrix", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    print("\\n" + "=" * 80)
    print("ADVANCED FEATURE ENGINEERING DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("\\nKey Insights:")
    print("1. Fractal dimension measures market complexity and roughness")
    print("2. Hurst exponent indicates persistence vs mean-reversion tendencies")
    print("3. Information-driven bars provide more robust sampling than time bars")
    print("4. Triple barrier method enables sophisticated meta-labeling")
    print("5. Fractional differentiation achieves stationarity while preserving memory")
    print("\\nThese advanced features form the foundation for sophisticated")
    print("machine learning models in quantitative finance!")


if __name__ == "__main__":
    main()
