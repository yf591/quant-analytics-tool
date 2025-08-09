"""
Simple Technical Indicators Demo

This script demonstrates the usage of technical indicators module
with manually created sample data.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.features.technical import TechnicalIndicators


def create_sample_data(days=252):
    """Create sample OHLCV data for demonstration"""
    # Set seed for reproducible results
    np.random.seed(42)

    # Create date range
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")

    # Generate realistic price data using random walk
    base_price = 150.0
    returns = np.random.normal(
        0.001, 0.02, days
    )  # Small positive drift, 2% daily volatility

    # Calculate cumulative prices
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Ensure positive prices

    # Create OHLCV data
    high_multiplier = 1 + np.abs(np.random.normal(0, 0.015, days))
    low_multiplier = 1 - np.abs(np.random.normal(0, 0.015, days))

    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p * h for p, h in zip(prices, high_multiplier)],
            "low": [p * l for p, l in zip(prices, low_multiplier)],
            "close": prices,
            "volume": np.random.randint(100000, 1000000, days),
        },
        index=dates,
    )

    return data


def main():
    """Demonstrate technical indicators functionality"""
    print("🚀 Technical Indicators Demo (Sample Data)")
    print("=" * 60)

    # Create sample data
    print("📊 Creating sample financial data...")
    data = create_sample_data(days=252)  # 1 year of trading days

    print(f"✅ Generated {len(data)} data points")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print()

    # Display basic data info
    print("📈 Basic Data Info:")
    print(f"Starting price: ${data['open'].iloc[0]:.2f}")
    print(f"Ending price: ${data['close'].iloc[-1]:.2f}")
    total_return = ((data["close"].iloc[-1] / data["open"].iloc[0]) - 1) * 100
    print(f"Total return: {total_return:.2f}%")
    print(f"Highest price: ${data['high'].max():.2f}")
    print(f"Lowest price: ${data['low'].min():.2f}")
    print(f"Average volume: {data['volume'].mean():,.0f}")
    print()

    # Initialize technical indicators
    ti = TechnicalIndicators()

    # Calculate all indicators
    print("🔧 Calculating Technical Indicators...")
    try:
        results = ti.calculate_all_indicators(data)
        print(f"✅ Successfully calculated {len(results)} indicators")
        print()

        # Display results
        print("📊 Technical Indicators Results:")
        print("-" * 50)

        # Moving Averages
        sma_20 = results["sma_20"].values.dropna()
        sma_50 = results["sma_50"].values.dropna()
        ema_12 = results["ema_12"].values.dropna()
        ema_26 = results["ema_26"].values.dropna()
        current_price = data["close"].iloc[-1]

        print(f"🔄 Moving Averages:")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  SMA(20): ${sma_20.iloc[-1]:.2f}")
        print(f"  SMA(50): ${sma_50.iloc[-1]:.2f}")
        print(f"  EMA(12): ${ema_12.iloc[-1]:.2f}")
        print(f"  EMA(26): ${ema_26.iloc[-1]:.2f}")

        # Moving Average signals
        if current_price > sma_20.iloc[-1]:
            print(f"  📈 Price is ABOVE SMA(20) - Bullish signal")
        else:
            print(f"  📉 Price is BELOW SMA(20) - Bearish signal")

        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            print(f"  📈 SMA(20) is ABOVE SMA(50) - Bullish trend")
        else:
            print(f"  📉 SMA(20) is BELOW SMA(50) - Bearish trend")

        print()

        # RSI Analysis
        rsi = results["rsi"].values.dropna()
        current_rsi = rsi.iloc[-1]

        print(f"📊 RSI Analysis:")
        print(f"  Current RSI: {current_rsi:.2f}")

        if current_rsi > 70:
            print(f"  🔴 OVERBOUGHT (RSI > 70) - Consider selling")
        elif current_rsi < 30:
            print(f"  🟢 OVERSOLD (RSI < 30) - Consider buying")
        else:
            print(f"  🟡 NEUTRAL (30 ≤ RSI ≤ 70) - No clear signal")

        # RSI trend
        rsi_trend = rsi.iloc[-5:].diff().mean()
        if rsi_trend > 0:
            print(f"  📈 RSI is trending UP (momentum gaining)")
        else:
            print(f"  📉 RSI is trending DOWN (momentum weakening)")

        print()

        # MACD Analysis
        macd_data = results["macd"].values.dropna()
        current_macd = macd_data["MACD"].iloc[-1]
        current_signal = macd_data["Signal"].iloc[-1]
        current_histogram = macd_data["Histogram"].iloc[-1]

        print(f"🌊 MACD Analysis:")
        print(f"  MACD Line: {current_macd:.4f}")
        print(f"  Signal Line: {current_signal:.4f}")
        print(f"  Histogram: {current_histogram:.4f}")

        if current_macd > current_signal:
            print(f"  📈 MACD ABOVE Signal - Bullish momentum")
        else:
            print(f"  📉 MACD BELOW Signal - Bearish momentum")

        # MACD crossover detection
        prev_histogram = macd_data["Histogram"].iloc[-2]
        if current_histogram > 0 and prev_histogram <= 0:
            print(f"  🚀 BULLISH CROSSOVER detected!")
        elif current_histogram < 0 and prev_histogram >= 0:
            print(f"  💥 BEARISH CROSSOVER detected!")

        print()

        # Bollinger Bands Analysis
        bb_data = results["bollinger_bands"].values.dropna()
        current_upper = bb_data["Upper"].iloc[-1]
        current_middle = bb_data["Middle"].iloc[-1]
        current_lower = bb_data["Lower"].iloc[-1]

        print(f"📊 Bollinger Bands Analysis:")
        print(f"  Upper Band: ${current_upper:.2f}")
        print(f"  Middle Band (SMA20): ${current_middle:.2f}")
        print(f"  Lower Band: ${current_lower:.2f}")
        print(f"  Current Price: ${current_price:.2f}")

        # Band position calculation
        band_position = (
            (current_price - current_lower) / (current_upper - current_lower) * 100
        )
        print(f"  Band Position: {band_position:.1f}%")

        if current_price > current_upper:
            print(f"  🔴 Price ABOVE upper band - Potentially overbought")
        elif current_price < current_lower:
            print(f"  🟢 Price BELOW lower band - Potentially oversold")
        else:
            print(f"  🟡 Price within normal range")

        # Band squeeze detection
        band_width = (current_upper - current_lower) / current_middle * 100
        print(f"  Band Width: {band_width:.2f}%")
        if band_width < 10:
            print(f"  ⚡ SQUEEZE detected - Breakout expected!")

        print()

        # Volatility Analysis (ATR)
        atr = results["atr"].values.dropna()
        current_atr = atr.iloc[-1]
        atr_percentage = (current_atr / current_price) * 100

        print(f"📏 Volatility Analysis (ATR):")
        print(f"  Current ATR: ${current_atr:.2f}")
        print(f"  ATR as % of price: {atr_percentage:.2f}%")

        # Volatility categorization
        if atr_percentage > 4:
            print(f"  🔥 VERY HIGH volatility - High risk/reward")
        elif atr_percentage > 2.5:
            print(f"  🔥 HIGH volatility - Increased risk")
        elif atr_percentage > 1.5:
            print(f"  🟡 MODERATE volatility - Normal market")
        else:
            print(f"  😴 LOW volatility - Quiet market")

        # ATR trend
        atr_trend = atr.iloc[-10:].mean()
        atr_recent = atr.iloc[-5:].mean()
        if atr_recent > atr_trend * 1.2:
            print(f"  📈 Volatility is INCREASING")
        elif atr_recent < atr_trend * 0.8:
            print(f"  📉 Volatility is DECREASING")

        print()

        # Stochastic Oscillator
        stoch_data = results["stochastic"].values.dropna()
        current_k = stoch_data["%K"].iloc[-1]
        current_d = stoch_data["%D"].iloc[-1]

        print(f"🎯 Stochastic Oscillator:")
        print(f"  %K: {current_k:.2f}")
        print(f"  %D: {current_d:.2f}")

        if current_k > 80:
            print(f"  🔴 OVERBOUGHT territory (>80)")
        elif current_k < 20:
            print(f"  🟢 OVERSOLD territory (<20)")
        else:
            print(f"  🟡 NEUTRAL territory (20-80)")

        # Stochastic crossover
        if current_k > current_d:
            print(f"  📈 %K above %D - Bullish signal")
        else:
            print(f"  📉 %K below %D - Bearish signal")

        print()

        # Williams %R
        williams_r = results["williams_r"].values.dropna()
        current_williams = williams_r.iloc[-1]

        print(f"📉 Williams %R:")
        print(f"  Current Williams %R: {current_williams:.2f}")

        if current_williams > -20:
            print(f"  🔴 OVERBOUGHT (> -20)")
        elif current_williams < -80:
            print(f"  🟢 OVERSOLD (< -80)")
        else:
            print(f"  🟡 NEUTRAL (-80 to -20)")

        print()

        # CCI Analysis
        cci = results["cci"].values.dropna()
        current_cci = cci.iloc[-1]

        print(f"🔄 Commodity Channel Index (CCI):")
        print(f"  Current CCI: {current_cci:.2f}")

        if current_cci > 100:
            print(f"  🔴 OVERBOUGHT (> 100)")
        elif current_cci < -100:
            print(f"  🟢 OVERSOLD (< -100)")
        else:
            print(f"  🟡 NEUTRAL (-100 to 100)")

        print()

        # Momentum Analysis
        momentum = results["momentum"].values.dropna()
        current_momentum = momentum.iloc[-1]

        print(f"🚀 Price Momentum (10-day):")
        print(f"  Momentum: ${current_momentum:.2f}")
        momentum_pct = (current_momentum / current_price) * 100
        print(f"  Momentum %: {momentum_pct:.2f}%")

        if momentum_pct > 5:
            print(f"  🚀 STRONG positive momentum")
        elif momentum_pct > 2:
            print(f"  📈 Positive momentum")
        elif momentum_pct < -5:
            print(f"  💥 STRONG negative momentum")
        elif momentum_pct < -2:
            print(f"  📉 Negative momentum")
        else:
            print(f"  🟡 Weak momentum")

        print()

        # Overall Signal Summary
        print("=" * 60)
        print("📋 TECHNICAL ANALYSIS SUMMARY")
        print("=" * 60)

        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0

        # Analyze each signal
        print("🔍 Signal Analysis:")

        # Moving Average signals
        if current_price > sma_20.iloc[-1]:
            print("  ✅ Price above SMA(20) - BULLISH")
            bullish_signals += 1
            signal_strength += 1
        else:
            print("  ❌ Price below SMA(20) - BEARISH")
            bearish_signals += 1
            signal_strength -= 1

        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            print("  ✅ SMA(20) above SMA(50) - BULLISH trend")
            bullish_signals += 1
            signal_strength += 1
        else:
            print("  ❌ SMA(20) below SMA(50) - BEARISH trend")
            bearish_signals += 1
            signal_strength -= 1

        # RSI signals
        if 30 <= current_rsi <= 70:
            print("  ✅ RSI in neutral zone - HEALTHY")
        elif current_rsi < 30:
            print("  ✅ RSI oversold - BULLISH opportunity")
            bullish_signals += 1
            signal_strength += 2
        else:
            print("  ❌ RSI overbought - BEARISH warning")
            bearish_signals += 1
            signal_strength -= 2

        # MACD signals
        if current_macd > current_signal:
            print("  ✅ MACD above Signal - BULLISH momentum")
            bullish_signals += 1
            signal_strength += 1
        else:
            print("  ❌ MACD below Signal - BEARISH momentum")
            bearish_signals += 1
            signal_strength -= 1

        # Bollinger Bands signals
        if current_price < current_lower:
            print("  ✅ Price below lower BB - BULLISH opportunity")
            bullish_signals += 1
            signal_strength += 2
        elif current_price > current_upper:
            print("  ❌ Price above upper BB - BEARISH warning")
            bearish_signals += 1
            signal_strength -= 2
        else:
            print("  ✅ Price within BB range - NEUTRAL")

        # Stochastic signals
        if current_k < 20:
            print("  ✅ Stochastic oversold - BULLISH opportunity")
            bullish_signals += 1
            signal_strength += 1
        elif current_k > 80:
            print("  ❌ Stochastic overbought - BEARISH warning")
            bearish_signals += 1
            signal_strength -= 1
        else:
            print("  ✅ Stochastic neutral - No extreme reading")

        print()
        print("📊 Signal Count:")
        print(f"  📈 Bullish signals: {bullish_signals}")
        print(f"  📉 Bearish signals: {bearish_signals}")
        print(f"  🎯 Signal strength: {signal_strength}")
        print()

        # Final recommendation
        if signal_strength >= 3:
            recommendation = "🟢 STRONG BUY"
        elif signal_strength >= 1:
            recommendation = "📈 BUY"
        elif signal_strength <= -3:
            recommendation = "🔴 STRONG SELL"
        elif signal_strength <= -1:
            recommendation = "📉 SELL"
        else:
            recommendation = "🟡 HOLD"

        print(f"💡 Overall Recommendation: {recommendation}")

        # Risk assessment
        if atr_percentage > 3:
            risk_level = "🔥 HIGH RISK"
        elif atr_percentage > 2:
            risk_level = "🟡 MODERATE RISK"
        else:
            risk_level = "🟢 LOW RISK"

        print(f"⚠️  Risk Level: {risk_level}")
        print()

        print("📚 Key Insights:")
        print(f"  • Price has {total_return:.1f}% total return in the period")
        print(f"  • Current volatility: {atr_percentage:.1f}% (ATR)")
        print(f"  • RSI momentum: {current_rsi:.1f}")
        print(f"  • Bollinger band position: {band_position:.1f}%")
        print()

        print("⚠️  DISCLAIMER:")
        print("   This analysis is for educational purposes only.")
        print("   Not financial advice. Always do your own research.")
        print("   Past performance does not guarantee future results.")

    except Exception as e:
        print(f"❌ Error calculating indicators: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
