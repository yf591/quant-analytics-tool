"""
Technical# Ad# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.collectors import YFinanceCollector
from src.features.technical import TechnicalIndicators to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.collectors import YFinanceCollector
from src.features.technical import TechnicalIndicatorscators Demo

This script demonstrates the usage of technical indicators module
with real financial data from Yahoo Finance.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.collectors import YFinanceCollector
from features.technical import TechnicalIndicators


def main():
    """Demonstrate technical indicators functionality"""
    print("🚀 Technical Indicators Demo")
    print("=" * 50)

    # Initialize data collector
    collector = YFinanceCollector()

    # Fetch sample data (Apple stock)
    symbol = "AAPL"
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # 1 year of data

    print(f"📊 Fetching data for {symbol}")
    print(f"Period: {start_date} to {end_date}")

    try:
        # Fetch data
        data = collector.fetch_data(
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        print(f"✅ Successfully fetched {len(data)} data points")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print()

        # Display basic data info
        print("📈 Basic Data Info:")
        print(f"Opening price: ${data['open'].iloc[0]:.2f}")
        print(f"Closing price: ${data['close'].iloc[-1]:.2f}")
        print(
            f"Price change: {((data['close'].iloc[-1] / data['open'].iloc[0]) - 1) * 100:.2f}%"
        )
        print(f"Highest price: ${data['high'].max():.2f}")
        print(f"Lowest price: ${data['low'].min():.2f}")
        print()

        # Initialize technical indicators
        ti = TechnicalIndicators()

        # Calculate all indicators
        print("🔧 Calculating Technical Indicators...")
        results = ti.calculate_all_indicators(data)

        print(f"✅ Successfully calculated {len(results)} indicators")
        print()

        # Display results
        print("📊 Technical Indicators Results:")
        print("-" * 40)

        # SMA Results
        sma_20 = results["sma_20"].values.dropna()
        sma_50 = results["sma_50"].values.dropna()
        current_price = data["close"].iloc[-1]

        print(f"🔄 Moving Averages:")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  SMA(20): ${sma_20.iloc[-1]:.2f}")
        print(f"  SMA(50): ${sma_50.iloc[-1]:.2f}")

        if current_price > sma_20.iloc[-1]:
            print(f"  📈 Price is ABOVE SMA(20) - Bullish signal")
        else:
            print(f"  📉 Price is BELOW SMA(20) - Bearish signal")

        print()

        # RSI Results
        rsi = results["rsi"].values.dropna()
        current_rsi = rsi.iloc[-1]

        print(f"📈 RSI Analysis:")
        print(f"  Current RSI: {current_rsi:.2f}")

        if current_rsi > 70:
            print(f"  🔴 OVERBOUGHT (RSI > 70)")
        elif current_rsi < 30:
            print(f"  🟢 OVERSOLD (RSI < 30)")
        else:
            print(f"  🟡 NEUTRAL (30 ≤ RSI ≤ 70)")

        print()

        # MACD Results
        macd_data = results["macd"].values.dropna()
        current_macd = macd_data["MACD"].iloc[-1]
        current_signal = macd_data["Signal"].iloc[-1]
        current_histogram = macd_data["Histogram"].iloc[-1]

        print(f"🌊 MACD Analysis:")
        print(f"  MACD Line: {current_macd:.4f}")
        print(f"  Signal Line: {current_signal:.4f}")
        print(f"  Histogram: {current_histogram:.4f}")

        if current_macd > current_signal:
            print(f"  📈 MACD is ABOVE Signal line - Bullish momentum")
        else:
            print(f"  📉 MACD is BELOW Signal line - Bearish momentum")

        print()

        # Bollinger Bands Results
        bb_data = results["bollinger_bands"].values.dropna()
        current_upper = bb_data["Upper"].iloc[-1]
        current_middle = bb_data["Middle"].iloc[-1]
        current_lower = bb_data["Lower"].iloc[-1]

        print(f"📊 Bollinger Bands Analysis:")
        print(f"  Upper Band: ${current_upper:.2f}")
        print(f"  Middle Band: ${current_middle:.2f}")
        print(f"  Lower Band: ${current_lower:.2f}")
        print(f"  Current Price: ${current_price:.2f}")

        if current_price > current_upper:
            print(f"  🔴 Price ABOVE upper band - Potentially overbought")
        elif current_price < current_lower:
            print(f"  🟢 Price BELOW lower band - Potentially oversold")
        else:
            print(f"  🟡 Price within normal range")

        # Calculate band position
        band_position = (
            (current_price - current_lower) / (current_upper - current_lower) * 100
        )
        print(f"  Band Position: {band_position:.1f}%")
        print()

        # ATR Results
        atr = results["atr"].values.dropna()
        current_atr = atr.iloc[-1]
        atr_percentage = (current_atr / current_price) * 100

        print(f"📏 Volatility Analysis (ATR):")
        print(f"  Current ATR: ${current_atr:.2f}")
        print(f"  ATR as % of price: {atr_percentage:.2f}%")

        if atr_percentage > 3:
            print(f"  🔥 HIGH volatility")
        elif atr_percentage < 1:
            print(f"  😴 LOW volatility")
        else:
            print(f"  🟡 MODERATE volatility")

        print()

        # Stochastic Results
        stoch_data = results["stochastic"].values.dropna()
        current_k = stoch_data["%K"].iloc[-1]
        current_d = stoch_data["%D"].iloc[-1]

        print(f"🎯 Stochastic Oscillator:")
        print(f"  %K: {current_k:.2f}")
        print(f"  %D: {current_d:.2f}")

        if current_k > 80:
            print(f"  🔴 OVERBOUGHT territory")
        elif current_k < 20:
            print(f"  🟢 OVERSOLD territory")
        else:
            print(f"  🟡 NEUTRAL territory")

        print()

        # Summary
        print("=" * 50)
        print("📋 SUMMARY")
        print("=" * 50)

        bullish_signals = 0
        bearish_signals = 0

        # Count signals
        if current_price > sma_20.iloc[-1]:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_rsi < 30:
            bullish_signals += 1
        elif current_rsi > 70:
            bearish_signals += 1

        if current_macd > current_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_price < current_lower:
            bullish_signals += 1
        elif current_price > current_upper:
            bearish_signals += 1

        print(f"📈 Bullish signals: {bullish_signals}")
        print(f"📉 Bearish signals: {bearish_signals}")

        if bullish_signals > bearish_signals:
            print(f"🟢 Overall sentiment: BULLISH")
        elif bearish_signals > bullish_signals:
            print(f"🔴 Overall sentiment: BEARISH")
        else:
            print(f"🟡 Overall sentiment: NEUTRAL")

        print()
        print("⚠️  Disclaimer: This is for educational purposes only.")
        print("   Not financial advice. Always do your own research.")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return


if __name__ == "__main__":
    main()
