#!/usr/bin/env python3
"""
Demo Phase 5 Week 14: Backtest UI Integration Testing

This demo tests the comprehensive backtesting UI integration functionality
including data preparation, strategy execution, and result processing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "streamlit_app"))

from src.backtesting import BacktestEngine, BuyAndHoldStrategy, MomentumStrategy
from streamlit_app.utils.backtest_utils import (
    BacktestDataPreparer,
    StrategyBuilder,
    BacktestResultProcessor,
)


def create_sample_ohlcv_data(symbol: str = "AAPL", days: int = 252) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""

    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")

    # Starting price
    initial_price = 150.0

    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    prices = [initial_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)

    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        # Add some intraday volatility
        volatility = 0.01
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))

        # Ensure OHLC consistency
        if i == 0:
            open_price = price
        else:
            open_price = prices[i - 1] * (1 + np.random.uniform(-0.005, 0.005))

        close_price = price

        # Adjust high and low to ensure consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        # Generate volume
        volume = int(np.random.uniform(1000000, 5000000))

        data.append(
            {
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(close_price, 2),
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data, index=dates)
    return df


def create_sample_metadata_structure(ohlcv_data: pd.DataFrame) -> dict:
    """Create sample metadata structure like feature cache"""

    return {
        "original_data": ohlcv_data,
        "features_dict": {
            "sma_20": ohlcv_data["Close"].rolling(20).mean(),
            "rsi": np.random.uniform(20, 80, len(ohlcv_data)),  # Mock RSI
        },
        "type": "technical_indicators",
        "config": {"symbol": "AAPL", "timeframe": "1D", "indicators": ["SMA", "RSI"]},
        "calculated_at": datetime.now(),
    }


def test_data_preparation():
    """Test data preparation functionality"""

    print("🧪 Testing Data Preparation...")

    # Create sample data
    sample_ohlcv = create_sample_ohlcv_data("AAPL", 100)
    sample_metadata = create_sample_metadata_structure(sample_ohlcv)

    # Initialize data preparer
    data_preparer = BacktestDataPreparer()

    # Test 1: Direct DataFrame processing
    print("\n📊 Test 1: Direct DataFrame Processing")
    try:
        result1 = data_preparer.prepare_feature_data(sample_ohlcv)
        print(f"✅ Direct DataFrame: {result1.shape}, Columns: {list(result1.columns)}")
    except Exception as e:
        print(f"❌ Direct DataFrame failed: {e}")
        return False

    # Test 2: Metadata structure processing
    print("\n📊 Test 2: Metadata Structure Processing")
    try:
        result2 = data_preparer.prepare_feature_data(sample_metadata)
        print(
            f"✅ Metadata processing: {result2.shape}, Columns: {list(result2.columns)}"
        )
    except Exception as e:
        print(f"❌ Metadata processing failed: {e}")
        return False

    # Test 3: Invalid data handling
    print("\n📊 Test 3: Invalid Data Handling")
    try:
        invalid_df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}  # No price data
        )
        result3 = data_preparer.prepare_feature_data(invalid_df)
        print("❌ Should have failed with invalid data")
        return False
    except ValueError as e:
        print(f"✅ Correctly handled invalid data: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


def test_strategy_building():
    """Test strategy building functionality"""

    print("\n🏗️ Testing Strategy Building...")

    # Initialize strategy builder
    strategy_builder = StrategyBuilder()

    # Test Buy and Hold strategy
    print("\n📈 Test: Buy and Hold Strategy")
    try:
        buy_hold_config = {"strategy_type": "Buy & Hold", "parameters": {}}
        strategy1 = strategy_builder.build_strategy(buy_hold_config, ["AAPL"])
        print(f"✅ Buy and Hold strategy created: {type(strategy1).__name__}")
    except Exception as e:
        print(f"❌ Buy and Hold strategy failed: {e}")
        return False

    # Test Momentum strategy
    print("\n📈 Test: Momentum Strategy")
    try:
        momentum_config = {
            "strategy_type": "Momentum",
            "parameters": {"short_window": 10, "long_window": 20},
        }
        strategy2 = strategy_builder.build_strategy(momentum_config, ["AAPL"])
        print(f"✅ Momentum strategy created: {type(strategy2).__name__}")
    except Exception as e:
        print(f"❌ Momentum strategy failed: {e}")
        return False

    return True


def test_end_to_end_backtest():
    """Test complete end-to-end backtest process"""

    print("\n🔄 Testing End-to-End Backtest...")

    # Create sample data
    sample_data = create_sample_ohlcv_data("AAPL", 100)

    # Initialize components
    data_preparer = BacktestDataPreparer()
    strategy_builder = StrategyBuilder()
    result_processor = BacktestResultProcessor()

    try:
        # Step 1: Prepare data
        print("📊 Step 1: Preparing data...")
        prepared_data = data_preparer.prepare_feature_data(sample_data)
        print(f"✅ Data prepared: {prepared_data.shape}")

        # Step 2: Initialize engine
        print("⚙️ Step 2: Initializing backtest engine...")
        engine = BacktestEngine(
            initial_capital=100000.0, commission_rate=0.001, slippage_rate=0.0005
        )
        engine.add_data("AAPL", prepared_data)
        print("✅ Engine initialized")

        # Step 3: Build strategy
        print("🎯 Step 3: Building strategy...")
        strategy_config = {"strategy_type": "Buy & Hold", "parameters": {}}
        strategy = strategy_builder.build_strategy(strategy_config, ["AAPL"])
        engine.set_strategy(strategy)
        print("✅ Strategy set")

        # Step 4: Run backtest
        print("🚀 Step 4: Running backtest...")

        # Set date range
        start_date = prepared_data.index[10]  # Skip first few days
        end_date = prepared_data.index[-1]

        results = engine.run_backtest(start_date=start_date, end_date=end_date)
        print("✅ Backtest completed")

        # Step 5: Process results
        print("📈 Step 5: Processing results...")
        backtest_config = {
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
        }

        processed_results = result_processor.process_results(engine, backtest_config)
        print("✅ Results processed")

        # Display summary
        print("\n📊 Backtest Summary:")
        summary = processed_results.get("summary", {})
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_date_conversion():
    """Test date conversion functionality"""

    print("\n📅 Testing Date Conversion...")

    from datetime import date
    import pandas as pd

    # Test different date formats
    test_dates = [
        date(2024, 1, 1),  # datetime.date
        datetime(2024, 1, 1),  # datetime.datetime
        pd.Timestamp("2024-01-01"),  # pandas.Timestamp
        "2024-01-01",  # string
    ]

    for i, test_date in enumerate(test_dates):
        try:
            # Simulate the conversion logic from the backtesting page
            converted_date = None

            if isinstance(test_date, date) and not isinstance(test_date, datetime):
                converted_date = pd.Timestamp.combine(test_date, datetime.min.time())
            elif hasattr(test_date, "date") and not isinstance(test_date, pd.Timestamp):
                converted_date = pd.Timestamp(test_date)
            elif isinstance(test_date, str):
                converted_date = pd.to_datetime(test_date)
            else:
                converted_date = pd.Timestamp(test_date)

            print(
                f"✅ Date conversion {i+1}: {type(test_date).__name__} -> {type(converted_date).__name__}"
            )

        except Exception as e:
            print(f"❌ Date conversion {i+1} failed: {e}")
            return False

    return True


def main():
    """Run comprehensive backtesting integration tests"""

    print("🚀 Demo Phase 5 Week 14: Backtest UI Integration Testing")
    print("=" * 60)

    tests = [
        ("Data Preparation", test_data_preparation),
        ("Strategy Building", test_strategy_building),
        ("Date Conversion", test_date_conversion),
        ("End-to-End Backtest", test_end_to_end_backtest),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Tests...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)

    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All tests passed! Backtest UI integration is ready.")
        return True
    else:
        print(f"\n⚠️ {len(tests) - passed} tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
