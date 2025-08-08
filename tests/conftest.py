"""
Test configuration for the quant-analytics-tool.

This module provides pytest configuration and common test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import tempfile
import os


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)  # For reproducible tests

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Generate OHLC from prices
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))

        # Ensure OHLC relationships are valid
        open_price = price * (1 + np.random.normal(0, 0.005))
        close_price = price * (1 + np.random.normal(0, 0.005))

        # Ensure high is the highest and low is the lowest
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        # Generate volume
        volume = np.random.randint(100000, 1000000)

        data.append(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close_price,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def invalid_ohlcv_data():
    """Create invalid OHLCV data for testing validation."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")

    data = {
        "Open": [
            100,
            101,
            -5,
            103,
            104,
            105,
            106,
            np.inf,
            108,
            109,
        ],  # Negative and infinite values
        "High": [
            95,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ],  # High < Open (invalid)
        "Low": [105, 99, 98, 97, 96, 95, 94, 93, 92, 91],  # Low > Open (invalid)
        "Close": [102, 100, 99, 101, 103, 104, 105, 106, 107, 108],
        "Volume": [
            1000,
            2000,
            -500,
            4000,
            5000,
            0,
            7000,
            8000,
            9000,
            10000,
        ],  # Negative volume
    }

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def temp_database():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_symbols():
    """Sample stock symbols for testing."""
    return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]


class TestDataHelper:
    """Helper class for generating test data."""

    @staticmethod
    def create_price_series(
        start_date: str = "2023-01-01",
        periods: int = 100,
        base_price: float = 100.0,
        volatility: float = 0.02,
    ) -> pd.Series:
        """Create a realistic price series."""
        dates = pd.date_range(start=start_date, periods=periods, freq="D")
        np.random.seed(42)

        returns = np.random.normal(0, volatility, periods)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        return pd.Series(prices, index=dates)

    @staticmethod
    def create_ohlcv_from_prices(prices: pd.Series) -> pd.DataFrame:
        """Create OHLCV data from a price series."""
        data = []

        for i, (date, price) in enumerate(prices.items()):
            # Add intraday volatility
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))

            if i == 0:
                open_price = price
                close_price = price
            else:
                open_price = prices.iloc[i - 1] * (1 + np.random.normal(0, 0.005))
                close_price = price

            # Ensure OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            volume = np.random.randint(100000, 1000000)

            data.append(
                {
                    "Open": open_price,
                    "High": high,
                    "Low": low,
                    "Close": close_price,
                    "Volume": volume,
                }
            )

        return pd.DataFrame(data, index=prices.index)

    @staticmethod
    def introduce_data_issues(df: pd.DataFrame, issue_type: str) -> pd.DataFrame:
        """Introduce specific data quality issues for testing."""
        df_copy = df.copy()

        if issue_type == "missing_values":
            # Introduce random missing values
            mask = np.random.random(df_copy.shape) < 0.05  # 5% missing
            df_copy = df_copy.mask(mask)

        elif issue_type == "invalid_ohlc":
            # Make some high prices lower than low prices
            invalid_rows = np.random.choice(len(df_copy), size=3, replace=False)
            for row in invalid_rows:
                df_copy.iloc[row, df_copy.columns.get_loc("High")] = (
                    df_copy.iloc[row]["Low"] * 0.9
                )

        elif issue_type == "extreme_values":
            # Introduce extreme price movements
            extreme_rows = np.random.choice(len(df_copy), size=2, replace=False)
            for row in extreme_rows:
                df_copy.iloc[
                    row, df_copy.columns.get_loc("Close")
                ] *= 10  # 1000% increase

        elif issue_type == "negative_values":
            # Introduce negative prices
            negative_rows = np.random.choice(len(df_copy), size=2, replace=False)
            for row in negative_rows:
                df_copy.iloc[row, df_copy.columns.get_loc("Low")] *= -1

        return df_copy
