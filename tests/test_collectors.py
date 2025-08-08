"""
Unit tests for data collectors module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.collectors import YFinanceCollector, DataRequest
from data.validators import ValidationLevel


class TestYFinanceCollector:
    """Test cases for YFinanceCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = YFinanceCollector()
        self.sample_request = DataRequest(
            symbol="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            interval="1d",
        )

    def test_initialization(self):
        """Test collector initialization."""
        assert self.collector.source_name == "yahoo"
        assert self.collector.rate_limit_delay == 1.0
        assert self.collector.validation_level == ValidationLevel.STANDARD

    def test_data_request_validation(self):
        """Test data request validation."""
        # Valid request
        assert self.collector._validate_request(self.sample_request) is True

        # Invalid symbol
        invalid_request = DataRequest(
            symbol="", start_date=datetime(2023, 1, 1), end_date=datetime(2023, 12, 31)
        )
        assert self.collector._validate_request(invalid_request) is False

        # Invalid date range
        invalid_date_request = DataRequest(
            symbol="AAPL",
            start_date=datetime(2023, 12, 31),
            end_date=datetime(2023, 1, 1),
        )
        assert self.collector._validate_request(invalid_date_request) is False

    @patch("data.collectors.yf.download")
    def test_successful_data_collection(self, mock_download):
        """Test successful data collection."""
        # Mock yfinance response
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [152.0, 153.0, 154.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [151.0, 152.0, 153.0],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        )

        mock_download.return_value = mock_data

        # Test data collection
        result = self.collector.collect_data(self.sample_request)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(
            col in result.columns for col in ["Open", "High", "Low", "Close", "Volume"]
        )

        # Verify yfinance was called correctly
        mock_download.assert_called_once()

    @patch("data.collectors.yf.download")
    def test_failed_data_collection(self, mock_download):
        """Test failed data collection."""
        # Mock yfinance exception
        mock_download.side_effect = Exception("Network error")

        # Test data collection
        result = self.collector.collect_data(self.sample_request)

        assert result is None

    @patch("data.collectors.yf.download")
    def test_empty_data_response(self, mock_download):
        """Test handling of empty data response."""
        # Mock empty DataFrame response
        mock_download.return_value = pd.DataFrame()

        # Test data collection
        result = self.collector.collect_data(self.sample_request)

        assert result is None

    @patch("data.collectors.yf.download")
    def test_data_normalization(self, mock_download):
        """Test data normalization features."""
        # Mock yfinance response with various column names
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [152.0, 153.0, 154.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [151.0, 152.0, 153.0],
                "Volume": [1000000, 1100000, 1200000],
                "Adj Close": [150.5, 151.5, 152.5],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        )

        mock_download.return_value = mock_data

        # Test data collection
        result = self.collector.collect_data(self.sample_request)

        assert result is not None

        # Check for derived features
        expected_derived = ["Returns", "Log_Returns", "Typical_Price", "True_Range"]
        for feature in expected_derived:
            assert feature in result.columns

        # Verify calculations
        assert not result["Returns"].iloc[1:].isna().all()  # Should have returns data
        assert not result["Typical_Price"].isna().all()  # Should have typical price

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        start_time = datetime.now()

        # Call rate limiting method
        self.collector._apply_rate_limit()

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        # Should take at least the rate limit delay
        assert elapsed >= self.collector.rate_limit_delay

    @patch("data.collectors.yf.download")
    def test_different_intervals(self, mock_download):
        """Test data collection with different intervals."""
        # Mock data for different intervals
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )

        mock_download.return_value = mock_data

        # Test different intervals
        intervals = ["1d", "1wk", "1mo"]

        for interval in intervals:
            request = DataRequest(
                symbol="AAPL",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                interval=interval,
            )

            result = self.collector.collect_data(request)
            assert result is not None

    @patch("data.collectors.yf.download")
    def test_data_validation_integration(self, mock_download):
        """Test integration with data validation."""
        # Mock data with some issues
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0, -5.0],  # Negative price
                "High": [152.0, 153.0, 154.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [151.0, 152.0, 153.0],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        )

        mock_download.return_value = mock_data

        # Create collector with strict validation
        strict_collector = YFinanceCollector(validation_level=ValidationLevel.STRICT)

        # Test data collection (should handle validation issues)
        result = strict_collector.collect_data(self.sample_request)

        # Data should still be returned even with validation warnings
        assert result is not None

    def test_get_supported_intervals(self):
        """Test getting supported intervals."""
        intervals = self.collector.get_supported_intervals()

        assert isinstance(intervals, list)
        assert len(intervals) > 0
        assert "1d" in intervals
        assert "1h" in intervals

    def test_get_source_info(self):
        """Test getting source information."""
        info = self.collector.get_source_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert info["name"] == "yahoo"


class TestDataRequest:
    """Test cases for DataRequest dataclass."""

    def test_data_request_creation(self):
        """Test DataRequest creation and defaults."""
        request = DataRequest(
            symbol="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        assert request.symbol == "AAPL"
        assert request.interval == "1d"  # Default interval
        assert request.start_date == datetime(2023, 1, 1)
        assert request.end_date == datetime(2023, 12, 31)

    def test_data_request_with_custom_interval(self):
        """Test DataRequest with custom interval."""
        request = DataRequest(
            symbol="GOOGL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            interval="1h",
        )

        assert request.interval == "1h"

    def test_data_request_validation_properties(self):
        """Test DataRequest validation helper properties."""
        request = DataRequest(
            symbol="MSFT",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        # Test date range calculation
        date_range = request.end_date - request.start_date
        assert date_range.days > 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__])
