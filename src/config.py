"""
Configuration settings for the Quant Analytics Tool
Simplified version without pydantic for initial setup
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any


class Settings:
    """Application settings with environment variable support"""

    def __init__(self):
        # Application Info
        self.app_name: str = "Quant Analytics Tool"
        self.app_version: str = "0.1.0"
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"

        # Paths
        self.project_root: Path = Path(__file__).parent.parent
        self.data_dir: Path = self.project_root / "data"
        self.models_dir: Path = self.project_root / "models"
        self.logs_dir: Path = self.project_root / "logs"

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Streamlit Configuration
        self.streamlit_page_title: str = "Quant Analytics Tool"
        self.streamlit_page_icon: str = "📊"
        self.streamlit_layout: str = "wide"


# Global settings instance
settings = Settings()


# Data source configurations
DATA_SOURCE_CONFIG = {
    "yfinance": {
        "rate_limit": 1,  # requests per second
        "max_period": "max",
        "supported_intervals": [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ],
    },
    "alpha_vantage": {
        "rate_limit": 5,  # requests per minute for free tier
        "base_url": "https://www.alphavantage.co/query",
        "supported_intervals": [
            "1min",
            "5min",
            "15min",
            "30min",
            "60min",
            "daily",
            "weekly",
            "monthly",
        ],
    },
    "polygon": {
        "rate_limit": 5,  # requests per minute for free tier
        "base_url": "https://api.polygon.io",
        "supported_intervals": ["1", "5", "15", "30", "60", "day", "week", "month"],
    },
}

# Market data configuration
MARKET_CONFIG = {
    "exchanges": {
        "US": ["NYSE", "NASDAQ", "AMEX"],
        "JP": ["TSE", "JASDAQ"],
        "crypto": ["binance", "coinbase", "kraken"],
    },
    "trading_hours": {
        "US": {"open": "09:30", "close": "16:00", "timezone": "America/New_York"},
        "JP": {"open": "09:00", "close": "15:00", "timezone": "Asia/Tokyo"},
        "crypto": {"open": "00:00", "close": "23:59", "timezone": "UTC"},
    },
    "holidays": {
        "US": [
            "New Year's Day",
            "Martin Luther King Jr. Day",
            "Presidents Day",
            "Good Friday",
            "Memorial Day",
            "Independence Day",
            "Labor Day",
            "Thanksgiving",
            "Christmas",
        ],
        "JP": [
            "New Year's Day",
            "Coming of Age Day",
            "National Foundation Day",
            "Vernal Equinox Day",
            "Showa Day",
            "Constitution Memorial Day",
            "Greenery Day",
            "Children's Day",
            "Marine Day",
            "Mountain Day",
            "Respect for the Aged Day",
            "Autumnal Equinox Day",
            "Health and Sports Day",
            "Culture Day",
            "Labor Thanksgiving Day",
            "Emperor's Birthday",
        ],
    },
}
