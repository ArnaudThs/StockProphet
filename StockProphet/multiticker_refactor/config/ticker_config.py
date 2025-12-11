"""
Ticker selection and date range configuration.
"""
import os

# =============================================================================
# TARGET CONFIGURATION
# =============================================================================
TARGET_TICKER = os.getenv("TARGET_TICKER", "AAPL")
SUPPORT_TICKERS = []  # Empty for now, can add supporting tickers later

# Multi-ticker configuration (for dashboard demo)
_tickers_str = os.getenv("TICKERS", "AAPL,MSFT,GOOGL")
TICKERS = [t.strip() for t in _tickers_str.split(",")]
MAX_TICKERS = 3  # Hard limit for demo performance

# =============================================================================
# DATE RANGES
# =============================================================================
START_DATE = os.getenv("START_DATE", "2020-01-01")
END_DATE = os.getenv("END_DATE", "2025-06-30")

# Sentiment configuration
INCLUDE_SENTIMENT = os.getenv("INCLUDE_SENTIMENT", "False").lower() == "true"
SENTIMENT_START_DATE = os.getenv("SENTIMENT_START_DATE", "2024-01-01")
SENTIMENT_END_DATE = os.getenv("SENTIMENT_END_DATE", "2025-06-30")
