"""
Data downloading and cleaning functions.
Source: External dataprep.py (download_prices, clean_raw)
"""
import numpy as np
import pandas as pd
import yfinance as yf

from ..config import MIN_HISTORY, HORIZON
from .cache import (
    load_yfinance_cache,
    save_yfinance_cache,
    should_invalidate_yfinance_cache
)


def compute_safe_window(start_date: str, end_date: str, min_history: int, horizon: int):
    """
    Compute safe date window that includes extra history for indicators.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    safe_start = start - pd.Timedelta(days=min_history)
    safe_end = end + pd.Timedelta(days=horizon) + pd.Timedelta(days=1)
    return safe_start, safe_end


def download_prices(target: str, support_tickers: list, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data for target and support tickers from yfinance.
    Uses disk cache to avoid re-downloading data.

    Args:
        target: Target ticker symbol (e.g., "AAPL")
        support_tickers: List of supporting ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_cache: Whether to use disk cache (default True)

    Returns:
        DataFrame with columns like AAPL_Open, AAPL_High, etc.
    """
    safe_start, safe_end = compute_safe_window(start_date, end_date, MIN_HISTORY, HORIZON)
    safe_start_str = safe_start.strftime("%Y-%m-%d")
    safe_end_str = safe_end.strftime("%Y-%m-%d")

    tickers = [target] + list(support_tickers)

    # Single ticker download with caching
    if len(tickers) == 1:
        ticker = tickers[0]

        # Check cache first
        if use_cache:
            # Invalidate cache if end_date is recent (< 24 hours old)
            if should_invalidate_yfinance_cache(end_date):
                pass  # Skip cache, download fresh data
            else:
                cached_df = load_yfinance_cache(ticker, safe_start_str, safe_end_str)
                if cached_df is not None:
                    return cached_df

        # Download data
        raw = yf.download(
            tickers=ticker,
            start=safe_start_str,
            end=safe_end_str,
            auto_adjust=False,
            progress=False
        )

        # Single ticker: yfinance might still return MultiIndex in some cases
        # Check if columns are MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            # Flatten MultiIndex: ('Close', 'AAPL') -> 'AAPL_Close'
            raw.columns = [f"{ticker}_{field}" for field, _ in raw.columns]
        else:
            # Simple columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            # Add ticker prefix manually
            raw.columns = [f"{ticker}_{col}" for col in raw.columns]

        # Remove Adj Close columns
        raw = raw[[c for c in raw.columns if "Adj" not in c]]
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index().copy()

        # Save to cache
        if use_cache:
            save_yfinance_cache(raw, ticker, safe_start_str, safe_end_str)

        return raw

    # Multiple tickers (no caching for simplicity - download together)
    else:
        raw = yf.download(
            tickers=tickers,
            start=safe_start_str,
            end=safe_end_str,
            auto_adjust=False,
            progress=False
        )

        # Multiple tickers: columns are like ('Close', 'AAPL') - flatten to 'AAPL_Close'
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [f"{ticker}_{field}" for field, ticker in raw.columns]

        # Remove Adj Close columns
        raw = raw[[c for c in raw.columns if "Adj" not in c]]
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index().copy()

        return raw


def clean_raw(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Clean raw OHLCV data: handle inf/nan, forward fill, drop remaining NaN.

    Args:
        df: Raw DataFrame from download_prices
        tickers: List of tickers to clean

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Identify OHLCV columns
    ohlcv = []
    for t in tickers:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            col = f"{t}_{field}"
            if col in df.columns:
                ohlcv.append(col)

    # Replace inf with NaN
    df[ohlcv] = df[ohlcv].replace([np.inf, -np.inf], np.nan)

    # Forward fill
    df[ohlcv] = df[ohlcv].ffill()

    # Drop rows with remaining NaN
    df = df.dropna(subset=ohlcv, how="any")

    # Ensure numeric types
    for col in ohlcv:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
