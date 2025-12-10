"""
Data downloading and cleaning functions.
Source: External dataprep.py (download_prices, clean_raw)
"""
import numpy as np
import pandas as pd
import yfinance as yf

from project_refactored.config import MIN_HISTORY, HORIZON


def compute_safe_window(start_date: str, end_date: str, min_history: int, horizon: int):
    """
    Compute safe date window that includes extra history for indicators.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    safe_start = start - pd.Timedelta(days=min_history)
    safe_end = end + pd.Timedelta(days=horizon) + pd.Timedelta(days=1)
    return safe_start, safe_end


def download_prices(target: str, support_tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download OHLCV data for target and support tickers from yfinance.

    Args:
        target: Target ticker symbol (e.g., "AAPL")
        support_tickers: List of supporting ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns like AAPL_Open, AAPL_High, etc.
    """
    safe_start, safe_end = compute_safe_window(start_date, end_date, MIN_HISTORY, HORIZON)

    tickers = [target] + list(support_tickers)

    # Download data
    raw = yf.download(
        tickers=tickers,
        start=safe_start.strftime("%Y-%m-%d"),
        end=safe_end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    # Handle MultiIndex columns (yfinance now always returns MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        # Columns are like ('Close', 'AAPL') - flatten to 'AAPL_Close'
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
