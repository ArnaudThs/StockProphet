"""
Data module for StockProphet.

This module handles:
- downloader.py: YFinance data fetching
- features.py: Technical indicators and feature engineering
- cache.py: Pipeline caching (YFinance + RNN predictions)
"""

from .downloader import download_prices, clean_raw
from .features import (
    add_all_technicals,
    add_calendar_macro,
    trim_date_range,
    clean_final_dataset,
    apply_shift_engine,
)
from .cache import (
    load_yfinance_cache,
    save_yfinance_cache,
    load_rnn_cache,
    save_rnn_cache,
    load_pipeline_cache,
    save_pipeline_cache,
    clear_cache,
    get_cache_stats,
)

__all__ = [
    # Downloader functions
    'download_prices',
    'clean_raw',

    # Feature engineering
    'add_all_technicals',
    'add_calendar_macro',
    'trim_date_range',
    'clean_final_dataset',
    'apply_shift_engine',

    # Cache functions
    'load_yfinance_cache',
    'save_yfinance_cache',
    'load_rnn_cache',
    'save_rnn_cache',
    'load_pipeline_cache',
    'save_pipeline_cache',
    'clear_cache',
    'get_cache_stats',
]
