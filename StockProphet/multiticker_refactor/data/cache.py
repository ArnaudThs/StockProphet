"""
Data caching utilities for yfinance OHLCV and RNN predictions.

Implements disk-based caching to speed up development iteration:
- YFinance cache: Skip re-downloading stock data
- RNN cache: Skip re-training RNN models (2-5 min per ticker)
"""
import os
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from ..config import DATA_DIR


# =============================================================================
# CACHE DIRECTORIES
# =============================================================================

YFINANCE_CACHE_DIR = DATA_DIR / "yfinance_cache"
RNN_CACHE_DIR = DATA_DIR / "rnn_cache"

# Create cache directories
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RNN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# YFINANCE CACHE
# =============================================================================

def get_yfinance_cache_key(ticker: str, start_date: str, end_date: str) -> str:
    """
    Generate cache key for yfinance data.

    Args:
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Cache key string
    """
    return f"{ticker}_{start_date}_{end_date}"


def get_yfinance_cache_path(cache_key: str) -> Path:
    """Get file path for yfinance cache."""
    return YFINANCE_CACHE_DIR / f"{cache_key}.parquet"


def load_yfinance_cache(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Load cached yfinance data if available.

    Args:
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Cached DataFrame or None if not found
    """
    cache_key = get_yfinance_cache_key(ticker, start_date, end_date)
    cache_path = get_yfinance_cache_path(cache_key)

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"   Warning: Failed to load cache for {ticker}: {e}")
            return None

    return None


def save_yfinance_cache(df: pd.DataFrame, ticker: str, start_date: str, end_date: str):
    """
    Save yfinance data to cache.

    Args:
        df: DataFrame to cache
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    cache_key = get_yfinance_cache_key(ticker, start_date, end_date)
    cache_path = get_yfinance_cache_path(cache_key)

    try:
        df.to_parquet(cache_path)
    except Exception as e:
        print(f"   Warning: Failed to save cache for {ticker}: {e}")


def should_invalidate_yfinance_cache(end_date: str, max_age_hours: int = 24) -> bool:
    """
    Check if yfinance cache should be invalidated based on end date recency.

    Args:
        end_date: End date (YYYY-MM-DD)
        max_age_hours: Maximum age in hours before re-downloading (default 24)

    Returns:
        True if cache should be invalidated (end_date is recent)
    """
    from datetime import datetime, timedelta

    end_dt = pd.to_datetime(end_date)
    now = datetime.now()

    # If end_date is within max_age_hours of now, invalidate cache
    age = now - end_dt
    return age < timedelta(hours=max_age_hours)


# =============================================================================
# RNN CACHE
# =============================================================================

def get_rnn_cache_key(
    ticker: str,
    start_date: str,
    end_date: str,
    window_size: int,
    epochs: int,
    probabilistic: bool,
    data_hash: str
) -> str:
    """
    Generate cache key for RNN predictions.

    Args:
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        window_size: LSTM window size
        epochs: Number of training epochs
        probabilistic: Whether using probabilistic LSTM
        data_hash: Hash of input data (to detect data changes)

    Returns:
        Cache key string
    """
    rnn_type = "prob" if probabilistic else "simple"
    # Truncate hash to 8 chars for readability
    short_hash = data_hash[:8]
    return f"rnn_{ticker}_{start_date}_{end_date}_w{window_size}_e{epochs}_{rnn_type}_{short_hash}"


def get_rnn_cache_path(cache_key: str) -> Path:
    """Get file path for RNN cache."""
    return RNN_CACHE_DIR / f"{cache_key}.npz"


def get_rnn_metadata_path(cache_key: str) -> Path:
    """Get file path for RNN cache metadata."""
    return RNN_CACHE_DIR / f"{cache_key}_meta.json"


def compute_data_hash(df: pd.DataFrame, ticker: str, verbose: bool = True) -> str:
    """
    Compute hash of ticker's data to detect changes.

    Args:
        df: DataFrame with ticker data
        ticker: Ticker symbol
        verbose: Print progress message

    Returns:
        MD5 hash string
    """
    if verbose:
        n_features = len([col for col in df.columns if col.startswith(f"{ticker}_")])
        print(f"   Computing data hash for cache validation ({len(df)} days × {n_features} features)...")

    # Get all columns for this ticker
    ticker_cols = [col for col in df.columns if col.startswith(f"{ticker}_")]

    # Hash the data
    data_bytes = df[ticker_cols].to_numpy().tobytes()
    hash_result = hashlib.md5(data_bytes).hexdigest()

    if verbose:
        print(f"   Hash computed: {hash_result[:8]}...")

    return hash_result


def load_rnn_cache(
    ticker: str,
    start_date: str,
    end_date: str,
    window_size: int,
    epochs: int,
    probabilistic: bool,
    data_hash: str
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load cached RNN predictions if available.

    Args:
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        window_size: LSTM window size
        epochs: Number of training epochs
        probabilistic: Whether using probabilistic LSTM
        data_hash: Hash of input data

    Returns:
        Dict with feature arrays or None if not found
    """
    cache_key = get_rnn_cache_key(
        ticker, start_date, end_date, window_size, epochs, probabilistic, data_hash
    )
    cache_path = get_rnn_cache_path(cache_key)
    meta_path = get_rnn_metadata_path(cache_key)

    if not cache_path.exists() or not meta_path.exists():
        return None

    try:
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Verify data hash matches (data hasn't changed)
        if metadata.get('data_hash') != data_hash:
            print(f"   Cache invalidated for {ticker} (data changed)")
            return None

        # Load predictions
        data = np.load(cache_path)

        if probabilistic:
            # Return feature_dict for probabilistic RNN
            feature_dict = {key: data[key] for key in data.files}
            return {'feature_dict': feature_dict}
        else:
            # Return predictions for simple RNN
            return {'predictions': data['predictions']}

    except Exception as e:
        print(f"   Warning: Failed to load RNN cache for {ticker}: {e}")
        return None


def save_rnn_cache(
    result: Dict[str, Any],
    ticker: str,
    start_date: str,
    end_date: str,
    window_size: int,
    epochs: int,
    probabilistic: bool,
    data_hash: str
):
    """
    Save RNN predictions to cache.

    Args:
        result: Dict with 'feature_dict' (probabilistic) or 'predictions' (simple)
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        window_size: LSTM window size
        epochs: Number of training epochs
        probabilistic: Whether using probabilistic LSTM
        data_hash: Hash of input data
    """
    cache_key = get_rnn_cache_key(
        ticker, start_date, end_date, window_size, epochs, probabilistic, data_hash
    )
    cache_path = get_rnn_cache_path(cache_key)
    meta_path = get_rnn_metadata_path(cache_key)

    try:
        # Save predictions
        if probabilistic:
            # Save all features from feature_dict
            np.savez_compressed(cache_path, **result['feature_dict'])
        else:
            # Save simple predictions
            np.savez_compressed(cache_path, predictions=result['predictions'])

        # Save metadata
        from datetime import datetime
        metadata = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'window_size': window_size,
            'epochs': epochs,
            'probabilistic': probabilistic,
            'data_hash': data_hash,
            'cached_at': datetime.now().isoformat()
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"   Warning: Failed to save RNN cache for {ticker}: {e}")


def clear_cache(cache_type: str = "all"):
    """
    Clear cached data.

    Args:
        cache_type: Type of cache to clear ("yfinance", "rnn", "pipeline", or "all")
    """
    if cache_type in ["yfinance", "all"]:
        for file in YFINANCE_CACHE_DIR.glob("*.parquet"):
            file.unlink()
        print(f"✅ Cleared yfinance cache ({YFINANCE_CACHE_DIR})")

    if cache_type in ["rnn", "all"]:
        for file in RNN_CACHE_DIR.glob("*"):
            file.unlink()
        print(f"✅ Cleared RNN cache ({RNN_CACHE_DIR})")

    if cache_type in ["pipeline", "all"]:
        for file in PIPELINE_CACHE_DIR.glob("*"):
            file.unlink()
        print(f"✅ Cleared pipeline cache ({PIPELINE_CACHE_DIR})")


# =============================================================================
# PIPELINE DATAFRAME CACHE (Complete Dataset)
# =============================================================================

PIPELINE_CACHE_DIR = DATA_DIR / "pipeline_cache"
PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_pipeline_cache_key(
    tickers: list,
    start_date: str,
    end_date: str,
    include_rnn: bool,
    include_sentiment: bool,
    probabilistic_rnn: bool
) -> str:
    """
    Generate cache key for complete pipeline DataFrame.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        include_rnn: Whether RNN features are included
        include_sentiment: Whether sentiment features are included
        probabilistic_rnn: Whether using probabilistic LSTM

    Returns:
        Cache key string
    """
    ticker_str = "_".join(sorted(tickers))
    rnn_str = "prob" if probabilistic_rnn else "simple" if include_rnn else "nornn"
    sent_str = "sent" if include_sentiment else "nosent"

    return f"pipeline_{ticker_str}_{start_date}_{end_date}_{rnn_str}_{sent_str}"


def get_pipeline_cache_path(cache_key: str) -> Path:
    """Get file path for pipeline cache."""
    return PIPELINE_CACHE_DIR / f"{cache_key}.parquet"


def get_pipeline_metadata_path(cache_key: str) -> Path:
    """Get file path for pipeline cache metadata."""
    return PIPELINE_CACHE_DIR / f"{cache_key}_meta.json"


def load_pipeline_cache(
    tickers: list,
    start_date: str,
    end_date: str,
    include_rnn: bool,
    include_sentiment: bool,
    probabilistic_rnn: bool
) -> Optional[tuple]:
    """
    Load cached pipeline DataFrame if available.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        include_rnn: Whether RNN features are included
        include_sentiment: Whether sentiment features are included
        probabilistic_rnn: Whether using probabilistic LSTM

    Returns:
        Tuple of (df, metadata) or None if not found
    """
    cache_key = get_pipeline_cache_key(
        tickers, start_date, end_date, include_rnn, include_sentiment, probabilistic_rnn
    )
    cache_path = get_pipeline_cache_path(cache_key)
    meta_path = get_pipeline_metadata_path(cache_key)

    if not cache_path.exists() or not meta_path.exists():
        return None

    try:
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Load DataFrame
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)

        print(f"✅ Loaded pipeline cache: {cache_key}")
        print(f"   Cached at: {metadata.get('cached_at', 'unknown')}")
        print(f"   Shape: {df.shape}")

        return df, metadata

    except Exception as e:
        print(f"⚠️  Warning: Failed to load pipeline cache: {e}")
        return None


def save_pipeline_cache(
    df: pd.DataFrame,
    metadata: dict,
    tickers: list,
    start_date: str,
    end_date: str,
    include_rnn: bool,
    include_sentiment: bool,
    probabilistic_rnn: bool
):
    """
    Save complete pipeline DataFrame to cache.

    Args:
        df: Complete DataFrame with all features
        metadata: Metadata dict (scalers, rnn_models, etc.)
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        include_rnn: Whether RNN features are included
        include_sentiment: Whether sentiment features are included
        probabilistic_rnn: Whether using probabilistic LSTM
    """
    cache_key = get_pipeline_cache_key(
        tickers, start_date, end_date, include_rnn, include_sentiment, probabilistic_rnn
    )
    cache_path = get_pipeline_cache_path(cache_key)
    meta_path = get_pipeline_metadata_path(cache_key)

    try:
        # Save DataFrame (parquet is fast and compressed)
        df.to_parquet(cache_path)

        # Prepare metadata for serialization (remove non-serializable items)
        save_metadata = {
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'include_rnn': include_rnn,
            'include_sentiment': include_sentiment,
            'probabilistic_rnn': probabilistic_rnn,
            'feature_cols': metadata.get('feature_cols', []),
            'train_end_idx': metadata.get('train_end_idx'),
            'val_end_idx': metadata.get('val_end_idx'),
            'shape': list(df.shape),
            'cached_at': pd.Timestamp.now().isoformat(),
            'cache_key': cache_key
        }

        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump(save_metadata, f, indent=2)

        print(f"✅ Saved pipeline cache: {cache_key}")
        print(f"   Size: {cache_path.stat().st_size / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"⚠️  Warning: Failed to save pipeline cache: {e}")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about cached data.

    Returns:
        Dict with cache statistics
    """
    yfinance_files = list(YFINANCE_CACHE_DIR.glob("*.parquet"))
    rnn_files = list(RNN_CACHE_DIR.glob("*.npz"))
    pipeline_files = list(PIPELINE_CACHE_DIR.glob("*.parquet"))

    yfinance_size = sum(f.stat().st_size for f in yfinance_files) / (1024 * 1024)  # MB
    rnn_size = sum(f.stat().st_size for f in rnn_files) / (1024 * 1024)  # MB
    pipeline_size = sum(f.stat().st_size for f in pipeline_files) / (1024 * 1024)  # MB

    return {
        'yfinance_cache': {
            'count': len(yfinance_files),
            'size_mb': round(yfinance_size, 2)
        },
        'rnn_cache': {
            'count': len(rnn_files),
            'size_mb': round(rnn_size, 2)
        },
        'pipeline_cache': {
            'count': len(pipeline_files),
            'size_mb': round(pipeline_size, 2)
        },
        'total_size_mb': round(yfinance_size + rnn_size + pipeline_size, 2)
    }
