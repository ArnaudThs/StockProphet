"""
Unified Data Pipeline (Single-Ticker and Multi-Ticker)

Builds feature dataset for single or multiple tickers with:
- Inner join data alignment (20% loss threshold, for multi-ticker)
- Per-ticker normalization (StandardScaler fit on train only)
- Parallel RNN training (one model per ticker)
- Feature-type grouping for cross-ticker learning

Usage:
    # Multi-ticker (main_multi.py):
    from multiticker_refactor.pipeline import build_multi_ticker_dataset

    df, metadata = build_multi_ticker_dataset(
        tickers=["AAPL", "MSFT", "GOOGL"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        include_rnn=True,
        include_sentiment=True
    )

    # Single-ticker (main.py, backward-compatible):
    from multiticker_refactor.pipeline import build_feature_dataset, prepare_for_ppo

    df = build_feature_dataset(include_rnn=True, include_sentiment=True)
    prices, signal_features, feature_cols = prepare_for_ppo(df)
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

from .config import (
    MAX_TICKERS,
    MAX_DATA_LOSS_PCT,
    PPO_TRAIN_RATIO,
    PPO_VAL_RATIO,
    LSTM_WINDOW_SIZE,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    PROB_LSTM_HORIZONS,
    POLYGON_API_KEY,
)
from .data.downloader import download_prices, clean_raw
from .data.features import add_all_technicals, add_calendar_macro
from .sentiment import add_sentiment_features
from .data.cache import (
    compute_data_hash,
    load_rnn_cache,
    save_rnn_cache,
    load_pipeline_cache,
    save_pipeline_cache
)
from .models.rnn import train_and_predict, train_and_predict_probabilistic


# =============================================================================
# STEP 1: DATA ALIGNMENT WITH VALIDATION
# =============================================================================

def align_ticker_data(ticker_dfs: Dict[str, pd.DataFrame], max_loss_pct: float = 0.20) -> pd.DataFrame:
    """
    Inner join with comprehensive data quality checks.

    Aligns multiple ticker DataFrames to common trading dates. Fails fast
    if data loss exceeds threshold or quality issues detected.

    Args:
        ticker_dfs: Dict of {ticker: DataFrame} with OHLCV data
        max_loss_pct: Maximum acceptable data loss (default 20%)

    Returns:
        Aligned DataFrame with all tickers

    Raises:
        ValueError: If data loss exceeds threshold or quality issues detected
    """
    tickers = list(ticker_dfs.keys())

    # 1. Get common dates (inner join)
    common_dates = set(ticker_dfs[tickers[0]].index)
    for ticker, df in ticker_dfs.items():
        common_dates &= set(df.index)

    common_dates = sorted(common_dates)

    # 2. Check data loss threshold
    max_possible = max(len(df) for df in ticker_dfs.values())
    data_loss_pct = 1 - (len(common_dates) / max_possible)

    if data_loss_pct > max_loss_pct:
        error_msg = [
            f"❌ Data alignment would lose {data_loss_pct:.1%} of data (threshold: {max_loss_pct:.1%}).",
            "\nTicker date ranges:"
        ]
        for ticker, df in ticker_dfs.items():
            error_msg.append(
                f"  {ticker}: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})"
            )
        error_msg.append(
            "\n💡 Suggestion: Choose tickers with similar listing dates, or adjust START_DATE/END_DATE in config.py"
        )
        raise ValueError('\n'.join(error_msg))

    # 3. Align all DataFrames to common dates
    aligned_dfs = {}
    for ticker, df in ticker_dfs.items():
        aligned_dfs[ticker] = df.loc[common_dates]

    # 4. Data quality checks (BEFORE merging, while columns are still unprefixed)
    for ticker, df in aligned_dfs.items():
        # Check what columns we have
        if 'Close' not in df.columns:
            # If columns are already prefixed, skip validation
            # (they were validated before download)
            continue

        # Check for invalid prices (price <= 0)
        invalid_prices = (df['Close'] <= 0).sum()
        if invalid_prices > 0:
            raise ValueError(
                f"❌ {ticker} has {invalid_prices} days with invalid prices (Close <= 0). "
                f"Data quality issue - check data source."
            )

        # Check for excessive zero-volume days (>1% of days)
        zero_volume = (df['Volume'] == 0).sum()
        if zero_volume / len(df) > 0.01:
            raise ValueError(
                f"❌ {ticker} has {zero_volume} zero-volume days ({zero_volume/len(df):.1%}). "
                f"Possibly delisted or data quality issue."
            )

    # 5. Check for date gaps (warn if gap > 5 trading days)
    date_diffs = np.diff([d.timestamp() for d in common_dates])
    max_gap_days = max(date_diffs) / 86400  # Convert seconds to days
    if max_gap_days > 5:
        print(f"⚠️  Warning: Maximum date gap is {max_gap_days:.0f} days (expected <5 for trading days)")

    print(f"✅ Aligned {len(tickers)} tickers: {len(common_dates)} common trading days")
    print(f"   Data loss: {data_loss_pct:.1%} (within {max_loss_pct:.1%} threshold)")

    # 6. Merge all DataFrames (columns already have ticker prefixes from download_prices)
    result = pd.DataFrame(index=common_dates)
    for ticker, df in aligned_dfs.items():
        for col in df.columns:
            # Columns are already prefixed (e.g., "AAPL_Close"), just copy them
            result[col] = df[col].values

    return result


# =============================================================================
# STEP 2: PER-TICKER TECHNICAL INDICATORS
# =============================================================================

def add_ticker_technicals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add technical indicators for a single ticker.

    Args:
        df: DataFrame with ticker OHLCV columns (e.g., AAPL_Close, AAPL_Volume)
        ticker: Ticker symbol

    Returns:
        DataFrame with technical indicator columns added
    """
    # Build a temporary single-ticker DataFrame for technical calculations
    temp_df = pd.DataFrame(index=df.index)
    temp_df['Close'] = df[f'{ticker}_Close']
    temp_df['High'] = df[f'{ticker}_High']
    temp_df['Low'] = df[f'{ticker}_Low']
    temp_df['Volume'] = df[f'{ticker}_Volume']
    temp_df['Open'] = df[f'{ticker}_Open']

    # Add technicals using existing functions
    # Pass empty list since temp_df has unprefixed columns (Close, High, Low, etc.)
    temp_df = add_all_technicals(temp_df, [])

    # Copy technical columns back to main df with ticker prefix
    for col in temp_df.columns:
        if col not in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[f'{ticker}_{col}'] = temp_df[col]

    return df


# =============================================================================
# STEP 3: PER-TICKER NORMALIZATION (FIT ON TRAIN ONLY)
# =============================================================================

def normalize_ticker_features(
    df: pd.DataFrame,
    ticker: str,
    train_end_idx: int
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize all features for a ticker using StandardScaler.
    Fit on TRAIN SET only to prevent data leakage.

    Args:
        df: DataFrame with all features
        ticker: Ticker symbol
        train_end_idx: Index where training set ends (for fit)

    Returns:
        Tuple of (normalized_df, scaler)
    """
    # Get all columns for this ticker
    ticker_cols = [col for col in df.columns if col.startswith(f"{ticker}_")]

    # Initialize scaler
    scaler = StandardScaler()

    # Fit scaler on TRAIN SET ONLY (critical: prevent data leakage)
    train_data = df[ticker_cols].iloc[:train_end_idx]
    scaler.fit(train_data)

    # Transform FULL dataset (train + val + test)
    df[ticker_cols] = scaler.transform(df[ticker_cols])

    return df, scaler


# =============================================================================
# STEP 4: PARALLEL RNN TRAINING (ONE MODEL PER TICKER)
# =============================================================================

def train_single_rnn(
    df: pd.DataFrame,
    ticker: str,
    target_col: str,
    probabilistic: bool,
    window_size: int,
    train_end_idx: int,
    start_date: str,
    end_date: str,
    use_cache: bool = True
) -> dict:
    """
    Train a single RNN for one ticker (worker function for multiprocessing).
    Uses disk cache to avoid re-training.

    IMPORTANT: Don't return Keras model - it can't be pickled across processes
    due to custom loss functions. Only return predictions.

    Returns:
        Dict with {feature_dict or predictions} (no model, no scaler)
    """
    # Compute data hash for cache validation
    data_hash = compute_data_hash(df, ticker)

    # Try to load from cache
    if use_cache:
        cached_result = load_rnn_cache(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            epochs=LSTM_EPOCHS,
            probabilistic=probabilistic,
            data_hash=data_hash
        )
        if cached_result is not None:
            print(f"   ✓ {ticker} RNN loaded from cache")
            return cached_result

    # Train RNN (cache miss)
    if probabilistic:
        feature_dict, model, scaler = train_and_predict_probabilistic(
            df, target_col, window_size=window_size
        )
        # Don't return model/scaler - can't be pickled with custom loss functions
        result = {'feature_dict': feature_dict}
    else:
        predictions, model, scaler = train_and_predict(
            df, target_col,
            window_size=window_size,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE
        )
        # Don't return model/scaler - can't be pickled
        result = {'predictions': predictions}

    # Save to cache
    if use_cache:
        save_rnn_cache(
            result=result,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            epochs=LSTM_EPOCHS,
            probabilistic=probabilistic,
            data_hash=data_hash
        )

    return result


def train_rnns_parallel(
    df: pd.DataFrame,
    tickers: List[str],
    start_date: str,
    end_date: str,
    probabilistic: bool = True,
    window_size: int = LSTM_WINDOW_SIZE,
    train_end_idx: int = None,
    use_cache: bool = True
) -> Dict[str, dict]:
    """
    Train one RNN per ticker in parallel using multiprocessing.
    Uses disk cache to avoid re-training.

    Args:
        df: DataFrame with normalized features
        tickers: List of ticker symbols
        start_date: Start date (for cache key)
        end_date: End date (for cache key)
        probabilistic: Use probabilistic multi-horizon LSTM
        window_size: LSTM window size
        train_end_idx: Index where training data ends
        use_cache: Whether to use disk cache

    Returns:
        Dict of {ticker: model_data} where model_data includes (model, scaler, predictions)
    """
    n_processes = min(len(tickers), cpu_count())
    print(f"   Training {len(tickers)} RNNs in parallel ({n_processes} CPU cores)...")
    print(f"   This may take several minutes per ticker (checking cache first)...\n")

    # Prepare arguments for each ticker
    train_args = []
    for ticker in tickers:
        target_col = f"{ticker}_Close"
        train_args.append((
            df, ticker, target_col, probabilistic, window_size, train_end_idx,
            start_date, end_date, use_cache
        ))

    # Train in parallel (or sequentially if only 1 ticker to avoid multiprocessing issues)
    if len(tickers) == 1:
        print(f"   Training single ticker sequentially (avoiding multiprocessing)...")
        results = [train_single_rnn(*train_args[0])]
    else:
        print(f"   Starting parallel training...")
        with Pool(processes=n_processes) as pool:
            results = pool.starmap(train_single_rnn, train_args)

    # Build results dict
    rnn_models = {}
    for ticker, result in zip(tickers, results):
        rnn_models[ticker] = result

    print()
    return rnn_models


def add_rnn_predictions_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    model_data: dict,
    probabilistic: bool = True
) -> pd.DataFrame:
    """
    Add RNN prediction columns for a single ticker.

    Args:
        df: DataFrame
        ticker: Ticker symbol
        model_data: Dict from train_single_rnn
        probabilistic: Whether using probabilistic LSTM

    Returns:
        DataFrame with RNN columns added (e.g., AAPL_rnn_mu_1d, AAPL_rnn_sigma_1d)
    """
    if probabilistic:
        # Add all probabilistic features
        for feature_name, values in model_data['feature_dict'].items():
            # Add ticker prefix to feature name
            ticker_feature_name = f"{ticker}_{feature_name}"
            df[ticker_feature_name] = values

        # Add confidence features (inverse of sigma)
        for horizon in PROB_LSTM_HORIZONS:
            sigma_col = f"{ticker}_rnn_sigma_{horizon}d"
            if sigma_col in df.columns:
                confidence_col = f"{ticker}_rnn_confidence_{horizon}d"
                df[confidence_col] = 1.0 / (df[sigma_col] + 1e-8)
    else:
        # Add simple point predictions
        df[f"{ticker}_rnn_pred_close"] = model_data['predictions']

    return df


# =============================================================================
# STEP 5: SENTIMENT (PER TICKER) - Polygon API + FinBERT
# =============================================================================

# Sentiment module handles:
# - News fetching from Polygon API (with caching)
# - Sentiment scoring with FinBERT (lazy-loaded, with caching)
# See: sentiment/pipeline.py


# =============================================================================
# STEP 6: FEATURE REGROUPING (BY TYPE FOR CROSS-TICKER LEARNING)
# =============================================================================

def regroup_features_by_type(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Regroup columns from ticker-grouped to feature-type-grouped.

    Before: [AAPL_Close, AAPL_RSI, MSFT_Close, MSFT_RSI, day_of_week, ...]
    After:  [AAPL_Close, MSFT_Close, AAPL_RSI, MSFT_RSI, day_of_week, ...]

    Args:
        df: DataFrame with ticker-prefixed columns
        tickers: List of ticker symbols

    Returns:
        DataFrame with columns regrouped by feature type
    """
    # 1. Extract all feature types (suffixes after ticker prefix)
    feature_types = set()
    for col in df.columns:
        if '_' in col and col.split('_')[0] in tickers:
            # Extract feature type (everything after first underscore)
            feature_type = '_'.join(col.split('_')[1:])
            feature_types.add(feature_type)

    # 2. Regroup: For each feature type, add all tickers' versions
    new_cols = []
    for feature_type in sorted(feature_types):
        for ticker in tickers:
            col = f"{ticker}_{feature_type}"
            if col in df.columns:
                new_cols.append(col)

    # 3. Add shared features (no ticker prefix) at the end
    shared_cols = [
        col for col in df.columns
        if '_' not in col or col.split('_')[0] not in tickers
    ]
    new_cols.extend(shared_cols)

    return df[new_cols]


# =============================================================================
# STEP 7: PREPARE FOR PPO (EXTRACT ARRAYS + METADATA)
# =============================================================================

def prepare_multi_ticker_for_ppo(
    df: pd.DataFrame,
    tickers: List[str],
    validate: bool = True
) -> dict:
    """
    Extract prices array and signal features for multi-ticker PPO environment.

    **IMPORTANT**: DataFrame must be ALREADY REGROUPED by feature type.

    Pipeline ordering:
        df = build_multi_ticker_dataset(...)      # Step 1: Add all features
        df = regroup_features_by_type(df, ...)    # Step 2: Regroup by type
        result = prepare_multi_ticker_for_ppo(df, tickers)  # Step 3: Extract

    Args:
        df: DataFrame with features ALREADY regrouped by type
        tickers: List of ticker symbols
        validate: Whether to validate data (default True)

    Returns:
        Dict with:
        - prices: np.ndarray, shape (n_timesteps, n_tickers)
        - signal_features: np.ndarray, shape (n_timesteps, n_features) - includes close prices
        - ticker_map: Dict[int, str] - Maps array index to ticker
        - feature_cols: List[str] - Feature column names
        - tickers: List[str] - Copy of input tickers
        - n_timesteps, n_tickers, n_features: Dimensions
    """
    # Extract close prices for all tickers (for prices array)
    close_cols = [f"{ticker}_Close" for ticker in tickers]

    if validate:
        # Check all close columns exist
        missing = [col for col in close_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing close price columns: {missing}")

        # Check for non-finite values in prices
        prices_df = df[close_cols]
        if not np.isfinite(prices_df.values).all():
            raise ValueError("Close prices contain NaN or inf values")

    # Extract prices array (n_timesteps, n_tickers)
    prices = df[close_cols].values.astype(np.float32)

    # Extract ALL features INCLUDING close prices (agent needs to see current prices)
    # Trading timeline: Agent trades AFTER close, sees today's close
    feature_cols = list(df.columns)
    signal_features = df[feature_cols].values.astype(np.float32)

    if validate:
        # Check for non-finite values in features
        if not np.isfinite(signal_features).all():
            raise ValueError("Signal features contain NaN or inf values")

    # Build ticker map (index -> ticker symbol)
    ticker_map = {i: ticker for i, ticker in enumerate(tickers)}

    # Get dimensions
    n_timesteps, n_features = signal_features.shape
    n_tickers = len(tickers)

    # Extract dates from DataFrame index
    dates = df.index.strftime('%Y-%m-%d').tolist()

    return {
        'prices': prices,
        'signal_features': signal_features,
        'ticker_map': ticker_map,
        'feature_cols': feature_cols,
        'tickers': tickers,
        'n_timesteps': n_timesteps,
        'n_tickers': n_tickers,
        'n_features': n_features,
        'dates': dates,  # Add dates list
    }


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def build_multi_ticker_dataset(
    tickers: List[str],
    start_date: str,
    end_date: str,
    include_rnn: bool = True,
    include_sentiment: bool = True,
    probabilistic_rnn: bool = True,
    use_cache: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Build complete multi-ticker feature dataset with all critical fixes applied.

    Steps:
    1. Download OHLCV for all tickers
    2. Align data (inner join with 20% threshold + quality checks)
    3. Add technical indicators per ticker
    4. Add calendar/macro features (shared across tickers)
    5. Normalize each ticker separately (fit on train only)
    6. Train RNNs in parallel (one per ticker)
    7. Add RNN predictions
    8. Add sentiment (optional)
    9. Regroup features by type (for cross-ticker learning)
    10. Clean final dataset

    Args:
        tickers: List of ticker symbols (max 3 for demo)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        include_rnn: Whether to train LSTMs
        include_sentiment: Whether to fetch sentiment
        probabilistic_rnn: Use probabilistic multi-horizon LSTM
        use_cache: Whether to use pipeline cache (default True)
        verbose: Print progress

    Returns:
        Tuple of (df, metadata) where:
        - df: DataFrame with all features (shape: (n_days, n_features))
        - metadata: Dict with {scalers, rnn_models, feature_cols, etc.}
    """
    if len(tickers) > MAX_TICKERS:
        raise ValueError(f"Maximum {MAX_TICKERS} tickers allowed (got {len(tickers)})")

    # =========================================================================
    # Try to load from cache first
    # =========================================================================
    if use_cache:
        cached = load_pipeline_cache(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            include_rnn=include_rnn,
            include_sentiment=include_sentiment,
            probabilistic_rnn=probabilistic_rnn
        )
        if cached is not None:
            df, metadata = cached
            # Restore non-serializable items (they're not critical for evaluation/training)
            metadata['scalers'] = {}  # Scalers not needed after normalization
            metadata['rnn_models'] = {}  # Models not needed after predictions added
            if verbose:
                print()
            return df, metadata

    if verbose:
        print("=" * 60)
        print("BUILDING MULTI-TICKER FEATURE DATASET")
        print("=" * 60)
        print(f"\nTickers: {tickers}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Include RNN: {include_rnn}")
        print(f"Include sentiment: {include_sentiment}\n")

    # =========================================================================
    # STEP 1: Download OHLCV data
    # =========================================================================
    if verbose:
        print("[1/9] Downloading OHLCV data...")

    ticker_dfs = {}
    for ticker in tickers:
        print(f"   Downloading {ticker}...", end=" ", flush=True)
        df = download_prices(ticker, [], start_date, end_date)
        df = clean_raw(df, [ticker])
        ticker_dfs[ticker] = df
        print(f"✓ ({len(df)} days)")
    print()

    # =========================================================================
    # STEP 2: Align data (inner join with quality checks)
    # =========================================================================
    if verbose:
        print("[2/9] Aligning ticker data (inner join)...")

    df = align_ticker_data(ticker_dfs, max_loss_pct=MAX_DATA_LOSS_PCT)

    # =========================================================================
    # STEP 3: Add technical indicators (per ticker)
    # =========================================================================
    if verbose:
        print("[3/9] Adding technical indicators...")

    for ticker in tickers:
        print(f"   Adding technicals for {ticker}...", end=" ", flush=True)
        df = add_ticker_technicals(df, ticker)
        print("✓")
    print()

    # =========================================================================
    # STEP 4: Add calendar/macro features (shared)
    # =========================================================================
    if verbose:
        print("[4/9] Adding calendar/macro features...", end=" ", flush=True)

    df = add_calendar_macro(df)

    if verbose:
        print("✓\n")

    # =========================================================================
    # STEP 5: Normalize each ticker separately (fit on train only)
    # =========================================================================
    if verbose:
        print("[5/9] Normalizing features (per ticker, train-only fit)...")

    total_len = len(df)
    train_end_idx = int(total_len * PPO_TRAIN_RATIO)  # 60%

    scalers = {}
    for ticker in tickers:
        print(f"   Normalizing {ticker}...", end=" ", flush=True)
        df, scaler = normalize_ticker_features(df, ticker, train_end_idx)
        scalers[ticker] = scaler
        print("✓")
    print()

    # =========================================================================
    # STEP 6: Train RNNs in parallel (one per ticker)
    # =========================================================================
    rnn_models = {}
    if include_rnn:
        if verbose:
            rnn_type = "Probabilistic Multi-Horizon" if probabilistic_rnn else "Simple"
            print(f"[6/9] Training {rnn_type} RNNs (parallel)...")

        rnn_models = train_rnns_parallel(
            df, tickers,
            start_date=start_date,
            end_date=end_date,
            probabilistic=probabilistic_rnn,
            window_size=LSTM_WINDOW_SIZE,
            train_end_idx=train_end_idx,
            use_cache=True
        )

        # =====================================================================
        # STEP 7: Add RNN predictions
        # =====================================================================
        if verbose:
            print("[7/9] Adding RNN predictions to dataset...")

        for ticker, model_data in rnn_models.items():
            print(f"   Adding predictions for {ticker}...", end=" ", flush=True)
            df = add_rnn_predictions_for_ticker(
                df, ticker, model_data, probabilistic=probabilistic_rnn
            )
            print("✓")
        print()
    else:
        if verbose:
            print("[6/9] Skipping RNN training")
            print("[7/9] Skipping RNN predictions")

    # =========================================================================
    # STEP 8: Add sentiment (Polygon + FinBERT, batch processing, optional)
    # =========================================================================
    if include_sentiment:
        if verbose:
            print("[8/9] Adding sentiment features...")

        df = add_sentiment_features(
            df=df,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            polygon_api_key=POLYGON_API_KEY,
            force_refresh=False,
            verbose=verbose
        )
    else:
        if verbose:
            print("[8/9] Skipping sentiment")

    # =========================================================================
    # STEP 9: Regroup features by type (cross-ticker learning)
    # =========================================================================
    if verbose:
        print("[9/9] Regrouping features by type...", end=" ", flush=True)

    df = regroup_features_by_type(df, tickers)

    if verbose:
        print("✓")
        print(f"   Cleaning NaN rows...", end=" ", flush=True)

    # Drop NaN rows (from rolling windows in technical indicators)
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)

    if verbose:
        print(f"✓ (removed {rows_before - rows_after} rows)\n")

    # Build metadata
    val_end_idx = train_end_idx + int(total_len * PPO_VAL_RATIO)
    metadata = {
        'tickers': tickers,
        'scalers': scalers,
        'rnn_models': rnn_models,
        'feature_cols': list(df.columns),
        'train_end_idx': train_end_idx,
        'val_end_idx': val_end_idx,
    }

    if verbose:
        print("\n" + "=" * 60)
        print(f"Final dataset shape: {df.shape}")
        print(f"Features per ticker: ~{len([c for c in df.columns if c.startswith(tickers[0] + '_')])}")
        print(f"Shared features: ~{len([c for c in df.columns if '_' not in c or c.split('_')[0] not in tickers])}")
        print("=" * 60 + "\n")

    # =========================================================================
    # Save to cache for next run
    # =========================================================================
    if use_cache:
        save_pipeline_cache(
            df=df,
            metadata=metadata,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            include_rnn=include_rnn,
            include_sentiment=include_sentiment,
            probabilistic_rnn=probabilistic_rnn
        )

    return df, metadata


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS (for single-ticker main.py)
# =============================================================================

def build_feature_dataset(
    include_rnn: bool = True,
    include_sentiment: bool = True,
    probabilistic_rnn: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for single-ticker pipeline (used by main.py).

    This calls the unified build_multi_ticker_dataset() with n_tickers=1.

    Args:
        include_rnn: Whether to train LSTM and add predictions
        include_sentiment: Whether to fetch and add sentiment
        probabilistic_rnn: If True, use Probabilistic Multi-Horizon LSTM
        verbose: Whether to print progress messages

    Returns:
        DataFrame with all features, ready for PPO training
    """
    from .config.ticker_config import TARGET_TICKER, START_DATE, END_DATE

    # Call unified pipeline with single ticker
    df, metadata = build_multi_ticker_dataset(
        tickers=[TARGET_TICKER],
        start_date=START_DATE,
        end_date=END_DATE,
        include_rnn=include_rnn,
        include_sentiment=include_sentiment,
        probabilistic_rnn=probabilistic_rnn,
        use_cache=True,
        verbose=verbose
    )

    return df


def prepare_for_ppo(df: pd.DataFrame) -> tuple:
    """
    Backward-compatible wrapper for single-ticker PPO prep (used by main.py).

    For single-ticker, returns (prices, signal_features, feature_cols) in old format.

    Args:
        df: Feature DataFrame from build_feature_dataset

    Returns:
        Tuple of (prices, signal_features, feature_cols)
    """
    from .config.ticker_config import TARGET_TICKER

    # Use the unified multi-ticker prepare function
    result = prepare_multi_ticker_for_ppo(df, tickers=[TARGET_TICKER], validate=True)

    # Extract single ticker's data (n_tickers=1, so prices is shape (n_timesteps, 1))
    prices = result['prices'][:, 0]  # Extract first ticker column -> (n_timesteps,)
    signal_features = result['signal_features']
    feature_cols = result['feature_cols']

    return prices, signal_features, feature_cols
