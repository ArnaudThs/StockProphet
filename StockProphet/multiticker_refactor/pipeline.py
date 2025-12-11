"""
Unified data pipeline that builds the complete feature dataset.
Integrates: OHLCV download, technical indicators, calendar features, RNN predictions, sentiment.
"""
import numpy as np
import pandas as pd

from multiticker_refactor.config import (
    TARGET_TICKER, SUPPORT_TICKERS, START_DATE, END_DATE,
    SENTIMENT_START_DATE, SENTIMENT_END_DATE,
    API_KEY_MASSIVE, LSTM_WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_MODEL_PATH, PROB_LSTM_MODEL_PATH
)
from multiticker_refactor.data.downloader import download_prices, clean_raw
from multiticker_refactor.data.features import (
    initialize_feature_registry, add_all_technicals, add_calendar_macro,
    generate_cross_ticker_features, apply_shift_engine, trim_date_range,
    clean_final_dataset, data_integrity_report, generate_markdown_feature_doc,
    register_feature
)
from multiticker_refactor.data.sentiment import fetch_daily_ticker_sentiment
from multiticker_refactor.models.rnn import (
    train_and_predict, save_model,
    train_and_predict_probabilistic, save_probabilistic_model
)


def add_rnn_predictions(df: pd.DataFrame, probabilistic: bool = True) -> pd.DataFrame:
    """
    Train RNN (LSTM) on close prices and add predictions as columns.

    Args:
        df: DataFrame with target_close column
        probabilistic: If True, use Probabilistic Multi-Horizon LSTM
                       If False, use simple LSTM with point predictions

    Returns:
        DataFrame with RNN prediction columns added:
        - If probabilistic: rnn_mu_1d, rnn_sigma_1d, rnn_prob_up_1d, etc.
        - If not: rnn_pred_close
    """
    target_col = f"{TARGET_TICKER}_Close"

    if probabilistic:
        # Use Probabilistic Multi-Horizon LSTM
        feature_dict, model, scaler = train_and_predict_probabilistic(
            df, target_col,
            window_size=LSTM_WINDOW_SIZE
        )

        # Add all probabilistic features to DataFrame
        for feature_name, values in feature_dict.items():
            df[feature_name] = values
            register_feature(feature_name, "no_shift")

        # Add RNN confidence features (inverse of sigma)
        # Agent can learn to trust high-confidence predictions more
        for horizon in [1, 5]:
            sigma_col = f"rnn_sigma_{horizon}d"
            if sigma_col in df.columns:
                confidence_col = f"rnn_confidence_{horizon}d"
                df[confidence_col] = 1.0 / (df[sigma_col] + 1e-8)
                register_feature(confidence_col, "no_shift")

        # Save model
        save_probabilistic_model(model, PROB_LSTM_MODEL_PATH)

        print(f"Added {len(feature_dict)} probabilistic RNN features + confidence metrics")
    else:
        # Use simple LSTM with point predictions
        predictions, model, scaler = train_and_predict(
            df, target_col,
            window_size=LSTM_WINDOW_SIZE,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE
        )

        df["rnn_pred_close"] = predictions
        save_model(model, LSTM_MODEL_PATH)
        register_feature("rnn_pred_close", "no_shift")

    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch sentiment data and merge into DataFrame.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with sentiment column added
    """
    try:
        df_sent = fetch_daily_ticker_sentiment(
            API_KEY_MASSIVE, TARGET_TICKER,
            SENTIMENT_START_DATE, SENTIMENT_END_DATE
        )

        # Map sentiment to DataFrame index
        sentiment_dict = df_sent["sentiment"].to_dict()
        df["sentiment"] = df.index.to_series().apply(
            lambda d: sentiment_dict.get(d.normalize(), 0)
        )

    except Exception as e:
        print(f"Sentiment fetch failed: {e}, filling with zeros")
        df["sentiment"] = 0

    # Register the feature
    register_feature("sentiment", "no_shift")

    return df


def build_feature_dataset(
    include_rnn: bool = True,
    include_sentiment: bool = True,
    probabilistic_rnn: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Master pipeline that builds the complete feature dataset.

    Steps:
    1. Downloads OHLCV data for AAPL (and support tickers if any)
    2. Adds technical indicators (Return, SMA, RSI, Stochastic, Entropy)
    3. Adds calendar/macro features (day_of_week, holidays, CPI/NFP distances)
    4. Optionally trains LSTM and adds RNN prediction columns
    5. Optionally fetches and adds sentiment column
    6. Applies shift engine and trims to date range
    7. Returns clean DataFrame ready for RL training

    Args:
        include_rnn: Whether to train LSTM and add predictions
        include_sentiment: Whether to fetch and add sentiment
        probabilistic_rnn: If True, use Probabilistic Multi-Horizon LSTM (mu, sigma, prob_up)
                          If False, use simple LSTM with point predictions
        verbose: Whether to print progress messages

    Returns:
        DataFrame with all features, ready for PPO training
    """
    tickers = [TARGET_TICKER] + SUPPORT_TICKERS

    # Step 1: Download raw OHLCV
    if verbose:
        print("=" * 60)
        print("BUILDING FEATURE DATASET")
        print("=" * 60)
        print(f"\nTarget: {TARGET_TICKER}")
        print(f"Support: {SUPPORT_TICKERS if SUPPORT_TICKERS else 'None'}")
        print(f"Date range: {START_DATE} to {END_DATE}")
        print("\n[1/7] Downloading price data...")

    df = download_prices(TARGET_TICKER, SUPPORT_TICKERS, START_DATE, END_DATE)
    df = clean_raw(df, tickers)

    if verbose:
        print(f"      Downloaded {len(df)} rows")

    # Step 2: Initialize registry and add target_close
    if verbose:
        print("[2/7] Initializing feature registry...")
    df = initialize_feature_registry(df, TARGET_TICKER, SUPPORT_TICKERS)

    # Step 3: Add technical indicators
    if verbose:
        print("[3/7] Adding technical indicators...")
    df = add_all_technicals(df, tickers)

    # Step 4: Add cross-ticker features (if support tickers exist)
    if SUPPORT_TICKERS:
        if verbose:
            print("[4/7] Adding cross-ticker features...")
        df = generate_cross_ticker_features(df, TARGET_TICKER, SUPPORT_TICKERS)
    else:
        if verbose:
            print("[4/7] Skipping cross-ticker features (no support tickers)")

    # Step 5: Add calendar/macro features
    if verbose:
        print("[5/7] Adding calendar/macro features...")
    df = add_calendar_macro(df)

    # Step 6: Add RNN predictions (trains LSTM first)
    if include_rnn:
        if verbose:
            rnn_type = "Probabilistic Multi-Horizon" if probabilistic_rnn else "Simple"
            print(f"[6/7] Training {rnn_type} RNN and generating predictions...")
        df = add_rnn_predictions(df, probabilistic=probabilistic_rnn)
    else:
        if verbose:
            print("[6/7] Skipping RNN predictions")

    # Step 7: Add sentiment
    if include_sentiment:
        if verbose:
            print("[7/7] Fetching sentiment data...")
        df = add_sentiment(df)
    else:
        if verbose:
            print("[7/7] Skipping sentiment")

    # Apply shift engine and trim dates
    if verbose:
        print("\nApplying shift engine and trimming dates...")
    df = apply_shift_engine(df)
    df = trim_date_range(df, START_DATE, END_DATE)

    # Clean final dataset
    df = clean_final_dataset(df)

    # Report
    if verbose:
        print("\n" + "=" * 60)
        data_integrity_report(df)
        generate_markdown_feature_doc()
        print("=" * 60)
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("=" * 60 + "\n")

    return df


def get_X_y(df: pd.DataFrame, target_column: str = 'target_close') -> tuple:
    """
    Split DataFrame into features (X) and target (y).

    Args:
        df: Feature DataFrame
        target_column: Name of target column

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def prepare_for_ppo(df: pd.DataFrame) -> tuple:
    """
    Prepare data for PPO training.

    Args:
        df: Feature DataFrame from build_feature_dataset

    Returns:
        Tuple of (prices, signal_features, feature_cols)
    """
    # Extract prices
    prices = df["target_close"].values.astype(np.float32)

    # Feature columns (exclude target)
    feature_cols = df.columns.drop("target_close")

    # Signal features matrix
    signal_features = df[feature_cols].values.astype(np.float32)

    return prices, signal_features, feature_cols
