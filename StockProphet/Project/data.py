import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from Project.param import *



import yfinance as yf
import numpy as np
import pandas as pd

def load_and_prepare_data(ticker: str, start: str, end: str,
                          test_size=0.15, valid_size=0.15,
                          drop_cols=None,
                          target_col=TARGET):
    """
    Load stock data, compute selected features, flatten columns,
    split into train/validation/test sets, and drop unnecessary columns.

    Returns:
        train_df, valid_df, test_df: Split and processed DataFrames
    """
    if drop_cols is None:
        drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'OpenInt']

    # -----------------------------
    # 1️⃣ Download data
    # -----------------------------
    df = yf.download(ticker, start=start, end=end)
    df = df.dropna()

    # Fill missing business dates
    df.index = pd.to_datetime(df.index)
    all_dates = pd.date_range(start=start, end=end, freq='B')
    df = df.reindex(all_dates)
    df = df.ffill()

    # -----------------------------
    # 2️⃣ Compute indicators
    # -----------------------------
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()

    EMA_12 = df['Close'].ewm(span=12, min_periods=12).mean()
    EMA_26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['MACD'] = EMA_12 - EMA_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()

    n = 14
    delta = df['Close'].diff()
    pricesUp = delta.clip(lower=0)
    pricesDown = -1 * delta.clip(upper=0)
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.rolling(n).mean()
    rs = rollUp / rollDown
    df['RSI'] = 100 - (100 / (1 + rs))

    df = df.dropna()

    # Flatten MultiIndex if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)) for col in df.columns]

    # -----------------------------
    # 3️⃣ Shift target for next-day prediction
    # -----------------------------
    df[target_col] = df[target_col].shift(-1)

    # Drop first rows due to rolling indicators & last row due to shift
    df = df.iloc[33:-1].copy()
    df.index = range(len(df))

    # -----------------------------
    # 4️⃣ Split into train/valid/test
    # -----------------------------
    total_len = df.shape[0]
    test_split_idx = int(total_len * (1 - test_size))
    valid_split_idx = int(total_len * (1 - (valid_size + test_size)))

    train_df = df.loc[:valid_split_idx].copy()
    valid_df = df.loc[valid_split_idx + 1:test_split_idx].copy()
    test_df = df.loc[test_split_idx + 1:].copy()

    # Drop unwanted columns if they exist
    for df_subset in [train_df, valid_df, test_df]:
        for col in drop_cols:
            if col in df_subset.columns:
                df_subset.drop(columns=[col], inplace=True)

    return train_df, valid_df, test_df


def split_features_labels(train_df, valid_df, test_df, target_col=TARGET):
    """
    Split train, validation, and test DataFrames into features (X) and labels (y).

    Args:
        train_df (pd.DataFrame): Training set
        valid_df (pd.DataFrame): Validation set
        test_df (pd.DataFrame): Test set
        target_col (str): Column to use as target variable

    Returns:
        X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    # Training set
    y_train = train_df[target_col].copy()
    X_train = train_df.drop(columns=[target_col])

    # Validation set
    y_valid = valid_df[target_col].copy()
    X_valid = valid_df.drop(columns=[target_col])

    # Test set
    y_test = test_df[target_col].copy()
    X_test = test_df.drop(columns=[target_col])

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def prepare_stock_data(ticker: str, start: str, end: str,
                       test_size=0.15, valid_size=0.15,
                       target_col=TARGET, drop_cols=None):
    """
    End-to-end pipeline: download stock data, compute indicators, scale,
    split into train/valid/test, drop unwanted columns, and return X/y sets.

    Args:
        ticker (str): Stock ticker symbol
        start (str): Start date (YYYY-MM-DD)
        end (str): End date (YYYY-MM-DD)
        test_size (float): Fraction of data for test set
        valid_size (float): Fraction of data for validation set
        target_col (str): Target column name
        drop_cols (list): Columns to drop from features

    Returns:
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler
    """
    # 1️⃣ Load and prepare the data (returns train/valid/test DataFrames)
    train_df, valid_df, test_df = load_and_prepare_data(
        ticker=ticker, start=start, end=end,
        test_size=test_size, valid_size=valid_size,
        drop_cols=drop_cols
    )

    # 2️⃣ Split into features (X) and labels (y)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_features_labels(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        target_col=target_col
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test
