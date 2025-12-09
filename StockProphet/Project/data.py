import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Project.param import *

import yfinance as yf
import numpy as np
import pandas as pd

def load_data(ticker, start_date, end_date):
    """
    Download yfinance data and flatten MultiIndex columns
    into a clean OHLCV DataFrame.
    """
    # 1. Download data
    data = yf.download(ticker, start=start_date, end=end_date)
    df = data.copy()

    # 2. Reset index so Date becomes a column
    df = df.reset_index()

    # 3. Flatten MultiIndex columns if needed
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        # --- AUTO CLEAN: Remove ticker suffix, keep only OHLCV ---
    clean_cols = {}
    for col in df.columns:
        if col.startswith("Date"):
            clean_cols[col] = "Date"
        elif col.endswith(f"_{ticker}"):
            clean_cols[col] = col.replace(f"_{ticker}", "")
        else:
            clean_cols[col] = col

    df = df.rename(columns=clean_cols)

    return df


def train_test_split_lstm(df, split_ratio=0.7, time_step=50):
    """
    Splits a DataFrame with columns ['Date', 'Close'] into
    scaled LSTM-ready X_train, y_train, X_test, y_test.
    Uses one scaler for both train and validation.
    """

    # --- Basic split ---
    length = len(df)
    length_train = round(length * split_ratio)

    train_df = df.iloc[:length_train][["date", "close"]].copy()
    val_df   = df.iloc[length_train:][["date", "close"]].copy()

    # Convert dates
    train_df["date"] = pd.to_datetime(train_df["date"])
    val_df["date"]   = pd.to_datetime(val_df["date"])

    # --- Scaling ---
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_close = train_df["close"].values.reshape(-1, 1)
    val_close   = val_df["close"].values.reshape(-1, 1)

    train_scaled = scaler.fit_transform(train_close)
    val_scaled   = scaler.transform(val_close)  # <-- Correct! transform, NOT fit

    # --- Create sequences ---
    def create_sequences(data, time_step):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i - time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y).reshape(-1, 1)

    X_train, y_train = create_sequences(train_scaled, time_step)
    X_test,  y_test  = create_sequences(val_scaled, time_step)

    # --- Reshape for LSTM ---
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

    return X_train, y_train, X_test, y_test, scaler
