import pandas as pd
import os
from Project.param import *
from Project.data import load_data, train_test_split_lstm
from Project.sentiment_analysis import fetch_daily_ticker_sentiment
from Project.model import *


def load_market_data(ticker: str):
    df = load_data(

        start_date=START_DATE,
        end_date=END_DATE
    )

    df = df.rename(columns={"Date": "date"})
    df_OHLC = df.sort_values("date").reset_index(drop=True)

    return df_OHLC


# -------------------------
# Utility: Build RNN predictions aligned to dates
# -------------------------
def build_rnn_predictions(df_ohlc: pd.DataFrame, window_size: int = WINDOW_SIZE,
                          epochs: int = LSTM_EPOCHS, batch_size: int = BATCH_SIZE,
                          force_retrain: bool = True) -> pd.Series:
    """
    Train LSTM on historical Close and produce a one-day-ahead prediction for each day
    where enough history exists. Returns a pd.Series indexed by date with predicted value
    in the same scale as the original Close (not scaled).
    """
    df_local = df_ohlc.copy()

    # Normalize date column name
    if "date" in df_local.columns:
        df_local = df_local.rename(columns={"date": "Date"})
    elif "Date" not in df_local.columns:
        raise ValueError("DataFrame must contain either 'Date' or 'date' column.")

    df_local = df_local[["Date", "Close"]].reset_index(drop=True)
    # Use the helper which returns X_train,y_train,X_test,y_test,scaler
    X_train, y_train, X_test, y_test, scaler = train_test_split_lstm(df_local)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, n_features)
    model = LSTM_model(input_shape)
    model = compile_LSTM(model)

    # Train model (if force_retrain or no saved model)
    if force_retrain or not os.path.exists(RNN_MODEL_SAVE):
        print("Training LSTM predictor...")
        train_LSTM(model, X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(RNN_MODEL_SAVE)
    else:
        print("Loading existing LSTM predictor...")
        from keras.models import load_model
        model = load_model(RNN_MODEL_SAVE)

    # Now generate sliding-window predictions across the full dataset
    closes = df_local["Close"].values.reshape(-1, 1)
    closes_scaled = scaler.transform(closes)  # use same scaler

    preds = []
    dates = []
    for end_idx in range(window_size, len(closes_scaled)):
        start_idx = end_idx - window_size
        seq = closes_scaled[start_idx:end_idx]  # shape (window_size, 1)
        seq = seq.reshape((1, seq.shape[0], seq.shape[1]))
        pred_scaled = model.predict(seq, verbose=0)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        preds.append(pred)
        dates.append(df_local.loc[end_idx, "Date"])  # prediction aligned to day end_idx

    preds_series = pd.Series(data=preds, index=pd.to_datetime(dates))
    preds_series.name = "rnn_pred_close"
    return preds_series


# -------------------------
# Utility: Merge OHLC / Sentiment / RNN preds and create lag-features
# -------------------------
def build_merged_dataframe(df_ohlc: pd.DataFrame, df_sentiment: pd.DataFrame,
                           rnn_preds: pd.Series, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    Date, Open, High, Low, Close, Volume, sentiment, rnn_pred_close,
    close_lag_1 .. close_lag_{window_size}, next_return (reward target).
    """
    # Normalize column names & Date
    df = df_ohlc.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Merge sentiment (df_sentiment is indexed by date)
    df_sent = df_sentiment.copy()
    if "date" in df_sent.columns or df_sent.index.name == "date":
        df_sent.index = pd.to_datetime(df_sent.index)
    df_sent = df_sent.rename(columns={df_sent.columns[0]: "sentiment"})
    df = df.join(df_sent, how="left")
    df["sentiment"] = df["sentiment"].fillna(0.0)

    # Merge RNN preds (already indexed by date)
    df = df.join(rnn_preds.rename("rnn_pred_close"), how="left")

    # compute returns (next day) for reward and intraday if needed
    df["close"] = df["Close"].astype(float)
    df["return"] = df["close"].pct_change()
    # next day return as target (reward reference)
    df["next_return"] = df["return"].shift(-1)

    # Create lag features for close (flattened)
    for i in range(1, window_size + 1):
        df[f"close_lag_{i}"] = df["close"].shift(i)

    # drop rows without enough history or without next_return
    df = df.dropna().reset_index()
    return df


def load_sentiment():
    # Step 1: Get ticker-specific sentiment entries
    df_daily = fetch_daily_ticker_sentiment(api_key = API_KEY_MASSIVE, ticker = SENTIMENT_TICKERS, start_date = SENTIMENT_START_DATE, end_date = SENTIMENT_END_DATE)

    # Final structure: date | sentiment
    return df_daily


def merge_all(ohlcv, lstm_pred, sentiment):
    df = ohlcv.merge(lstm_pred, on="date", how="left")
    df = df.merge(sentiment, on="date", how="left")

    return df
