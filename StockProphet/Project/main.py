import pandas as pd
import os
from Project.param import *
from Project.data import load_data, train_test_split_lstm
from Project.sentiment_analysis import fetch_daily_ticker_sentiment
from Project.model import LSTM_model, compile_LSTM, train_LSTM
from keras.models import load_model

# -------------------------
# Load OHLC Data
# -------------------------
def load_market_data(ticker: str):
    df = load_data(ticker, START_DATE, END_DATE)
    df.columns = [c.lower() for c in df.columns]  # lowercase all columns
    df = df.sort_values("date").reset_index(drop=True)
    return df

# -------------------------
# Build RNN Predictions
# -------------------------
def build_rnn_predictions(df_ohlc: pd.DataFrame, window_size: int = WINDOW_SIZE,
                          epochs: int = LSTM_EPOCHS, batch_size: int = BATCH_SIZE,
                          force_retrain: bool = True) -> pd.Series:

    df_local = df_ohlc.copy()
    df_local = df_local[["date", "close"]].reset_index(drop=True)

    # Train/test split for LSTM
    X_train, y_train, X_test, y_test, scaler = train_test_split_lstm(df_local)

    # Build LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = LSTM_model(input_shape)
    model = compile_LSTM(model)

    # Train or load model
    if force_retrain or not os.path.exists(RNN_MODEL_SAVE):
        print("Training LSTM predictor...")
        train_LSTM(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
        model.save(RNN_MODEL_SAVE)
    else:
        print("Loading existing LSTM predictor...")
        model = load_model(RNN_MODEL_SAVE)

    # Sliding-window predictions
    closes = df_local["close"].values.reshape(-1, 1)
    closes_scaled = scaler.transform(closes)
    preds, dates = [], []

    for end_idx in range(window_size, len(closes_scaled)):
        start_idx = end_idx - window_size
        seq = closes_scaled[start_idx:end_idx].reshape(1, window_size, 1)
        pred_scaled = model.predict(seq, verbose=0)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        preds.append(pred)
        dates.append(df_local.loc[end_idx, "date"])

    preds_series = pd.Series(data=preds, index=pd.to_datetime(dates))
    preds_series.name = "rnn_pred_close"
    return preds_series

# -------------------------
# Load Sentiment
# -------------------------
def load_sentiment():
    try:
        df_sent = fetch_daily_ticker_sentiment(
            api_key=API_KEY_MASSIVE,
            ticker=SENTIMENT_TICKERS,
            start_date=SENTIMENT_START_DATE,
            end_date=SENTIMENT_END_DATE
        )
    except Exception as e:
        print("Sentiment fetch failed, filling zeros:", e)
        full_dates = pd.date_range(SENTIMENT_START_DATE, SENTIMENT_END_DATE, freq="D")
        df_sent = pd.DataFrame(0, index=full_dates, columns=["sentiment"])
        df_sent.index.name = "date"

    df_sent.columns = [c.lower() for c in df_sent.columns]
    df_sent.index = pd.to_datetime(df_sent.index)
    return df_sent

# -------------------------
# Merge OHLC, RNN, Sentiment and create features
# -------------------------
def build_merged_dataframe(df_ohlc: pd.DataFrame, df_sentiment: pd.DataFrame,
                           rnn_preds: pd.Series, window_size: int = WINDOW_SIZE) -> pd.DataFrame:

    df = df_ohlc.copy()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Merge sentiment
    df_sentiment.index = pd.to_datetime(df_sentiment.index)
    df = df.join(df_sentiment, how="left")
    df["sentiment"] = df["sentiment"].fillna(0.0)

    # Merge RNN predictions
    df = df.join(rnn_preds.rename("rnn_pred_close"), how="left")

    # Compute returns
    df["return"] = df["close"].pct_change()
    df["next_return"] = df["return"].shift(-1)

    # Create lag features
    for i in range(1, window_size + 1):
        df[f"close_lag_{i}"] = df["close"].shift(i)

    df = df.dropna().reset_index()
    return df

# -------------------------
# Main Orchestration
# -------------------------
def main():
    print("Loading OHLC data...")
    df_ohlc = load_market_data(TICKER)

    print("Computing RNN predictions...")
    rnn_preds = build_rnn_predictions(df_ohlc, window_size=WINDOW_SIZE, epochs=LSTM_EPOCHS)

    print("Fetching sentiment...")
    df_sent = load_sentiment()

    print("Merging datasets...")
    df_merged = build_merged_dataframe(df_ohlc, df_sent, rnn_preds, window_size=WINDOW_SIZE)
    print(f"Merged dataset shape: {df_merged.shape}")

    print("Saving merged dataset to merged_dataset.csv...")
    df_merged.to_csv("merged_dataset.csv", index=False)

if __name__ == "__main__":
    main()
