import pandas as pd
from Project.param import *
from Project.data import load_data
from Project.sentiment_analysis import (
    get_daily_sentiment_filled,
    get_filtered_massive_sentiment_simple
)

def load_market_data(ticker: str):
    df = load_data(
        ticker=ticker,
        start_date=START_DATE,
        end_date=END_DATE
    )

    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    return df


def build_lstm_prediction_df(y_pred, time_step):
    all_dates = pd.date_range(start=START_DATE, end=END_DATE)

    pred_dates = all_dates[time_step:]

    df_pred = pd.DataFrame({
        "date": pred_dates,
        "lstm_pred": y_pred.flatten()
    })

    return df_pred


def load_sentiment():
    # Step 1: Get ticker-specific sentiment entries
    df_ticker = get_filtered_massive_sentiment_simple()

    # Step 2: Convert to daily sentiment & fill missing dates
    df_daily = get_daily_sentiment_filled(df_ticker)

    # Step 3: Reset index so we get a column "date"
    df_daily = df_daily.reset_index()  # index becomes "date"
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    # Final structure: date | sentiment
    return df_daily


def merge_all(ohlcv, lstm_pred, sentiment):
    df = ohlcv.merge(lstm_pred, on="date", how="left")
    df = df.merge(sentiment, on="date", how="left")

    return df
