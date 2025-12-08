import time
from massive import RESTClient
from massive.rest.models import TickerNews
import pandas as pd
from Project.param import *

def fetch_ticker_sentiments(api_key: str, ticker: str, start_date: str, end_date: str, sleep: float = 0.2) -> pd.DataFrame:
    """
    Fetch news sentiments for a specific ticker and date range from Massive API.
    Returns a DataFrame indexed by `published_utc` containing only `sentiment`.
    """
    client = RESTClient(api_key)
    news_data = []

    generator = client.list_ticker_news(
        ticker=ticker,
        published_utc_gte=f"{start_date}T00:00:00Z",
        published_utc_lte=f"{end_date}T23:59:59Z",
        order="asc",
        limit=50,
        sort="published_utc",
    )

    for n in generator:
        if isinstance(n, TickerNews):
            for insight in n.insights:
                if insight.ticker == ticker and insight.sentiment:
                    news_data.append({
                        "published_utc": n.published_utc,
                        "sentiment": insight.sentiment
                    })
        time.sleep(sleep)

    df = pd.DataFrame(news_data)
    df["published_utc"] = pd.to_datetime(df["published_utc"])
    df.set_index("published_utc", inplace=True)
    return df


def get_daily_sentiment_filled(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Convert textual sentiment to numeric (-1, 0, 1), compute daily sum,
    and fill missing dates with 0.
    """
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    if df.empty:
        return pd.DataFrame(0, index=full_dates, columns=["sentiment"])

    df_num = df.copy()
    df_num.index = pd.to_datetime(df_num.index)

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df_num["sentiment"] = df_num["sentiment"].map(sentiment_map).fillna(0)

    df_daily = df_num.groupby(df_num.index.date)["sentiment"].sum()
    df_daily = df_daily.reindex(full_dates.date, fill_value=0)
    df_daily.index = pd.to_datetime(df_daily.index)
    df_daily.index.name = "date"
    df_daily = df_daily.to_frame()
    return df_daily


def fetch_daily_ticker_sentiment(api_key: str, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Wrapper function to fetch ticker sentiments and return daily numeric sentiment.
    """
    df_sentiments = fetch_ticker_sentiments(api_key, ticker, start_date, end_date)
    df_daily = get_daily_sentiment_filled(df_sentiments, start_date, end_date)
    return df_daily
