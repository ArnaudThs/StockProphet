import requests
import pandas as pd
from datetime import datetime
from Project.param import *

def fetch_massive_news_df(api_key: str, start_date: str, end_date: str = None, order: str = "asc") -> pd.DataFrame:
    """
    Fetch news from Massive API for a specific date range and return as a pandas DataFrame.

    Args:
        api_key (str): Your Massive API key.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        order (str, optional): 'asc' or 'desc' for sorting by published date. Default is 'desc'.

    Returns:
        pd.DataFrame: DataFrame with news articles.
    """
    url = "https://api.massive.com/v2/reference/news"

    # Convert end_date to today if not provided
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    # Convert to ISO format for the API
    start_iso = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "order": order,
        "sort": "published_utc",
        "limit": 1000,  # Max per request
        "published_utc.gt": start_iso,
        "published_utc.lt": end_iso,
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        if "results" in data:
            df = pd.DataFrame(data["results"])
            return df
        else:
            print("No results found in the API response.")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


    return df_news


def filter_news_by_tickers(df_news: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Filter a Massive API news DataFrame by tickers.

    Args:
        df_news (pd.DataFrame): DataFrame returned by fetch_massive_news_df.
        tickers (list): List of tickers to filter (e.g., ["AAPL", "MSFT"]).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only news for the specified tickers.
    """
    # If df is empty → return empty df
    if df_news.empty:
        return df_news

    # Ensure tickers column exists
    if "tickers" not in df_news.columns:
        print("Warning: 'tickers' column missing in the data.")
        return pd.DataFrame()

    # Normalize tickers (case-insensitive)
    tickers = [t.upper() for t in tickers]

    # Filter rows containing at least one of the tickers
    filtered_df = df_news[
        df_news["tickers"].apply(
            lambda lst: any(t in lst for t in tickers) if isinstance(lst, list) else False
        )
    ]

    return filtered_df

def explode_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the filtered news DataFrame and explodes each insight entry
    into separate rows with: date, ticker, title, sentiment, reasoning.

    Returns:
        pd.DataFrame: A transformed DataFrame ready for RL pipelines.
    """

    all_rows = []

    for _, row in df.iterrows():
        date = row["published_utc"]
        title = row.get("title", None)
        insights = row.get("insights", [])

        # Skip rows without insights
        if not isinstance(insights, list):
            continue

        for item in insights:
            ticker = item.get("ticker")
            sentiment = item.get("sentiment")
            reasoning = item.get("sentiment_reasoning")

            all_rows.append({
                "date": date,
                "ticker": ticker,
                "title": title,
                "sentiment": sentiment,
                "sentiment_reasoning": reasoning
            })

    # Convert to DataFrame
    df_out = pd.DataFrame(all_rows)

    # Index by date
    if not df_out.empty:
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out = df_out.set_index("date").sort_index()

    return df_out

def get_filtered_massive_sentiment_simple():
    """
    Fetch Massive news, explode insights, and return only AAPL sentiment with date index as YYYY-MM-DD.

    Returns:
        pd.DataFrame: columns ['ticker', 'sentiment'], index = date as YYYY-MM-DD
    """
    # Step 1: fetch news
    df_news = fetch_massive_news_df(
        api_key=API_KEY_MASSIVE,
        start_date=SENTIMENT_START_DATE,
        end_date=SENTIMENT_END_DATE
    )

    # Step 2: filter by ticker
    df_filtered = filter_news_by_tickers(df_news, [SENTIMENT_TICKERS])

    # Step 3: explode insights
    df_exploded = explode_insights(df_filtered)

    if df_exploded.empty:
        return df_exploded

    # Step 4: keep only AAPL, ticker and sentiment
    df_ticker = df_exploded[df_exploded['ticker'].str.upper() == 'AAPL'][['ticker', 'sentiment']].copy()

    # Step 5: format index as YYYY-MM-DD
    df_ticker.index = df_ticker.index.strftime('%Y-%m-%d')

    return df_ticker


def get_daily_sentiment_filled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert textual sentiment to numeric (-1, 0, 1), compute daily average,
    and fill missing dates with 0.
    """
    if df.empty:
        # Return full date range with zeros if input is empty
        full_dates = pd.date_range(start=SENTIMENT_START_DATE, end=SENTIMENT_END_DATE, freq='D')
        return pd.DataFrame(0, index=full_dates, columns=["sentiment"])

    # Ensure index is datetime
    df_num = df.copy()
    df_num.index = pd.to_datetime(df_num.index)

    # Map sentiment to numeric
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df_num["sentiment"] = df_num["sentiment"].map(sentiment_map)

    # Group by date and compute daily average
    df_daily = df_num.groupby(df_num.index)["sentiment"].mean().to_frame(name="sentiment")

    # Create full date range
    full_dates = pd.date_range(start=SENTIMENT_START_DATE, end=SENTIMENT_END_DATE, freq='D')

    # Reindex to include all dates, fill missing with 0
    df_daily = df_daily.reindex(full_dates, fill_value=0)
    df_daily.index.name = "date"

    return df_daily
