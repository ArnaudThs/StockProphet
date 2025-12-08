import requests
import pandas as pd
from datetime import datetime

from Project.param import (
    API_KEY_MASSIVE,
    SENTIMENT_START_DATE,
    SENTIMENT_END_DATE,
    SENTIMENT_TICKERS,
    SENTIMENT_API_LIMIT,
    SENTIMENT_MAX_PAGES
)

# ============================================================
# 1. FETCH NEWS FROM MASSIVE API
# ============================================================

def fetch_massive_news_df(
    api_key: str,
    start_date: str = SENTIMENT_START_DATE,
    end_date: str = SENTIMENT_END_DATE,
    limit: int = SENTIMENT_API_LIMIT,
    max_pages: int = SENTIMENT_MAX_PAGES
) -> pd.DataFrame:
    """
    Fetch news from Massive API using proper date filters.
    Massive supports:
        - published_utc.gte
        - published_utc.lte
        - ticker filter
    """

    url = "https://api.massive.com/v2/reference/news"

    params = {
        "apiKey": api_key,
        "limit": min(limit, 1000),
        "order": "desc",
        "sort": "published_utc",
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
    }

    all_items = []
    page = 0

    while True:
        page += 1
        print(f"Fetching page {page}...")

        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            break

        all_items.extend(results)

        # stop if we have enough items
        if len(all_items) >= limit:
            break

        # pagination
        next_url = data.get("next_url")
        if not next_url or page >= max_pages:
            break

        url = next_url
        params = {}  # next_url already contains cursor

    df = pd.DataFrame(all_items)

    if not df.empty:
        df["published_utc"] = pd.to_datetime(df["published_utc"])
        df = df.sort_values("published_utc")

    return df


# ============================================================
# 2. FILTER NEWS BY TICKERS
# ============================================================

def filter_news_by_tickers(df_news: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Keep only news containing at least one of the requested tickers."""
    if df_news.empty:
        return df_news

    if "tickers" not in df_news.columns:
        print("Warning: 'tickers' column missing in news data.")
        return pd.DataFrame()

    tickers = [t.upper() for t in tickers]

    return df_news[
        df_news["tickers"].apply(
            lambda lst: any(t in lst for t in tickers) if isinstance(lst, list) else False
        )
    ]


# ============================================================
# 3. EXPLODE INSIGHTS → ONE ROW PER SENTIMENT SIGNAL
# ============================================================

def explode_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list of insights into individual rows (ticker, sentiment, reasoning)."""
    rows = []

    for _, row in df.iterrows():
        published = row.get("published_utc")
        title = row.get("title")
        insights = row.get("insights", [])

        if not isinstance(insights, list):
            continue

        for item in insights:
            rows.append({
                "date": published,
                "ticker": item.get("ticker"),
                "title": title,
                "sentiment": item.get("sentiment"),
                "sentiment_reasoning": item.get("sentiment_reasoning"),
            })

    df_out = pd.DataFrame(rows)

    if not df_out.empty:
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out = df_out.set_index("date").sort_index()

    return df_out


# ============================================================
# 4. HIGH-LEVEL WRAPPER USING param.py CONFIG
# ============================================================

def get_sentiment_dataset(
    api_key: str = API_KEY_MASSIVE,
    tickers: list = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Master function to fetch + filter + explode sentiment data.
    Uses default parameters from param.py unless overridden.
    """

    tickers = tickers or SENTIMENT_TICKERS
    start_date = start_date or SENTIMENT_START_DATE
    end_date = end_date or SENTIMENT_END_DATE

    print(f"Fetching sentiment for {tickers} from {start_date} to {end_date}...")

    df_news = fetch_massive_news_df(api_key, start_date, end_date)
    df_filtered = filter_news_by_tickers(df_news, tickers)
    df_sentiment = explode_insights(df_filtered)

    print(f"Loaded {len(df_sentiment)} sentiment rows for {tickers}.")
    return df_sentiment
