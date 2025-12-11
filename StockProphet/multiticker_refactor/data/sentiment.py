"""
Sentiment data fetching from Massive API with local caching.
Source: sentiment_analysis.py
"""
import time
import pandas as pd
from pathlib import Path
from massive import RESTClient
from massive.rest.models import TickerNews
from urllib3.exceptions import MaxRetryError

from ..config import SENTIMENT_CACHE_DIR


def _get_cache_path(ticker: str) -> Path:
    """Get the cache file path for a ticker."""
    return SENTIMENT_CACHE_DIR / f"{ticker}_sentiment.parquet"


def _load_cached_sentiment(ticker: str) -> pd.DataFrame:
    """Load cached sentiment data for a ticker if it exists."""
    cache_path = _get_cache_path(ticker)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame(columns=["sentiment"])


def _save_cached_sentiment(ticker: str, df: pd.DataFrame) -> None:
    """Save sentiment data to cache."""
    SENTIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(ticker)
    df.to_parquet(cache_path)
    print(f"Cached {len(df)} sentiment records for {ticker} at {cache_path}")


def _fetch_sentiment_from_api(
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
    sleep: float = 0.1,
    max_retries: int = 5,
    max_articles: int = 1000
) -> pd.DataFrame:
    """
    Fetch news sentiments for a specific ticker and date range from Massive API.
    Returns a DataFrame indexed by `published_utc` containing only `sentiment`.
    Includes exponential backoff for rate limiting (429 errors).
    """
    client = RESTClient(api_key)
    news_data = []
    article_count = 0

    for attempt in range(max_retries):
        try:
            generator = client.list_ticker_news(
                ticker=ticker,
                published_utc_gte=f"{start_date}T00:00:00Z",
                published_utc_lte=f"{end_date}T23:59:59Z",
                order="asc",
                limit=50,
                sort="published_utc",
            )

            for n in generator:
                article_count += 1
                if isinstance(n, TickerNews) and n.insights:
                    for insight in n.insights:
                        if insight.ticker == ticker and insight.sentiment:
                            news_data.append({
                                "published_utc": n.published_utc,
                                "sentiment": insight.sentiment
                            })
                # Stop if we've checked enough articles without finding sentiment
                if article_count >= max_articles and len(news_data) == 0:
                    print(f"Checked {article_count} articles, no sentiment data found. Stopping.")
                    break
                if sleep > 0:
                    time.sleep(sleep)
            break  # Success, exit retry loop

        except (MaxRetryError, Exception) as e:
            if "429" in str(e) or "too many" in str(e).lower():
                wait_time = sleep * (2 ** attempt)
                print(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise

    df = pd.DataFrame(news_data)
    if not df.empty:
        df["published_utc"] = pd.to_datetime(df["published_utc"])
        df.set_index("published_utc", inplace=True)
    return df


def fetch_ticker_sentiments(
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
    sleep: float = 1.0,
    max_retries: int = 5,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch news sentiments with local caching.
    Only fetches missing date ranges from the API.

    Args:
        api_key: Massive API key
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sleep: Seconds to wait between API requests
        max_retries: Number of retries on rate limit
        force_refresh: If True, ignore cache and fetch all data fresh
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if force_refresh:
        print(f"Force refresh: fetching all data for {ticker} from {start_date} to {end_date}")
        df_new = _fetch_sentiment_from_api(api_key, ticker, start_date, end_date, sleep, max_retries)
        if not df_new.empty:
            _save_cached_sentiment(ticker, df_new)
        return df_new

    # Load existing cache
    df_cached = _load_cached_sentiment(ticker)

    if df_cached.empty:
        print(f"No cache found for {ticker}. Fetching {start_date} to {end_date}...")
        df_new = _fetch_sentiment_from_api(api_key, ticker, start_date, end_date, sleep, max_retries)
        if not df_new.empty:
            _save_cached_sentiment(ticker, df_new)
        return df_new

    # Determine missing date ranges
    cached_min = df_cached.index.min()
    cached_max = df_cached.index.max()

    dfs_to_concat = [df_cached]

    # Fetch earlier data if needed
    if start_dt < cached_min:
        earlier_end = (cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Fetching earlier data for {ticker}: {start_date} to {earlier_end}")
        df_earlier = _fetch_sentiment_from_api(api_key, ticker, start_date, earlier_end, sleep, max_retries)
        if not df_earlier.empty:
            dfs_to_concat.append(df_earlier)

    # Fetch later data if needed
    if end_dt > cached_max:
        later_start = (cached_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Fetching later data for {ticker}: {later_start} to {end_date}")
        df_later = _fetch_sentiment_from_api(api_key, ticker, later_start, end_date, sleep, max_retries)
        if not df_later.empty:
            dfs_to_concat.append(df_later)

    # Merge and save
    if len(dfs_to_concat) > 1:
        df_merged = pd.concat(dfs_to_concat).sort_index()
        df_merged = df_merged[~df_merged.index.duplicated(keep='first')]
        _save_cached_sentiment(ticker, df_merged)
    else:
        df_merged = df_cached
        print(f"Using cached data for {ticker} ({len(df_cached)} records)")

    # Filter to requested range
    mask = (df_merged.index >= start_dt) & (df_merged.index <= end_dt + pd.Timedelta(days=1))
    return df_merged[mask]


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


def fetch_daily_ticker_sentiment(
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Wrapper function to fetch ticker sentiments and return daily numeric sentiment.
    Uses local cache to avoid repeated API calls.
    """
    df_sentiments = fetch_ticker_sentiments(api_key, ticker, start_date, end_date, force_refresh=force_refresh)
    df_daily = get_daily_sentiment_filled(df_sentiments, start_date, end_date)
    return df_daily


def clear_sentiment_cache(ticker: str = None) -> None:
    """
    Clear cached sentiment data.

    Args:
        ticker: If provided, clear only that ticker's cache. Otherwise clear all.
    """
    if ticker:
        cache_path = _get_cache_path(ticker)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache for {ticker}")
    else:
        if SENTIMENT_CACHE_DIR.exists():
            for f in SENTIMENT_CACHE_DIR.glob("*.parquet"):
                f.unlink()
            print("Cleared all sentiment cache")
