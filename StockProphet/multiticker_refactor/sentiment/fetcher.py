"""
News Fetcher with Polygon API

Fetches news articles with caching and Apple-style content filtering.
"""

import requests
import pandas as pd
import time
import re
from pathlib import Path
from typing import List, Dict, Optional


# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "sentiment_cache"


def _get_cache_path(ticker: str, data_type: str = "raw") -> Path:
    """Get cache file path for a ticker."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker}_news_{data_type}.parquet"


def _load_cached_news(ticker: str) -> pd.DataFrame:
    """Load cached raw news if it exists."""
    cache_path = _get_cache_path(ticker, "raw")
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Ensure timezone-naive for consistency
        df['published_utc'] = pd.to_datetime(df['published_utc']).dt.tz_localize(None)
        return df
    return pd.DataFrame(columns=['id', 'published_utc', 'title', 'description', 'ticker'])


def _save_cached_news(ticker: str, df: pd.DataFrame) -> None:
    """Save raw news to cache."""
    cache_path = _get_cache_path(ticker, "raw")
    df.to_parquet(cache_path, index=False)
    print(f"  💾 Cached {len(df)} articles for {ticker}")


def _article_mentions_ticker(article: dict, ticker: str) -> bool:
    """
    Check if article actually mentions the ticker.

    Filters out irrelevant articles (e.g., 'pineapple' for AAPL).
    Uses word boundary regex to avoid false matches.
    """
    # Check ticker is in Polygon's ticker list
    tickers_list = article.get("tickers", [])
    if ticker not in tickers_list:
        return False

    # Get text content
    title = article.get("title", "") or ""
    description = article.get("description", "") or ""
    text = (title + " " + description).lower()

    # Check for ticker mentions (case-insensitive word boundary)
    ticker_pattern = rf"\b{re.escape(ticker.lower())}\b"
    if re.search(ticker_pattern, text):
        return True

    # For common stocks, also check company name
    # Add more mappings as needed
    company_names = {
        "AAPL": ["apple"],
        "MSFT": ["microsoft"],
        "GOOGL": ["google", "alphabet"],
        "AMZN": ["amazon"],
        "TSLA": ["tesla"],
        "META": ["meta", "facebook"],
        "NVDA": ["nvidia"],
    }

    if ticker in company_names:
        for name in company_names[ticker]:
            name_pattern = rf"\b{re.escape(name)}\b"
            if re.search(name_pattern, text):
                return True

    return False


def fetch_polygon_news(
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 5,
    rate_limit_sleep: int = 12
) -> List[Dict]:
    """
    Fetch news articles from Polygon API with retry logic.

    Args:
        api_key: Polygon API key
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_retries: Max retries on rate limit
        rate_limit_sleep: Seconds between requests

    Returns:
        List of article dictionaries
    """
    base_url = "https://api.polygon.io/v2/reference/news"
    articles = []

    params = {
        'ticker': ticker,
        'published_utc.gte': start_date,
        'published_utc.lte': end_date,
        'limit': 1000,
        'apiKey': api_key,
        'sort': 'published_utc',
        'order': 'asc'
    }

    next_url = None
    retries = 0

    while True:
        try:
            if next_url:
                url = next_url
                response = requests.get(url)
            else:
                response = requests.get(base_url, params=params)

            # Handle rate limiting (429)
            if response.status_code == 429:
                if retries >= max_retries:
                    print(f"  ⚠️  Max retries reached for {ticker}, stopping fetch")
                    break

                wait_time = rate_limit_sleep * (2 ** retries)  # Exponential backoff
                print(f"  ⚠️  Rate limited. Waiting {wait_time}s (retry {retries + 1}/{max_retries})...")
                time.sleep(wait_time)
                retries += 1
                continue

            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if not results:
                break

            # Filter for relevant articles
            filtered = [art for art in results if _article_mentions_ticker(art, ticker)]
            articles.extend(filtered)

            print(f"  📰 Fetched {len(filtered)}/{len(results)} relevant articles (total: {len(articles)})")

            # Check for next page
            next_url = data.get('next_url')
            if next_url:
                # Add API key to next_url
                separator = '&' if '?' in next_url else '?'
                next_url += f'{separator}apiKey={api_key}'
            else:
                break

            # Respect rate limit
            time.sleep(rate_limit_sleep)
            retries = 0  # Reset retry counter on success

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request error for {ticker}: {e}")
            break

    return articles


def get_news_for_ticker(
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Get news articles with caching.

    Only fetches missing date ranges from API.

    Args:
        api_key: Polygon API key
        ticker: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        force_refresh: Ignore cache and fetch fresh

    Returns:
        DataFrame with columns: ['id', 'published_utc', 'title', 'description', 'ticker']
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if force_refresh:
        print(f"  🔄 Force refresh for {ticker}")
        articles = fetch_polygon_news(api_key, ticker, start_date, end_date)
        df = _articles_to_dataframe(articles, ticker)
        if not df.empty:
            _save_cached_news(ticker, df)
        return df

    # Load cache
    df_cached = _load_cached_news(ticker)

    if df_cached.empty:
        # No cache, fetch everything
        print(f"  📥 No cache for {ticker}, fetching {start_date} to {end_date}")
        articles = fetch_polygon_news(api_key, ticker, start_date, end_date)
        df = _articles_to_dataframe(articles, ticker)
        if not df.empty:
            _save_cached_news(ticker, df)
        return df

    # Determine missing ranges
    cached_min = df_cached['published_utc'].min()
    cached_max = df_cached['published_utc'].max()

    dfs_to_concat = [df_cached]

    # Fetch earlier data if needed
    if start_dt < cached_min:
        earlier_end = (cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  📥 Fetching earlier data for {ticker}: {start_date} to {earlier_end}")
        articles = fetch_polygon_news(api_key, ticker, start_date, earlier_end)
        df_earlier = _articles_to_dataframe(articles, ticker)
        if not df_earlier.empty:
            dfs_to_concat.append(df_earlier)

    # Fetch later data if needed
    if end_dt > cached_max:
        later_start = (cached_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  📥 Fetching later data for {ticker}: {later_start} to {end_date}")
        articles = fetch_polygon_news(api_key, ticker, later_start, end_date)
        df_later = _articles_to_dataframe(articles, ticker)
        if not df_later.empty:
            dfs_to_concat.append(df_later)

    # Merge and save
    if len(dfs_to_concat) > 1:
        df_merged = pd.concat(dfs_to_concat, ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=['id']).sort_values('published_utc')
        _save_cached_news(ticker, df_merged)
    else:
        df_merged = df_cached
        print(f"  ✅ Using cached data for {ticker} ({len(df_cached)} articles)")

    # Filter to requested range
    mask = (df_merged['published_utc'] >= start_dt) & (df_merged['published_utc'] <= end_dt)
    return df_merged[mask].reset_index(drop=True)


def _articles_to_dataframe(articles: List[Dict], ticker: str) -> pd.DataFrame:
    """Convert article list to DataFrame."""
    if not articles:
        return pd.DataFrame(columns=['id', 'published_utc', 'title', 'description', 'ticker'])

    data = []
    for art in articles:
        data.append({
            'id': art.get('id', ''),
            'published_utc': art.get('published_utc', ''),
            'title': art.get('title', '') or '',
            'description': art.get('description', '') or '',
            'ticker': ticker
        })

    df = pd.DataFrame(data)
    # Ensure timezone-naive for consistency
    df['published_utc'] = pd.to_datetime(df['published_utc']).dt.tz_localize(None)
    return df
