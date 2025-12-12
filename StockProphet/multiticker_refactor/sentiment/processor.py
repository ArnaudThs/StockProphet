"""
FinBERT Sentiment Processor

Analyzes sentiment of news articles using FinBERT.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "sentiment_cache"


def _get_cache_path(ticker: str) -> Path:
    """Get cache path for processed sentiment."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker}_sentiment_processed.parquet"


def _load_cached_sentiment(ticker: str) -> pd.DataFrame:
    """Load cached sentiment if exists."""
    cache_path = _get_cache_path(ticker)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df['published_utc'] = pd.to_datetime(df['published_utc'])
        return df
    return pd.DataFrame(columns=['id', 'published_utc', 'sentiment_label', 'sentiment_score'])


def _save_cached_sentiment(ticker: str, df: pd.DataFrame) -> None:
    """Save processed sentiment to cache."""
    cache_path = _get_cache_path(ticker)
    df.to_parquet(cache_path, index=False)
    print(f"  💾 Cached sentiment for {ticker} ({len(df)} articles)")


def load_finbert_model(device: Optional[int] = None):
    """
    Load FinBERT model (once for all tickers).

    Args:
        device: GPU device index (None for CPU, 0 for first GPU)

    Returns:
        FinBERT sentiment pipeline
    """
    MODEL_NAME = "yiyanghkust/finbert-tone"

    print("  🤖 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    finbert = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=256,
        device=device  # None = CPU, 0 = GPU
    )

    print("  ✅ FinBERT loaded")
    return finbert


def analyze_sentiment(
    df_news: pd.DataFrame,
    finbert_pipeline,
    ticker: str,
    batch_size: int = 32,
    force_reprocess: bool = False
) -> pd.DataFrame:
    """
    Analyze sentiment using FinBERT with caching.

    Args:
        df_news: DataFrame with columns ['id', 'published_utc', 'title', 'description']
        finbert_pipeline: Loaded FinBERT pipeline
        ticker: Stock ticker
        batch_size: Batch size for processing
        force_reprocess: Reprocess even if cached

    Returns:
        DataFrame with sentiment columns added
    """
    if df_news.empty:
        return df_news

    # Load cached sentiment
    df_cached = _load_cached_sentiment(ticker)

    if force_reprocess or df_cached.empty:
        # Process all articles
        df_result = _process_all_articles(df_news, finbert_pipeline, batch_size)
        _save_cached_sentiment(ticker, df_result)
        return df_result

    # Find articles not in cache
    cached_ids = set(df_cached['id'].values)
    new_articles = df_news[~df_news['id'].isin(cached_ids)]

    if new_articles.empty:
        # All cached
        print(f"  ✅ All {len(df_news)} articles already processed for {ticker}")
        return df_cached[df_cached['id'].isin(df_news['id'])].copy()

    # Process new articles
    print(f"  🔄 Processing {len(new_articles)} new articles for {ticker}")
    df_new_processed = _process_all_articles(new_articles, finbert_pipeline, batch_size)

    # Merge with cache
    df_merged = pd.concat([df_cached, df_new_processed], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['id'])
    _save_cached_sentiment(ticker, df_merged)

    # Return only requested articles
    return df_merged[df_merged['id'].isin(df_news['id'])].copy()


def _process_all_articles(df_news: pd.DataFrame, finbert_pipeline, batch_size: int) -> pd.DataFrame:
    """Process articles with FinBERT."""
    df = df_news.copy()

    # Prepare text: title + description
    df['text_for_sentiment'] = (
        df['title'].fillna('') + ". " + df['description'].fillna('')
    ).str.strip()

    # Filter empty text
    df = df[df['text_for_sentiment'] != ''].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=['id', 'published_utc', 'sentiment_label', 'sentiment_score'])

    # Run FinBERT in batches
    texts = df['text_for_sentiment'].tolist()
    sentiments = []
    scores = []

    print(f"  🤖 Running FinBERT on {len(texts)} articles...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        results = finbert_pipeline(batch_texts)

        for res in results:
            # Normalize label
            label = res['label'].lower().strip()
            if label.startswith('pos'):
                label = 'positive'
            elif label.startswith('neg'):
                label = 'negative'
            elif label.startswith('neu'):
                label = 'neutral'

            sentiments.append(label)
            scores.append(float(res['score']))

        # Progress indicator
        if (i + batch_size) % 160 == 0:  # Every ~5 batches
            print(f"    Processed {min(i + batch_size, len(texts))}/{len(texts)} articles...")

    df['sentiment_label'] = sentiments
    df['sentiment_score'] = scores

    # Keep only essential columns
    return df[['id', 'published_utc', 'sentiment_label', 'sentiment_score']].copy()


def aggregate_daily_sentiment(
    df_sentiment: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Aggregate sentiment to daily scores.

    Maps: positive → +1, negative → -1, neutral → 0
    Aggregates: sum of all sentiment values per day
    Fills missing dates with 0

    Args:
        df_sentiment: DataFrame with sentiment_label column
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with daily sentiment sums, indexed by date
    """
    if df_sentiment.empty:
        # Return all zeros (timezone-naive)
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D', tz=None)
        return pd.DataFrame({'sentiment': 0}, index=full_dates)

    df = df_sentiment.copy()
    # Ensure timezone-aware timestamps are converted to timezone-naive
    df['published_utc'] = pd.to_datetime(df['published_utc']).dt.tz_localize(None)

    # Map to numeric
    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['sentiment_num'] = df['sentiment_label'].map(sentiment_map).fillna(0)

    # Group by date
    df['date'] = df['published_utc'].dt.date
    df_daily = df.groupby('date')['sentiment_num'].sum().reset_index()
    df_daily.columns = ['date', 'sentiment']

    # Create full date range (timezone-naive)
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D', tz=None)

    # Reindex to fill missing dates
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.set_index('date')
    df_daily = df_daily.reindex(full_dates, fill_value=0)
    df_daily.index.name = 'date'

    return df_daily
