"""
Sentiment Pipeline Integration

Main entry point for adding sentiment features to the multi-ticker pipeline.
"""

import pandas as pd
from typing import List
from .fetcher import get_news_for_ticker
from .processor import load_finbert_model, analyze_sentiment, aggregate_daily_sentiment


def add_sentiment_features(
    df: pd.DataFrame,
    tickers: List[str],
    start_date: str,
    end_date: str,
    polygon_api_key: str,
    force_refresh: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add sentiment features to multi-ticker DataFrame.

    This function:
    1. Fetches news from Polygon API (with caching)
    2. Checks cache for processed sentiment (CRITICAL: FinBERT only loads if needed)
    3. Analyzes NEW articles with FinBERT (lazy-loads model on first uncached article)
    4. Aggregates to daily sentiment scores (sum of article sentiments)
    5. Adds {TICKER}_Sentiment columns to DataFrame

    Performance:
    - FinBERT model only loads if uncached articles exist (saves ~3-5 seconds)
    - Uses cached parquet files when all articles are already processed

    Error Handling:
    - If sentiment fails for any ticker, fills with 0 and prints warning
    - Does not break the pipeline

    Args:
        df: Multi-ticker DataFrame (with datetime index or 'date' column)
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        polygon_api_key: Polygon.io API key
        force_refresh: Ignore cache and fetch fresh (default: False)
        verbose: Print progress (default: True)

    Returns:
        DataFrame with {TICKER}_Sentiment columns added

    Example:
        >>> df = add_sentiment_features(
        ...     df=df,
        ...     tickers=['AAPL', 'MSFT'],
        ...     start_date='2020-01-01',
        ...     end_date='2025-12-10',
        ...     polygon_api_key='YOUR_KEY'
        ... )
        >>> # df now has columns: AAPL_Sentiment, MSFT_Sentiment
    """
    if verbose:
        print("\\n" + "=" * 60)
        print("ADDING SENTIMENT FEATURES")
        print("=" * 60)

    # Ensure df has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")

    # Process each ticker
    # NOTE: Only load FinBERT if needed (checked per ticker below)
    finbert = None
    failed_tickers = []

    for ticker in tickers:
        try:
            if verbose:
                print(f"\\n[{ticker}] Fetching and processing sentiment...")

            # 1. Fetch news articles
            df_news = get_news_for_ticker(
                api_key=polygon_api_key,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh
            )

            if df_news.empty:
                if verbose:
                    print(f"  ⚠️  No news articles found for {ticker}, filling with 0")
                df[f'{ticker}_Sentiment'] = 0
                continue

            # 2. Check if FinBERT model is needed (only load if uncached articles exist)
            from .processor import _load_cached_sentiment
            df_cached = _load_cached_sentiment(ticker)
            cached_ids = set(df_cached['id'].values) if not df_cached.empty else set()
            new_articles = df_news[~df_news['id'].isin(cached_ids)]

            # Only load FinBERT if there are new articles to process
            if not new_articles.empty and finbert is None:
                if verbose:
                    print(f"  🤖 Loading FinBERT model (needed for {len(new_articles)} new articles)...")
                try:
                    finbert = load_finbert_model(device=None)
                except Exception as e:
                    print(f"  ❌ Failed to load FinBERT model: {e}")
                    print(f"     Skipping sentiment features (will fill with 0)")
                    for remaining_ticker in tickers:
                        if f'{remaining_ticker}_Sentiment' not in df.columns:
                            df[f'{remaining_ticker}_Sentiment'] = 0
                    return df

            # 3. Analyze sentiment with FinBERT
            df_sentiment = analyze_sentiment(
                df_news=df_news,
                finbert_pipeline=finbert,
                ticker=ticker,
                batch_size=32,
                force_reprocess=force_refresh
            )

            # 4. Aggregate to daily sentiment
            df_daily = aggregate_daily_sentiment(
                df_sentiment=df_sentiment,
                start_date=start_date,
                end_date=end_date
            )

            # 4. Join to main DataFrame (capital S for Sentiment to match convention)
            # Ensure both indices are timezone-naive for join
            df_daily_naive = df_daily.copy()
            df_daily_naive.index = pd.to_datetime(df_daily_naive.index).tz_localize(None)

            df = df.join(df_daily_naive['sentiment'].rename(f'{ticker}_Sentiment'), how='left')
            df[f'{ticker}_Sentiment'] = df[f'{ticker}_Sentiment'].fillna(0).astype(int)

            if verbose:
                non_zero = (df[f'{ticker}_Sentiment'] != 0).sum()
                print(f"  ✅ Added {ticker}_Sentiment ({non_zero}/{len(df)} non-zero days)")

        except Exception as e:
            if verbose:
                print(f"  ❌ Failed to process sentiment for {ticker}: {e}")
                print(f"     Filling {ticker}_Sentiment with 0")
                import traceback
                traceback.print_exc()
            df[f'{ticker}_Sentiment'] = 0
            failed_tickers.append(ticker)

    # Final summary
    if verbose:
        print("\\n" + "=" * 60)
        if failed_tickers:
            print(f"⚠️  Sentiment processing failed for: {', '.join(failed_tickers)}")
        else:
            print("✅ Sentiment features added successfully for all tickers")
        print("=" * 60)

    return df


def sense_check_finbert(
    ticker: str = "AAPL",
    n_samples: int = 100,
    polygon_api_key: str = None,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-10"
) -> pd.DataFrame:
    """
    Sense check FinBERT sentiment on first N articles.

    Used for validation only - not part of main pipeline.

    Args:
        ticker: Ticker to test
        n_samples: Number of articles to check
        polygon_api_key: Polygon API key
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with articles and sentiment for manual review
    """
    print("=" * 60)
    print(f"FINBERT SENSE CHECK - {ticker}")
    print("=" * 60)

    # Fetch news
    print(f"\\nFetching news for {ticker}...")
    df_news = get_news_for_ticker(
        api_key=polygon_api_key,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        force_refresh=False
    )

    if df_news.empty:
        print("No articles found!")
        return pd.DataFrame()

    # Take first N
    df_sample = df_news.head(n_samples).copy()
    print(f"\\nAnalyzing {len(df_sample)} articles with FinBERT...\\n")

    # Load FinBERT
    finbert = load_finbert_model(device=None)

    # Process
    df_sentiment = analyze_sentiment(
        df_news=df_sample,
        finbert_pipeline=finbert,
        ticker=ticker,
        batch_size=32,
        force_reprocess=False
    )

    # Merge for review
    df_review = df_sample.merge(df_sentiment, on='id', how='left')

    # Display sample
    print("\\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)

    for idx, row in df_review.head(20).iterrows():
        print(f"\\n[{idx + 1}] {row['published_utc_x'].strftime('%Y-%m-%d')}")
        print(f"Title: {row['title'][:100]}...")
        print(f"Sentiment: {row['sentiment_label']} (confidence: {row['sentiment_score']:.3f})")
        print("-" * 60)

    # Summary statistics
    print("\\n" + "=" * 60)
    print("SENTIMENT DISTRIBUTION")
    print("=" * 60)
    print(df_review['sentiment_label'].value_counts())

    print("\\n" + "=" * 60)
    print("AVERAGE CONFIDENCE BY SENTIMENT")
    print("=" * 60)
    print(df_review.groupby('sentiment_label')['sentiment_score'].mean())

    return df_review[['published_utc_x', 'title', 'description', 'sentiment_label', 'sentiment_score']].rename(
        columns={'published_utc_x': 'published_utc'}
    )
