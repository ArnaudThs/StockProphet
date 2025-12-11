# Sentiment Analysis Module

Polygon API news fetching + FinBERT sentiment analysis integrated into the multi-ticker pipeline.

## Features

- ✅ **Polygon API Integration**: Fetches news articles with content filtering
- ✅ **FinBERT Sentiment**: Professional-grade financial sentiment analysis (97%+ confidence)
- ✅ **Two-Layer Caching**: Raw news + processed sentiment cached separately
- ✅ **Error Isolation**: Failures don't break the pipeline (fills with 0)
- ✅ **Multi-Ticker Support**: Processes all tickers in parallel
- ✅ **Smart Updates**: Only fetches missing date ranges

## Architecture

```
sentiment/
├── __init__.py          # Module exports
├── fetcher.py           # Polygon API + caching
├── processor.py         # FinBERT + sentiment caching
├── pipeline.py          # Integration with main pipeline
└── README.md            # This file

../sentiment_cache/      # Cache directory
├── {TICKER}_news_raw.parquet          # Raw news articles
└── {TICKER}_sentiment_processed.parquet  # FinBERT results
```

## Usage

### Option 1: Integrated into Pipeline

```python
from multiticker_refactor.pipeline_multi import build_multi_ticker_dataset

df, metadata = build_multi_ticker_dataset(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2025-12-10',
    include_sentiment=True,  # Enable sentiment
    verbose=True
)

# df now has columns: AAPL_Sentiment, MSFT_Sentiment, GOOGL_Sentiment
```

### Option 2: Standalone

```python
from multiticker_refactor.sentiment import add_sentiment_features
from multiticker_refactor.config import POLYGON_API_KEY

# Add sentiment to existing DataFrame
df = add_sentiment_features(
    df=df,
    tickers=['AAPL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2025-12-10',
    polygon_api_key=POLYGON_API_KEY,
    verbose=True
)
```

## How It Works

### 1. News Fetching (fetcher.py)

- Fetches from Polygon API with pagination
- **Content filtering**: Removes irrelevant articles (e.g., "pineapple" for AAPL)
- Uses regex word boundaries to match ticker/company name
- Caches raw articles as Parquet files
- Only fetches missing date ranges on subsequent runs

### 2. Sentiment Analysis (processor.py)

- Uses **FinBERT** (yiyanghkust/finbert-tone) - financial sentiment BERT
- Processes title + description
- Returns: `positive`, `negative`, or `neutral` with confidence score
- Caches processed sentiment separately
- Batch processing (32 articles at a time)

### 3. Daily Aggregation

- Maps sentiment to numeric: `positive → +1`, `negative → -1`, `neutral → 0`
- **Aggregation**: Simple sum of all article sentiments per day
- Fills missing dates with 0
- Joins to main DataFrame as `{TICKER}_Sentiment`

## Configuration

In [config.py](../config.py):

```python
# Enable sentiment
INCLUDE_SENTIMENT = True

# API key
POLYGON_API_KEY = "your_key_here"

# Date range (uses same as main pipeline)
START_DATE = "2020-01-01"
END_DATE = "2025-12-10"
```

## Performance

- **First run**: ~2-3 minutes per ticker (fetches all news + FinBERT processing)
- **Subsequent runs**: ~10-30 seconds (uses cache, only processes new articles)
- **FinBERT model size**: ~400MB (downloaded once)

## Cache Management

### Clear all caches:
```bash
rm -rf StockProphet/multiticker_refactor/sentiment_cache/
```

### Clear specific ticker:
```bash
rm StockProphet/multiticker_refactor/sentiment_cache/AAPL_*
```

### Force refresh (ignores cache):
```python
df = add_sentiment_features(
    df=df,
    tickers=['AAPL'],
    start_date='2020-01-01',
    end_date='2025-12-10',
    polygon_api_key=POLYGON_API_KEY,
    force_refresh=True  # Re-fetch everything
)
```

## Error Handling

The module is designed to **never break the pipeline**:

1. **FinBERT fails to load** → Skips sentiment, fills all tickers with 0
2. **Polygon API fails for a ticker** → Fills that ticker with 0, continues with others
3. **No articles found** → Fills with 0
4. **Processing error** → Fills with 0, prints warning

All errors are logged with warnings, and a summary is printed at the end.

## Validation Results

**FinBERT Sense Check** (100 AAPL articles):
- Distribution: 42% neutral, 41% positive, 17% negative
- Average confidence: **97.1%**
- Accuracy: **Excellent** (correctly identifies market sentiment, regulatory concerns, earnings previews)
- Verdict: ✅ **Fit for purpose** for financial sentiment analysis

## Example Output

```
============================================================
ADDING SENTIMENT FEATURES
============================================================

[AAPL] Fetching and processing sentiment...
  📥 No cache for AAPL, fetching 2020-01-01 to 2025-12-10
  📰 Fetched 310/1000 relevant articles (total: 310)
  📰 Fetched 382/1000 relevant articles (total: 692)
  ...
  💾 Cached 7142 articles for AAPL
  🤖 Loading FinBERT model...
  ✅ FinBERT loaded
  🤖 Running FinBERT on 7142 articles...
  💾 Cached sentiment for AAPL (7142 articles)
  ✅ Added AAPL_Sentiment (1521/2000 non-zero days)

[MSFT] Fetching and processing sentiment...
  ✅ Using cached data for MSFT (5234 articles)
  ✅ All 5234 articles already processed for MSFT
  ✅ Added MSFT_Sentiment (1342/2000 non-zero days)

============================================================
✅ Sentiment features added successfully for all tickers
============================================================
```

## Troubleshooting

**Import errors:**
```bash
pip install transformers torch pandas requests pyarrow
```

**Rate limiting (Polygon API):**
- Free tier: 5 requests/minute
- Uses exponential backoff automatically
- Consider upgrading API plan for faster fetching

**GPU acceleration:**
Edit [processor.py](processor.py):
```python
finbert = load_finbert_model(device=0)  # Use GPU 0
```

**Memory issues:**
Reduce batch size in [processor.py](processor.py):
```python
analyze_sentiment(..., batch_size=16)  # Reduce from 32
```
