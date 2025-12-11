# Pipeline DataFrame Caching Guide

## Overview

The pipeline now includes **3-tier caching** to speed up development iteration:

1. **YFinance Cache** - Cached OHLCV data (skips re-downloading)
2. **RNN Cache** - Cached RNN predictions (skips 2-5 min training per ticker)
3. **Pipeline Cache** - **NEW: Cached complete DataFrame** (skips entire pipeline)

## Pipeline Cache

### What Gets Cached

The complete final DataFrame from `build_multi_ticker_dataset()` including:
- All OHLCV data (aligned across tickers)
- Technical indicators
- Calendar/macro features
- Normalized features
- RNN predictions (if enabled)
- Sentiment features (if enabled)
- Metadata (feature columns, train/val/test split indices)

### Cache Key

Cache invalidates (rebuilds) when ANY input parameter changes:

```python
# These parameters define the cache key:
- tickers: List of ticker symbols (order doesn't matter)
- start_date: Start date (YYYY-MM-DD)
- end_date: End date (YYYY-MM-DD)
- include_rnn: Whether RNN features are included
- include_sentiment: Whether sentiment features are included
- probabilistic_rnn: Whether using probabilistic LSTM

# Example cache key:
pipeline_AAPL_GOOGL_MSFT_2020-01-01_2025-06-30_prob_nosent.parquet
```

### Performance Benefits

**First run** (no cache):
```
[1/9] Downloading OHLCV data...        ~30-60 seconds
[2/9] Aligning ticker data...          ~1 second
[3/9] Adding technical indicators...   ~5-10 seconds
[4/9] Adding calendar/macro...         ~1 second
[5/9] Normalizing features...          ~1 second
[6/9] Training RNNs (parallel)...      ~2-5 min per ticker
[7/9] Adding RNN predictions...        ~1 second
[8/9] Adding sentiment...              ~10-30 seconds (if enabled)
[9/9] Regrouping features...           ~1 second

TOTAL: ~3-8 minutes (3 tickers with RNN)
```

**Second run** (with cache):
```
✅ Loaded pipeline cache: pipeline_AAPL_GOOGL_MSFT_...
   Cached at: 2025-12-11T10:30:45
   Shape: (1234, 156)

TOTAL: ~1-2 seconds ⚡
```

**Speedup: 100-200× faster** for subsequent runs with same parameters!

## Usage

### Enable Caching (Default)

```python
from multiticker_refactor.pipeline_multi import build_multi_ticker_dataset

# Cache is enabled by default
df, metadata = build_multi_ticker_dataset(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2025-06-30",
    include_rnn=True,
    include_sentiment=False,
    use_cache=True  # Default
)
```

### Disable Caching

```python
# Force rebuild (ignore cache)
df, metadata = build_multi_ticker_dataset(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2025-06-30",
    use_cache=False  # Disable cache
)
```

## Cache Management

### View Cache Statistics

```bash
cd StockProphet
python -m multiticker_refactor.cache_cli --stats
```

Output:
```
============================================================
CACHE STATISTICS
============================================================

YFinance Cache:
  Files: 3
  Size:  1.2 MB

RNN Cache:
  Files: 6
  Size:  15.3 MB

Pipeline Cache:
  Files: 2
  Size:  45.7 MB

Total Cache Size: 62.2 MB
============================================================
```

### Clear Cache

```bash
# Clear all caches
python -m multiticker_refactor.cache_cli --clear all

# Clear only pipeline cache (keep RNN/yfinance)
python -m multiticker_refactor.cache_cli --clear pipeline

# Clear only RNN cache
python -m multiticker_refactor.cache_cli --clear rnn

# Clear only yfinance cache
python -m multiticker_refactor.cache_cli --clear yfinance
```

## When Cache Invalidates (Rebuilds)

The pipeline cache will **automatically invalidate** when:

1. **Tickers change** - Different stocks
   ```python
   # Cache miss (different tickers)
   df1 = build_multi_ticker_dataset(tickers=["AAPL", "MSFT"], ...)
   df2 = build_multi_ticker_dataset(tickers=["AAPL", "GOOGL"], ...)
   ```

2. **Date range changes** - Different start/end dates
   ```python
   # Cache miss (different dates)
   df1 = build_multi_ticker_dataset(..., start_date="2020-01-01", end_date="2024-12-31")
   df2 = build_multi_ticker_dataset(..., start_date="2020-01-01", end_date="2025-06-30")
   ```

3. **Features change** - RNN or sentiment toggled
   ```python
   # Cache miss (different features)
   df1 = build_multi_ticker_dataset(..., include_rnn=True, include_sentiment=False)
   df2 = build_multi_ticker_dataset(..., include_rnn=True, include_sentiment=True)
   ```

4. **RNN type changes** - Probabilistic vs simple LSTM
   ```python
   # Cache miss (different RNN type)
   df1 = build_multi_ticker_dataset(..., probabilistic_rnn=True)
   df2 = build_multi_ticker_dataset(..., probabilistic_rnn=False)
   ```

## Cache Location

```
StockProphet/multiticker_refactor/data_cache/
├── yfinance_cache/          # OHLCV data
│   └── AAPL_2020-01-01_2025-06-30.parquet
├── rnn_cache/               # RNN predictions
│   ├── rnn_AAPL_2020-01-01_2025-06-30_w50_e20_prob_abc123.npz
│   └── rnn_AAPL_2020-01-01_2025-06-30_w50_e20_prob_abc123_meta.json
└── pipeline_cache/          # Complete DataFrames (NEW)
    ├── pipeline_AAPL_GOOGL_MSFT_2020-01-01_2025-06-30_prob_nosent.parquet
    └── pipeline_AAPL_GOOGL_MSFT_2020-01-01_2025-06-30_prob_nosent_meta.json
```

## Testing

Test the caching system:

```bash
cd StockProphet
python test_pipeline_cache.py
```

This will:
1. Clear pipeline cache
2. Build DataFrame from scratch (slow)
3. Load from cache (fast)
4. Verify data integrity
5. Test cache invalidation
6. Show performance metrics

Expected output:
```
First run:  180.5s
Second run: 1.2s
Speedup:    150.4x faster

✅ Cache saves ~150x time on repeat runs
✅ Data integrity verified: True
✅ Cache invalidation works: Different params trigger rebuild
```

## Best Practices

1. **Keep cache enabled** during development (default behavior)
2. **Clear cache** when you change pipeline code or add new features
3. **Monitor cache size** - Large date ranges create large cache files
4. **Use different parameters** for experiments - Each combo gets its own cache

## Troubleshooting

**Problem: Cache not being used**
- Check parameters match exactly (tickers, dates, flags)
- Verify cache files exist: `python -m multiticker_refactor.cache_cli --stats`

**Problem: Outdated cache**
- Clear cache: `python -m multiticker_refactor.cache_cli --clear pipeline`
- Or force rebuild: `use_cache=False`

**Problem: Disk space**
- Check size: `python -m multiticker_refactor.cache_cli --stats`
- Clear old caches: `python -m multiticker_refactor.cache_cli --clear all`

## Technical Details

### Storage Format
- **DataFrame**: Parquet (fast, compressed, preserves dtypes)
- **Metadata**: JSON (human-readable, lightweight)

### What's NOT Cached
- Scalers (not needed after normalization applied)
- RNN models (not needed after predictions added)
- Temporary computation artifacts

### Cache Hit Detection
1. Compute cache key from input parameters
2. Check if `pipeline_{key}.parquet` and `pipeline_{key}_meta.json` exist
3. Load both files
4. Restore metadata dict
5. Return (df, metadata)

### Cache Miss (Rebuild)
1. Run full 9-step pipeline
2. Save DataFrame as parquet
3. Save metadata as JSON
4. Return (df, metadata)
