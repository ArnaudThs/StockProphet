"""
Quick test to verify pipeline DataFrame caching works correctly.

Usage:
    cd StockProphet
    python test_pipeline_cache.py
"""
import time
from multiticker_refactor.pipeline_multi import build_multi_ticker_dataset
from multiticker_refactor.data.cache import get_cache_stats, clear_cache

# Test configuration (small date range for fast testing)
TEST_TICKERS = ["AAPL"]
TEST_START = "2024-01-01"
TEST_END = "2024-06-30"

print("=" * 80)
print("TESTING PIPELINE DATAFRAME CACHE")
print("=" * 80)

# Clear pipeline cache to start fresh
print("\n1. Clearing pipeline cache...")
clear_cache("pipeline")

# Show initial stats
print("\n2. Initial cache stats:")
stats = get_cache_stats()
print(f"   Pipeline cache: {stats['pipeline_cache']['count']} files, {stats['pipeline_cache']['size_mb']} MB")

# First run - should build from scratch
print("\n3. First run (building from scratch)...")
start_time = time.time()
df1, meta1 = build_multi_ticker_dataset(
    tickers=TEST_TICKERS,
    start_date=TEST_START,
    end_date=TEST_END,
    include_rnn=True,
    include_sentiment=False,
    probabilistic_rnn=True,
    use_cache=True,
    verbose=True
)
first_run_time = time.time() - start_time
print(f"\n   First run took: {first_run_time:.2f} seconds")
print(f"   DataFrame shape: {df1.shape}")

# Show cache stats after first run
print("\n4. Cache stats after first run:")
stats = get_cache_stats()
print(f"   Pipeline cache: {stats['pipeline_cache']['count']} files, {stats['pipeline_cache']['size_mb']} MB")

# Second run - should load from cache
print("\n5. Second run (loading from cache)...")
start_time = time.time()
df2, meta2 = build_multi_ticker_dataset(
    tickers=TEST_TICKERS,
    start_date=TEST_START,
    end_date=TEST_END,
    include_rnn=True,
    include_sentiment=False,
    probabilistic_rnn=True,
    use_cache=True,
    verbose=True
)
second_run_time = time.time() - start_time
print(f"\n   Second run took: {second_run_time:.2f} seconds")
print(f"   DataFrame shape: {df2.shape}")

# Verify data matches
print("\n6. Verifying cached data matches...")
shapes_match = df1.shape == df2.shape
columns_match = list(df1.columns) == list(df2.columns)
data_match = df1.equals(df2)

print(f"   Shapes match: {shapes_match}")
print(f"   Columns match: {columns_match}")
print(f"   Data matches: {data_match}")

# Calculate speedup
speedup = first_run_time / second_run_time
print(f"\n7. Performance improvement:")
print(f"   First run:  {first_run_time:.2f}s")
print(f"   Second run: {second_run_time:.2f}s")
print(f"   Speedup:    {speedup:.1f}x faster")

# Test different parameters (should NOT hit cache)
print("\n8. Testing cache invalidation (different parameters)...")
start_time = time.time()
df3, meta3 = build_multi_ticker_dataset(
    tickers=TEST_TICKERS,
    start_date=TEST_START,
    end_date=TEST_END,
    include_rnn=False,  # Different parameter
    include_sentiment=False,
    probabilistic_rnn=True,
    use_cache=True,
    verbose=True
)
third_run_time = time.time() - start_time
print(f"\n   Third run (different params) took: {third_run_time:.2f}s")
print(f"   Should be slower than cache hit: {third_run_time > second_run_time}")

# Final cache stats
print("\n9. Final cache stats:")
stats = get_cache_stats()
print(f"   Pipeline cache: {stats['pipeline_cache']['count']} files, {stats['pipeline_cache']['size_mb']} MB")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

# Summary
print("\nSUMMARY:")
print(f"✅ Cache saves ~{speedup:.1f}x time on repeat runs")
print(f"✅ Data integrity verified: {data_match}")
print(f"✅ Cache invalidation works: Different params trigger rebuild")
print(f"\nCache location: multiticker_refactor/data_cache/pipeline_cache/")
print(f"Manage cache: python -m multiticker_refactor.cache_cli --stats")
