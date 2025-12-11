"""
Quick test script to verify caching implementation works.

Usage:
    python test_caching.py
"""
import time
import numpy as np
import pandas as pd
from StockProphet.multiticker_refactor.data.cache import (
    save_yfinance_cache,
    load_yfinance_cache,
    save_rnn_cache,
    load_rnn_cache,
    compute_data_hash,
    get_cache_stats,
    clear_cache
)


def test_yfinance_cache():
    """Test yfinance caching."""
    print("\n" + "=" * 60)
    print("TESTING YFINANCE CACHE")
    print("=" * 60)

    # Create dummy data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    df = pd.DataFrame({
        'AAPL_Open': np.random.randn(len(dates)) + 100,
        'AAPL_High': np.random.randn(len(dates)) + 102,
        'AAPL_Low': np.random.randn(len(dates)) + 98,
        'AAPL_Close': np.random.randn(len(dates)) + 100,
        'AAPL_Volume': np.random.randint(1e6, 1e8, len(dates))
    }, index=dates)

    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    # Save to cache
    print(f"\n1. Saving {ticker} data to cache...")
    save_yfinance_cache(df, ticker, start_date, end_date)
    print("   ✓ Saved")

    # Load from cache
    print(f"\n2. Loading {ticker} data from cache...")
    cached_df = load_yfinance_cache(ticker, start_date, end_date)
    print(f"   ✓ Loaded: {cached_df.shape}")

    # Verify data matches
    assert cached_df is not None, "Cache load failed"
    assert len(cached_df) == len(df), "Cache size mismatch"
    print("   ✓ Data verified")

    print("\n✅ YFinance cache test PASSED")


def test_rnn_cache():
    """Test RNN caching."""
    print("\n" + "=" * 60)
    print("TESTING RNN CACHE")
    print("=" * 60)

    # Create dummy RNN predictions
    n_timesteps = 100
    feature_dict = {
        'rnn_mu_1d': np.random.randn(n_timesteps),
        'rnn_sigma_1d': np.abs(np.random.randn(n_timesteps)),
        'rnn_prob_up_1d': np.random.rand(n_timesteps),
        'rnn_mu_5d': np.random.randn(n_timesteps),
        'rnn_sigma_5d': np.abs(np.random.randn(n_timesteps)),
        'rnn_prob_up_5d': np.random.rand(n_timesteps),
    }

    result = {'feature_dict': feature_dict}

    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    window_size = 50
    epochs = 20
    probabilistic = True
    data_hash = "abc12345"

    # Save to cache
    print(f"\n1. Saving {ticker} RNN predictions to cache...")
    save_rnn_cache(
        result=result,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        epochs=epochs,
        probabilistic=probabilistic,
        data_hash=data_hash
    )
    print("   ✓ Saved")

    # Load from cache
    print(f"\n2. Loading {ticker} RNN predictions from cache...")
    cached_result = load_rnn_cache(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        epochs=epochs,
        probabilistic=probabilistic,
        data_hash=data_hash
    )
    print(f"   ✓ Loaded: {len(cached_result['feature_dict'])} features")

    # Verify data matches
    assert cached_result is not None, "Cache load failed"
    assert 'feature_dict' in cached_result, "Missing feature_dict"
    assert len(cached_result['feature_dict']) == len(feature_dict), "Feature count mismatch"
    print("   ✓ Data verified")

    # Test cache invalidation (wrong data hash)
    print(f"\n3. Testing cache invalidation (wrong data hash)...")
    invalid_result = load_rnn_cache(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        epochs=epochs,
        probabilistic=probabilistic,
        data_hash="different_hash"
    )
    assert invalid_result is None, "Cache should be invalidated"
    print("   ✓ Cache correctly invalidated")

    print("\n✅ RNN cache test PASSED")


def test_cache_stats():
    """Test cache statistics."""
    print("\n" + "=" * 60)
    print("TESTING CACHE STATS")
    print("=" * 60)

    stats = get_cache_stats()
    print(f"\nYFinance Cache: {stats['yfinance_cache']['count']} files, {stats['yfinance_cache']['size_mb']} MB")
    print(f"RNN Cache: {stats['rnn_cache']['count']} files, {stats['rnn_cache']['size_mb']} MB")
    print(f"Total: {stats['total_size_mb']} MB")

    assert stats['yfinance_cache']['count'] >= 1, "Expected at least 1 yfinance cache file"
    assert stats['rnn_cache']['count'] >= 1, "Expected at least 1 RNN cache file"

    print("\n✅ Cache stats test PASSED")


def main():
    print("\n" + "=" * 60)
    print("CACHE IMPLEMENTATION TEST SUITE")
    print("=" * 60)

    # Clear cache before testing
    print("\nClearing existing cache...")
    clear_cache("all")

    # Run tests
    test_yfinance_cache()
    test_rnn_cache()
    test_cache_stats()

    # Final summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)

    # Show final stats
    stats = get_cache_stats()
    print(f"\nFinal cache size: {stats['total_size_mb']} MB")
    print("\nCache is ready for use!")
    print("\nTo clear cache, run:")
    print("  python -m StockProphet.multiticker_refactor.cache_cli --clear all")
    print()


if __name__ == "__main__":
    main()
