"""
Cache management CLI utility.

Usage:
    # Show cache statistics
    python -m multiticker_refactor.cache_cli --stats

    # Clear all caches
    python -m multiticker_refactor.cache_cli --clear all

    # Clear only yfinance cache
    python -m multiticker_refactor.cache_cli --clear yfinance

    # Clear only RNN cache
    python -m multiticker_refactor.cache_cli --clear rnn

    # Clear only pipeline cache
    python -m multiticker_refactor.cache_cli --clear pipeline
"""
import argparse
from .data.cache import clear_cache, get_cache_stats


def main():
    parser = argparse.ArgumentParser(description="Cache management utility")

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics"
    )

    parser.add_argument(
        "--clear",
        type=str,
        choices=["all", "yfinance", "rnn", "pipeline"],
        help="Clear cache (all/yfinance/rnn/pipeline)"
    )

    args = parser.parse_args()

    if args.stats:
        stats = get_cache_stats()
        print("\n" + "=" * 60)
        print("CACHE STATISTICS")
        print("=" * 60)
        print(f"\nYFinance Cache:")
        print(f"  Files: {stats['yfinance_cache']['count']}")
        print(f"  Size:  {stats['yfinance_cache']['size_mb']} MB")
        print(f"\nRNN Cache:")
        print(f"  Files: {stats['rnn_cache']['count']}")
        print(f"  Size:  {stats['rnn_cache']['size_mb']} MB")
        print(f"\nPipeline Cache:")
        print(f"  Files: {stats['pipeline_cache']['count']}")
        print(f"  Size:  {stats['pipeline_cache']['size_mb']} MB")
        print(f"\nTotal Cache Size: {stats['total_size_mb']} MB")
        print("=" * 60 + "\n")

    if args.clear:
        print()
        clear_cache(args.clear)
        print()


if __name__ == "__main__":
    main()
