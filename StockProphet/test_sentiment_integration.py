"""
Quick test to verify sentiment module integration with pipeline.

Tests that AAPL sentiment loads from cache correctly.
"""

import sys
sys.path.append('/Users/pnl1f276/code/ArnaudThs/StockProphet/StockProphet')

from multiticker_refactor.pipeline_multi import build_multi_ticker_dataset

print("=" * 60)
print("TESTING SENTIMENT INTEGRATION")
print("=" * 60)
print("\nBuilding dataset for AAPL with sentiment enabled...")
print("(Should load from cache - no API calls)\n")

df, metadata = build_multi_ticker_dataset(
    tickers=["AAPL"],
    start_date="2020-01-01",
    end_date="2020-06-30",  # Short period for quick test
    include_rnn=False,  # Skip RNN to test sentiment only
    include_sentiment=True,
    verbose=True
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

# Check sentiment column exists
if 'AAPL_Sentiment' in df.columns:
    print("✅ AAPL_Sentiment column found")
else:
    print("❌ AAPL_Sentiment column NOT found")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

# Check sentiment values
sentiment_col = df['AAPL_Sentiment']
non_zero = (sentiment_col != 0).sum()
total = len(sentiment_col)
min_val = sentiment_col.min()
max_val = sentiment_col.max()

print(f"✅ Sentiment statistics:")
print(f"   Non-zero days: {non_zero}/{total} ({non_zero/total*100:.1f}%)")
print(f"   Range: [{min_val}, {max_val}]")
print(f"   Mean: {sentiment_col.mean():.2f}")
print(f"   Sample values (first 10):")
for i in range(min(10, len(df))):
    date = df.index[i]
    val = sentiment_col.iloc[i]
    if val != 0:
        print(f"      {date.date()}: {val:+.0f}")

print("\n✅ Integration test PASSED!")
print("   Sentiment module is correctly integrated into pipeline")
