"""
Cache AAPL sentiment from existing database.

Reads from AAPL_daily_sentiment_v2.db and converts to the cache format
used by the sentiment module.
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Paths
DB_PATH = "Project/AAPL_daily_sentiment_v2.db"
CACHE_DIR = Path("multiticker_refactor/sentiment_cache")
OUTPUT_PATH = CACHE_DIR / "AAPL_sentiment_processed.parquet"

print("=" * 60)
print("CACHING AAPL SENTIMENT FROM DATABASE")
print("=" * 60)

# Create cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Read from database
print(f"\nReading from {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)
df_db = pd.read_sql_query("SELECT * FROM daily_sentiment ORDER BY date", conn)
conn.close()

print(f"Loaded {len(df_db)} rows from database")
print(f"Date range: {df_db['date'].min()} to {df_db['date'].max()}")
print(f"Non-zero days: {(df_db['daily_sentiment'] != 0).sum()}")

# Convert to sentiment module format
# The database has daily aggregated sentiment already
# We need to create a format compatible with the sentiment module's cache

# Create a fake processed sentiment DataFrame
# Since we already have daily aggregates, we'll create one "article" per day
# with the aggregate sentiment as if it came from that many articles

print("\nConverting to cache format...")

# Expand: for each day with sentiment != 0, create pseudo-articles
rows = []
for idx, row in df_db.iterrows():
    date_str = row['date']
    sentiment_sum = int(row['daily_sentiment'])

    if sentiment_sum != 0:
        # Create pseudo-articles to represent the aggregate
        # If sentiment is +3, create 3 positive articles
        # If sentiment is -2, create 2 negative articles

        if sentiment_sum > 0:
            # Positive articles
            for i in range(sentiment_sum):
                rows.append({
                    'id': f"db_cached_{date_str}_{i}",
                    'published_utc': pd.to_datetime(date_str),
                    'sentiment_label': 'positive',
                    'sentiment_score': 1.0
                })
        else:
            # Negative articles
            for i in range(abs(sentiment_sum)):
                rows.append({
                    'id': f"db_cached_{date_str}_{i}",
                    'published_utc': pd.to_datetime(date_str),
                    'sentiment_label': 'negative',
                    'sentiment_score': 1.0
                })

df_cache = pd.DataFrame(rows)

print(f"Created {len(df_cache)} pseudo-article records")
print(f"Positive: {(df_cache['sentiment_label'] == 'positive').sum()}")
print(f"Negative: {(df_cache['sentiment_label'] == 'negative').sum()}")

# Save to cache
print(f"\nSaving to {OUTPUT_PATH}...")
df_cache.to_parquet(OUTPUT_PATH, index=False)

print("\n✅ AAPL sentiment cached successfully!")
print(f"   Location: {OUTPUT_PATH}")
print(f"   The sentiment module will now use this cache for AAPL")
print("\n" + "=" * 60)
