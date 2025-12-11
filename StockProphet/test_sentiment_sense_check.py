"""
Sense check FinBERT sentiment analysis.

Tests the first 100 news articles for AAPL to verify FinBERT accuracy.
"""

import sys
sys.path.append('/Users/pnl1f276/code/ArnaudThs/StockProphet/StockProphet')

from multiticker_refactor.sentiment.pipeline import sense_check_finbert
from multiticker_refactor.config import POLYGON_API_KEY, START_DATE, END_DATE

if __name__ == "__main__":
    df_review = sense_check_finbert(
        ticker="AAPL",
        n_samples=100,
        polygon_api_key=POLYGON_API_KEY,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Save for manual review
    if not df_review.empty:
        output_path = "sentiment_sense_check_results.csv"
        df_review.to_csv(output_path, index=False)
        print(f"\n✅ Full results saved to: {output_path}")
        print(f"   Review this file to validate FinBERT accuracy")
