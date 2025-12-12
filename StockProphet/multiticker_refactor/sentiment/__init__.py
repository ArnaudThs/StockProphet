"""
Sentiment Analysis Module

Fetches news from Polygon API, analyzes sentiment with FinBERT,
and integrates into the multi-ticker pipeline.
"""

from .pipeline import add_sentiment_features

__all__ = ['add_sentiment_features']
