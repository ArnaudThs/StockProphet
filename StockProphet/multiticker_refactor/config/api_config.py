"""
API keys and credentials configuration.

SECURITY NOTE: Never commit real API keys to version control.
Use environment variables or .env file for actual credentials.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from multiticker_refactor directory
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# =============================================================================
# API KEYS
# =============================================================================

# Polygon API - for fetching news articles (sentiment analysis)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Validate required keys
if not POLYGON_API_KEY:
    import warnings
    warnings.warn(
        "POLYGON_API_KEY not set. Sentiment features will be unavailable. "
        "Set POLYGON_API_KEY environment variable or create .env file.",
        UserWarning
    )
