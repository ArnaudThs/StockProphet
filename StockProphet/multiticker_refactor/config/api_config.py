"""
API keys and credentials configuration.

SECURITY NOTE: Never commit real API keys to version control.
Use environment variables or .env file for actual credentials.
"""
import os

# =============================================================================
# API KEYS
# =============================================================================
API_KEY_MASSIVE = os.getenv("API_KEY_MASSIVE", "")

# Validate required keys
if not API_KEY_MASSIVE:
    import warnings
    warnings.warn(
        "API_KEY_MASSIVE not set. Sentiment features will be unavailable. "
        "Set API_KEY_MASSIVE environment variable or create .env file.",
        UserWarning
    )
