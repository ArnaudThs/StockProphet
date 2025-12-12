"""
File paths and directory configuration.
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # multiticker_refactor/
MODELS_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data_cache"
DOCS_DIR = BASE_DIR / "docs"

# Model save paths
LSTM_MODEL_PATH = MODELS_DIR / "lstm_rnn.keras"
PROB_LSTM_MODEL_PATH = MODELS_DIR / "prob_lstm.keras"
PPO_MODEL_PATH = MODELS_DIR / "ppo_trading.zip"
RECURRENT_PPO_MODEL_PATH = MODELS_DIR / "recurrent_ppo_trading.zip"
VEC_NORMALIZE_PATH = MODELS_DIR / "vec_normalize.pkl"

# Data cache paths
SENTIMENT_CACHE_DIR = DATA_DIR / "sentiment_cache"

# Documentation
FEATURE_DOC_PATH = DOCS_DIR / "feature_documentation.md"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
