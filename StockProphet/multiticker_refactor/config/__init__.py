"""
Configuration package for StockProphet multi-ticker system.

This package organizes configuration into logical modules:
- ticker_config.py: Ticker selection and date ranges
- model_config.py: LSTM and PPO hyperparameters
- env_config.py: Trading environment parameters
- paths.py: File paths and directories
- api_config.py: API keys and credentials
"""

from .ticker_config import *
from .model_config import *
from .env_config import *
from .paths import *
from .api_config import *

__all__ = [
    # Ticker configuration
    'TARGET_TICKER', 'SUPPORT_TICKERS', 'TICKERS', 'MAX_TICKERS',
    'START_DATE', 'END_DATE',
    'INCLUDE_SENTIMENT', 'SENTIMENT_START_DATE', 'SENTIMENT_END_DATE',

    # Model configuration
    'LSTM_WINDOW_SIZE', 'LSTM_EPOCHS', 'LSTM_BATCH_SIZE', 'LSTM_TRAIN_RATIO',
    'PROB_LSTM_HORIZONS', 'PROB_LSTM_EPOCHS', 'PROB_LSTM_UNITS',
    'PPO_WINDOW_SIZE', 'PPO_TIMESTEPS', 'PPO_TRAIN_RATIO', 'PPO_VAL_RATIO',
    'PPO_LEARNING_RATE', 'PPO_N_STEPS', 'PPO_BATCH_SIZE', 'PPO_N_EPOCHS',
    'PPO_GAMMA', 'PPO_GAE_LAMBDA', 'PPO_CLIP_RANGE',
    'PPO_ENT_COEF_START', 'PPO_ENT_COEF_END',
    'PPO_EVAL_FREQ', 'PPO_CHECKPOINT_FREQ',
    'VECNORMALIZE_NORM_OBS', 'VECNORMALIZE_NORM_REWARD',
    'VECNORMALIZE_CLIP_OBS', 'VECNORMALIZE_CLIP_REWARD',

    # Environment configuration
    'ENV_TYPE', 'CONTINUOUS_ENV_VERSION',
    'INITIAL_CAPITAL', 'TRANSACTION_FEE_PCT', 'SHORT_BORROW_RATE',
    'REWARD_VOLATILITY_WINDOW',
    'TREND_REWARD_MULTIPLIER', 'CONVICTION_REWARD', 'EXIT_TIMING_REWARD', 'PATIENCE_REWARD',
    'CONTINUOUS_ENV_CONFIG', 'REWARD_CONFIG',
    'MIN_HISTORY', 'HORIZON', 'USE_HMM', 'FILLER',
    'MAX_DATA_LOSS_PCT',

    # Paths
    'BASE_DIR', 'MODELS_DIR', 'DATA_DIR', 'DOCS_DIR',
    'LSTM_MODEL_PATH', 'PROB_LSTM_MODEL_PATH',
    'PPO_MODEL_PATH', 'RECURRENT_PPO_MODEL_PATH', 'VEC_NORMALIZE_PATH',
    'SENTIMENT_CACHE_DIR', 'FEATURE_DOC_PATH',

    # API configuration
    'API_KEY_MASSIVE',
]
