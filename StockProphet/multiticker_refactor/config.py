"""
Configuration parameters for the StockProphet pipeline.
"""
from pathlib import Path

# =============================================================================
# TARGET CONFIGURATION
# =============================================================================
TARGET_TICKER = "AAPL"
SUPPORT_TICKERS = []  # Empty for now, can add supporting tickers later

# Multi-ticker configuration (for dashboard demo)
TICKERS = ["AAPL", "MSFT", "GOOGL"]  # Maximum 3 tickers for demo
MAX_TICKERS = 3  # Hard limit for demo performance

# =============================================================================
# DATE RANGES
# =============================================================================
START_DATE = "2020-01-01"
END_DATE = "2025-06-30"

# Sentiment configuration
INCLUDE_SENTIMENT = False  # Set to True to include sentiment features
SENTIMENT_START_DATE = "2024-01-01"  # Sentiment API only available from 2024
SENTIMENT_END_DATE = "2025-06-30"

# =============================================================================
# RNN (LSTM) PARAMETERS
# =============================================================================
LSTM_WINDOW_SIZE = 50  # Sequence length for LSTM
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32
LSTM_TRAIN_RATIO = 0.7  # 70% train, 30% test for LSTM

# Probabilistic Multi-Horizon LSTM
PROB_LSTM_HORIZONS = [1, 5]  # Prediction horizons in days (t+1, t+5)
PROB_LSTM_EPOCHS = 30
PROB_LSTM_UNITS = 64

# =============================================================================
# PPO PARAMETERS
# =============================================================================
PPO_WINDOW_SIZE = 10  # Not used with MlpLstmPolicy (policy maintains internal LSTM state)
PPO_TIMESTEPS = 200_000

# Train/Val/Test Split (60/20/20)
PPO_TRAIN_RATIO = 0.6  # 60% for training (changed from 0.8)
PPO_VAL_RATIO = 0.2    # 20% for validation (hyperparameter tuning)
# Test implicit: 20% = 1 - PPO_TRAIN_RATIO - PPO_VAL_RATIO

# PPO Hyperparameters (optimized for trading)
PPO_LEARNING_RATE = 3e-4       # Learning rate
PPO_N_STEPS = 512              # Steps per update (more frequent updates for market responsiveness)
PPO_BATCH_SIZE = 64            # Minibatch size
PPO_N_EPOCHS = 10              # Gradient descent epochs per update
PPO_GAMMA = 0.99               # Discount factor
PPO_GAE_LAMBDA = 0.95          # GAE lambda
PPO_CLIP_RANGE = 0.2           # PPO clipping parameter
PPO_ENT_COEF_START = 0.05      # Initial entropy coefficient (exploration)
PPO_ENT_COEF_END = 0.01        # Final entropy coefficient (convergence)

# Evaluation and checkpointing
PPO_EVAL_FREQ = 10_000         # Evaluate on validation set every N steps
PPO_CHECKPOINT_FREQ = 10_000   # Save checkpoint every N steps

# VecNormalize configuration
VECNORMALIZE_NORM_OBS = True   # Normalize observations
VECNORMALIZE_NORM_REWARD = False  # DON'T normalize rewards (masks learning signal)
VECNORMALIZE_CLIP_OBS = 5.0    # Clip observations to [-5, 5] after normalization
VECNORMALIZE_CLIP_REWARD = None  # Don't clip rewards

# Environment type: 'discrete' or 'continuous'
ENV_TYPE = 'continuous'

# Continuous environment version: 'v1' (basic) or 'v2' (trend-adaptive rewards)
CONTINUOUS_ENV_VERSION = 'v2'  # Set to 'v2' to enable trend-following reward structure

# =============================================================================
# CONTINUOUS ENVIRONMENT PARAMETERS
# =============================================================================
INITIAL_CAPITAL = 10_000.0  # Starting capital in dollars

# Transaction costs
TRANSACTION_FEE_PCT = 0.001  # 0.1% fee on trade value (applied to NET trades only)
SHORT_BORROW_RATE = 0.0003   # 0.03% daily borrow cost for short positions

# Risk management
REWARD_VOLATILITY_WINDOW = 30  # Window for computing recent volatility (Sharpe-like reward)

# V2 Reward parameters (trend-adaptive behavior)
TREND_REWARD_MULTIPLIER = 2.0   # Bonus for trading WITH detected trends
CONVICTION_REWARD = 0.5         # Bonus for large positions when profitable
EXIT_TIMING_REWARD = 1.0        # Bonus for exiting before reversals
PATIENCE_REWARD = 0.2           # Reward for staying flat in noisy periods

# Continuous environment reward/cost configuration (legacy dict for compatibility)
CONTINUOUS_ENV_CONFIG = {
    'fee': TRANSACTION_FEE_PCT,
    'short_borrow_rate': SHORT_BORROW_RATE,
}

# =============================================================================
# DISCRETE ENVIRONMENT PARAMETERS (Legacy)
# =============================================================================
# Reward configuration for discrete FlexibleTradingEnv
REWARD_CONFIG = {
    'fee': 0.0005,              # 0.05% per trade
    'holding_cost': 0.0,
    'short_borrow_cost': 0.0,
    'reward_scaling': 100.0,    # Scale up tiny log-returns
    'trade_penalty': 0.001,     # Reduce churning
    'profit_bonus': 0.5,        # 50% bonus on profits
    'trend_following_bonus': 0.0001,
}

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
MIN_HISTORY = 102  # Minimum history for technicals (at least 51 for indicators)
HORIZON = 30
USE_HMM = False  # HMM feature disabled by default
FILLER = 99999   # Sentinel for missing macro distances

# Multi-ticker data alignment
MAX_DATA_LOSS_PCT = 0.20  # Maximum acceptable data loss in inner join (20%)

# =============================================================================
# API KEYS
# =============================================================================
API_KEY_MASSIVE = "SiV7GQdKTF2ZtrAr1xNSrnNYP11dKCAC"

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
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
