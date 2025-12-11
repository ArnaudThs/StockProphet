"""
Trading environment configuration.
"""
import os

# Environment type: 'discrete' or 'continuous'
ENV_TYPE = os.getenv("ENV_TYPE", "continuous")

# Continuous environment version: 'v1' (basic) or 'v2' (trend-adaptive rewards)
CONTINUOUS_ENV_VERSION = 'v2'  # Set to 'v2' to enable trend-following reward structure

# =============================================================================
# CONTINUOUS ENVIRONMENT PARAMETERS
# =============================================================================
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000.0"))

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
