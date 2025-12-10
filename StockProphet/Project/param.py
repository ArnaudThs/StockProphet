START_DATE ="2010-01-01"
END_DATE ="2025-06-30"
TICKER ="AAPL"

# Massive API Key (ideally loaded from .env)
API_KEY_MASSIVE ="SiV7GQdKTF2ZtrAr1xNSrnNYP11dKCAC"

# Independent date range for news sentiment
SENTIMENT_START_DATE ="2025-03-01"

SENTIMENT_END_DATE ="2025-06-30"


# One or more tickers to extrract sentiment from Massive API
SENTIMENT_TICKERS ="AAPL"

# Maximum Massive API limit (1000)
SENTIMENT_API_LIMIT =1000

# Massive pagination available
SENTIMENT_MAX_PAGES =1  # Massive pagination available

#RNN Parameters
WINDOW_SIZE = 50
TRAIN_RATIO = 0.7
BATCH_SIZE = 32
LSTM_EPOCHS = 20
RNN_MODEL_SAVE = "lstm_rnn.keras"
RNN_MODEL_PATH = "lstm_rnn.keras"


#PPO Param
PPO_MODEL_PATH = "ppo_trader"
MOVEMENT_BONUS = 0.5
PPO_TIMESTEPS = 200_000  # Minimum!
ENT_COEF = 0.2  # Increase entropy for more exploration
CASH_PENALTY_RATE = 0.0002     # INCREASED from 0.00005
HOLDING_PENALTY = 0.15          # NEW: Penalize holding cash

# --- CONSTANTS ---
WINDOW_SIZE = 50
CASH_DEPRECIATION_RATE = 0.00005

TRANSACTION_COST_PCT = 0.0003  # REDUCED further for more trading
CASH_PENALTY_RATE = 0.0005     # INCREASED significantly
