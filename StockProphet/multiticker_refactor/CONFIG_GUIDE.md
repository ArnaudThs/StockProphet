# Configuration Guide

All system parameters are centralized in `config.py` for easy tuning.

## Configuration Organization

### 1. Target Configuration
```python
TARGET_TICKER = "AAPL"              # Single-ticker mode (legacy)
SUPPORT_TICKERS = []                # Supporting tickers (legacy)

# Multi-ticker configuration
TICKERS = ["AAPL", "MSFT", "GOOGL"]  # List of tickers to trade (max 3)
MAX_TICKERS = 3                      # Hard limit for demo performance
```

**Usage**: Multi-ticker system trades all tickers in `TICKERS` simultaneously.

---

### 2. Date Ranges
```python
START_DATE = "2020-01-01"           # Data start date
END_DATE = "2025-06-30"             # Data end date

# Sentiment API (only available from 2024)
SENTIMENT_START_DATE = "2024-01-01"
SENTIMENT_END_DATE = "2025-06-30"
```

**Note**: ~5 years of data = ~1250 trading days per ticker

---

### 3. RNN (LSTM) Parameters
```python
LSTM_WINDOW_SIZE = 50               # Sequence length for LSTM training
LSTM_EPOCHS = 20                    # Training epochs (simple RNN)
LSTM_BATCH_SIZE = 32                # Batch size
LSTM_TRAIN_RATIO = 0.7              # 70% train, 30% test for RNN

# Probabilistic Multi-Horizon LSTM
PROB_LSTM_HORIZONS = [1, 5]         # Prediction horizons: t+1 and t+5 days
PROB_LSTM_EPOCHS = 30               # More epochs for probabilistic model
PROB_LSTM_UNITS = 64                # LSTM units
```

**Output per ticker**:
- `{TICKER}_rnn_mu_1d`, `{TICKER}_rnn_sigma_1d`, `{TICKER}_rnn_prob_up_1d`
- `{TICKER}_rnn_mu_5d`, `{TICKER}_rnn_sigma_5d`, `{TICKER}_rnn_prob_up_5d`
- `{TICKER}_rnn_confidence_1d` = 1 / (sigma + 1e-8)

---

### 4. PPO Training Parameters
```python
PPO_WINDOW_SIZE = 10                # Not used with MlpLstmPolicy
PPO_TIMESTEPS = 200_000             # Total training steps

# Train/Val/Test Split (60/20/20)
PPO_TRAIN_RATIO = 0.6               # 60% for training
PPO_VAL_RATIO = 0.2                 # 20% for validation (hyperparameter tuning)
# Test: 20% implicit (1 - train - val)
```

**Example**: 1250 days → 750 train, 250 val, 250 test

---

### 5. PPO Hyperparameters
```python
PPO_LEARNING_RATE = 3e-4            # Learning rate (standard for PPO)
PPO_N_STEPS = 512                   # Steps per update (4× more frequent than default 2048)
PPO_BATCH_SIZE = 64                 # Minibatch size
PPO_N_EPOCHS = 10                   # Gradient descent epochs per update
PPO_GAMMA = 0.99                    # Discount factor (standard)
PPO_GAE_LAMBDA = 0.95               # GAE lambda (standard)
PPO_CLIP_RANGE = 0.2                # PPO clipping parameter (standard)

# Entropy coefficient schedule (exploration decay)
PPO_ENT_COEF_START = 0.05           # Initial (high exploration)
PPO_ENT_COEF_END = 0.01             # Final (convergence)
# Actual schedule: start * (1 - progress * 0.8)
```

**Why more frequent updates?**
- Markets change quickly → agent needs to adapt faster
- `n_steps=512` means update every 512 environment steps
- Default 2048 is too slow for dynamic trading environments

---

### 6. Evaluation and Checkpointing
```python
PPO_EVAL_FREQ = 10_000              # Evaluate on validation set every N steps
PPO_CHECKPOINT_FREQ = 10_000        # Save checkpoint every N steps
```

**Result**: 20 checkpoints + 20 evaluations for 200k timesteps

---

### 7. VecNormalize Configuration
```python
VECNORMALIZE_NORM_OBS = True        # Normalize observations (CRITICAL: yes)
VECNORMALIZE_NORM_REWARD = False    # DON'T normalize rewards (masks learning signal)
VECNORMALIZE_CLIP_OBS = 5.0         # Clip observations to [-5, 5] after normalization
VECNORMALIZE_CLIP_REWARD = None     # Don't clip rewards
```

**Why not normalize rewards?**
- Reward normalization masks the learning signal
- Big wins/losses are important signals for the agent
- We want agent to see raw PnL changes

---

### 8. Environment Type
```python
ENV_TYPE = 'continuous'             # Use continuous action space
                                    # (vs 'discrete' for legacy environment)
```

---

### 9. Continuous Environment Parameters
```python
INITIAL_CAPITAL = 10_000.0          # Starting capital in dollars

# Transaction costs
TRANSACTION_FEE_PCT = 0.001         # 0.1% fee (applied to NET trades only)
SHORT_BORROW_RATE = 0.0003          # 0.03% daily borrow cost for shorts

# Risk management
REWARD_VOLATILITY_WINDOW = 30       # Window for computing recent volatility
```

**Transaction fee examples**:
- Trade $1000 → fee = $1
- Rebalance from 50% AAPL to 30% AAPL → only pay fee on NET change (20% of portfolio)

**Short borrow cost examples**:
- Short $1000 worth of MSFT → daily cost = $0.30
- Annual cost ≈ 11% (realistic for retail trading)

**Reward calculation**:
```python
pnl_pct = (portfolio_after - portfolio_before) / initial_capital
recent_volatility = std(last_30_returns)
reward = pnl_pct / (recent_volatility + 1e-8)  # Sharpe-like
```

---

### 10. Feature Engineering
```python
MIN_HISTORY = 102                   # Minimum history for technical indicators
HORIZON = 30                        # Prediction horizon (legacy, not used)
USE_HMM = False                     # HMM features disabled
FILLER = 99999                      # Sentinel for missing macro distances

# Multi-ticker data alignment
MAX_DATA_LOSS_PCT = 0.20            # Maximum 20% data loss in inner join
```

**Data alignment**:
- Inner join: Keep only dates where ALL tickers have data
- If alignment would lose >20% data → fail with error
- Prevents silent data loss from mismatched ticker date ranges

---

### 11. API Keys
```python
API_KEY_MASSIVE = "SiV7GQdKTF2ZtrAr1xNSrnNYP11dKCAC"  # Massive News API
```

**Usage**: Fetch sentiment data via Massive API (2024+ only)

---

### 12. Paths
```python
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
```

All directories created automatically on import.

---

## How Modules Use Config

### pipeline_multi.py
**Imports**:
- `MAX_TICKERS` - Validate ticker count
- `MAX_DATA_LOSS_PCT` - Data alignment threshold
- `PPO_TRAIN_RATIO`, `PPO_VAL_RATIO` - Train/val/test split
- `LSTM_WINDOW_SIZE`, `LSTM_EPOCHS`, `LSTM_BATCH_SIZE` - RNN training
- `PROB_LSTM_HORIZONS` - Multi-horizon predictions
- `API_KEY_MASSIVE`, `SENTIMENT_START_DATE`, `SENTIMENT_END_DATE` - Sentiment

**Usage**: Build multi-ticker dataset with all features

---

### main_multi.py
**Imports**:
- `TICKERS`, `START_DATE`, `END_DATE` - Default data range
- `INITIAL_CAPITAL` - Starting capital
- `TRANSACTION_FEE_PCT`, `SHORT_BORROW_RATE` - Environment costs
- `REWARD_VOLATILITY_WINDOW` - Risk-adjusted reward
- `PPO_*` - All PPO hyperparameters
- `VECNORMALIZE_*` - Observation normalization

**Usage**: Train/evaluate PPO agent with configured parameters

---

### envs/multi_asset_env.py
**Parameters accepted** (passed from main_multi.py):
- `initial_capital` ← `INITIAL_CAPITAL`
- `transaction_fee_pct` ← `TRANSACTION_FEE_PCT`
- `short_borrow_rate` ← `SHORT_BORROW_RATE`
- `reward_volatility_window` ← `REWARD_VOLATILITY_WINDOW`

**Usage**: Create environment with correct costs and reward scaling

---

## Tuning Guidelines

### For Faster Training
```python
PPO_TIMESTEPS = 100_000              # Reduce from 200k
PPO_EVAL_FREQ = 20_000               # Less frequent evaluation
PPO_CHECKPOINT_FREQ = 20_000         # Less frequent checkpoints
LSTM_EPOCHS = 15                     # Reduce from 20 (simple RNN)
PROB_LSTM_EPOCHS = 20                # Reduce from 30 (probabilistic)
```

### For Better Performance (Longer Training)
```python
PPO_TIMESTEPS = 500_000              # Increase from 200k
PPO_LEARNING_RATE = 1e-4             # Lower learning rate (more stable)
PPO_N_STEPS = 1024                   # Larger batch collection
PPO_ENT_COEF_START = 0.1             # More exploration initially
```

### For Less Risk-Averse Agent
```python
TRANSACTION_FEE_PCT = 0.0005         # Lower fees (from 0.001)
SHORT_BORROW_RATE = 0.0001           # Lower borrow cost (from 0.0003)
REWARD_VOLATILITY_WINDOW = 60        # Longer window = smoother reward
```

### For More Tickers (WARNING: Slow)
```python
MAX_TICKERS = 5                      # Increase from 3
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
# Note: 5 tickers = 5× RNN training time (even with parallelization)
```

---

## Configuration Validation

**On startup**, the system validates:
1. `len(TICKERS) <= MAX_TICKERS`
2. Date ranges valid (`START_DATE < END_DATE`)
3. Train/val/test ratios sum to ≤ 1.0
4. All required directories exist (created automatically)

**During data pipeline**:
1. Data alignment loses ≤ `MAX_DATA_LOSS_PCT` (else error)
2. All tickers have valid OHLCV data (no NaN, price > 0)
3. Normalized features are finite (no NaN/inf)

---

## Quick Reference: What to Tune First

| Goal | Parameters to Change |
|------|---------------------|
| **Faster training** | `PPO_TIMESTEPS`, `LSTM_EPOCHS`, `PROB_LSTM_EPOCHS` |
| **Better performance** | `PPO_LEARNING_RATE`, `PPO_ENT_COEF_START`, `PPO_TIMESTEPS` |
| **More exploration** | `PPO_ENT_COEF_START` (increase to 0.1) |
| **Less churning** | `TRANSACTION_FEE_PCT` (increase to 0.002) |
| **Longer-term trading** | `PROB_LSTM_HORIZONS = [1, 10, 20]` |
| **Different tickers** | `TICKERS`, `START_DATE` (ensure overlap) |
| **More data** | `START_DATE` (go back further, e.g., 2015-01-01) |

---

## Example: Quick Test Configuration

For a fast test run (verify pipeline works):

```python
# config.py modifications for quick test
PPO_TIMESTEPS = 10_000               # Just 10k steps
LSTM_EPOCHS = 5                      # Minimal RNN training
PROB_LSTM_EPOCHS = 10                # Minimal probabilistic training
PPO_EVAL_FREQ = 5_000                # Less frequent eval
PPO_CHECKPOINT_FREQ = 5_000          # Less frequent checkpoints
START_DATE = "2023-01-01"            # Just 2 years of data
```

Run: `python -m multiticker_refactor.main_multi --mode full --timesteps 10000`

Expected time: ~30-45 minutes (depends on CPU cores for parallel RNN training)

---

For full configuration reference, see [config.py](config.py).
