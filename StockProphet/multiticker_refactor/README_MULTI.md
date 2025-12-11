# Multi-Ticker Trading System

Stock prediction and trading system for multiple tickers (up to 3) with:
- Multi-asset continuous environment (explicit cash allocation)
- Per-ticker LSTM predictions (probabilistic multi-horizon)
- Per-ticker normalization (StandardScaler fit on train only)
- Feature-type grouping for cross-ticker learning
- PPO with MlpLstmPolicy (recurrent temporal learning)

## Quick Start

```bash
# Train on 3 tickers (default: AAPL, MSFT, GOOGL)
cd StockProphet
python -m multiticker_refactor.main_multi --mode train --timesteps 200000

# Evaluate trained model
python -m multiticker_refactor.main_multi --mode evaluate

# Full pipeline (train + evaluate)
python -m multiticker_refactor.main_multi --mode full --timesteps 200000

# Custom tickers
python -m multiticker_refactor.main_multi --mode train --tickers AAPL TSLA NVDA --timesteps 200000
```

## Architecture

```
Multi-Ticker Pipeline (pipeline_multi.py)
├── Download OHLCV for all tickers
├── Align data (inner join, 20% threshold)
├── Add technical indicators (per ticker)
├── Add calendar/macro features (shared)
├── Normalize per ticker (fit on train only)
├── Train RNNs in parallel (one per ticker)
├── Add RNN predictions
├── Add sentiment (optional)
└── Regroup features by type
        ↓
Multi-Asset Environment (envs/multi_asset_env.py)
├── Action: Box(-1, 1, shape=(n_tickers+1,))
│   - First n_tickers: position weights (signed)
│   - Last: cash allocation
│   - Normalized by sum(abs(action)) = 1.0
├── Observation: Box(-inf, inf, shape=(n_features + n_tickers + 1,))
│   - Market features (prices, indicators, RNN preds)
│   - Portfolio state: [pos_weight_1, ..., pos_weight_n, cash_frac]
└── Reward: Risk-adjusted (Sharpe-like)
        ↓
PPO Agent (stable-baselines3)
├── MlpLstmPolicy (internal LSTM for temporal learning)
├── Entropy schedule (0.05 → 0.01)
└── Risk-adjusted rewards
```

## Key Design Decisions

### 1. Action Space (Explicit Cash Allocation)
- **Dimensions**: `(n_tickers + 1,)` = 4 for 3 tickers
- **Components**: `[pos_AAPL, pos_MSFT, pos_GOOGL, cash]`
- **Normalization**: `action / sum(abs(action))` ensures sum = 1.0
- **Shorts**: Negative weights → inverted returns

### 2. Portfolio Calculation
```python
# Step 1: Portfolio value BEFORE rebalancing
portfolio_before = cash + sum(shares * prices)

# Step 2: Convert weights to shares
target_dollars = action_weights[:n_tickers] * portfolio_before
target_shares = target_dollars / prices
target_cash = action_weights[-1] * portfolio_before

# Step 3: Apply fees (NET trades only)
for i in range(n_tickers):
    if net_trade_shares[i] != 0:
        fee = abs(net_trade_shares[i]) * prices[i] * FEE_RATE
        cash -= fee

# Step 4: Apply short borrow costs (daily)
for i in range(n_tickers):
    if shares[i] < 0:
        borrow_cost = abs(shares[i]) * prices[i] * BORROW_RATE
        cash -= borrow_cost
```

### 3. Observation Space (No Redundancy)
- **Market features**: All tickers' prices, indicators, RNN predictions
- **Portfolio state**: `[pos_weight_1, ..., pos_weight_n, cash_frac]` (4 features)
- **No portfolio_frac**: Prevents information leakage, agent learns value relationships

### 4. Data Pipeline Critical Fixes
1. **Ticker alignment**: Inner join with 20% loss threshold + quality checks
2. **Normalization**: Per-ticker StandardScaler, fit on train only (prevent leakage)
3. **Train/Val/Test**: 60/20/20 split for hyperparameter tuning
4. **Feature naming**: `{TICKER}_{feature}` convention
5. **Pipeline ordering**: `build → regroup → prepare` (MUST follow this order)
6. **Close prices**: INCLUDED in signal_features (agent trades after close)

### 5. RNN Training
- **Per-ticker models**: One LSTM per ticker (ticker-specific patterns)
- **Parallel training**: multiprocessing.Pool for 2.5-3× speedup
- **Probabilistic output**: μ, σ, prob_up for t+1 and t+5 horizons

## File Structure

```
multiticker_refactor/
├── README_MULTI.md              # This file
├── config.py                    # Configuration (tickers, dates, hyperparameters)
├── pipeline_multi.py            # Multi-ticker data pipeline
├── main_multi.py                # Main entry point
├── data/
│   ├── downloader.py            # yfinance data fetching
│   ├── features.py              # Technical indicators, calendar features
│   └── sentiment.py             # Massive API sentiment
├── models/
│   ├── rnn.py                   # LSTM models (simple + probabilistic)
│   └── ppo.py                   # PPO model creation
├── envs/
│   ├── multi_asset_env.py       # Multi-ticker continuous environment
│   └── trading_env.py           # Single-ticker environment (legacy)
└── saved_models_multi/          # Trained models, VecNormalize stats
```

## Configuration (config.py)

Key parameters:
```python
TICKERS = ["AAPL", "MSFT", "GOOGL"]  # Max 3 for demo
MAX_TICKERS = 3
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# Train/Val/Test Split
PPO_TRAIN_RATIO = 0.6  # 60% training
PPO_VAL_RATIO = 0.2    # 20% validation (for hyperparameter tuning)
# Test: 20% (final evaluation)

# Environment
INITIAL_CAPITAL = 10_000  # Starting capital ($)
ENV_TYPE = 'continuous'   # Use multi_asset_env

# PPO
PPO_TIMESTEPS = 200_000
PPO_WINDOW_SIZE = 10  # Not used with MlpLstmPolicy

# RNN
LSTM_WINDOW_SIZE = 30
PROB_LSTM_HORIZONS = [1, 5]  # 1-day and 5-day predictions

# Data Alignment
MAX_DATA_LOSS_PCT = 0.20  # Maximum 20% data loss from inner join
```

## Command Line Options

```bash
# Mode selection
--mode train|evaluate|full  # Operating mode

# Tickers
--tickers AAPL MSFT GOOGL   # List of tickers (max 3)

# Training
--timesteps 200000          # Number of PPO training steps

# Date range
--start-date 2019-01-01
--end-date 2024-12-31

# Feature flags
--no-rnn                    # Skip RNN training
--no-sentiment              # Skip sentiment data
--simple-rnn                # Use simple RNN (not probabilistic)
--recurrent                 # Use RecurrentPPO (requires sb3-contrib)
```

## Output Files

After training, these files are created:

```
saved_models_multi/
├── ppo_multi_trading.zip       # Trained PPO model
├── vec_normalize_multi.pkl     # VecNormalize stats (obs normalization)
└── metadata_multi.npy          # Metadata (scalers, RNN models, etc.)

ppo_multi_best_model/
└── best_model.zip              # Best model (by validation reward)

ppo_multi_checkpoints/
├── ppo_multi_trading_model_10000_steps.zip
├── ppo_multi_trading_model_20000_steps.zip
└── ...                         # Checkpoints every 10k steps

ppo_multi_logs/
└── PPO_1/                      # TensorBoard logs

ppo_multi_eval_logs/
└── evaluations.npz             # Validation set evaluation results
```

## TensorBoard Monitoring

```bash
tensorboard --logdir=./ppo_multi_logs/
```

## Trading Timeline

**Important**: Agent trades AFTER market close, BEFORE next open.

```
Day 1: Market closes at 4pm → Agent sees close price → Agent rebalances portfolio
Day 2: Market opens at 9:30am → Portfolio executes at open prices → Day continues
Day 2: Market closes at 4pm → Agent sees new close price → Agent rebalances again
```

This means:
- Agent observations INCLUDE today's close price
- Agent makes decisions based on all info up to and including today's close
- Trades execute at next open (not modeled explicitly, assumed instant)

## Example Workflow

```bash
# 1. Train on default tickers (AAPL, MSFT, GOOGL)
python -m multiticker_refactor.main_multi --mode train --timesteps 200000

# 2. Monitor training
tensorboard --logdir=./ppo_multi_logs/

# 3. Evaluate on test set
python -m multiticker_refactor.main_multi --mode evaluate

# 4. Try different tickers
python -m multiticker_refactor.main_multi --mode full --tickers AAPL TSLA NVDA --timesteps 200000
```

## Differences from Single-Ticker (`project_refactored`)

| Feature | Single-Ticker | Multi-Ticker |
|---------|--------------|--------------|
| **Tickers** | 1 (AAPL) | Up to 3 |
| **Action space** | Discrete or Box(-1,1) | Box(-1,1, shape=(n+1)) |
| **Cash** | Implicit | Explicit (agent controls) |
| **Normalization** | Single scaler | Per-ticker scalers |
| **RNN training** | Sequential | Parallel (multiprocessing) |
| **Feature grouping** | N/A | By type (cross-ticker learning) |
| **Data alignment** | N/A | Inner join with 20% threshold |
| **Portfolio calc** | Simple | Weight → shares conversion |

## Troubleshooting

### Error: "Data alignment would lose >20% of data"
**Cause**: Tickers have very different date ranges (e.g., one IPO'd recently)
**Fix**: Choose tickers with similar listing dates, or adjust START_DATE in config.py

### Error: "Missing close price columns"
**Cause**: Pipeline ordering issue - forgot to call `regroup_features_by_type()`
**Fix**: Always follow order: `build → regroup → prepare`

### Error: "sb3-contrib not installed" (when using --recurrent)
**Fix**: Install with `pip install sb3-contrib` or remove --recurrent flag

### Warning: "Maximum date gap is X days"
**Cause**: Market holidays or data gaps
**Impact**: Usually harmless if < 10 days

## Performance Tips

1. **Start small**: Test with 100k timesteps first, then scale to 200k
2. **Use TensorBoard**: Monitor `train/entropy_loss` and `eval/mean_reward`
3. **Adjust entropy schedule**: If agent converges too fast, increase initial entropy
4. **Feature selection**: After initial run, consider reducing from ~160 to ~12 essential features
5. **Hyperparameter tuning**: Use validation set (20%) to tune learning rate, n_steps, etc.

## Next Steps

1. **Feature selection**: Reduce from ~160 to ~12 essential features (40-60% faster training)
2. **Hyperparameter tuning**: Use validation set to optimize PPO hyperparameters
3. **Streamlit dashboard**: Visualize trades, portfolio evolution, RNN predictions
4. **Extended evaluation**: Sharpe ratio, maximum drawdown, vs Buy & Hold comparison

---

For questions or issues, see the main [CLAUDE.md](../CLAUDE.md) for full design documentation.
