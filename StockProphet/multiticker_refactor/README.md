# StockProphet Multi-Ticker Refactor

Clean, modular implementation of the StockProphet trading system with support for both single-ticker and multi-ticker portfolios.

## Quick Start

```bash
# Single-ticker training
python -m multiticker_refactor.main --mode train --timesteps 200000

# Multi-ticker training
python -m multiticker_refactor.main_multi --mode train --timesteps 200000

# Evaluate trained model
python -m multiticker_refactor.main --mode evaluate
```

## Project Structure

```
multiticker_refactor/
├── config/                    # Configuration (organized by category)
│   ├── __init__.py           # Main config exports
│   ├── ticker_config.py      # Ticker selection, date ranges
│   ├── model_config.py       # LSTM & PPO hyperparameters
│   ├── env_config.py         # Trading environment parameters
│   ├── paths.py              # File paths
│   └── api_config.py         # API keys (sentiment data)
│
├── data/                      # Data pipeline modules
│   ├── __init__.py           # Data module exports
│   ├── downloader.py         # YFinance data fetching
│   ├── features.py           # Technical indicators & feature engineering
│   └── cache.py              # Pipeline caching (YFinance + RNN predictions)
│
├── models/                    # ML models
│   ├── __init__.py           # Model exports
│   ├── rnn.py                # LSTM price prediction (simple + probabilistic)
│   └── ppo.py                # PPO reinforcement learning agent
│
├── envs/                      # Trading environments
│   ├── __init__.py           # Environment exports
│   └── trading_env.py        # UnifiedTradingEnv (single + multi-ticker)
│
├── feature_selection/         # Feature selection tools
├── sentiment/                 # Sentiment analysis (optional)
├── streamlit_demo/            # Demo dashboard
│
├── pipeline.py                # Single-ticker data pipeline
├── pipeline_multi.py          # Multi-ticker data pipeline
├── main.py                    # Single-ticker entry point
├── main_multi.py              # Multi-ticker entry point
└── evaluate.py                # Model evaluation utilities
```

## Key Modules

### config/ - All configuration
- ticker_config.py: Which tickers, date ranges
- model_config.py: LSTM/PPO hyperparameters
- env_config.py: Trading settings (fees, capital, rewards)

### data/ - Data pipeline
- downloader.py: YFinance OHLCV data
- features.py: Technical indicators
- cache.py: Caching (YFinance, RNN predictions)

### models/ - Machine learning
- rnn.py: LSTM price prediction
- ppo.py: PPO agent training

### envs/ - Trading environment
- trading_env.py: UnifiedTradingEnv (single + multi-ticker)
  - Continuous action space
  - Risk-adjusted rewards
  - Transaction fees, short borrowing
  - Bankruptcy termination

## Usage Examples

### Single-Ticker
```bash
python -m multiticker_refactor.main --mode train --timesteps 200000
python -m multiticker_refactor.main --mode evaluate
```

### Multi-Ticker
```bash
python -m multiticker_refactor.main_multi --mode train --tickers AAPL MSFT GOOGL
python -m multiticker_refactor.main_multi --mode evaluate
```

### Feature Selection
```bash
python -m multiticker_refactor.feature_selection.main --ticker AAPL --stage full
```

## Configuration

Edit `config/` files:

```python
# config/ticker_config.py
TARGET_TICKER = "AAPL"
TICKERS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = "2020-01-01"

# config/model_config.py
PPO_TIMESTEPS = 200_000
PPO_TRAIN_RATIO = 0.6  # 60% train, 20% val, 20% test

# config/env_config.py
INITIAL_CAPITAL = 10_000.0
TRANSACTION_FEE_PCT = 0.001  # 0.1%
```

## Output

Trained models saved to `saved_models/`:
- ppo_trading.zip
- vec_normalize.pkl
- prob_lstm.keras

---

## Design Notes

### Why Two Pipelines?

**pipeline.py** (single-ticker):
- Simple, straightforward data flow
- Single RNN model
- Used by `main.py`

**pipeline_multi.py** (multi-ticker):
- Complex: data alignment, per-ticker normalization
- Parallel RNN training (multiprocessing)
- Feature regrouping for cross-ticker learning
- Used by `main_multi.py` and feature selection

Consolidating would create a messy file with complex branching logic. Keeping them separate makes each easier to understand.
