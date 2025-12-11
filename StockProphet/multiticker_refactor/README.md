# StockProphet Multi-Ticker Trading System

Self-contained, cloud-deployable trading system combining LSTM price predictions with PPO reinforcement learning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (copy .env.example to .env and fill in values)
cp .env.example .env

# Train multi-ticker agent
python -m multiticker_refactor.main_multi --mode train --timesteps 200000

# Evaluate trained agent
python -m multiticker_refactor.main_multi --mode evaluate

# Run Streamlit dashboard
streamlit run streamlit_demo/app.py
```

## Architecture

```
Data Pipeline (pipeline_multi.py)
├── Download OHLCV for multiple tickers (yfinance)
├── Align data across tickers (inner join with quality checks)
├── Technical indicators per ticker (RSI, SMA, etc.)
├── Calendar/macro features (shared across tickers)
├── Per-ticker normalization (StandardScaler, fit on train only)
├── Train RNNs in parallel (one LSTM per ticker)
└── Add sentiment (optional, Massive API)
        ↓
Trading Environment (UnifiedTradingEnv)
├── Supports single-ticker (n_tickers=1) and multi-ticker (n_tickers>1)
├── Action space: Box(-1, 1, shape=(n_tickers+1,)) for position weights + cash
├── Dollar-based portfolio tracking with transaction fees
├── Short position support with borrow costs
└── Risk-adjusted rewards (Sharpe-like)
        ↓
PPO Agent (stable-baselines3)
├── Standard PPO with MLP policy
└── RecurrentPPO with LSTM policy (optional, requires sb3-contrib)
```

## Project Structure

```
multiticker_refactor/
├── config/                   # Configuration package
│   ├── __init__.py          # Re-exports all config
│   ├── ticker_config.py     # Ticker selection and date ranges
│   ├── model_config.py      # LSTM and PPO hyperparameters
│   ├── env_config.py        # Trading environment parameters
│   ├── paths.py             # File paths and directories
│   └── api_config.py        # API keys and credentials
├── data/                     # Data fetching and features
│   ├── downloader.py        # yfinance data fetching
│   ├── features.py          # Technical indicators, calendar features
│   ├── sentiment.py         # Massive API sentiment
│   └── cache.py             # Data caching utilities
├── models/                   # Model implementations
│   ├── rnn.py               # LSTM models (simple + probabilistic)
│   └── ppo.py               # PPO model creation and training
├── envs/                     # Trading environments
│   ├── trading_env_unified.py  # UnifiedTradingEnv (single + multi-ticker)
│   ├── trading_env.py       # Environment factory functions
│   └── multi_asset_env.py   # Multi-ticker wrappers
├── streamlit_demo/           # Dashboard (optional)
│   ├── app.py               # Streamlit UI
│   └── utils.py             # Helper functions
├── pipeline_multi.py         # Multi-ticker data pipeline
├── main_multi.py             # CLI entry point for multi-ticker
├── evaluate.py               # Agent evaluation and metrics
├── requirements.txt          # Python dependencies
├── Dockerfile                # Cloud deployment
├── .env.example              # Environment variables template
├── README.md                 # This file
└── config.py                 # LEGACY: Backward compatibility wrapper
```

## Configuration

All configuration is organized in the `config/` package:

### Ticker Configuration (`config/ticker_config.py`)
- `TARGET_TICKER`: Single ticker for single-ticker mode (default: "AAPL")
- `TICKERS`: List of tickers for multi-ticker mode (default: ["AAPL", "MSFT", "GOOGL"])
- `START_DATE`, `END_DATE`: Date range for training data
- `INCLUDE_SENTIMENT`: Enable sentiment features (requires API key)

### Model Configuration (`config/model_config.py`)
- `PPO_TIMESTEPS`: Training steps (default: 200,000)
- `PPO_TRAIN_RATIO`, `PPO_VAL_RATIO`: Train/val/test split (60/20/20)
- `LSTM_WINDOW_SIZE`, `LSTM_EPOCHS`: LSTM hyperparameters
- `PROB_LSTM_HORIZONS`: RNN prediction horizons ([1, 5] days)

### Environment Configuration (`config/env_config.py`)
- `ENV_TYPE`: Environment type (default: "continuous")
- `INITIAL_CAPITAL`: Starting capital (default: $10,000)
- `TRANSACTION_FEE_PCT`: Transaction fee rate (default: 0.1%)
- `SHORT_BORROW_RATE`: Short borrow cost (default: 0.03%/day)

### API Configuration (`config/api_config.py`)
- `API_KEY_MASSIVE`: Massive API key for sentiment (set in `.env` file)

**SECURITY NOTE**: Never commit real API keys! Use environment variables or `.env` file.

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# API Keys
API_KEY_MASSIVE=your_key_here

# Ticker Configuration
TICKERS=AAPL,MSFT,GOOGL

# Training Parameters
PPO_TIMESTEPS=200000
INITIAL_CAPITAL=10000.0
```

## Docker Deployment

```bash
# Build image
docker build -t stockprophet-multiticker .

# Train agent
docker run stockprophet-multiticker

# Evaluate agent
docker run stockprophet-multiticker python -m multiticker_refactor.main_multi --mode evaluate

# Run Streamlit dashboard
docker run -p 8501:8501 stockprophet-multiticker streamlit run streamlit_demo/app.py
```

## Key Features

### Multi-Ticker Support
- Train on 1-3 tickers simultaneously
- Separate LSTM per ticker (parallel training)
- Unified environment for single/multi-ticker cases
- Cross-ticker feature learning

### Risk-Adjusted Trading
- Sharpe-like reward calculation (return / volatility)
- Transaction fees on NET trades only
- Short position support with borrow costs
- Bankruptcy termination when equity ≤ $0

### Self-Contained Deployment
- No external file dependencies
- All code in single folder
- Cloud-ready Docker container
- Environment variable configuration

## Migration from Old Code

Old code importing from `multiticker_refactor.config` will see a deprecation warning:

```python
# Old (deprecated):
from multiticker_refactor.config import TARGET_TICKER, PPO_TIMESTEPS

# New (recommended):
from multiticker_refactor.config import TARGET_TICKER, PPO_TIMESTEPS
```

Both work the same way, but the new config package provides better organization.

## Development

```bash
# Run tests (TODO: Phase 8)
pytest

# Format code
black .

# Type checking
mypy .
```

## License

See LICENSE file in root repository.
