# Feature Selection Module

Reduces ~50 features → 12 essential features using a two-stage approach:

1. **Statistical Screening** (~5 min): Fast elimination using correlation, mutual information, Random Forest
2. **RL Validation** (~25 min): Validate with RecurrentPPO training

## Quick Start

### Run Full Pipeline

```bash
cd StockProphet
python -m multiticker_refactor.feature_selection.main --ticker AAPL --stage full
```

This will:
- Build dataset for AAPL
- Run statistical screening (50 → 20 features)
- Run RL validation (20 → 12 features)
- Save results to `multiticker_refactor/feature_selection/results/`

### Run Stages Separately

**Stage 1: Statistical Screening Only**
```bash
python -m multiticker_refactor.feature_selection.main --ticker AAPL --stage statistical
```

**Stage 2: RL Validation Only**
```bash
python -m multiticker_refactor.feature_selection.main \
    --ticker AAPL \
    --stage rl \
    --input multiticker_refactor/feature_selection/results/statistical_results.json
```

## Configuration

Edit `feature_selection/config.py` to adjust:

```python
TARGET_FEATURE_COUNT = 12            # Final number of features
INTERMEDIATE_COUNT = 20              # Shortlist after statistical stage
RL_VALIDATION_TIMESTEPS = 50_000     # Training timesteps (increase to 100k for final run)
RL_VALIDATION_SEEDS = 3              # Random seeds for averaging
```

## How It Works

### Stage 1: Statistical Screening

Computes 5 scores for each feature:

1. **Correlation with returns**: Spearman correlation with next-day returns
2. **Mutual information**: MI with price direction (up/down)
3. **Random Forest importance**: Feature importance from RF classifier
4. **Variance**: Standard deviation (low variance = uninformative)
5. **Redundancy**: Max correlation with other features (high = redundant)

Composite score:
```python
score = 0.3 * corr + 0.3 * MI + 0.3 * RF + 0.1 * variance - 0.2 * redundancy
```

**Output**: Top 20 features ranked by composite score

### Stage 2: RL Validation

Tests different feature subsets using RecurrentPPO:

- `top_20_all`: All 20 from statistical stage
- `top_15`: Top 15
- `top_12`: Top 12 (target)
- `top_10`: Top 10

Each configuration trained with multiple random seeds and evaluated on:
- **Sharpe ratio** (primary metric)
- Test return
- Max drawdown

**Output**: Best performing feature subset (usually 10-12 features)

## Output Files

All saved to `multiticker_refactor/feature_selection/results/`:

- `statistical_results.json` - Statistical scores for all features
- `rl_validation_results.json` - RL training results for each configuration
- `selected_features.json` - **Final selected features** (use this!)

## Example: Using Selected Features

```python
import json

# Load selected features
with open('multiticker_refactor/feature_selection/results/selected_features.json', 'r') as f:
    selection = json.load(f)

selected_features = selection['features']
print(f"Selected {len(selected_features)} features")

# Use in pipeline
df, metadata = build_multi_ticker_dataset(...)
df_selected = df[selected_features + ['target_close']]  # Keep target
```

## Advanced Options

### Adjust Training Time

Faster (10 min total):
```bash
python -m multiticker_refactor.feature_selection.main \
    --ticker AAPL \
    --stage full \
    --timesteps 30000 \
    --seeds 1
```

More thorough (60 min total):
```bash
python -m multiticker_refactor.feature_selection.main \
    --ticker AAPL \
    --stage full \
    --timesteps 100000 \
    --seeds 5
```

### Test Multiple Tickers

```bash
for ticker in AAPL MSFT GOOGL; do
    python -m multiticker_refactor.feature_selection.main \
        --ticker $ticker \
        --stage full \
        --output-dir feature_selection/results/$ticker
done
```

Then compare results and select features that work across all tickers.

## Troubleshooting

**Import errors**: Make sure you're in `StockProphet/` directory and using `-m` flag

**Out of memory**: Reduce `--timesteps` or `--seeds`

**Poor results**: Try different date ranges or increase `--timesteps`

## Integration with Main Pipeline

After selecting features, update `config.py`:

```python
# In multiticker_refactor/config.py
SELECTED_FEATURES = [
    'AAPL_rnn_mu_1d',
    'AAPL_rnn_sigma_1d',
    'AAPL_RSI',
    # ... rest of selected features
]

# In pipeline_multi.py, filter features:
df_filtered = df[SELECTED_FEATURES + [f'{ticker}_Close' for ticker in TICKERS]]
```
