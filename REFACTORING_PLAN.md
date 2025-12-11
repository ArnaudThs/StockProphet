# StockProphet Refactoring Plan

## Overview
This document outlines the refactoring strategy to transform the StockProphet project into a production-ready, well-organized Python data science project.

## Current State Analysis

### Current Structure (Problematic)
```
StockProphet/
├── multiticker_refactor/     # Main codebase (inconsistent naming)
├── project_refactored/        # Legacy single-ticker code
├── Project/                   # Old legacy code
├── Notebook/                  # Jupyter notebooks (unorganized)
├── ppo_*/                     # Model outputs scattered across root
└── Various cache directories scattered everywhere
```

### Issues Identified
1. **Naming Inconsistency**: `multiticker_refactor` vs `project_refactored` vs `Project`
2. **Scattered Outputs**: Model checkpoints, logs, caches in multiple locations
3. **Code Duplication**: pipeline.py vs pipeline_multi.py with overlapping logic
4. **Hard-coded Paths**: API keys, paths throughout codebase
5. **Large Functions**: Several 200+ line functions that should be broken down
6. **Missing Tests**: No systematic unit tests
7. **Poor Documentation**: Inconsistent docstrings, outdated README
8. **No Cloud Readiness**: No Docker, environment config, or deployment scripts

## Target Architecture

### New Directory Structure
```
StockProphet/
├── README.md                  # Comprehensive project documentation
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── Dockerfile                 # Container for deployment
├── docker-compose.yml         # Local development setup
├── .gitignore                 # Ignore data/cache/models
│
├── configs/                   # All configuration files
│   ├── __init__.py
│   ├── base_config.py         # Core parameters
│   ├── data_config.py         # Data sources, dates
│   ├── model_config.py        # RNN, PPO hyperparameters
│   └── env_config.py          # Trading environment parameters
│
├── src/                       # All source code
│   ├── __init__.py
│   │
│   ├── data/                  # Data acquisition & preprocessing
│   │   ├── __init__.py
│   │   ├── fetchers/          # Data source fetchers
│   │   │   ├── __init__.py
│   │   │   ├── yfinance_fetcher.py
│   │   │   ├── polygon_fetcher.py  # News API
│   │   │   └── cache.py       # Unified caching layer
│   │   ├── processors/        # Data processing
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py     # Data cleaning
│   │   │   └── aligner.py     # Multi-ticker alignment
│   │   └── loaders.py         # Data loading utilities
│   │
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical.py       # Technical indicators (RSI, SMA, etc.)
│   │   ├── calendar.py        # Calendar/macro features
│   │   ├── sentiment/         # Sentiment analysis module
│   │   │   ├── __init__.py
│   │   │   ├── processor.py   # FinBERT scoring
│   │   │   └── aggregator.py  # Daily aggregation
│   │   ├── rnn_features.py    # RNN prediction features
│   │   └── selection/         # Feature selection
│   │       ├── __init__.py
│   │       ├── statistical.py # Statistical screening
│   │       ├── rl_validation.py
│   │       └── elimination.py # Backward elimination
│   │
│   ├── models/                # Machine learning models
│   │   ├── __init__.py
│   │   ├── rnn/               # LSTM price prediction
│   │   │   ├── __init__.py
│   │   │   ├── simple_lstm.py
│   │   │   ├── probabilistic_lstm.py
│   │   │   └── trainer.py     # Training logic
│   │   └── rl/                # Reinforcement learning
│   │       ├── __init__.py
│   │       ├── ppo_trainer.py
│   │       ├── evaluator.py
│   │       └── callbacks.py   # Training callbacks
│   │
│   ├── envs/                  # Trading environments
│   │   ├── __init__.py
│   │   ├── base_env.py        # Base trading environment
│   │   ├── continuous_env.py  # Continuous action space
│   │   └── multi_asset_env.py # Multi-ticker environment
│   │
│   ├── utils/                 # Shared utilities
│   │   ├── __init__.py
│   │   ├── logging.py         # Logging configuration
│   │   ├── metrics.py         # Trading metrics (Sharpe, etc.)
│   │   ├── visualization.py   # Plotting utilities
│   │   └── validation.py      # Data validation
│   │
│   └── pipelines/             # End-to-end pipelines
│       ├── __init__.py
│       ├── data_pipeline.py   # Data prep pipeline
│       └── training_pipeline.py  # Training pipeline
│
├── scripts/                   # Executable scripts
│   ├── train_single.py        # Train single-ticker model
│   ├── train_multi.py         # Train multi-ticker model
│   ├── evaluate.py            # Evaluate trained model
│   ├── feature_selection.py   # Run feature selection
│   └── clear_cache.py         # Cache management
│
├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   ├── test_data/             # Data pipeline tests
│   ├── test_features/         # Feature engineering tests
│   ├── test_models/           # Model tests
│   └── test_envs/             # Environment tests
│
├── notebooks/                 # Jupyter notebooks (organized)
│   ├── exploratory/           # Data exploration
│   ├── experiments/           # Model experiments
│   └── visualization/         # Results visualization
│
├── data/                      # Data directory (gitignored except .gitkeep)
│   ├── raw/                   # Raw downloaded data
│   ├── processed/             # Processed datasets
│   └── cache/                 # Cached intermediate results
│       ├── yfinance/
│       ├── news/
│       ├── sentiment/
│       ├── rnn/
│       └── pipeline/
│
├── models/                    # Trained models (gitignored)
│   ├── lstm/                  # LSTM models
│   ├── ppo/                   # PPO agents
│   │   ├── checkpoints/
│   │   ├── best/
│   │   └── logs/
│   └── feature_selection/     # Feature selection results
│
├── outputs/                   # Run outputs (gitignored)
│   ├── evaluation/            # Evaluation results
│   ├── plots/                 # Generated plots
│   └── logs/                  # Application logs
│
└── streamlit_app/             # Streamlit dashboard (isolated)
    ├── app.py
    ├── pages/                 # Multi-page app
    │   ├── dashboard.py
    │   └── configuration.py
    └── utils.py               # Dashboard utilities
```

## Migration Strategy

### Phase 1: Foundation (Priority 1)
1. **Create new directory structure**
   - Create all directories with proper `__init__.py` files
   - Add `.gitkeep` for empty directories that should exist in git

2. **Extract configuration**
   - Split `multiticker_refactor/config.py` into modular configs
   - Create `.env.example` for sensitive data (API keys)
   - Add environment variable loading with `python-dotenv`

3. **Set up dependency management**
   - Create comprehensive `requirements.txt`
   - Add `requirements-dev.txt` for development dependencies
   - Create `Dockerfile` and `docker-compose.yml`

### Phase 2: Data Layer (Priority 1)
4. **Reorganize data fetching**
   - Move `data/downloader.py` → `src/data/fetchers/yfinance_fetcher.py`
   - Move `sentiment/fetcher.py` → `src/data/fetchers/polygon_fetcher.py`
   - Consolidate `data/cache.py` → `src/data/fetchers/cache.py` (unified caching)

5. **Reorganize data processing**
   - Extract alignment logic → `src/data/processors/aligner.py`
   - Extract cleaning logic → `src/data/processors/cleaner.py`
   - Remove code duplication between pipeline.py and pipeline_multi.py

### Phase 3: Feature Engineering (Priority 1)
6. **Modularize feature engineering**
   - Move `data/features.py` → split into:
     - `src/features/technical.py` (RSI, SMA, etc.)
     - `src/features/calendar.py` (holidays, CPI, NFP)
   - Move `sentiment/` → `src/features/sentiment/`
   - Move `models/rnn.py` → `src/models/rnn/` and extract RNN features logic

7. **Reorganize feature selection**
   - Move `feature_selection/` → `src/features/selection/`
   - Modularize large functions (statistical_selector.py is 400+ lines)

### Phase 4: Models (Priority 2)
8. **Reorganize RNN code**
   - Split `models/rnn.py` (600+ lines) into:
     - `src/models/rnn/simple_lstm.py`
     - `src/models/rnn/probabilistic_lstm.py`
     - `src/models/rnn/trainer.py`

9. **Reorganize PPO code**
   - Move `models/ppo.py` → `src/models/rl/ppo_trainer.py`
   - Move `train_ppo.py` → merge into `src/models/rl/ppo_trainer.py`
   - Move `evaluate.py` → `src/models/rl/evaluator.py`

### Phase 5: Environments (Priority 2)
10. **Clean up environment code**
    - Move `envs/` → `src/envs/`
    - Extract shared logic from trading_env.py and multi_asset_env.py → base_env.py
    - Simplify environment version logic (v1/v2 complexity)

### Phase 6: Utilities & Pipelines (Priority 2)
11. **Create shared utilities**
    - Extract metrics calculation → `src/utils/metrics.py`
    - Extract logging setup → `src/utils/logging.py`
    - Extract validation logic → `src/utils/validation.py`

12. **Create pipeline modules**
    - Consolidate pipeline.py and pipeline_multi.py → `src/pipelines/data_pipeline.py`
    - Create `src/pipelines/training_pipeline.py`

### Phase 7: Scripts & CLI (Priority 3)
13. **Create executable scripts**
    - `scripts/train_single.py` - Clean CLI for single-ticker training
    - `scripts/train_multi.py` - Clean CLI for multi-ticker training
    - `scripts/evaluate.py` - Evaluation script
    - `scripts/feature_selection.py` - Feature selection workflow
    - `scripts/clear_cache.py` - Replace cache_cli.py

### Phase 8: Testing (Priority 3)
14. **Add unit tests**
    - Data fetching and caching tests
    - Feature engineering tests
    - Environment tests
    - Model training tests (fast, small dataset)

### Phase 9: Documentation (Priority 3)
15. **Comprehensive documentation**
    - New README.md with architecture, installation, usage
    - Docstrings for all functions (Google style)
    - Module-level documentation
    - API documentation with examples

### Phase 10: Cleanup (Priority 4)
16. **Remove legacy code**
    - Delete `project_refactored/` entirely
    - Delete `Project/` entirely
    - Move useful notebooks to `notebooks/` and delete rest
    - Clean up root-level cache directories

17. **Final polish**
    - Remove all commented code
    - Remove unused imports
    - Consistent naming conventions
    - Code formatting with `black`

## Code Quality Improvements

### 1. Modularization Targets
These functions are too large and should be broken down:

**pipeline_multi.py**: `build_multi_ticker_dataset()` (300+ lines)
- Break into: fetch_data(), add_features(), train_rnns(), add_sentiment()

**models/rnn.py**: `train_and_predict_probabilistic()` (200+ lines)
- Break into: prepare_data(), build_model(), train(), predict()

**feature_selection/statistical_selector.py**: `compute_all_scores()` (150+ lines)
- Break into: compute_correlation(), compute_mutual_info(), compute_rf_importance()

**feature_selection/backward_elimination.py**: `run_backward_elimination()` (250+ lines)
- Break into: run_phase(), evaluate_features(), log_phase_results()

### 2. Code Duplication to Remove
- **Cache logic**: Duplicated across data/cache.py, sentiment/processor.py, models/rnn.py
- **Data cleaning**: Similar logic in pipeline.py and pipeline_multi.py
- **Metrics calculation**: Duplicated in evaluate.py and rl_validator.py
- **Environment setup**: Duplicated env creation logic

### 3. Hard-coded Values to Extract
```python
# Currently hard-coded:
POLYGON_API_KEY = "SiV7GQdKTF2ZtrAr1xNSrnNYP11dKCAC"  # → .env
save_path = "/Users/.../StockProphet/..."              # → use relative paths

# Should be:
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
save_path = Path(__file__).parent / "models" / "lstm"
```

### 4. Naming Convention Standardization
- **Modules**: snake_case (consistent)
- **Classes**: PascalCase (consistent)
- **Functions**: snake_case (consistent)
- **Constants**: UPPER_CASE (mostly consistent, needs enforcement)
- **Private**: _leading_underscore (inconsistent, needs cleanup)

## Environment Configuration

### .env.example
```bash
# API Keys
POLYGON_API_KEY=your_polygon_api_key_here

# Paths (optional, defaults to project structure)
DATA_DIR=./data
MODELS_DIR=./models
OUTPUTS_DIR=./outputs

# Logging
LOG_LEVEL=INFO
LOG_FILE=./outputs/logs/stockprophet.log

# Training
CUDA_VISIBLE_DEVICES=-1  # Use CPU by default
```

## Testing Strategy

### Unit Tests (Priority)
1. **Data fetchers**: Mock API calls, test caching
2. **Data processors**: Test alignment, cleaning with sample data
3. **Technical indicators**: Verify calculations against known values
4. **Feature selection**: Test with toy datasets
5. **Metrics**: Verify Sharpe, returns, max drawdown calculations

### Integration Tests
1. **Data pipeline**: End-to-end with small date range
2. **Training pipeline**: Fast training run with tiny dataset
3. **Evaluation**: Ensure metrics are computed correctly

## Documentation Requirements

### Docstring Template (Google Style)
```python
def function_name(param1: type, param2: type) -> return_type:
    """Short one-line description.

    Longer description explaining what the function does, including
    any important implementation details or caveats.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is invalid
        RuntimeError: When computation fails

    Example:
        >>> result = function_name(10, 'test')
        >>> print(result)
        42
    """
```

### README.md Structure
1. **Overview**: What is StockProphet?
2. **Features**: Key capabilities
3. **Architecture**: High-level system design
4. **Installation**: Step-by-step setup
5. **Quick Start**: Minimal working example
6. **Usage**: Training, evaluation, feature selection
7. **Configuration**: How to customize
8. **Project Structure**: Directory layout explanation
9. **Development**: How to contribute
10. **License**: MIT

## Dockerfile Strategy

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create data/models/outputs directories
RUN mkdir -p data/raw data/processed data/cache models outputs

# Default command
CMD ["python", "scripts/train_single.py", "--help"]
```

## Success Criteria

### Must Have
- ✅ Clean directory structure matching data science best practices
- ✅ All configuration extracted to configs/ and .env
- ✅ No hard-coded paths or API keys in source code
- ✅ All functions have comprehensive docstrings
- ✅ No functions > 100 lines (except unavoidable complexity)
- ✅ No code duplication
- ✅ Unit tests for critical functions (>50% coverage target)
- ✅ Comprehensive README with usage examples
- ✅ Dockerfile and docker-compose.yml working
- ✅ All legacy code removed

### Nice to Have
- ⭐ >80% test coverage
- ⭐ Type hints for all functions
- ⭐ Pre-commit hooks for formatting (black, isort, flake8)
- ⭐ CI/CD pipeline (GitHub Actions)
- ⭐ Sphinx documentation
- ⭐ Performance benchmarks

## Migration Checklist

### Phase 1: Foundation ✓
- [ ] Create directory structure
- [ ] Create .env.example
- [ ] Create requirements.txt
- [ ] Create Dockerfile
- [ ] Split config.py into modular configs

### Phase 2: Data Layer ✓
- [ ] Move and refactor data fetchers
- [ ] Consolidate caching logic
- [ ] Move and refactor data processors
- [ ] Add unit tests for data layer

### Phase 3: Feature Engineering ✓
- [ ] Move and split feature engineering code
- [ ] Move sentiment module
- [ ] Move feature selection module
- [ ] Add unit tests for features

### Phase 4: Models ✓
- [ ] Split RNN code into modules
- [ ] Refactor PPO code
- [ ] Add model training tests

### Phase 5: Environments ✓
- [ ] Move environments
- [ ] Extract base environment
- [ ] Add environment tests

### Phase 6: Utilities & Pipelines ✓
- [ ] Create utility modules
- [ ] Consolidate pipelines
- [ ] Add pipeline tests

### Phase 7: Scripts & CLI ✓
- [ ] Create train_single.py
- [ ] Create train_multi.py
- [ ] Create evaluate.py
- [ ] Create feature_selection.py

### Phase 8: Testing ✓
- [ ] Write unit tests (target >50% coverage)
- [ ] Write integration tests
- [ ] Test Docker build

### Phase 9: Documentation ✓
- [ ] Add docstrings to all functions
- [ ] Write comprehensive README
- [ ] Create usage examples
- [ ] Document architecture

### Phase 10: Cleanup ✓
- [ ] Remove legacy code
- [ ] Remove commented code
- [ ] Format all code with black
- [ ] Final review and commit

## Estimated Effort

**Total**: ~8-12 hours of focused work

- Phase 1 (Foundation): 1 hour
- Phase 2 (Data Layer): 2 hours
- Phase 3 (Features): 2 hours
- Phase 4 (Models): 1.5 hours
- Phase 5 (Environments): 1 hour
- Phase 6 (Utils/Pipelines): 1 hour
- Phase 7 (Scripts): 0.5 hours
- Phase 8 (Testing): 1.5 hours
- Phase 9 (Documentation): 1.5 hours
- Phase 10 (Cleanup): 1 hour

## Next Steps

1. Get user approval on this plan
2. Start with Phase 1 (Foundation)
3. Commit after each phase completes
4. Use feature branch: `refactor/modularization` ✓ (created)
5. Merge to main when all phases complete
