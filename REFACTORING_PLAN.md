# StockProphet Refactoring Plan (multiticker_refactor)

## Overview
This document outlines the refactoring strategy to transform the `multiticker_refactor` folder into a production-ready, **self-contained**, well-organized Python data science project that can be deployed to the cloud **without any external file dependencies**.

## Scope
**IMPORTANT**: This refactoring applies **ONLY to the `StockProphet/multiticker_refactor/` folder**. All changes must be contained within this directory.

## Current State Analysis

### Current Structure
```
StockProphet/multiticker_refactor/
├── config.py                    # All configuration in one file
├── pipeline.py                  # Legacy single-ticker pipeline
├── pipeline_multi.py            # Multi-ticker pipeline (300+ lines)
├── main.py                      # Single-ticker CLI
├── main_multi.py                # Multi-ticker CLI
├── train_ppo.py                 # PPO training logic
├── evaluate.py                  # Evaluation logic
├── cache_cli.py                 # Cache management
│
├── data/                        # Data fetching & processing
│   ├── downloader.py
│   ├── features.py
│   └── cache.py
│
├── models/                      # ML models
│   ├── rnn.py                   # LSTM (600+ lines)
│   └── ppo.py                   # PPO trainer
│
├── envs/                        # Trading environments
│   ├── trading_env.py           # ⚠️ Imports from gym_anytrading (EXTERNAL!)
│   └── multi_asset_env.py
│
├── sentiment/                   # Sentiment analysis
│   ├── fetcher.py               # Polygon API
│   ├── processor.py             # FinBERT
│   └── pipeline.py
│
├── feature_selection/           # Feature selection module
│   ├── main.py
│   ├── statistical_selector.py
│   ├── rl_validator.py
│   └── backward_elimination.py
│
├── streamlit_demo/              # Dashboard (isolated)
│   ├── app.py
│   └── utils.py
│
└── data_cache/                  # Cached data (gitignored)
```

### Critical Issues Identified

1. **🚨 EXTERNAL DEPENDENCY**: `envs/trading_env.py` imports from `gym_anytrading` (outside the folder)
   - **BLOCKER for cloud deployment**
   - Must copy required gym-anytrading files into the project

2. **Naming Inconsistency**: Multiple pipelines (pipeline.py vs pipeline_multi.py)

3. **Scattered Configuration**: All config in one massive config.py file

4. **Large Functions**: Several 200-300+ line functions

5. **Code Duplication**: Cache logic repeated across multiple files

6. **Hard-coded Paths**: API keys, absolute paths in codebase

7. **Missing Tests**: No systematic unit tests

8. **Poor Documentation**: Inconsistent docstrings

## Target Architecture

### New Directory Structure
```
multiticker_refactor/
├── README.md                   # Comprehensive documentation
├── requirements.txt            # Python dependencies (pip-installable)
├── .env.example                # Environment variable template
├── Dockerfile                  # Container for deployment
├── .gitignore                  # Ignore data/cache/models
│
├── configs/                    # Modular configuration
│   ├── __init__.py
│   ├── data_config.py          # Data sources, dates, tickers
│   ├── model_config.py         # RNN, PPO hyperparameters
│   ├── env_config.py           # Trading environment params
│   └── paths_config.py         # Path management (relative paths only)
│
├── src/                        # All source code
│   ├── __init__.py
│   │
│   ├── data/                   # Data acquisition & preprocessing
│   │   ├── __init__.py
│   │   ├── fetchers/
│   │   │   ├── __init__.py
│   │   │   ├── yfinance_fetcher.py
│   │   │   ├── polygon_fetcher.py
│   │   │   └── cache.py        # Unified caching layer
│   │   └── processors/
│   │       ├── __init__.py
│   │       ├── cleaner.py
│   │       └── aligner.py
│   │
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical.py        # RSI, SMA, indicators
│   │   ├── calendar.py         # Holidays, CPI, NFP
│   │   ├── sentiment/
│   │   │   ├── __init__.py
│   │   │   ├── fetcher.py
│   │   │   ├── processor.py
│   │   │   └── aggregator.py
│   │   ├── rnn_features.py     # RNN prediction features
│   │   └── selection/
│   │       ├── __init__.py
│   │       ├── statistical.py
│   │       ├── rl_validation.py
│   │       └── elimination.py
│   │
│   ├── models/                 # Machine learning models
│   │   ├── __init__.py
│   │   ├── rnn/
│   │   │   ├── __init__.py
│   │   │   ├── simple_lstm.py
│   │   │   ├── probabilistic_lstm.py
│   │   │   └── trainer.py
│   │   └── rl/
│   │       ├── __init__.py
│   │       ├── ppo_trainer.py
│   │       ├── evaluator.py
│   │       └── callbacks.py
│   │
│   ├── envs/                   # Trading environments (SELF-CONTAINED)
│   │   ├── __init__.py
│   │   ├── base_env.py         # Base trading env logic
│   │   ├── flexible_env.py     # ← Copied from gym-anytrading
│   │   ├── continuous_env.py   # ← Copied from gym-anytrading
│   │   ├── continuous_env_v2.py # ← Copied from gym-anytrading
│   │   └── multi_asset_env.py  # Multi-ticker environment
│   │
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py
│   │   ├── logging_config.py
│   │   ├── metrics.py          # Sharpe, returns, drawdown
│   │   ├── visualization.py
│   │   └── validation.py
│   │
│   └── pipelines/              # End-to-end workflows
│       ├── __init__.py
│       ├── data_pipeline.py    # Consolidate pipeline.py + pipeline_multi.py
│       └── training_pipeline.py
│
├── scripts/                    # Executable CLI scripts
│   ├── train_single.py         # Train single-ticker model
│   ├── train_multi.py          # Train multi-ticker model
│   ├── evaluate.py             # Evaluate trained model
│   ├── feature_selection.py    # Run feature selection
│   └── clear_cache.py          # Cache management
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── test_envs/
│
├── notebooks/                  # Jupyter notebooks (optional)
│   ├── exploratory/
│   └── experiments/
│
├── data_cache/                 # Cached data (gitignored)
│   ├── yfinance/
│   ├── news/
│   ├── sentiment/
│   ├── rnn/
│   └── pipeline/
│
├── saved_models/               # Trained models (gitignored)
│   ├── lstm/
│   ├── ppo/
│   └── feature_selection/
│
├── outputs/                    # Run outputs (gitignored)
│   ├── evaluation/
│   ├── plots/
│   └── logs/
│
└── streamlit_app/              # Dashboard (isolated)
    ├── app.py
    └── utils.py
```

## Migration Strategy

### Phase 0: Self-Containment (CRITICAL - Priority 0)

**MUST DO FIRST** before any other refactoring:

1. **Copy gym-anytrading environments into project**
   - Locate gym-anytrading source (likely in parent directory or site-packages)
   - Copy required files to `multiticker_refactor/envs/`:
     - `flexible_env.py`
     - `continuous_env.py`
     - `continuous_env_v2.py`
     - Any base classes they depend on

2. **Update imports in `envs/trading_env.py`**:
   ```python
   # OLD (external dependency):
   from gym_anytrading.envs.flexible_env import FlexibleTradingEnv

   # NEW (self-contained):
   from .flexible_env import FlexibleTradingEnv
   ```

3. **Test** that project works without gym-anytrading installed:
   ```bash
   pip uninstall gym-anytrading  # Should still work after this!
   python -m multiticker_refactor.main_multi --help
   ```

4. **Commit** self-containment changes before proceeding

### Phase 1: Foundation (Priority 1)

1. **Create new directory structure within multiticker_refactor/**
   - Create all directories with proper `__init__.py` files
   - Add `.gitkeep` for empty directories

2. **Extract configuration**
   - Split `config.py` into modular configs in `configs/`
   - Create `.env.example` for sensitive data (API keys)
   - Add environment variable loading with `python-dotenv`
   - **Use relative paths only** (no hard-coded absolute paths)

3. **Set up dependency management**
   - Create comprehensive `requirements.txt`
   - Create `Dockerfile` for cloud deployment
   - Ensure all dependencies are pip-installable

### Phase 2: Data Layer (Priority 1)

4. **Reorganize data fetching**
   - Move `data/downloader.py` → `src/data/fetchers/yfinance_fetcher.py`
   - Move `sentiment/fetcher.py` → `src/data/fetchers/polygon_fetcher.py`
   - Consolidate `data/cache.py` → `src/data/fetchers/cache.py` (unified caching)

5. **Reorganize data processing**
   - Extract alignment logic → `src/data/processors/aligner.py`
   - Extract cleaning logic → `src/data/processors/cleaner.py`

### Phase 3: Feature Engineering (Priority 1)

6. **Modularize feature engineering**
   - Move `data/features.py` → split into:
     - `src/features/technical.py` (RSI, SMA, etc.)
     - `src/features/calendar.py` (holidays, CPI, NFP)
   - Move `sentiment/` → `src/features/sentiment/`
   - Move `models/rnn.py` → `src/models/rnn/` and extract RNN features

7. **Reorganize feature selection**
   - Move `feature_selection/` → `src/features/selection/`
   - Break down large functions

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
    - Keep gym-anytrading files in `src/envs/` (already copied in Phase 0)
    - Extract shared logic → `src/envs/base_env.py`
    - Simplify environment version logic

### Phase 6: Utilities & Pipelines (Priority 2)

11. **Create shared utilities**
    - Extract metrics → `src/utils/metrics.py`
    - Extract logging → `src/utils/logging_config.py`
    - Extract validation → `src/utils/validation.py`

12. **Create pipeline modules**
    - Consolidate `pipeline.py` + `pipeline_multi.py` → `src/pipelines/data_pipeline.py`
    - Create `src/pipelines/training_pipeline.py`

### Phase 7: Scripts & CLI (Priority 3)

13. **Create executable scripts**
    - `scripts/train_single.py` - Clean CLI for single-ticker
    - `scripts/train_multi.py` - Clean CLI for multi-ticker
    - `scripts/evaluate.py` - Evaluation script
    - `scripts/feature_selection.py` - Feature selection workflow
    - `scripts/clear_cache.py` - Replace cache_cli.py

### Phase 8: Testing (Priority 3)

14. **Add unit tests**
    - Data fetching and caching tests
    - Feature engineering tests
    - Environment tests
    - Model training tests

### Phase 9: Documentation (Priority 3)

15. **Comprehensive documentation**
    - New README.md with architecture, installation, usage
    - Docstrings for all functions (Google style)
    - Module-level documentation

### Phase 10: Cleanup (Priority 4)

16. **Remove unused code**
    - Remove `pipeline.py` (superseded by src/pipelines/data_pipeline.py)
    - Remove `cache_cli.py` (superseded by scripts/clear_cache.py)
    - Remove all commented code
    - Remove unused imports

## Self-Containment Requirements

### MUST HAVE (Deployment Blockers)
- ✅ No imports from outside `multiticker_refactor/` folder
- ✅ No file system dependencies on parent directories
- ✅ All gym-anytrading code copied into project
- ✅ All paths are relative (use `Path(__file__).parent`)
- ✅ API keys from environment variables only
- ✅ `pip install -r requirements.txt` installs all dependencies

### Verification Test
```bash
# This MUST work from a fresh clone:
cd multiticker_refactor
pip install -r requirements.txt
python -m scripts.train_multi --help  # Should run without errors
```

## Code Quality Improvements

### 1. Modularization Targets

**pipeline_multi.py**: `build_multi_ticker_dataset()` (300+ lines)
- Break into: `fetch_data()`, `add_features()`, `train_rnns()`, `add_sentiment()`

**models/rnn.py**: `train_and_predict_probabilistic()` (200+ lines)
- Break into: `prepare_data()`, `build_model()`, `train()`, `predict()`

**feature_selection/statistical_selector.py**: `compute_all_scores()` (150+ lines)
- Break into: `compute_correlation()`, `compute_mutual_info()`, `compute_rf_importance()`

**feature_selection/backward_elimination.py**: `run_backward_elimination()` (250+ lines)
- Break into: `run_phase()`, `evaluate_features()`, `log_phase_results()`

### 2. Code Duplication to Remove
- Cache logic: Duplicated across `data/cache.py`, `sentiment/processor.py`, `models/rnn.py`
- Data cleaning: Similar logic in `pipeline.py` and `pipeline_multi.py`
- Metrics calculation: Duplicated in `evaluate.py` and `rl_validator.py`

### 3. Hard-coded Values to Extract
```python
# Currently hard-coded:
POLYGON_API_KEY = "SiV7GQdKTF2ZtrAr1xNSrnNYP11dKCAC"  # → .env
save_path = "/Users/.../multiticker_refactor/..."     # → relative paths

# Should be:
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
save_path = Path(__file__).parent / "saved_models" / "lstm"
```

## Success Criteria

### Must Have (Deployment Blockers)
- ✅ Project is 100% self-contained (no external file dependencies)
- ✅ Clean directory structure matching data science best practices
- ✅ All configuration extracted to `configs/` and `.env`
- ✅ No hard-coded paths or API keys in source code
- ✅ All functions have comprehensive docstrings
- ✅ No functions > 100 lines (except unavoidable complexity)
- ✅ No code duplication
- ✅ Unit tests for critical functions (>50% coverage target)
- ✅ Comprehensive README with usage examples
- ✅ Dockerfile working and tested
- ✅ Can deploy to cloud without any manual file copying

### Nice to Have
- ⭐ >80% test coverage
- ⭐ Type hints for all functions
- ⭐ Pre-commit hooks for formatting
- ⭐ CI/CD pipeline
- ⭐ Performance benchmarks

## Migration Checklist

### Phase 0: Self-Containment ✓ (CRITICAL)
- [ ] Locate gym-anytrading source files
- [ ] Copy required env files to `envs/`
- [ ] Update imports in `trading_env.py`
- [ ] Test without gym-anytrading installed
- [ ] Commit self-containment changes

### Phase 1: Foundation ✓
- [ ] Create directory structure within multiticker_refactor/
- [ ] Create `.env.example`
- [ ] Create `requirements.txt`
- [ ] Create `Dockerfile`
- [ ] Split `config.py` into modular configs

### Phase 2-10: (Same as before, but all within multiticker_refactor/)

## Estimated Effort

**Total**: ~10-14 hours of focused work

- Phase 0 (Self-Containment): **2 hours** ← NEW, CRITICAL
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

1. ✅ User approval on this plan
2. **START WITH PHASE 0** - Make project self-contained
3. Proceed with Phase 1 (Foundation)
4. Commit after each phase completes
5. Use feature branch: `refactor/modularization` ✅ (created)
6. Test deployment to cloud when complete
