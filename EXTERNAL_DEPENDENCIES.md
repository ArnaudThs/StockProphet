# External Dependencies Analysis

## Critical Finding: `gym-anytrading` Dependency

The `multiticker_refactor` project currently depends on `gym_anytrading`, which appears to be an external module outside the project folder.

### Location of Import
- **File**: `multiticker_refactor/envs/trading_env.py`
- **Lines**:
  ```python
  from gym_anytrading.envs.flexible_env import FlexibleTradingEnv
  from gym_anytrading.envs.continuous_env import ContinuousTradingEnv
  from gym_anytrading.envs.continuous_env_v2 import ContinuousTradingEnvV2
  ```

### Required Action
**The gym-anytrading environments MUST be copied into the `multiticker_refactor` project** to make it self-contained and deployable to cloud without external dependencies.

## All Python Package Dependencies

Based on analysis of imports in `multiticker_refactor/`:

### Data Science / ML Libraries
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing (stats, spearmanr, norm)
- `sklearn` (scikit-learn) - Machine learning utilities
  - `StandardScaler`, `MinMaxScaler` - Normalization
  - `RandomForestClassifier` - Feature importance
  - `mutual_info_classif` - Mutual information

### Deep Learning
- `tensorflow` / `keras` - LSTM models
  - `keras.layers` (LSTM, Dense, Input)
  - `keras.models` (Sequential, Model, load_model)
  - `keras.backend`

### Reinforcement Learning
- `gymnasium` - OpenAI Gym environments
- `stable-baselines3` - PPO implementation
  - `DummyVecEnv`, `VecNormalize`
  - `EvalCallback`, `CheckpointCallback`, `BaseCallback`
  - `get_schedule_fn`
- `sb3-contrib` - RecurrentPPO (LSTM policy)

### NLP / Sentiment Analysis
- `transformers` (HuggingFace) - FinBERT
  - `AutoTokenizer`, `AutoModelForSequenceClassification`, `pipeline`

### Data Sources
- `yfinance` - Stock price data
- `requests` - HTTP requests (Polygon API)

### Utilities
- `holidays` - Holiday calendar
- `hashlib` - Cache hashing
- `json` - JSON serialization
- `argparse` - CLI arguments
- `pathlib.Path` - Path handling
- `multiprocessing` - Parallel processing

### Visualization
- `matplotlib.pyplot` - Basic plotting
- `plotly` - Interactive plots
  - `plotly.express`
  - `plotly.graph_objects`
- `streamlit` - Dashboard (streamlit_demo/)

### Standard Library (Python built-in)
- `os`, `sys`, `time`, `re`
- `typing` - Type hints

## Action Items for Self-Containment

1. **CRITICAL**: Copy `gym_anytrading` environments into `multiticker_refactor/envs/`
   - Likely location: `../gym-anytrading/gym_anytrading/envs/`
   - Required files:
     - `flexible_env.py`
     - `continuous_env.py`
     - `continuous_env_v2.py`
     - Any base classes they depend on

2. **Create requirements.txt** with all pip-installable dependencies

3. **Update imports** in `trading_env.py`:
   ```python
   # OLD (external dependency):
   from gym_anytrading.envs.flexible_env import FlexibleTradingEnv

   # NEW (self-contained):
   from .flexible_env import FlexibleTradingEnv
   ```

4. **Test** that project works without any external folder dependencies

## Verification Command
```bash
# This should work from anywhere after refactor:
cd multiticker_refactor
pip install -r requirements.txt
python -m main_multi --help  # Should run without errors
```
