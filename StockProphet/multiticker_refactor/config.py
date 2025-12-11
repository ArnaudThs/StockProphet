"""
LEGACY: This file is deprecated.

For new code, import from the config package instead:
    from multiticker_refactor.config import TARGET_TICKER, PPO_TIMESTEPS, etc.

This file is kept for backward compatibility with existing code.
All configuration is now organized in the config/ package:
- config/ticker_config.py: Ticker selection and date ranges
- config/model_config.py: LSTM and PPO hyperparameters
- config/env_config.py: Trading environment parameters
- config/paths.py: File paths and directories
- config/api_config.py: API keys and credentials
"""
import warnings
warnings.warn(
    "Importing from multiticker_refactor.config is deprecated. "
    "Use 'from multiticker_refactor.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new config package for backward compatibility
from .config import *  # noqa: F401, F403
