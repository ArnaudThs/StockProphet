"""
Models module for StockProphet.

This module handles:
- rnn.py: LSTM price prediction models (simple + probabilistic multi-horizon)
- ppo.py: PPO reinforcement learning agent training
- constrained_policy.py: Custom policy with squashed Gaussian for bounded actions
"""

from .rnn import train_and_predict, train_and_predict_probabilistic
from .ppo import create_model, train_model, create_continuous_model
from .constrained_policy import ConstrainedRecurrentPolicy, get_constrained_policy_kwargs

__all__ = [
    'train_and_predict',
    'train_and_predict_probabilistic',
    'create_model',
    'train_model',
    'create_continuous_model',
    'ConstrainedRecurrentPolicy',
    'get_constrained_policy_kwargs',
]
