"""
LSTM and PPO model hyperparameters.
"""
import os

# =============================================================================
# RNN (LSTM) PARAMETERS
# =============================================================================
LSTM_WINDOW_SIZE = 50  # Sequence length for LSTM
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32
LSTM_TRAIN_RATIO = 0.7  # 70% train, 30% test for LSTM

# Probabilistic Multi-Horizon LSTM
PROB_LSTM_HORIZONS = [1, 5]  # Prediction horizons in days (t+1, t+5)
PROB_LSTM_EPOCHS = 30
PROB_LSTM_UNITS = 64

# =============================================================================
# PPO PARAMETERS
# =============================================================================
PPO_WINDOW_SIZE = 10  # Not used with MlpLstmPolicy (policy maintains internal LSTM state)
PPO_TIMESTEPS = int(os.getenv("PPO_TIMESTEPS", "200000"))

# Train/Val/Test Split (60/20/20)
PPO_TRAIN_RATIO = 0.6  # 60% for training
PPO_VAL_RATIO = 0.2    # 20% for validation (hyperparameter tuning)
# Test implicit: 20% = 1 - PPO_TRAIN_RATIO - PPO_VAL_RATIO

# PPO Hyperparameters (optimized for trading)
PPO_LEARNING_RATE = 3e-4       # Learning rate
PPO_N_STEPS = 512              # Steps per update (more frequent updates for market responsiveness)
PPO_BATCH_SIZE = 64            # Minibatch size
PPO_N_EPOCHS = 10              # Gradient descent epochs per update
PPO_GAMMA = 0.99               # Discount factor
PPO_GAE_LAMBDA = 0.95          # GAE lambda
PPO_CLIP_RANGE = 0.2           # PPO clipping parameter
PPO_ENT_COEF_START = 0.05      # Initial entropy coefficient (exploration)
PPO_ENT_COEF_END = 0.01        # Final entropy coefficient (convergence)

# Evaluation and checkpointing
PPO_EVAL_FREQ = 10_000         # Evaluate on validation set every N steps
PPO_CHECKPOINT_FREQ = 10_000   # Save checkpoint every N steps

# VecNormalize configuration
VECNORMALIZE_NORM_OBS = True   # Normalize observations
VECNORMALIZE_NORM_REWARD = False  # DON'T normalize rewards (masks learning signal)
VECNORMALIZE_CLIP_OBS = 5.0    # Clip observations to [-5, 5] after normalization
VECNORMALIZE_CLIP_REWARD = None  # Don't clip rewards
