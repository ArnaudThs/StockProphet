"""
PPO (Proximal Policy Optimization) agent for trading.
Source: Reinforcement.ipynb PPO setup and training cells

Supports both standard PPO and RecurrentPPO (with LSTM policy).
"""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

# RecurrentPPO from sb3-contrib (optional)
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False
    RecurrentPPO = None

from project_refactored.config import (
    PPO_TIMESTEPS, PPO_MODEL_PATH, VEC_NORMALIZE_PATH, RECURRENT_PPO_MODEL_PATH,
    ENV_TYPE
)


class SyncNormCallback(BaseCallback):
    """
    Callback to sync VecNormalize stats from train_env to eval_env.
    This ensures the evaluation uses the same normalization as training.
    """

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


def create_ppo_model(
    env: VecNormalize,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.95,
    gae_lambda: float = 0.9,
    ent_coef: float = 0.05,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    seed: int = 42,
    tensorboard_log: str = None,
    verbose: int = 1
) -> PPO:
    """
    Create a PPO model with tuned hyperparameters for trading.

    Key hyperparameters:
    - n_steps=2048: Longer rollouts capture multi-day patterns
    - ent_coef=0.05: Higher exploration (default is 0.01)
    - gamma=0.95: Shorter horizon focus for trading
    - net_arch=[256, 256]: Wider networks for complex patterns

    Args:
        env: VecNormalize wrapped training environment
        learning_rate: Learning rate (can be schedule)
        n_steps: Steps to run before update
        batch_size: Minibatch size
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Entropy coefficient for exploration
        clip_range: PPO clip range
        n_epochs: Number of epochs per update
        seed: Random seed
        tensorboard_log: Path for tensorboard logs
        verbose: Verbosity level

    Returns:
        PPO model
    """
    # Learning rate schedule: linear decay
    def lr_schedule(progress):
        return learning_rate * (1 - progress * 0.5)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed
    )

    return model


def create_callbacks(
    train_env: VecNormalize,
    eval_env: VecNormalize,
    eval_freq: int = 5000,
    checkpoint_freq: int = 10000,
    best_model_save_path: str = "./ppo_best_model/",
    log_path: str = "./ppo_eval_logs/",
    checkpoint_path: str = "./ppo_checkpoints/"
) -> list:
    """
    Create training callbacks.

    Args:
        train_env: Training environment
        eval_env: Evaluation environment
        eval_freq: Evaluation frequency (steps)
        checkpoint_freq: Checkpoint frequency (steps)
        best_model_save_path: Path to save best model
        log_path: Path for evaluation logs
        checkpoint_path: Path for checkpoints

    Returns:
        List of callbacks
    """
    # Sync normalization stats
    sync_callback = SyncNormCallback(train_env, eval_env)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_path,
        name_prefix="ppo_trading_model"
    )

    return [sync_callback, eval_callback, checkpoint_callback]


def train_ppo(
    model: PPO,
    total_timesteps: int = PPO_TIMESTEPS,
    callbacks: list = None
) -> PPO:
    """
    Train the PPO model.

    Args:
        model: PPO model to train
        total_timesteps: Total training timesteps
        callbacks: List of callbacks

    Returns:
        Trained model
    """
    print(f"\nTraining PPO for {total_timesteps:,} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    return model


def save_ppo_model(model: PPO, train_env: VecNormalize, model_path: str = None, vec_path: str = None):
    """
    Save PPO model and VecNormalize stats.

    Args:
        model: Trained PPO model
        train_env: Training environment with normalization stats
        model_path: Path to save model
        vec_path: Path to save VecNormalize stats
    """
    if model_path is None:
        model_path = PPO_MODEL_PATH
    if vec_path is None:
        vec_path = VEC_NORMALIZE_PATH

    model.save(model_path)
    train_env.save(vec_path)

    print(f"Model saved to {model_path}")
    print(f"VecNormalize saved to {vec_path}")


def load_ppo_model(model_path: str = None) -> PPO:
    """
    Load a saved PPO model.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded PPO model
    """
    if model_path is None:
        model_path = PPO_MODEL_PATH

    return PPO.load(model_path)


# =============================================================================
# RECURRENT PPO (LSTM Policy)
# =============================================================================

def create_recurrent_ppo_model(
    env: VecNormalize,
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 128,
    gamma: float = 0.95,
    gae_lambda: float = 0.9,
    ent_coef: float = 0.05,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    lstm_hidden_size: int = 64,
    n_lstm_layers: int = 1,
    seed: int = 42,
    tensorboard_log: str = None,
    verbose: int = 1
):
    """
    Create a RecurrentPPO model with LSTM policy for trading.

    RecurrentPPO uses MlpLstmPolicy which maintains LSTM hidden state
    across timesteps, allowing the policy to learn temporal patterns directly.

    Key differences from standard PPO:
    - Uses MlpLstmPolicy instead of MlpPolicy
    - n_steps should be smaller (sequences processed with LSTM)
    - Requires LSTM state management during evaluation

    Args:
        env: VecNormalize wrapped training environment
        learning_rate: Learning rate
        n_steps: Steps per rollout (smaller for LSTM)
        batch_size: Minibatch size (should be multiple of n_steps)
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Entropy coefficient
        clip_range: PPO clip range
        n_epochs: Epochs per update
        lstm_hidden_size: LSTM hidden layer size
        n_lstm_layers: Number of LSTM layers
        seed: Random seed
        tensorboard_log: Tensorboard log path
        verbose: Verbosity

    Returns:
        RecurrentPPO model
    """
    if not RECURRENT_PPO_AVAILABLE:
        raise ImportError(
            "RecurrentPPO requires sb3-contrib. "
            "Install with: pip install sb3-contrib"
        )

    # Learning rate schedule
    def lr_schedule(progress):
        return learning_rate * (1 - progress * 0.5)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=False,  # Separate LSTMs for policy and value
            enable_critic_lstm=True,
        ),
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed
    )

    return model


def train_recurrent_ppo(
    model,
    total_timesteps: int = PPO_TIMESTEPS,
    callbacks: list = None
):
    """
    Train a RecurrentPPO model.

    Args:
        model: RecurrentPPO model
        total_timesteps: Total training steps
        callbacks: List of callbacks

    Returns:
        Trained model
    """
    print(f"\nTraining RecurrentPPO for {total_timesteps:,} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    return model


def save_recurrent_ppo_model(model, train_env: VecNormalize,
                              model_path: str = None, vec_path: str = None):
    """
    Save RecurrentPPO model and VecNormalize stats.

    Args:
        model: Trained RecurrentPPO model
        train_env: Training environment
        model_path: Model save path
        vec_path: VecNormalize save path
    """
    if model_path is None:
        model_path = RECURRENT_PPO_MODEL_PATH
    if vec_path is None:
        vec_path = VEC_NORMALIZE_PATH

    model.save(model_path)
    train_env.save(vec_path)

    print(f"RecurrentPPO model saved to {model_path}")
    print(f"VecNormalize saved to {vec_path}")


def load_recurrent_ppo_model(model_path: str = None):
    """
    Load a saved RecurrentPPO model.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded RecurrentPPO model
    """
    if not RECURRENT_PPO_AVAILABLE:
        raise ImportError(
            "RecurrentPPO requires sb3-contrib. "
            "Install with: pip install sb3-contrib"
        )

    if model_path is None:
        model_path = RECURRENT_PPO_MODEL_PATH

    return RecurrentPPO.load(model_path)


def create_model(
    env: VecNormalize,
    recurrent: bool = False,
    tensorboard_log: str = None,
    verbose: int = 1,
    **kwargs
):
    """
    Factory function to create either PPO or RecurrentPPO model.

    Args:
        env: Training environment
        recurrent: If True, create RecurrentPPO; else create PPO
        tensorboard_log: Tensorboard log path
        verbose: Verbosity
        **kwargs: Additional arguments passed to model constructor

    Returns:
        PPO or RecurrentPPO model
    """
    if recurrent:
        return create_recurrent_ppo_model(
            env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs
        )
    else:
        return create_ppo_model(
            env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs
        )


def train_model(model, total_timesteps: int = PPO_TIMESTEPS, callbacks: list = None):
    """
    Train a PPO or RecurrentPPO model.

    Args:
        model: PPO or RecurrentPPO model
        total_timesteps: Training steps
        callbacks: Callbacks list

    Returns:
        Trained model
    """
    model_type = type(model).__name__
    print(f"\nTraining {model_type} for {total_timesteps:,} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    return model


def save_model(model, train_env: VecNormalize, model_path: str = None, vec_path: str = None):
    """
    Save a PPO or RecurrentPPO model.

    Args:
        model: Trained model
        train_env: Training environment
        model_path: Model save path
        vec_path: VecNormalize save path
    """
    is_recurrent = RECURRENT_PPO_AVAILABLE and isinstance(model, RecurrentPPO)

    if model_path is None:
        model_path = RECURRENT_PPO_MODEL_PATH if is_recurrent else PPO_MODEL_PATH
    if vec_path is None:
        vec_path = VEC_NORMALIZE_PATH

    model.save(model_path)
    train_env.save(vec_path)

    model_type = "RecurrentPPO" if is_recurrent else "PPO"
    print(f"{model_type} model saved to {model_path}")
    print(f"VecNormalize saved to {vec_path}")


def load_model(model_path: str, recurrent: bool = False):
    """
    Load a saved PPO or RecurrentPPO model.

    Args:
        model_path: Path to model
        recurrent: If True, load as RecurrentPPO

    Returns:
        Loaded model
    """
    if recurrent:
        return load_recurrent_ppo_model(model_path)
    else:
        return load_ppo_model(model_path)


# =============================================================================
# CONTINUOUS ACTION SPACE PPO
# =============================================================================

def create_continuous_ppo_model(
    env: VecNormalize,
    learning_rate: float = 5e-4,
    n_steps: int = 512,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.05,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    seed: int = 42,
    tensorboard_log: str = None,
    verbose: int = 1
) -> PPO:
    """
    Create a PPO model for continuous action spaces.

    For continuous control, PPO uses a Gaussian policy where the network
    outputs mean and (log) std for each action dimension.

    Optimized hyperparameters for faster learning:
    - learning_rate=5e-4: Slightly more aggressive (from 3e-4)
    - n_steps=512: More frequent updates (from 2048)
    - ent_coef=0.05 with decay: High exploration early, convergence late
    - Tighter clipping for observations

    Args:
        env: VecNormalize wrapped training environment
        learning_rate: Learning rate
        n_steps: Steps to run before update (reduced for responsiveness)
        batch_size: Minibatch size
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Initial entropy coefficient (decays during training)
        clip_range: PPO clip range
        n_epochs: Number of epochs per update
        seed: Random seed
        tensorboard_log: Path for tensorboard logs
        verbose: Verbosity level

    Returns:
        PPO model configured for continuous action space
    """
    # Learning rate schedule: linear decay
    def lr_schedule(progress):
        return learning_rate * (1 - progress * 0.5)

    # Entropy schedule: decay from high exploration to low
    def ent_schedule(progress):
        return ent_coef * (1 - progress * 0.8)  # 0.05 → 0.01

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_schedule,
        clip_range=clip_range,
        n_epochs=n_epochs,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-2.0,  # Start with lower action variance
        ),
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed
    )

    return model


def create_continuous_recurrent_ppo_model(
    env: VecNormalize,
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 128,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    lstm_hidden_size: int = 64,
    n_lstm_layers: int = 1,
    seed: int = 42,
    tensorboard_log: str = None,
    verbose: int = 1
):
    """
    Create a RecurrentPPO model for continuous action spaces.

    Combines LSTM policy with continuous action output for temporal
    pattern learning with position sizing.

    Args:
        env: VecNormalize wrapped training environment
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Entropy coefficient
        clip_range: PPO clip range
        n_epochs: Epochs per update
        lstm_hidden_size: LSTM hidden layer size
        n_lstm_layers: Number of LSTM layers
        seed: Random seed
        tensorboard_log: Tensorboard log path
        verbose: Verbosity

    Returns:
        RecurrentPPO model for continuous action space
    """
    if not RECURRENT_PPO_AVAILABLE:
        raise ImportError(
            "RecurrentPPO requires sb3-contrib. "
            "Install with: pip install sb3-contrib"
        )

    # Learning rate schedule
    def lr_schedule(progress):
        return learning_rate * (1 - progress * 0.5)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=False,
            enable_critic_lstm=True,
        ),
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed
    )

    return model
