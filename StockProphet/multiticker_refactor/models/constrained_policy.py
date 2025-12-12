"""
Constrained Recurrent Policy for bounded action spaces.

Fixes the std explosion problem by:
1. Clamping log_std to prevent explosion/collapse
2. Better initialization of log_std

This ensures the Gaussian distribution stays reasonable and actions
don't all collapse to the clip bounds due to std explosion.

NOTE: We keep the standard DiagGaussianDistribution (with clipping) but
constrain the std to stay in a reasonable range [0.14, 1.65]. This is
simpler and more stable than using SquashedDiagGaussianDistribution.
"""

import torch
from typing import Dict, Any, List
from gymnasium import spaces

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy


class ConstrainedRecurrentPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent Actor-Critic policy with constrained log_std.

    Key differences from default RecurrentActorCriticPolicy:
    1. Clamps log_std to [LOG_STD_MIN, LOG_STD_MAX] to prevent explosion
    2. Initializes log_std to a reasonable starting value

    This prevents the policy from outputting the same action every step
    due to std becoming so large that all samples hit clip bounds.

    Uses the standard DiagGaussianDistribution (with clipping) but ensures
    std stays in a reasonable range where clipping doesn't dominate.
    """

    # Bounds for log_std to prevent explosion/collapse
    LOG_STD_MIN = -2.0   # std >= 0.135 (prevents collapse)
    LOG_STD_MAX = 0.5    # std <= 1.65 (prevents explosion)
    LOG_STD_INIT = -0.5  # std ≈ 0.6 (reasonable starting exploration)

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        log_std_init: float = None,
        **kwargs
    ):
        # Use our default if not specified
        if log_std_init is None:
            log_std_init = self.LOG_STD_INIT

        # Store for later use
        self._custom_log_std_init = log_std_init

        # Initialize parent (will call _build)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            log_std_init=log_std_init,
            **kwargs
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the network. Parent creates DiagGaussianDistribution.
        We just re-initialize log_std to our custom value.
        """
        # Call parent _build first (creates self.action_dist as DiagGaussian)
        super()._build(lr_schedule)

        # Re-initialize log_std with our custom value
        if hasattr(self, 'log_std'):
            with torch.no_grad():
                self.log_std.fill_(self._custom_log_std_init)
            print(f"[ConstrainedPolicy] Initialized log_std to {self._custom_log_std_init}")
            print(f"[ConstrainedPolicy] log_std bounds: [{self.LOG_STD_MIN}, {self.LOG_STD_MAX}]")

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Get action distribution from latent features with clamped std.

        This is the key fix: we clamp log_std before creating the distribution.
        This keeps std in [0.14, 1.65] range where the Gaussian samples
        won't all hit the clip bounds.

        With std=0.6 and mean near 0, roughly:
        - 68% of samples within [-0.6, 0.6]
        - 95% of samples within [-1.2, 1.2]
        - Only ~5% hit the clip bounds at ±1
        """
        mean_actions = self.action_net(latent_pi)

        # CRITICAL: Clamp log_std to prevent explosion/collapse
        clamped_log_std = torch.clamp(
            self.log_std,
            min=self.LOG_STD_MIN,
            max=self.LOG_STD_MAX
        )

        # Use the standard DiagGaussianDistribution (parent created it)
        # Actions will still be clipped to [-1, 1] but with reasonable std,
        # most samples will be within bounds
        return self.action_dist.proba_distribution(mean_actions, clamped_log_std)


def get_constrained_policy_kwargs(
    net_arch: Dict[str, List[int]] = None,
    lstm_hidden_size: int = 64,
    n_lstm_layers: int = 1,
    log_std_init: float = -0.5,
    log_std_min: float = -2.0,
    log_std_max: float = 0.5,
) -> Dict[str, Any]:
    """
    Get policy_kwargs for RecurrentPPO with constrained policy.

    Args:
        net_arch: Network architecture for pi and vf
        lstm_hidden_size: LSTM hidden size
        n_lstm_layers: Number of LSTM layers
        log_std_init: Initial log_std value (default -0.5 → std ≈ 0.6)
        log_std_min: Minimum log_std (default -2.0 → std >= 0.135)
        log_std_max: Maximum log_std (default 0.5 → std <= 1.65)

    Returns:
        Dict of policy kwargs for RecurrentPPO
    """
    if net_arch is None:
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    # Update class constants if custom bounds provided
    ConstrainedRecurrentPolicy.LOG_STD_MIN = log_std_min
    ConstrainedRecurrentPolicy.LOG_STD_MAX = log_std_max
    ConstrainedRecurrentPolicy.LOG_STD_INIT = log_std_init

    return dict(
        net_arch=net_arch,
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
        shared_lstm=False,
        enable_critic_lstm=True,
        log_std_init=log_std_init,
    )
