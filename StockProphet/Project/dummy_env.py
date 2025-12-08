import gymnasium as gym
import numpy as np

class DummyEnv(gym.Env):
    def __init__(self, n_features=20):
        super().__init__()

        self.n_features = n_features
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)  # 0 = Sell, 1 = Buy

        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs = np.zeros(self.n_features, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1

        obs = np.random.randn(self.n_features).astype(np.float32)
        reward = float(np.random.randn())
        terminated = self.current_step > 200
        truncated = False

        return obs, reward, terminated, truncated, {}
