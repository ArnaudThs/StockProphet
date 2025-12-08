from stable_baselines3 import PPO

import sys, os
sys.path.append(os.path.abspath("/../../gym-anytrading"))

from gym_anytrading.envs.flexible_env import FlexibleEnv

def train():

    env = FlexibleEnv(
        df=None,
        window_size=50,
        frame_bound=(50, 300)
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        verbose=1
    )

    model.learn(total_timesteps=50_000)
    model.save("ppo_flexible_env_model")

    print("Training complete!")

if __name__ == "__main__":
    train()
