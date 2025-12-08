from stable_baselines3 import PPO
from Project.dummy_env import DummyEnv

def train():
    env = DummyEnv(n_features=20)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        verbose=1
    )

    model.learn(total_timesteps=50_000)
    model.save("ppo_dummy_model")

    print("Training complete!")

if __name__ == "__main__":
    train()
