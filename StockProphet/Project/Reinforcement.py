#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import sys
sys.path.append("/Users/pnl1f276/code/bennystu/trend_surgeon/Trend-Surgeon-Time-Series/ts_boilerplate")

from gym_anytrading.envs.flexible_env import FlexibleTradingEnv
from dataprep import build_feature_dataset, get_X_y


# In[18]:


df = build_feature_dataset()


# In[19]:


df.head()


# In[20]:


prices = df["target_close"].values.astype(np.float32)

feature_cols = df.columns.drop("target_close")
signal_features = df[feature_cols].values.astype(np.float32)


# In[21]:


window_size = 30
frame_bound = (window_size, len(df))


# In[22]:


def make_env():
    def _init():
        return FlexibleTradingEnv(
            df=df,
            prices=prices,
            signal_features=signal_features,
            window_size=window_size,
            frame_bound=frame_bound,
            fee=0.0005,
            holding_cost=0.00001,
            render_mode=None
        )
    return _init


# In[23]:


train_env = DummyVecEnv([make_env()])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_reward=np.inf)

eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)


# In[24]:


model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./ppo_logs/"
)


# In[25]:


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./ppo_best_model/",
    log_path="./ppo_eval_logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./ppo_checkpoints/",
    name_prefix="ppo_trading_model"
)


# In[26]:


model.learn(
    total_timesteps=200_000,
    callback=[eval_callback, checkpoint_callback]
)

model.save("ppo_trading_final")
train_env.save("vec_normalize.pkl")


# In[27]:


eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
eval_env.training = False

model = PPO.load("ppo_trading_final")

obs = eval_env.reset()
done = False
profits = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    profits.append(info[0]["total_profit"])

plt.plot(profits)
plt.title("Agent Performance")
plt.xlabel("Step")
plt.ylabel("Portfolio Value")
plt.grid()
plt.show()


# In[34]:


import numpy as np
import matplotlib.pyplot as plt


def evaluate_agent(model, vec_env, episodes=1):
    """
    Clean evaluation for SB3 PPO + VecNormalize + custom trading env.

    What this produces:
    -------------------------------------------------------------
    • PPO final equity
    • Buy & Hold benchmark
    • Total trades
    • Sharpe-like metric
    • Max drawdown
    • Plot: Price (left axis), PPO equity & Buy & Hold (right axis)

    IMPORTANT:
    The environment stores reward as cumulative log-return.
    We exponentiate cumulative PnL → true portfolio equity.
    """

    # Extract underlying env (NOT normalized)
    base_env = vec_env.venv.envs[0].unwrapped
    raw_prices = base_env.prices

    print("\n=== EVALUATION STARTED ===")

    for ep in range(episodes):

        obs = vec_env.reset()
        done = False

        cumulative_log_pnl = []
        positions = []
        steps = 0

        while not done:
            # Deterministic policy for evaluation
            action, _ = model.predict(obs, deterministic=True)

            # SB3 VecNormalize → returns obs, reward, done, info (4 values)
            obs, reward, done, info = vec_env.step(action)

            reward = float(reward[0])
            done = bool(done[0])
            info = info[0]

            cumulative_log_pnl.append(info["total_profit"])  # log-return PnL
            positions.append(info["position"])
            steps += 1

        # -----------------------------
        # Convert log returns → equity
        # -----------------------------
        cumulative_log_pnl = np.array(cumulative_log_pnl)
        ppo_equity = np.exp(cumulative_log_pnl)  # start ≈ 1.0

        # -----------------------------
        # Buy & Hold benchmark
        # -----------------------------
        price_segment = raw_prices[:steps]
        buy_hold_equity = price_segment / price_segment[0]

        # -----------------------------
        # Metrics (computed on equity)
        # -----------------------------
        final_equity = ppo_equity[-1]
        returns = np.diff(np.log(ppo_equity + 1e-12))

        sharpe_like = np.mean(returns) / (np.std(returns) + 1e-8)
        mdd = np.max(np.maximum.accumulate(ppo_equity) - ppo_equity)
        trades = int(np.sum(np.diff(positions) != 0))

        print(f"\nEpisode {ep+1} Results")
        print(f"• Final Equity:        {final_equity:.2f}")
        print(f"• Total Steps:         {steps}")
        print(f"• Total Trades:        {trades}")
        print(f"• Sharpe-like Metric:  {sharpe_like:.4f}")
        print(f"• Max Drawdown:        {mdd:.2f}")

        # -----------------------------
        # Plotting
        # -----------------------------
        fig, ax1 = plt.subplots(figsize=(13, 6))

        # Price on left
        ax1.plot(price_segment, color="blue", alpha=0.5, label="Price")
        ax1.set_ylabel("Price", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Equity curves on right
        ax2 = ax1.twinx()
        ax2.plot(ppo_equity, color="green", linewidth=2, label="PPO Portfolio")
        ax2.plot(buy_hold_equity, color="gray", linestyle="--", label="Buy & Hold")
        ax2.set_ylabel("Equity", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        fig.suptitle(f"PPO Evaluation — Episode {ep+1}")
        ax1.grid(alpha=0.3)
        ax2.legend(loc="upper left")

        plt.show()

    print("\n=== EVALUATION COMPLETE ===\n")


# In[35]:


vec_env = DummyVecEnv([lambda: FlexibleTradingEnv(
    df=df,
    prices=prices,
    signal_features=signal_features,
    window_size=window_size,
    frame_bound=frame_bound
)])

# Load normalization statistics
vec_env = VecNormalize.load("vec_normalize.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("ppo_trading_final")

evaluate_agent(model, vec_env, episodes=1)


# In[ ]:




