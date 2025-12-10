"""
Quick test: Does inverting actions make the agent profitable?
"""
import numpy as np
import sys
sys.path.insert(0, 'StockProphet')

from multiticker_refactor.config import *
from multiticker_refactor.envs.multi_asset_env import create_train_val_test_envs
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

# Load metadata
metadata = np.load('./StockProphet/saved_models_multi/metadata_multi.npy', allow_pickle=True).item()
ppo_data = metadata['ppo_data']
prices_array = ppo_data['prices']
signal_features = ppo_data['signal_features']
ticker_map = ppo_data['ticker_map']

# Create test env
_, _, test_env = create_train_val_test_envs(
    prices=prices_array,
    signal_features=signal_features,
    ticker_map=ticker_map,
    train_ratio=PPO_TRAIN_RATIO,
    val_ratio=PPO_VAL_RATIO,
    initial_capital=INITIAL_CAPITAL,
    transaction_fee_pct=TRANSACTION_FEE_PCT,
    short_borrow_rate=SHORT_BORROW_RATE,
    reward_volatility_window=REWARD_VOLATILITY_WINDOW
)

test_env_vec = DummyVecEnv([lambda: test_env])
test_env_vec = VecNormalize.load('./StockProphet/saved_models_multi/vec_normalize_multi.pkl', test_env_vec)
test_env_vec.training = False
test_env_vec.norm_reward = False

# Load model
model = RecurrentPPO.load('./StockProphet/saved_models_multi/ppo_multi_trading.zip')

print("=" * 70)
print("TESTING: INVERTED ACTIONS")
print("=" * 70)

obs = test_env_vec.reset()
done = False
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)

    # INVERT THE ACTIONS (flip sign)
    inverted_action = -action

    obs, reward, done, info = test_env_vec.step(inverted_action)
    step_count += 1

    if step_count % 50 == 0:
        print(f"Step {step_count}: Portfolio value = ${info[0]['portfolio_value']:,.2f}")

print("\n" + "=" * 70)
print("INVERTED ACTION RESULTS")
print("=" * 70)
print(f"\nTotal steps: {step_count}")
print(f"Final portfolio value: ${info[0]['portfolio_value']:,.2f}")
print(f"Total profit: ${info[0]['total_profit']:,.2f}")
print(f"Return: {(info[0]['total_profit'] / INITIAL_CAPITAL) * 100:.2f}%")
