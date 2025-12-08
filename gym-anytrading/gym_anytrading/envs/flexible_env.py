import numpy as np

from .trading_env import TradingEnv, Actions, Positions
from .stocks_env import StocksEnv


class FlexibleTradingEnv(StocksEnv):
    def __init__(self, df, prices, signal_features, window_size, frame_bound, fee=0.001):
        self._prices = prices
        self._signal_features = signal_features
        self.fee = fee   # <-- your fee
        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound)

    # KEEP this
    def _process_data(self):
        return self._prices, self._signal_features

    # REPLACE default reward
    def _calculate_reward(self, action):
        price_now = self.prices[self._current_tick]
        price_prev = self.prices[self._current_tick - 1]
        price_diff = price_now - price_prev

        position_factor = 1 if self._position == Positions.Long else -1
        reward = position_factor * price_diff

        trade = (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        )

        if trade:
            reward -= self.fee * price_now

        return reward

    # REPLACE default profit update
    def _update_profit(self, action):
        price_now = self.prices[self._current_tick]
        price_prev = self.prices[self._current_tick - 1]

        if self._position == Positions.Long:
            factor = price_now / price_prev
        else:
            factor = price_prev / price_now

        trade = (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        )

        if trade:
            factor *= (1 - self.fee)

        self._total_profit *= factor
