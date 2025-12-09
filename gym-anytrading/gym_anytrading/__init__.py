from gymnasium.envs.registration import register
from copy import deepcopy

from . import datasets

register(
    id='stocks-v0',
    entry_point='gym_anytrading.envs:StocksEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)
