"""
Evaluation functions for RecurrentPPO trading agents.
Source: Reinforcement.ipynb evaluate_agent function

Supports:
- RecurrentPPO (with LSTM state management)
- Discrete environments (FlexibleTradingEnv)
- Continuous environments (ContinuousTradingEnv) with dollar-based metrics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO

from .config import ENV_TYPE, INITIAL_CAPITAL


def _is_recurrent_model(model) -> bool:
    """Check if model is a RecurrentPPO model."""
    return isinstance(model, RecurrentPPO)


def _is_continuous_env(base_env) -> bool:
    """Check if environment is continuous (has position_weight attribute)."""
    return hasattr(base_env, '_position_weight')


def evaluate_agent(model, vec_env: VecNormalize, episodes: int = 1, fee: float = 0.0005) -> dict:
    """
    Comprehensive evaluation for a trading agent.

    Supports both standard PPO and RecurrentPPO models.
    For RecurrentPPO, properly manages LSTM hidden states.

    Includes:
    - Equity curve with transaction costs
    - Multiple benchmarks (Buy & Hold, Short & Hold)
    - Detailed performance metrics
    - Position analysis
    - Drawdown visualization
    - Trade markers

    Args:
        model: Trained PPO or RecurrentPPO model
        vec_env: VecNormalize wrapped test environment
        episodes: Number of evaluation episodes
        fee: Transaction fee per trade

    Returns:
        Dictionary with evaluation metrics and data
    """
    # Get raw underlying environment
    base_env = vec_env.venv.envs[0].unwrapped
    all_prices = base_env.prices.astype(float)
    start_tick = base_env._start_tick

    # Check if using RecurrentPPO
    is_recurrent = _is_recurrent_model(model)
    model_type = "RecurrentPPO" if is_recurrent else "PPO"

    print("\n" + "=" * 60)
    print(f"EVALUATION STARTED ({model_type})")
    print("=" * 60)
    print(f"Environment starts at tick {start_tick}")

    for ep in range(episodes):

        obs = vec_env.reset()
        done = False

        # Initialize LSTM states for RecurrentPPO
        if is_recurrent:
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)
        else:
            lstm_state = None
            episode_start = None

        # Track equity and positions indexed by tick for proper alignment
        equity_by_tick = {}
        equity_gross_by_tick = {}
        positions_by_tick = {}

        positions = []
        ticks = []
        actions_taken = []

        # Start portfolio at 1.0
        current_equity_gross = 1.0
        current_equity_net = 1.0
        last_position = 0

        # Get initial tick after reset
        current_tick = base_env._current_tick
        ticks.append(current_tick)
        equity_by_tick[current_tick] = current_equity_net
        equity_gross_by_tick[current_tick] = current_equity_gross
        positions_by_tick[current_tick] = 0
        positions.append(0)

        while not done:
            # Predict with LSTM state management for RecurrentPPO
            if is_recurrent:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)  # Not episode start after first step
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = vec_env.step(action)

            done = bool(done[0])
            info = info[0]

            current_tick = info["tick"]
            pos = info["position"]

            ticks.append(current_tick)
            positions.append(pos)
            positions_by_tick[current_tick] = pos
            actions_taken.append(int(action[0]) if hasattr(action, '__len__') else int(action))

            # Update equity using actual price indices
            if len(ticks) > 1:
                prev_tick = ticks[-2]
                price_ratio = all_prices[current_tick] / all_prices[prev_tick]

                # Gross equity (no fees)
                current_equity_gross *= price_ratio ** last_position

                # Net equity (with fees)
                current_equity_net *= price_ratio ** last_position
                if pos != last_position:
                    if last_position == 0:
                        current_equity_net *= (1 - fee)
                    elif pos == 0:
                        current_equity_net *= (1 - fee)
                    else:  # flip
                        current_equity_net *= (1 - 2 * fee)

            equity_by_tick[current_tick] = current_equity_net
            equity_gross_by_tick[current_tick] = current_equity_gross
            last_position = pos

        # ============================================
        # BENCHMARKS - aligned by tick
        # ============================================
        first_tick = ticks[0]
        last_tick = ticks[-1]

        # Create aligned arrays using actual ticks
        aligned_ticks = sorted(equity_by_tick.keys())
        equity_net = [equity_by_tick[t] for t in aligned_ticks]
        equity_gross = [equity_gross_by_tick[t] for t in aligned_ticks]

        # Buy & Hold: computed at each tick visited
        first_price = all_prices[first_tick]
        buy_hold = [all_prices[t] / first_price for t in aligned_ticks]
        short_hold = [first_price / all_prices[t] for t in aligned_ticks]

        # Price segment at each tick
        price_segment = [all_prices[t] for t in aligned_ticks]

        # ============================================
        # TRADE ANALYSIS
        # ============================================
        positions_arr = np.array(positions)
        position_changes = np.diff(positions_arr)
        trade_indices = np.where(position_changes != 0)[0] + 1

        # Calculate per-trade returns
        trade_returns = []
        if len(trade_indices) > 1:
            for i in range(len(trade_indices) - 1):
                start_idx = trade_indices[i]
                end_idx = trade_indices[i + 1]
                pos_during = positions_arr[start_idx]

                start_price = all_prices[ticks[start_idx]]
                end_price = all_prices[ticks[end_idx]]

                if pos_during == 1:  # Long
                    ret = (end_price / start_price) - 1
                elif pos_during == -1:  # Short
                    ret = (start_price / end_price) - 1
                else:
                    ret = 0
                trade_returns.append(ret)

        trade_returns = np.array(trade_returns)
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]

        # ============================================
        # METRICS
        # ============================================
        equity_arr = np.array(equity_net)
        buy_hold_arr = np.array(buy_hold)
        returns = np.diff(np.log(equity_arr + 1e-12))

        # Core metrics
        total_return = (equity_arr[-1] / equity_arr[0]) - 1
        total_return_gross = (equity_gross[-1] / equity_gross[0]) - 1
        bh_return = (buy_hold_arr[-1] / buy_hold_arr[0]) - 1
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        # Drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max
        max_dd = np.max(drawdowns)

        # Calmar ratio
        calmar = (total_return * 252 / len(returns)) / (max_dd + 1e-8) if max_dd > 0 else 0

        # Trade metrics
        n_trades = len(trade_returns)
        win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0
        avg_win = np.mean(winning_trades) * 100 if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) * 100 if len(losing_trades) > 0 else 0
        profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades)) if len(losing_trades) > 0 and np.sum(losing_trades) != 0 else np.inf

        # Position analysis
        long_pct = np.sum(positions_arr == 1) / len(positions_arr) * 100
        short_pct = np.sum(positions_arr == -1) / len(positions_arr) * 100
        flat_pct = np.sum(positions_arr == 0) / len(positions_arr) * 100

        # Alignment check
        ppo_vs_bh = equity_arr - buy_hold_arr
        ahead_count = np.sum(ppo_vs_bh > 0)
        behind_count = np.sum(ppo_vs_bh < 0)

        # ============================================
        # PRINT RESULTS
        # ============================================
        print(f"\n{'─' * 60}")
        print(f"EPISODE {ep + 1} RESULTS")
        print(f"{'─' * 60}")

        print(f"\n📊 RETURNS")
        print(f"   Total Return (net):    {total_return * 100:+.2f}%")
        print(f"   Total Return (gross):  {total_return_gross * 100:+.2f}%")
        print(f"   Buy & Hold Return:     {bh_return * 100:+.2f}%")
        print(f"   Short & Hold Return:   {(short_hold[-1] - 1) * 100:+.2f}%")
        print(f"   Outperformance vs B&H: {(total_return - bh_return) * 100:+.2f}%")

        print(f"\n📉 ALIGNMENT CHECK")
        print(f"   Steps where PPO > B&H: {ahead_count} ({100 * ahead_count / len(equity_arr):.1f}%)")
        print(f"   Steps where PPO < B&H: {behind_count} ({100 * behind_count / len(equity_arr):.1f}%)")

        print(f"\n📈 RISK METRICS")
        print(f"   Sharpe Ratio (ann.):   {sharpe:.3f}")
        print(f"   Max Drawdown:          {max_dd * 100:.2f}%")
        print(f"   Calmar Ratio:          {calmar:.3f}")

        print(f"\n🔄 TRADE ANALYSIS")
        print(f"   Total Trades:          {n_trades}")
        print(f"   Win Rate:              {win_rate:.1f}%")
        print(f"   Avg Winning Trade:     {avg_win:+.2f}%")
        print(f"   Avg Losing Trade:      {avg_loss:+.2f}%")
        print(f"   Profit Factor:         {profit_factor:.2f}")

        print(f"\n⚖️  POSITION DISTRIBUTION")
        print(f"   Long:  {long_pct:5.1f}%  {'█' * int(long_pct / 5)}")
        print(f"   Short: {short_pct:5.1f}%  {'█' * int(short_pct / 5)}")
        print(f"   Flat:  {flat_pct:5.1f}%  {'█' * int(flat_pct / 5)}")

        # ============================================
        # PLOT 1: EQUITY CURVES
        # ============================================
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        x_axis = np.arange(len(aligned_ticks))

        # Panel 1: Price + Equity
        ax1 = axes[0]
        ax1_twin = ax1.twinx()

        ax1.plot(x_axis, price_segment, color="blue", alpha=0.4, linewidth=1, label="Price")
        ax1.set_ylabel("Price", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax1_twin.plot(x_axis, equity_net, color="green", linewidth=2, label="PPO (net)")
        ax1_twin.plot(x_axis, equity_gross, color="lightgreen", linewidth=1, linestyle="--", label="PPO (gross)", alpha=0.7)
        ax1_twin.plot(x_axis, buy_hold, color="gray", linewidth=1.5, linestyle="--", label="Buy & Hold")
        ax1_twin.plot(x_axis, short_hold, color="red", linewidth=1, linestyle=":", alpha=0.5, label="Short & Hold")
        ax1_twin.axhline(y=1.0, color="black", linestyle="-", alpha=0.3)
        ax1_twin.set_ylabel("Equity", color="green")
        ax1_twin.tick_params(axis="y", labelcolor="green")
        ax1_twin.legend(loc="upper left")

        ax1.set_title("Equity Curves vs Benchmarks (Aligned by Tick)", fontsize=12, fontweight="bold")
        ax1.grid(alpha=0.3)

        # Panel 2: Drawdown
        ax2 = axes[1]
        ax2.fill_between(x_axis, 0, -drawdowns * 100, color="red", alpha=0.3)
        ax2.plot(x_axis, -drawdowns * 100, color="red", linewidth=1)
        ax2.axhline(y=-max_dd * 100, color="darkred", linestyle="--", label=f"Max DD: {max_dd * 100:.1f}%")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Underwater Curve (Drawdown)", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)

        # Panel 3: Position over time with trade markers
        ax3 = axes[2]

        for i in range(len(x_axis) - 1):
            if i < len(positions):
                if positions[i] == 1:
                    ax3.axvspan(i, i + 1, alpha=0.3, color="green")
                elif positions[i] == -1:
                    ax3.axvspan(i, i + 1, alpha=0.3, color="red")

        ax3.plot(x_axis, price_segment, color="blue", linewidth=1, alpha=0.7)

        for idx in trade_indices:
            if idx < len(x_axis):
                marker = "^" if positions[idx] == 1 else "v"
                color = "green" if positions[idx] == 1 else "red"
                ax3.scatter(idx, price_segment[idx], marker=marker, color=color, s=50, zorder=5)

        ax3.set_ylabel("Price")
        ax3.set_xlabel("Step")
        ax3.set_title("Position & Trade Markers (▲=Long, ▼=Short)", fontsize=12, fontweight="bold")
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ============================================
        # PLOT 2: TRADE RETURN DISTRIBUTION
        # ============================================
        if len(trade_returns) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            ax1 = axes[0]
            ax1.hist(trade_returns * 100, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
            ax1.axvline(x=0, color="black", linestyle="--")
            ax1.axvline(x=np.mean(trade_returns) * 100, color="green", linestyle="-", label=f"Mean: {np.mean(trade_returns) * 100:.2f}%")
            ax1.set_xlabel("Trade Return (%)")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Trade Return Distribution", fontsize=12, fontweight="bold")
            ax1.legend()
            ax1.grid(alpha=0.3)

            ax2 = axes[1]
            cum_trade_returns = np.cumprod(1 + trade_returns)
            ax2.plot(cum_trade_returns, color="green", linewidth=2)
            ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Trade #")
            ax2.set_ylabel("Cumulative Return")
            ax2.set_title("Cumulative Returns by Trade", fontsize=12, fontweight="bold")
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60 + "\n")

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "equity_curve": equity_net,
        "buy_hold": buy_hold,
        "positions": positions,
        "ticks": aligned_ticks,
        "ppo_ahead_pct": 100 * ahead_count / len(equity_arr),
        "ppo_behind_pct": 100 * behind_count / len(equity_arr)
    }


def quick_evaluate(model, vec_env: VecNormalize) -> dict:
    """
    Quick evaluation without plotting.

    Supports both PPO and RecurrentPPO models.

    Args:
        model: Trained PPO or RecurrentPPO model
        vec_env: VecNormalize wrapped test environment

    Returns:
        Dictionary with basic metrics
    """
    base_env = vec_env.venv.envs[0].unwrapped
    all_prices = base_env.prices.astype(float)

    # Check if using RecurrentPPO
    is_recurrent = _is_recurrent_model(model)

    obs = vec_env.reset()
    done = False
    equity = 1.0
    n_trades = 0
    last_pos = 0
    ticks = [base_env._current_tick]

    # Initialize LSTM states for RecurrentPPO
    if is_recurrent:
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

    while not done:
        # Predict with LSTM state management for RecurrentPPO
        if is_recurrent:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, done, info = vec_env.step(action)
        done = bool(done[0])

        tick = info[0]["tick"]
        pos = info[0]["position"]
        ticks.append(tick)

        if len(ticks) > 1:
            price_ratio = all_prices[tick] / all_prices[ticks[-2]]
            equity *= price_ratio ** last_pos

        if pos != last_pos:
            n_trades += 1

        last_pos = pos

    # Buy & Hold
    bh_return = (all_prices[ticks[-1]] / all_prices[ticks[0]]) - 1
    total_return = equity - 1

    return {
        "total_return": total_return,
        "buy_hold_return": bh_return,
        "outperformance": total_return - bh_return,
        "n_trades": n_trades
    }


# =============================================================================
# CONTINUOUS ENVIRONMENT EVALUATION (Dollar-based)
# =============================================================================

def evaluate_continuous_agent(model, vec_env: VecNormalize, episodes: int = 1) -> dict:
    """
    Comprehensive evaluation for continuous trading agent with dollar-based metrics.

    Designed for ContinuousTradingEnv with Box(-1, 1) action space.

    Includes:
    - Dollar-based equity curve
    - Position weight over time
    - Comparison with Buy & Hold
    - Transaction costs from actual trading
    - Risk metrics

    Args:
        model: Trained PPO or RecurrentPPO model
        vec_env: VecNormalize wrapped test environment
        episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation metrics and data
    """
    base_env = vec_env.venv.envs[0].unwrapped
    all_prices = base_env.prices.astype(float)
    initial_capital = base_env.initial_capital
    start_tick = base_env._start_tick

    is_recurrent = _is_recurrent_model(model)
    model_type = "RecurrentPPO" if is_recurrent else "PPO"

    print("\n" + "=" * 60)
    print(f"CONTINUOUS EVALUATION ({model_type})")
    print("=" * 60)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Environment starts at tick {start_tick}")

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False

        # Initialize LSTM states for RecurrentPPO
        if is_recurrent:
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)

        # Tracking arrays
        ticks = []
        portfolio_values = []
        position_weights = []
        actions_taken = []
        cash_values = []
        shares_held = []

        # Get initial state
        current_tick = base_env._current_tick
        ticks.append(current_tick)
        portfolio_values.append(initial_capital)
        position_weights.append(0.0)
        cash_values.append(initial_capital)
        shares_held.append(0.0)

        while not done:
            # Predict action
            if is_recurrent:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = vec_env.step(action)

            done = bool(done[0])
            info = info[0]

            ticks.append(info["tick"])
            portfolio_values.append(info["portfolio_value"])
            position_weights.append(info["position_weight"])
            cash_values.append(info["cash"])
            shares_held.append(info["shares"])
            actions_taken.append(float(action[0]) if hasattr(action, '__len__') else float(action))

        # Convert to arrays
        portfolio_arr = np.array(portfolio_values)
        position_arr = np.array(position_weights)
        ticks_arr = np.array(ticks)

        # ============================================
        # BENCHMARKS
        # ============================================
        first_tick = ticks[0]
        first_price = all_prices[first_tick]

        # Buy & Hold: fully invested from start
        buy_hold = [initial_capital * all_prices[t] / first_price for t in ticks]
        buy_hold_arr = np.array(buy_hold)

        # Short & Hold
        short_hold = [initial_capital * first_price / all_prices[t] for t in ticks]
        short_hold_arr = np.array(short_hold)

        # Price segment
        price_segment = [all_prices[t] for t in ticks]

        # ============================================
        # METRICS
        # ============================================
        final_value = portfolio_arr[-1]
        total_profit = final_value - initial_capital
        total_return = total_profit / initial_capital
        bh_final = buy_hold_arr[-1]
        bh_return = (bh_final - initial_capital) / initial_capital

        # Daily returns
        returns = np.diff(portfolio_arr) / portfolio_arr[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        # Drawdown
        running_max = np.maximum.accumulate(portfolio_arr)
        drawdowns = (running_max - portfolio_arr) / running_max
        max_dd = np.max(drawdowns)
        max_dd_dollar = np.max(running_max - portfolio_arr)

        # Trade count (significant position changes)
        position_changes = np.abs(np.diff(position_arr))
        n_trades = np.sum(position_changes > 0.05)  # Count changes > 5%

        # Position statistics
        avg_position = np.mean(np.abs(position_arr))
        avg_long = np.mean(position_arr[position_arr > 0]) if np.any(position_arr > 0) else 0
        avg_short = np.mean(position_arr[position_arr < 0]) if np.any(position_arr < 0) else 0
        time_long = np.sum(position_arr > 0.1) / len(position_arr) * 100
        time_short = np.sum(position_arr < -0.1) / len(position_arr) * 100
        time_neutral = np.sum(np.abs(position_arr) <= 0.1) / len(position_arr) * 100

        # Comparison
        ahead_count = np.sum(portfolio_arr > buy_hold_arr)
        behind_count = np.sum(portfolio_arr < buy_hold_arr)

        # ============================================
        # PRINT RESULTS
        # ============================================
        print(f"\n{'─' * 60}")
        print(f"EPISODE {ep + 1} RESULTS")
        print(f"{'─' * 60}")

        print(f"\n💰 DOLLAR RETURNS")
        print(f"   Starting Capital:      ${initial_capital:,.2f}")
        print(f"   Final Portfolio:       ${final_value:,.2f}")
        print(f"   Total Profit/Loss:     ${total_profit:+,.2f}")
        print(f"   Return:                {total_return * 100:+.2f}%")

        print(f"\n📊 BENCHMARKS")
        print(f"   Buy & Hold Final:      ${bh_final:,.2f}")
        print(f"   Buy & Hold Return:     {bh_return * 100:+.2f}%")
        print(f"   Outperformance:        ${final_value - bh_final:+,.2f} ({(total_return - bh_return) * 100:+.2f}%)")

        print(f"\n📈 RISK METRICS")
        print(f"   Sharpe Ratio (ann.):   {sharpe:.3f}")
        print(f"   Max Drawdown:          {max_dd * 100:.2f}% (${max_dd_dollar:,.2f})")

        print(f"\n⚖️  POSITION ANALYSIS")
        print(f"   Avg Position Size:     {avg_position * 100:.1f}%")
        print(f"   Time Long (>10%):      {time_long:.1f}%")
        print(f"   Time Short (<-10%):    {time_short:.1f}%")
        print(f"   Time Neutral:          {time_neutral:.1f}%")
        print(f"   Significant Trades:    {n_trades}")

        print(f"\n📉 ALIGNMENT")
        print(f"   Steps ahead of B&H:    {ahead_count} ({100 * ahead_count / len(portfolio_arr):.1f}%)")
        print(f"   Steps behind B&H:      {behind_count} ({100 * behind_count / len(portfolio_arr):.1f}%)")

        # ============================================
        # PLOT 1: PORTFOLIO VALUE CURVES
        # ============================================
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        x_axis = np.arange(len(ticks))

        # Panel 1: Portfolio Value vs Benchmarks
        ax1 = axes[0]
        ax1.plot(x_axis, portfolio_arr, color="green", linewidth=2, label="Agent Portfolio")
        ax1.plot(x_axis, buy_hold_arr, color="gray", linewidth=1.5, linestyle="--", label="Buy & Hold")
        ax1.plot(x_axis, short_hold_arr, color="red", linewidth=1, linestyle=":", alpha=0.5, label="Short & Hold")
        ax1.axhline(y=initial_capital, color="black", linestyle="-", alpha=0.3, label="Initial Capital")
        ax1.fill_between(x_axis, initial_capital, portfolio_arr,
                        where=portfolio_arr >= initial_capital, color="green", alpha=0.2)
        ax1.fill_between(x_axis, initial_capital, portfolio_arr,
                        where=portfolio_arr < initial_capital, color="red", alpha=0.2)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(f"Portfolio Performance: ${initial_capital:,.0f} → ${final_value:,.0f} ({total_return * 100:+.1f}%)",
                     fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Panel 2: Position Weight Over Time
        ax2 = axes[1]
        ax2.fill_between(x_axis, 0, position_arr, where=np.array(position_arr) >= 0,
                        color="green", alpha=0.4, label="Long")
        ax2.fill_between(x_axis, 0, position_arr, where=np.array(position_arr) < 0,
                        color="red", alpha=0.4, label="Short")
        ax2.plot(x_axis, position_arr, color="blue", linewidth=1)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax2.axhline(y=1, color="green", linestyle="--", alpha=0.3)
        ax2.axhline(y=-1, color="red", linestyle="--", alpha=0.3)
        ax2.set_ylabel("Position Weight")
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_title("Position Weight Over Time (+ = Long, - = Short)", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.3)

        # Panel 3: Drawdown
        ax3 = axes[2]
        ax3.fill_between(x_axis, 0, -drawdowns * 100, color="red", alpha=0.3)
        ax3.plot(x_axis, -drawdowns * 100, color="red", linewidth=1)
        ax3.axhline(y=-max_dd * 100, color="darkred", linestyle="--",
                   label=f"Max DD: {max_dd * 100:.1f}% (${max_dd_dollar:,.0f})")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_xlabel("Step")
        ax3.set_title("Underwater Curve (Drawdown)", fontsize=12, fontweight="bold")
        ax3.legend(loc="lower left")
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ============================================
        # PLOT 2: POSITION DISTRIBUTION & ACTIONS
        # ============================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Position weight distribution
        ax1 = axes[0]
        ax1.hist(position_arr, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
        ax1.axvline(x=0, color="black", linestyle="--")
        ax1.axvline(x=np.mean(position_arr), color="green", linestyle="-",
                   label=f"Mean: {np.mean(position_arr):.2f}")
        ax1.set_xlabel("Position Weight")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Position Weight Distribution", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Actions taken distribution
        ax2 = axes[1]
        ax2.hist(actions_taken, bins=50, color="purple", edgecolor="white", alpha=0.7)
        ax2.axvline(x=0, color="black", linestyle="--")
        ax2.axvline(x=np.mean(actions_taken), color="green", linestyle="-",
                   label=f"Mean: {np.mean(actions_taken):.2f}")
        ax2.set_xlabel("Action (Target Position)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Actions Taken Distribution", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60 + "\n")

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_profit": total_profit,
        "total_return": total_return,
        "buy_hold_return": bh_return,
        "outperformance": total_return - bh_return,
        "outperformance_dollar": final_value - bh_final,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_dollar": max_dd_dollar,
        "n_trades": n_trades,
        "avg_position": avg_position,
        "time_long_pct": time_long,
        "time_short_pct": time_short,
        "time_neutral_pct": time_neutral,
        "portfolio_history": portfolio_arr.tolist(),
        "position_history": position_arr.tolist(),
        "buy_hold_history": buy_hold_arr.tolist(),
        "ticks": ticks,
    }


def evaluate_agent_auto(model, vec_env: VecNormalize, episodes: int = 1, fee: float = 0.0005) -> dict:
    """
    Automatically select evaluation function based on environment type.

    Args:
        model: Trained PPO or RecurrentPPO model
        vec_env: VecNormalize wrapped test environment
        episodes: Number of evaluation episodes
        fee: Transaction fee (for discrete env)

    Returns:
        Dictionary with evaluation metrics
    """
    base_env = vec_env.venv.envs[0].unwrapped

    if _is_continuous_env(base_env):
        return evaluate_continuous_agent(model, vec_env, episodes)
    else:
        return evaluate_agent(model, vec_env, episodes, fee)
