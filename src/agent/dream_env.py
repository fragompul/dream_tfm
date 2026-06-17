"""
D.R.E.A.M. Portfolio Management Environment.

Gymnasium-compatible environment for multi-asset portfolio management using
Deep Reinforcement Learning.  Integrates sentiment signals and volatility
forecasts from upstream pipeline modules to support informed capital allocation.

Architecture
------------
Action space:
    Box([-10, 10], shape=(N+1,)) — N+1 logits mapped to portfolio weights
    via softmax.  The first N outputs correspond to asset allocation weights;
    the (N+1)-th output represents cash.  Softmax guarantees weights sum to
    1.0 by construction, eliminating the need for ad-hoc normalisation or
    dead zones.

Observation space:
    Depends on the sentiment mode detected from the data feed:

    "macro" mode — obs_dim = 3 + 3*N:
        [norm_portfolio, cash_weight, sentiment_macro,
         z_ret_0, scaled_vol_0, weight_0,
         z_ret_1, scaled_vol_1, weight_1, ...]

    "per_asset" mode — obs_dim = 2 + 4*N:
        [norm_portfolio, cash_weight,
         z_ret_0, scaled_vol_0, sentiment_0, weight_0,
         z_ret_1, scaled_vol_1, sentiment_1, weight_1, ...]

    Each asset's current portfolio weight (value / portfolio_value) is used
    instead of the absolute normalised position.  This is scale-invariant
    and stays in [0, 1] regardless of how the portfolio grows or contracts.

Reward function:
    Information Ratio vs equal-weight benchmark.  The base reward measures
    how much the agent outperforms (or underperforms) a passive equal-weight
    buy-and-hold strategy on the same assets:

        base = ((step_return - bh_return) / daily_vol) * scale

    where bh_return = mean(log_returns) across the N assets in the episode.

    This formulation is zero-centred around the benchmark:
      - Beating B&H    → positive reward
      - Matching B&H   → zero reward
      - Cash when market rises → negative reward (explicit opportunity cost)

    Additional components:
      - MDD penalty:  -MDD_COEF * current_mdd  (if MDD > threshold)
      - Turnover penalty:  -TURNOVER_COEF * sum(|delta_w_i|)
      - Cash penalty:  -CASH_PENALTY * max(0, min_exposure - total_exposure)
        Enforces a minimum investment mandate: the agent must maintain
        at least `min_exposure` fraction of the portfolio invested.
        This is an operational constraint standard in fund management,
        not a strategy hint.

Configurable parameters:
    turnover_coef:  Override via __init__ (per_asset: 0.01, macro: 0.03).
    cash_penalty:   Override via __init__ (0.0 = disabled).
    min_exposure:   Override via __init__ (default 0.5 = 50% minimum).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class DreamEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Reward hyperparameters
    # ------------------------------------------------------------------
    REWARD_VOL_SCALE: float     = 1.0
    REWARD_MDD_THRESHOLD: float = 0.10
    REWARD_MDD_COEF: float      = 1.0
    REWARD_TURNOVER_COEF: float = 0.02
    REWARD_CASH_PENALTY: float  = 0.0    # configurable; 0.1 recommended for empirical
    REWARD_MIN_EXPOSURE: float  = 0.5    # threshold below which cash penalty activates
    VOL_ANN_FLOOR: float        = 0.05

    # ------------------------------------------------------------------
    # Observation normalisation
    # ------------------------------------------------------------------
    OBS_RETURN_WINDOW: int = 20
    OBS_VOL_SCALE: float   = 0.5

    # ------------------------------------------------------------------
    # Transaction mechanics
    # ------------------------------------------------------------------
    COMMISSION_RATE: float        = 0.001
    DELTA_WEIGHT_THRESHOLD: float = 0.02

    def __init__(
        self,
        data_generator,
        initial_balance: float = 10_000.0,
        turnover_coef: float | None = None,
        cash_penalty: float | None = None,
        min_exposure: float | None = None,
    ):
        """
        Args:
            data_generator:  WyckoffMockData or EmpiricalDataFeed instance.
            initial_balance: Starting portfolio value in currency units.
            turnover_coef:   Override for REWARD_TURNOVER_COEF.
                             Recommended: 0.01 (per_asset) | 0.03 (macro).
            cash_penalty:    Override for REWARD_CASH_PENALTY.
                             Recommended: 0.0 (synthetic) | 0.1 (empirical).
            min_exposure:    Override for REWARD_MIN_EXPOSURE.
                             Minimum fraction of portfolio that must be invested
                             before the cash penalty activates.  Default 0.5.
        """
        super().__init__()

        self.data_gen        = data_generator
        self.initial_balance = initial_balance

        if turnover_coef is not None:
            self.REWARD_TURNOVER_COEF = turnover_coef
        if cash_penalty is not None:
            self.REWARD_CASH_PENALTY = cash_penalty
        if min_exposure is not None:
            self.REWARD_MIN_EXPOSURE = min_exposure

        self.num_assets: int = getattr(self.data_gen, "num_assets", 1)
        self.max_steps: int  = self.data_gen.steps - 1

        self.sentiment_mode: str = getattr(
            self.data_gen, "sentiment_mode", "macro"
        )

        # ---- Action space: N assets + 1 cash logits ---------------------
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.num_assets + 1,),
            dtype=np.float32,
        )

        # ---- Observation space ------------------------------------------
        if self.sentiment_mode == "macro":
            obs_dim = 3 + 3 * self.num_assets
        else:
            obs_dim = 2 + 4 * self.num_assets

        low  = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim,  np.inf, dtype=np.float32)
        low[0] = 0.0   # norm_portfolio >= 0
        low[1] = 0.0   # cash_weight    >= 0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ---- Accounting state -------------------------------------------
        self.current_step: int           = 0
        self.balance: float              = initial_balance
        self.position_sizes: np.ndarray  = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value: float      = initial_balance
        self.peak_portfolio_value: float = initial_balance
        self.current_mdd: float          = 0.0
        self._return_history: deque      = deque(maxlen=self.OBS_RETURN_WINDOW)

        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.data_gen.generate_new_market()

        self.max_steps = (
            self.data_gen.current_market_steps - 1
            if hasattr(self.data_gen, "current_market_steps")
            else self.data_gen.steps - 1
        )

        self.current_step         = 0
        self.balance              = self.initial_balance
        self.position_sizes       = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value      = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.current_mdd          = 0.0
        self._return_history.clear()

        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading step: rebalance, mark-to-market, compute reward."""

        prices, log_returns, sentiment, volatility = self.data_gen.get_step_data(
            self.current_step
        )

        # 1. Target weights via softmax
        target_weights = self._action_to_weights(action)

        # 2. Pre-trade snapshot
        prev_portfolio_value = self.balance + float(
            np.sum(self.position_sizes * prices)
        )
        if prev_portfolio_value > 0:
            current_weights = (self.position_sizes * prices) / prev_portfolio_value
        else:
            current_weights = np.zeros(self.num_assets, dtype=np.float32)
        delta_weights = target_weights - current_weights

        # 3. Trade execution (sells first to free cash for buys)
        self._execute_sells(prices, delta_weights, prev_portfolio_value)
        self._execute_buys(prices, delta_weights, prev_portfolio_value)

        # 4. Mark to market
        self.portfolio_value = self.balance + float(
            np.sum(self.position_sizes * prices)
        )
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        self.current_mdd = (
            (self.peak_portfolio_value - self.portfolio_value)
            / self.peak_portfolio_value
        )

        # 5. Reward
        step_return = (
            (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            if prev_portfolio_value > 0 else 0.0
        )
        exposure_ratios = (
            (self.position_sizes * prices) / self.portfolio_value
            if self.portfolio_value > 0
            else np.zeros(self.num_assets, dtype=np.float32)
        )
        reward = self._compute_reward(
            step_return=step_return,
            log_returns=log_returns,
            volatility=volatility,
            delta_weights=delta_weights,
            exposure_ratios=exposure_ratios,
        )

        # 6. Advance
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        info = {
            "portfolio_value": self.portfolio_value,
            "mdd":             self.current_mdd,
            "step_return":     step_return,
            "tickers": (
                self.data_gen.current_tickers
                if hasattr(self.data_gen, "current_tickers")
                else "SYNTHETIC"
            ),
        }
        return self._get_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        Build the flat observation vector.

        Uses scale-invariant features: portfolio weights instead of absolute
        positions, and rolling z-scored returns to normalise across regimes.
        """
        prices, log_returns, sentiment, volatility = self.data_gen.get_step_data(
            self.current_step
        )

        # Rolling z-score of log returns
        self._return_history.append(log_returns.copy())
        if len(self._return_history) > 1:
            hist = np.array(self._return_history)
            z_log_returns = (log_returns - hist.mean(axis=0)) / (
                hist.std(axis=0) + 1e-8
            )
        else:
            z_log_returns = np.zeros(self.num_assets, dtype=np.float32)

        scaled_vol     = np.clip(volatility / self.OBS_VOL_SCALE, 0.0, 4.0)
        norm_portfolio = self.portfolio_value / self.initial_balance
        cash_weight    = (
            self.balance / self.portfolio_value
            if self.portfolio_value > 0 else 1.0
        )

        if self.portfolio_value > 0:
            asset_weights = (self.position_sizes * prices) / self.portfolio_value
        else:
            asset_weights = np.zeros(self.num_assets, dtype=np.float32)

        if self.sentiment_mode == "macro":
            obs = [norm_portfolio, cash_weight, float(sentiment)]
            for i in range(self.num_assets):
                obs.extend([
                    float(z_log_returns[i]),
                    float(scaled_vol[i]),
                    float(asset_weights[i]),
                ])
        else:
            obs = [norm_portfolio, cash_weight]
            for i in range(self.num_assets):
                obs.extend([
                    float(z_log_returns[i]),
                    float(scaled_vol[i]),
                    float(sentiment[i]),
                    float(asset_weights[i]),
                ])

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        step_return: float,
        log_returns: np.ndarray,
        volatility: np.ndarray,
        delta_weights: np.ndarray,
        exposure_ratios: np.ndarray,
    ) -> float:
        """
        Information Ratio vs equal-weight benchmark with operational constraints.

        Components:
            base       = ((step_return - bh_return) / daily_vol) * scale
            mdd_pen    = -MDD_COEF * current_mdd          (if MDD > threshold)
            turn_pen   = -TURNOVER_COEF * sum(|delta_w|)
            cash_pen   = -CASH_PENALTY * shortfall         (if exposure < min)

        The benchmark term (bh_return = mean of log_returns across assets)
        makes the reward zero-centred: positive when outperforming, negative
        when underperforming, and explicitly negative when holding cash while
        the market rises.

        The cash penalty enforces a minimum investment mandate — a standard
        operational constraint in fund management.  It does not prescribe
        which assets to hold, only that the agent must be invested above
        the minimum threshold.
        """
        ann_vol        = max(float(np.mean(volatility)), self.VOL_ANN_FLOOR)
        daily_vol      = ann_vol / np.sqrt(252)
        bh_step_return = float(np.mean(log_returns))

        reward = (
            (step_return - bh_step_return) / daily_vol
        ) * self.REWARD_VOL_SCALE

        # Drawdown penalty
        if self.current_mdd > self.REWARD_MDD_THRESHOLD:
            reward -= self.current_mdd * self.REWARD_MDD_COEF

        # Turnover penalty
        reward -= self.REWARD_TURNOVER_COEF * float(np.sum(np.abs(delta_weights)))

        # Cash penalty (minimum investment mandate)
        if self.REWARD_CASH_PENALTY > 0:
            total_exposure = float(np.sum(exposure_ratios))
            shortfall = max(0.0, self.REWARD_MIN_EXPOSURE - total_exposure)
            reward -= self.REWARD_CASH_PENALTY * shortfall

        return float(reward)

    # ------------------------------------------------------------------
    # Action mapping (softmax)
    # ------------------------------------------------------------------

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert N+1 raw policy logits to N asset weights via softmax.

        The last logit is the cash allocation.  After softmax, the first N
        probabilities become asset weights; the remainder stays as uninvested
        balance.  Guarantees all weights in (0, 1) and sum <= 1.0 with smooth
        differentiable gradients.
        """
        shifted = action - np.max(action)
        e       = np.exp(shifted)
        probs   = e / e.sum()
        return probs[:self.num_assets].astype(np.float32)

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute_sells(
        self,
        prices: np.ndarray,
        delta_weights: np.ndarray,
        prev_portfolio_value: float,
    ) -> None:
        """Process sell orders for assets whose target weight decreased."""
        for i in range(self.num_assets):
            if delta_weights[i] < -self.DELTA_WEIGHT_THRESHOLD:
                units = min(
                    abs(delta_weights[i]) * prev_portfolio_value / prices[i],
                    self.position_sizes[i],
                )
                if units > 0:
                    self.position_sizes[i] -= units
                    self.balance += units * prices[i] * (1.0 - self.COMMISSION_RATE)

    def _execute_buys(
        self,
        prices: np.ndarray,
        delta_weights: np.ndarray,
        prev_portfolio_value: float,
    ) -> None:
        """Process buy orders for assets whose target weight increased."""
        for i in range(self.num_assets):
            if delta_weights[i] > self.DELTA_WEIGHT_THRESHOLD:
                invest = min(
                    delta_weights[i] * prev_portfolio_value,
                    self.balance,
                )
                if invest > 0:
                    self.balance -= invest
                    self.position_sizes[i] += (
                        invest * (1.0 - self.COMMISSION_RATE) / prices[i]
                    )
