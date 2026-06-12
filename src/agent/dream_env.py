import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class DreamEnv(gym.Env):
    """
    D.R.E.A.M. Portfolio Management Environment (Phase 5).

    Supports two sentiment modes, automatically detected from the data feed:

    "macro"  (EmpiricalDataFeed or WyckoffMockData with sentiment_mode="macro")
        Observation space: 3 + 3*N dimensions
            [norm_balance, norm_portfolio, sentiment_macro,
             z_ret_0, scaled_vol_0, norm_pos_0,
             z_ret_1, scaled_vol_1, norm_pos_1, ...]

    "per_asset"  (WyckoffMockData with sentiment_mode="per_asset")
        Observation space: 2 + 4*N dimensions
            [norm_balance, norm_portfolio,
             z_ret_0, scaled_vol_0, sentiment_0, norm_pos_0,
             z_ret_1, scaled_vol_1, sentiment_1, norm_pos_1, ...]

        Placing the per-asset sentiment adjacent to each asset's price signals
        helps the MLP associate the sentiment directly with the corresponding
        asset's dynamics, rather than forcing it to learn this association
        across a globally shared slot.

    All other mechanics (reward, trade execution, MDD tracking) are unchanged
    from Phase 4.
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Reward hyperparameters
    # ------------------------------------------------------------------
    REWARD_VOL_SCALE: float    = 1.0
    REWARD_MDD_THRESHOLD: float = 0.10
    REWARD_MDD_COEF: float     = 1.0
    REWARD_TURNOVER_COEF: float = 0.05
    REWARD_POSITIVE_BONUS: float = 0.05
    VOL_ANN_FLOOR: float       = 0.05

    # ------------------------------------------------------------------
    # Observation normalisation
    # ------------------------------------------------------------------
    OBS_RETURN_WINDOW: int  = 20
    OBS_VOL_SCALE: float    = 0.5

    # ------------------------------------------------------------------
    # Transaction mechanics
    # ------------------------------------------------------------------
    COMMISSION_RATE: float       = 0.001
    DELTA_WEIGHT_THRESHOLD: float = 0.02
    DEAD_ZONE_LOW: float         = 0.05
    DEAD_ZONE_HIGH: float        = 0.95

    def __init__(self, data_generator, initial_balance: float = 10_000.0):
        """
        Args:
            data_generator: WyckoffMockData or EmpiricalDataFeed instance.
            initial_balance: Starting portfolio value in currency units.
        """
        super().__init__()

        self.data_gen = data_generator
        self.initial_balance = initial_balance
        self.num_assets: int = getattr(self.data_gen, "num_assets", 1)
        self.max_steps: int = self.data_gen.steps - 1

        # Detect sentiment mode from the data feed
        self.sentiment_mode: str = getattr(
            self.data_gen, "sentiment_mode", "macro"
        )

        # ---- Action space ------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
        )

        # ---- Observation space -------------------------------------------
        # macro:     3 + 3*N  (one shared sentiment slot)
        # per_asset: 2 + 4*N  (one sentiment slot per asset, inside its block)
        if self.sentiment_mode == "macro":
            obs_dim = 3 + 3 * self.num_assets
        else:
            obs_dim = 2 + 4 * self.num_assets

        low  = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim,  np.inf, dtype=np.float32)
        low[0] = 0.0   # norm_balance  >= 0
        low[1] = 0.0   # norm_portfolio >= 0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ---- Accounting state --------------------------------------------
        self.current_step: int = 0
        self.balance: float = initial_balance
        self.position_sizes: np.ndarray = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value: float = initial_balance
        self.peak_portfolio_value: float = initial_balance
        self.current_mdd: float = 0.0

        self._return_history: deque = deque(maxlen=self.OBS_RETURN_WINDOW)

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

        self.current_step = 0
        self.balance = self.initial_balance
        self.position_sizes = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.current_mdd = 0.0
        self._return_history.clear()

        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one trading step.

        Pipeline:
            1. Map raw action → target weights (dead-zone + no-leverage).
            2. Snapshot portfolio value before trades.
            3. Execute sells, then buys.
            4. Mark to market; update peak and MDD.
            5. Compute reward.
            6. Advance step counter.
        """
        prices, log_returns, sentiment, volatility = self.data_gen.get_step_data(
            self.current_step
        )

        # 1. Weights
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

        # 3. Trades
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
            if prev_portfolio_value > 0
            else 0.0
        )
        exposure_ratios = (
            (self.position_sizes * prices) / self.portfolio_value
            if self.portfolio_value > 0
            else np.zeros(self.num_assets, dtype=np.float32)
        )
        reward = self._compute_reward(
            step_return=step_return,
            volatility=volatility,
            delta_weights=delta_weights,
            exposure_ratios=exposure_ratios,
        )

        # 6. Advance
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        info = {
            "portfolio_value": self.portfolio_value,
            "mdd": self.current_mdd,
            "step_return": step_return,
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
        Builds the flat observation vector.

        macro mode  (obs_dim = 3 + 3*N):
            [norm_balance, norm_portfolio, sentiment_macro,
             z_ret_0, scaled_vol_0, norm_pos_0, ...]

        per_asset mode  (obs_dim = 2 + 4*N):
            [norm_balance, norm_portfolio,
             z_ret_0, scaled_vol_0, sentiment_0, norm_pos_0,
             z_ret_1, scaled_vol_1, sentiment_1, norm_pos_1, ...]

        Placing per-asset sentiment inside each asset's feature block
        (rather than in a separate global section) keeps spatially related
        features contiguous, which aids learning in shallow MLPs.
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

        scaled_vol  = np.clip(volatility / self.OBS_VOL_SCALE, 0.0, 4.0)
        norm_balance    = self.balance / self.initial_balance
        norm_portfolio  = self.portfolio_value / self.initial_balance

        if self.sentiment_mode == "macro":
            # sentiment is a float scalar
            obs = [norm_balance, norm_portfolio, float(sentiment)]
            for i in range(self.num_assets):
                norm_pos = (
                    self.position_sizes[i] * prices[i]
                ) / self.initial_balance
                obs.extend(
                    [float(z_log_returns[i]), float(scaled_vol[i]), float(norm_pos)]
                )
        else:
            # sentiment is np.ndarray of shape (num_assets,)
            obs = [norm_balance, norm_portfolio]
            for i in range(self.num_assets):
                norm_pos = (
                    self.position_sizes[i] * prices[i]
                ) / self.initial_balance
                obs.extend(
                    [
                        float(z_log_returns[i]),
                        float(scaled_vol[i]),
                        float(sentiment[i]),
                        float(norm_pos),
                    ]
                )

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        step_return: float,
        volatility: np.ndarray,
        delta_weights: np.ndarray,
        exposure_ratios: np.ndarray,
    ) -> float:
        """
        Scalar reward for a single trading step.

        Components:
            base_reward      = (step_return / daily_vol) * REWARD_VOL_SCALE
            drawdown_penalty = current_mdd * REWARD_MDD_COEF  (if > threshold)
            turnover_penalty = REWARD_TURNOVER_COEF * Σ|Δw|
            positive_bonus   = REWARD_POSITIVE_BONUS  (if profitable & exposed)
        """
        ann_vol   = max(float(np.mean(volatility)), self.VOL_ANN_FLOOR)
        daily_vol = ann_vol / np.sqrt(252)
        reward    = (step_return / daily_vol) * self.REWARD_VOL_SCALE

        if self.current_mdd > self.REWARD_MDD_THRESHOLD:
            reward -= self.current_mdd * self.REWARD_MDD_COEF

        reward -= self.REWARD_TURNOVER_COEF * float(np.sum(np.abs(delta_weights)))

        if step_return > 0.0 and float(np.sum(exposure_ratios)) > 0.1:
            reward += self.REWARD_POSITIVE_BONUS

        return float(reward)

    # ------------------------------------------------------------------
    # Trade execution helpers
    # ------------------------------------------------------------------

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Maps raw policy actions in [-1, 1]^N to valid weights in [0, 1]^N.

        Steps:
            1. Linear rescaling [-1, 1] → [0, 1].
            2. Dead-zone snapping.
            3. Sum normalisation to enforce no-leverage constraint.
        """
        w = (action + 1.0) / 2.0
        w = np.where(w < self.DEAD_ZONE_LOW,  0.0, w)
        w = np.where(w > self.DEAD_ZONE_HIGH, 1.0, w)
        s = w.sum()
        return w / s if s > 1.0 else w

    def _execute_sells(
        self,
        prices: np.ndarray,
        delta_weights: np.ndarray,
        prev_portfolio_value: float,
    ) -> None:
        """Processes sell orders (assets whose target weight decreased)."""
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
        """Processes buy orders (assets whose target weight increased)."""
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
