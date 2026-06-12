import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class DreamEnv(gym.Env):
    """
    D.R.E.A.M. Portfolio Management Environment — Phase 6.

    Architectural changes vs Phase 5
    ---------------------------------

    1. Action space: softmax + explicit cash slot
       The policy emits N+1 logits (N assets + 1 cash).  Softmax converts
       them to a valid probability distribution summing to 1.0 by
       construction.  No dead zones or ad-hoc normalisation needed.
       The cash logit weight stays as balance (not invested).

    2. Observation space: current_weight replaces norm_pos
       Each asset's weight (value / portfolio_value) replaces the old
       norm_pos (value / initial_balance).  current_weight is
       scale-invariant — stays in [0,1] regardless of portfolio drift.
       cash_weight is added as an explicit global feature.
       norm_balance is dropped (redundant given norm_portfolio + cash_weight).

    3. Observation layout
       macro mode    obs_dim = 3 + 3*N:
           [norm_portfolio, cash_weight, sentiment_macro,
            z_ret_0, scaled_vol_0, weight_0, ...]

       per_asset mode  obs_dim = 2 + 4*N:
           [norm_portfolio, cash_weight,
            z_ret_0, scaled_vol_0, sentiment_0, weight_0, ...]

    4. Dead zones removed
       Softmax output is already a proper distribution; dead zones are
       not needed and would distort the gradient signal.

    5. REWARD_POSITIVE_BONUS removed (0.05 -> 0.0)
       The discontinuous bonus contributed to critic collapse (ev ~= 0)
       in macro-mode experiments.

    6. REWARD_TURNOVER_COEF configurable via turnover_coef __init__ arg.
       Recommended: 0.01 for per_asset, 0.03 for macro.
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Reward hyperparameters
    # ------------------------------------------------------------------
    REWARD_VOL_SCALE: float     = 1.0
    REWARD_MDD_THRESHOLD: float = 0.10
    REWARD_MDD_COEF: float      = 1.0
    REWARD_TURNOVER_COEF: float = 0.02
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
    ):
        """
        Args:
            data_generator: WyckoffMockData or EmpiricalDataFeed instance.
            initial_balance: Starting portfolio value in currency units.
            turnover_coef:  Override for REWARD_TURNOVER_COEF.
                            per_asset: 0.01 | macro: 0.03
        """
        super().__init__()

        self.data_gen        = data_generator
        self.initial_balance = initial_balance
        if turnover_coef is not None:
            self.REWARD_TURNOVER_COEF = turnover_coef

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
        # macro:     3 + 3*N
        # per_asset: 2 + 4*N
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
        """
        Executes one trading step.

        1. Softmax(action) -> target weights for N assets (cash implicit).
        2. Snapshot portfolio value before trades.
        3. Execute sells, then buys.
        4. Mark to market; update peak and MDD.
        5. Compute reward.
        6. Advance step counter.
        """
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
            if prev_portfolio_value > 0 else 0.0
        )
        reward = self._compute_reward(
            step_return=step_return,
            volatility=volatility,
            delta_weights=delta_weights,
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
        macro mode  (obs_dim = 3 + 3*N):
            [norm_portfolio, cash_weight, sentiment_macro,
             z_ret_0, scaled_vol_0, weight_0, ...]

        per_asset mode  (obs_dim = 2 + 4*N):
            [norm_portfolio, cash_weight,
             z_ret_0, scaled_vol_0, sentiment_0, weight_0, ...]
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

        # Scale-invariant asset weights
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
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        step_return: float,
        volatility: np.ndarray,
        delta_weights: np.ndarray,
    ) -> float:
        """
        base_reward      = (step_return / daily_vol) * REWARD_VOL_SCALE
        drawdown_penalty = current_mdd * REWARD_MDD_COEF  (if MDD > threshold)
        turnover_penalty = REWARD_TURNOVER_COEF * sum(|delta_w_i|)
        """
        ann_vol   = max(float(np.mean(volatility)), self.VOL_ANN_FLOOR)
        daily_vol = ann_vol / np.sqrt(252)
        reward    = (step_return / daily_vol) * self.REWARD_VOL_SCALE

        if self.current_mdd > self.REWARD_MDD_THRESHOLD:
            reward -= self.current_mdd * self.REWARD_MDD_COEF

        reward -= self.REWARD_TURNOVER_COEF * float(np.sum(np.abs(delta_weights)))

        return float(reward)

    # ------------------------------------------------------------------
    # Action -> weights (softmax)
    # ------------------------------------------------------------------

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Converts N+1 raw logits to N asset weights via softmax.

        The last logit is the cash slot.  After softmax, the first N
        probabilities are the asset weights; the agent holds
        (1 - sum(asset_weights)) as uninvested balance.

        Softmax guarantees:
          - all weights in (0, 1)
          - sum of all N+1 probabilities = 1.0
          - asset weights sum to (1 - cash_prob) <= 1.0
          - smooth differentiable gradients throughout
        """
        shifted = action - np.max(action)   # numerical stability
        e       = np.exp(shifted)
        probs   = e / e.sum()               # shape: (num_assets + 1,)
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
