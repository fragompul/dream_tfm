"""
inference.py — D.R.E.A.M. Inference Engine, Phase 6 (multi-asset PPO)
======================================================================

Overview
--------
This module exposes ``DreamInferenceEngine``, the stateful decision layer of
the D.R.E.A.M. (Deep Reinforcement Ensemble for Asset Management) pipeline.
It wraps a trained Stable-Baselines3 PPO policy and its associated
VecNormalize statistics, accepts raw market inputs, preprocesses them to
exactly reproduce the observation vector seen during training, and returns
portfolio rebalancing orders ready for broker execution.

The engine is self-contained: it does **not** depend on ``DreamEnv``,
``WyckoffMockData``, or ``EmpiricalDataFeed`` at inference time.  All
feature-engineering logic that was applied inside ``DreamEnv._get_observation``
during training is replicated here, ensuring distribution consistency between
training and deployment.

Pipeline position
-----------------
The engine sits at the end of the three-phase D.R.E.A.M. pipeline:

    Phase 1 (FinBERT / rsLoRA)
        Raw financial news  →  raw sentiment score  ∈ [0, 1]
                                        │
    Phase 2 (Volatility ensemble)       │
        Historical prices   →  annualised volatility forecast  ∈ ℝ⁺  (per asset)
                                        │
                            ┌───────────┘
    Phase 3 (this module)   │
        prices, volatility, sentiment, broker state
                            │
                            ▼
                   DreamInferenceEngine
                            │
                            ▼
                   rebalancing orders  →  broker / execution layer

Usage modes
-----------
The engine supports two operational modes, selected via the
``sentiment_normalized`` constructor flag.

Mode 1 — Live / empirical  (``sentiment_normalized=False``, **default**)
    Intended for production use with the full D.R.E.A.M. pipeline.
    ``sentiment_raw`` must be the **raw FinBERT output** in [0, 1].
    The engine normalises it to a z-score using the training-set statistics
    (µ = 0.143970, σ = 0.114915) before building the observation vector.

    Example::

        from src.agent.inference import DreamInferenceEngine
        import numpy as np

        engine = DreamInferenceEngine(
            model_path     = "test_modelos/dream_synthetic_macro_N3.zip",
            vecnorm_path   = "test_modelos/dream_synthetic_macro_N3_vecnorm.pkl",
            num_assets     = 3,
            sentiment_mode = "macro",   # one FinBERT score per day
        )

        # Call once per trading day, in chronological order
        result = engine.process_market_tick(
            prices         = np.array([450.12, 380.55, 291.80]),  # closing prices
            volatility     = np.array([0.182, 0.217, 0.154]),     # Phase 2 output
            sentiment_raw  = 0.187,                               # raw FinBERT [0,1]
            position_sizes = np.array([2.1, 0.0, 5.3]),          # units held
            balance        = 8_200.0,                             # cash in account
        )

Mode 2 — Synthetic demonstration  (``sentiment_normalized=True``)
    Intended for testing, validation, and demonstration without running
    the upstream Phase 1 / Phase 2 models.  ``sentiment_raw`` must be the
    **z-scored sentiment** produced directly by ``WyckoffMockData.get_step_data()``.
    No normalisation is applied.

    Example::

        from src.agent.inference import DreamInferenceEngine
        from src.agent.data_feed import WyckoffMockData
        import numpy as np

        data_gen = WyckoffMockData(steps=1000, num_assets=3, sentiment_mode="macro")
        data_gen.generate_new_market()

        engine = DreamInferenceEngine(
            model_path           = "test_modelos/dream_synthetic_macro_N3.zip",
            vecnorm_path         = "test_modelos/dream_synthetic_macro_N3_vecnorm.pkl",
            num_assets           = 3,
            sentiment_mode       = "macro",
            sentiment_normalized = True,   # sentiment from WyckoffMockData is z-score
        )
        engine.reset()

        position_sizes = np.zeros(3)
        balance        = 10_000.0

        for t in range(data_gen.steps - 1):
            prices, _, sentiment_z, volatility = data_gen.get_step_data(t)
            result = engine.process_market_tick(
                prices         = prices,
                volatility     = volatility,
                sentiment_raw  = sentiment_z,
                position_sizes = position_sizes,
                balance        = balance,
            )
            # apply orders to position_sizes / balance here ...

Input specification
-------------------
All inputs are passed to ``process_market_tick()``:

prices : np.ndarray, shape (N,), dtype float32
    Raw closing prices for the current step, one per asset.
    Log returns are computed **internally** from consecutive ticks
    (``log(price_t / price_{t-1})``).  On the first call after ``reset()``,
    log returns are set to zero.  The engine must be called once per
    trading day in strict chronological order.

volatility : np.ndarray, shape (N,), dtype float32
    Annualised volatility forecast for each asset, produced by the Phase 2
    ensemble.  Each component is **independent** — asset i carries its own
    volatility estimate regardless of the sentiment mode.
    The engine applies the following transform internally:
        ``scaled_vol_i = clip(volatility_i / 0.5, 0.0, 4.0)``

sentiment_raw : float  (macro mode)  |  np.ndarray shape (N,)  (per_asset mode)
    Sentiment signal.  Interpretation depends on ``sentiment_normalized``:
      - ``False`` (live):      raw FinBERT score ∈ [0, 1].
                               Normalised internally: z = (raw − µ) / σ.
      - ``True``  (synthetic): z-score from WyckoffMockData ∈ ℝ.
                               Passed through without modification.
    In **macro** mode a single scalar covers all N assets (mirrors
    EmpiricalDataFeed, where one FinBERT z-score is shared across tickers).
    In **per_asset** mode each asset has an independent sentiment value.

position_sizes : np.ndarray, shape (N,), dtype float32
    Number of units of each asset currently held in the portfolio.
    Provided by the broker or portfolio tracker.

balance : float
    Uninvested cash available, in currency units.
    Provided by the broker or portfolio tracker.

Output specification
--------------------
``process_market_tick()`` returns a ``dict`` with the following keys:

target_weights : np.ndarray, shape (N,)
    Target allocation for each asset as a fraction of total portfolio value.
    Derived from softmax over the N+1 policy logits:
        weights = softmax(action)[0:N]
    Cash allocation is implicit: cash_weight = 1 − sum(target_weights).

cash_weight : float
    Fraction of portfolio to hold as uninvested cash.
    Always equals ``1.0 − sum(target_weights)``.

orders : list of dict
    One entry per asset whose weight change exceeds ``delta_threshold`` (2%).
    Each dict contains:
        "asset"        : int   — zero-based asset index.
        "action"       : str   — "BUY" or "SELL".
        "delta_weight" : float — signed weight change (positive = buy).
        "amount_usd"   : float — trade size in currency units.
    Assets within the dead-zone (|Δw| ≤ delta_threshold) are omitted.

portfolio_value : float
    Total portfolio value at the current tick:
        portfolio_value = balance + sum(position_sizes * prices)

Observation vector layout
--------------------------
The engine reproduces the ``DreamEnv._get_observation()`` vector exactly.

macro mode  (obs_dim = 3 + 3·N):
    A single global sentiment scalar is prepended to the vector.
    Individual assets do NOT have a sentiment slot.

    [norm_portfolio, cash_weight, sentiment_z,
     z_ret_0, scaled_vol_0, weight_0,
     z_ret_1, scaled_vol_1, weight_1,
     ...  (z_ret_i, scaled_vol_i, weight_i  repeated for each asset)]

per_asset mode  (obs_dim = 2 + 4·N):
    No global sentiment.  Each asset carries its own sentiment value,
    interleaved with its return, volatility and weight features.

    [norm_portfolio, cash_weight,
     z_ret_0, scaled_vol_0, sentiment_z_0, weight_0,
     z_ret_1, scaled_vol_1, sentiment_z_1, weight_1,
     ...  (z_ret_i, scaled_vol_i, sentiment_z_i, weight_i  repeated)]

Where:
    norm_portfolio = portfolio_value / initial_balance
    cash_weight    = balance / portfolio_value
    z_ret_i        = rolling z-score of log_return_i  (window = 20)
    scaled_vol_i   = clip(volatility_i / 0.5, 0, 4)
    weight_i       = (position_sizes_i * prices_i) / portfolio_value
    sentiment_z    = (sentiment_raw − µ) / σ      [live mode]
                   = sentiment_raw                 [synthetic mode, pre-normalised]
    sentiment_z_i  = per-asset equivalent of sentiment_z  (per_asset mode only)

Notes
-----
- ``reset()`` must be called at the start of each new trading session or
  synthetic episode.  It clears the rolling return history and the stored
  previous prices.
- VecNormalize statistics are frozen at load time (``training=False``).
  The running mean/variance computed during synthetic training are applied
  as-is to both synthetic and empirical inputs.  For empirically-trained
  models (``vecnorm_path=None``) no normalisation is applied.
- The engine is CPU-only (``device="cpu"``) to avoid GPU memory overhead
  in production deployments.
"""

from __future__ import annotations

from collections import deque
from typing import Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Sentiment normalisation constants
# Must match config.py and the training pipeline exactly.
# Source: mean and std of raw FinBERT output over the full training split.
# Computed in: notebooks/10_data_agent.ipynb
# Applied as: z = (raw_score − SENTIMENT_MEAN) / SENTIMENT_STD
# ---------------------------------------------------------------------------
_SENTIMENT_MEAN: float = 0.143970
_SENTIMENT_STD:  float = 0.114915


# ---------------------------------------------------------------------------
# Minimal stub environment
# ---------------------------------------------------------------------------

class _StubEnv(gym.Env):
    """
    Minimal Gymnasium environment used solely to satisfy the signature of
    ``VecNormalize.load()``, which requires a vectorised environment wrapper
    even when no actual stepping will occur.  All methods are no-ops.
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(act_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class DreamInferenceEngine:
    """
    Stateful inference wrapper for a trained D.R.E.A.M. Phase 6 PPO agent.

    Handles all input preprocessing internally to guarantee that the
    observation vector passed to the policy at inference time is
    numerically identical to the one produced by ``DreamEnv._get_observation``
    during training.

    Internal preprocessing pipeline (per tick):
        1. Compute log returns from consecutive raw prices.
        2. Normalise raw FinBERT sentiment to z-score (live mode only).
        3. Compute portfolio value and current asset weights.
        4. Build the flat observation vector (rolling z-ret, scaled vol,
           sentiment, weights) in the layout expected by the policy.
        5. Apply frozen VecNormalize statistics (if applicable).
        6. Run deterministic policy inference.
        7. Map N+1 logits to asset weights via softmax.
        8. Generate BUY / SELL orders for weight changes above the dead-zone.

    Parameters
    ----------
    model_path : str
        Path to the ``.zip`` file produced by ``PPO.save()``.
    vecnorm_path : str or None
        Path to the ``_vecnorm.pkl`` file produced alongside the model.
        Pass ``None`` for models trained without VecNormalize
        (i.e. ``USE_VECNORM=False``, empirical training mode).
    num_assets : int
        Number of assets N.  Must match the value used during training.
    sentiment_mode : {"macro", "per_asset"}, default "macro"
        Observation layout selector.  Must match the mode used during
        training.
        - ``"macro"``    : one shared sentiment scalar; obs_dim = 3 + 3·N.
        - ``"per_asset"``: one sentiment value per asset; obs_dim = 2 + 4·N.
    sentiment_normalized : bool, default False
        Controls sentiment preprocessing:
        - ``False`` (live / empirical): ``sentiment_raw`` is treated as
          raw FinBERT output ∈ [0, 1] and normalised internally.
        - ``True``  (synthetic / demo): ``sentiment_raw`` is treated as a
          z-score already (e.g. direct output of
          ``WyckoffMockData.get_step_data()``) and passed through unchanged.
    initial_balance : float, default 10_000.0
        Reference portfolio value used to compute the scale-invariant
        ``norm_portfolio`` feature.  Should match the value used during
        training (``DreamEnv`` default: 10 000).
    return_window : int, default 20
        Length of the rolling window for log-return z-scoring.
        Must match ``DreamEnv.OBS_RETURN_WINDOW``.
    vol_scale : float, default 0.5
        Denominator for volatility scaling before clipping.
        Must match ``DreamEnv.OBS_VOL_SCALE``.
    commission_rate : float, default 0.001
        Round-trip commission fraction applied to trade size computation.
        Must match ``DreamEnv.COMMISSION_RATE``.
    delta_threshold : float, default 0.02
        Minimum absolute weight change required to generate an order.
        Changes smaller than this threshold are suppressed (dead-zone).
        Must match ``DreamEnv.DELTA_WEIGHT_THRESHOLD``.
    """

    # ---- Mirror DreamEnv constants (must stay in sync) -------------------
    OBS_RETURN_WINDOW: int = 20
    OBS_VOL_SCALE: float   = 0.5
    COMMISSION_RATE: float = 0.001
    DELTA_THRESHOLD: float = 0.02

    def __init__(
        self,
        model_path: str,
        vecnorm_path: str | None,
        num_assets: int,
        sentiment_mode: str = "macro",
        sentiment_normalized: bool = False,
        initial_balance: float = 10_000.0,
        return_window: int = OBS_RETURN_WINDOW,
        vol_scale: float = OBS_VOL_SCALE,
        commission_rate: float = COMMISSION_RATE,
        delta_threshold: float = DELTA_THRESHOLD,
    ) -> None:
        if sentiment_mode not in ("macro", "per_asset"):
            raise ValueError(
                f"sentiment_mode must be 'macro' or 'per_asset', got '{sentiment_mode}'."
            )

        self.num_assets           = num_assets
        self.sentiment_mode       = sentiment_mode
        self.sentiment_normalized = sentiment_normalized
        self.initial_balance      = initial_balance
        self.vol_scale            = vol_scale
        self.commission_rate      = commission_rate
        self.delta_threshold      = delta_threshold

        # Observation dimension — mirrors DreamEnv exactly
        self._obs_dim = (
            3 + 3 * num_assets if sentiment_mode == "macro"
            else 2 + 4 * num_assets
        )

        # Load PPO policy weights
        self.model = PPO.load(model_path, device="cpu")

        # Load VecNormalize and freeze running statistics
        self._vecnorm: VecNormalize | None = None
        if vecnorm_path is not None:
            stub = DummyVecEnv([lambda: _StubEnv(self._obs_dim, num_assets + 1)])
            vn   = VecNormalize.load(vecnorm_path, stub)
            vn.training    = False  # do not update running stats at inference
            vn.norm_reward = False  # reward normalisation is irrelevant at inference
            self._vecnorm  = vn

        # Rolling log-return history (stateful across ticks)
        self._return_history: deque = deque(maxlen=return_window)

        # Previous closing prices for log-return computation
        self._prev_prices: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the internal rolling state.

        Must be called at the start of each new trading session or synthetic
        episode.  Clears the log-return history window and the stored previous
        prices, so that the first call to ``process_market_tick()`` after
        ``reset()`` produces zero log returns (cold start).
        """
        self._return_history.clear()
        self._prev_prices = None

    def process_market_tick(
        self,
        prices: np.ndarray,
        volatility: np.ndarray,
        sentiment_raw: Union[float, np.ndarray],
        position_sizes: np.ndarray,
        balance: float,
    ) -> dict:
        """
        Process one market tick and return portfolio rebalancing orders.

        Must be called **once per trading day in strict chronological order**.
        The engine maintains internal state between calls (rolling return
        history, previous prices); calling out of order or skipping days
        will corrupt the observation features.

        Parameters
        ----------
        prices : np.ndarray, shape (N,)
            Raw closing prices for the current step, one value per asset.
        volatility : np.ndarray, shape (N,)
            Annualised volatility forecast for each asset, as produced by
            Phase 2.  Each component is independently estimated per asset.
        sentiment_raw : float or np.ndarray
            Sentiment signal.  Interpretation controlled by
            ``sentiment_normalized``:
            - Live mode  (``sentiment_normalized=False``):
              raw FinBERT output ∈ [0, 1].
              macro → scalar float; per_asset → array of shape (N,).
            - Synthetic mode (``sentiment_normalized=True``):
              z-score from ``WyckoffMockData.get_step_data()``.
              macro → scalar float; per_asset → array of shape (N,).
        position_sizes : np.ndarray, shape (N,)
            Number of units of each asset currently held.
        balance : float
            Uninvested cash available, in currency units.

        Returns
        -------
        dict
            target_weights : np.ndarray, shape (N,)
                Target allocation per asset as a fraction of portfolio value.
            cash_weight : float
                Implied cash allocation = 1 − sum(target_weights).
            orders : list of dict
                One entry per asset with |Δweight| > delta_threshold.
                Keys: "asset" (int), "action" ("BUY"|"SELL"),
                      "delta_weight" (float), "amount_usd" (float).
                Assets within the dead-zone are omitted.
            portfolio_value : float
                Total portfolio value = balance + sum(position_sizes × prices).
        """
        prices         = np.asarray(prices,        dtype=np.float32)
        volatility     = np.asarray(volatility,     dtype=np.float32)
        position_sizes = np.asarray(position_sizes, dtype=np.float32)

        # 1. Log returns from consecutive closing prices
        if self._prev_prices is not None:
            log_returns = np.log(
                prices / np.maximum(self._prev_prices, 1e-8)
            ).astype(np.float32)
        else:
            log_returns = np.zeros(self.num_assets, dtype=np.float32)
        self._prev_prices = prices.copy()

        # 2. Sentiment preprocessing
        if self.sentiment_normalized:
            # Synthetic mode: WyckoffMockData output is already z-scored
            sentiment_z = sentiment_raw
        else:
            # Live mode: normalise raw FinBERT score to z-score
            if self.sentiment_mode == "macro":
                sentiment_z = float(
                    (float(sentiment_raw) - _SENTIMENT_MEAN) / _SENTIMENT_STD
                )
            else:
                arr = np.asarray(sentiment_raw, dtype=np.float32)
                sentiment_z = (arr - _SENTIMENT_MEAN) / _SENTIMENT_STD

        # 3. Portfolio accounting
        asset_values    = position_sizes * prices
        portfolio_value = float(balance + asset_values.sum())

        if portfolio_value > 0:
            asset_weights = asset_values / portfolio_value
            cash_weight   = balance / portfolio_value
        else:
            asset_weights = np.zeros(self.num_assets, dtype=np.float32)
            cash_weight   = 1.0

        # 4. Build observation vector (mirrors DreamEnv._get_observation)
        obs = self._build_observation(
            log_returns     = log_returns,
            volatility      = volatility,
            sentiment_z     = sentiment_z,
            asset_weights   = asset_weights,
            cash_weight     = cash_weight,
            portfolio_value = portfolio_value,
        )

        # 5. Apply frozen VecNormalize statistics (synthetic models only)
        if self._vecnorm is not None:
            obs = self._vecnorm.normalize_obs(obs.reshape(1, -1)).flatten()

        # 6. Deterministic policy inference
        action, _ = self.model.predict(obs, deterministic=True)

        # 7. Map N+1 logits to asset weights via softmax
        target_weights = self._softmax_to_weights(action)

        # 8. Generate orders for assets crossing the dead-zone threshold
        orders        = []
        delta_weights = target_weights - asset_weights

        for i in range(self.num_assets):
            dw = float(delta_weights[i])
            if abs(dw) <= self.delta_threshold:
                continue
            if dw > 0:
                amount = min(dw * portfolio_value, balance)
                orders.append({
                    "asset":        i,
                    "action":       "BUY",
                    "delta_weight": dw,
                    "amount_usd":   float(amount),
                })
            else:
                sell_value = abs(dw) * portfolio_value
                units      = min(sell_value / prices[i], position_sizes[i])
                orders.append({
                    "asset":        i,
                    "action":       "SELL",
                    "delta_weight": dw,
                    "amount_usd":   float(units * prices[i]),
                })

        return {
            "target_weights":  target_weights,
            "cash_weight":     float(1.0 - target_weights.sum()),
            "orders":          orders,
            "portfolio_value": portfolio_value,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        log_returns: np.ndarray,
        volatility: np.ndarray,
        sentiment_z: Union[float, np.ndarray],
        asset_weights: np.ndarray,
        cash_weight: float,
        portfolio_value: float,
    ) -> np.ndarray:
        """
        Construct the flat observation vector consumed by the PPO policy.

        This method is a stateful mirror of ``DreamEnv._get_observation()``:
        it appends ``log_returns`` to the internal rolling history and
        computes z-scores relative to that history.  The resulting vector
        layout is identical to the one the policy observed during training.

        Parameters
        ----------
        log_returns     : np.ndarray (N,) — log returns for the current step.
        volatility      : np.ndarray (N,) — annualised volatility per asset.
        sentiment_z     : float or np.ndarray — z-scored sentiment.
        asset_weights   : np.ndarray (N,) — current portfolio weights.
        cash_weight     : float — current cash fraction.
        portfolio_value : float — current total portfolio value.

        Returns
        -------
        np.ndarray, shape (obs_dim,), dtype float32.
        """
        # Rolling z-score of log returns
        self._return_history.append(log_returns.copy())
        if len(self._return_history) > 1:
            hist  = np.array(self._return_history)
            z_ret = (log_returns - hist.mean(axis=0)) / (hist.std(axis=0) + 1e-8)
        else:
            z_ret = np.zeros(self.num_assets, dtype=np.float32)

        scaled_vol     = np.clip(volatility / self.vol_scale, 0.0, 4.0)
        norm_portfolio = portfolio_value / self.initial_balance

        if self.sentiment_mode == "macro":
            obs = [norm_portfolio, cash_weight, float(sentiment_z)]
            for i in range(self.num_assets):
                obs.extend([
                    float(z_ret[i]),
                    float(scaled_vol[i]),
                    float(asset_weights[i]),
                ])
        else:
            sentiment_z = np.asarray(sentiment_z, dtype=np.float32)
            obs = [norm_portfolio, cash_weight]
            for i in range(self.num_assets):
                obs.extend([
                    float(z_ret[i]),
                    float(scaled_vol[i]),
                    float(sentiment_z[i]),
                    float(asset_weights[i]),
                ])

        return np.array(obs, dtype=np.float32)

    def _softmax_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert N+1 raw policy logits to N asset weights via softmax.

        Mirror of ``DreamEnv._action_to_weights()``.  The last logit
        represents the cash allocation; after softmax the first N
        probabilities become the asset target weights.

        Parameters
        ----------
        action : np.ndarray, shape (N+1,) — raw logits from the policy.

        Returns
        -------
        np.ndarray, shape (N,), dtype float32 — asset target weights.
        """
        shifted = action - np.max(action)   # numerical stability
        e       = np.exp(shifted)
        probs   = e / e.sum()
        return probs[:self.num_assets].astype(np.float32)
