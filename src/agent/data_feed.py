import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class WyckoffMockData:
    """
    Synthetic multi-asset market data generator driven by a global systemic core.

    Supports two sentiment modes controlled by the `sentiment_mode` parameter:

    "macro"  (default, backward-compatible)
        A single scalar sentiment signal is shared by all assets at each step.
        This mirrors the EmpiricalDataFeed setup where FinBERT produces one
        macroeconomic z-score regardless of the number of assets in the portfolio.
        get_step_data() returns sentiment as a float scalar, identical to the
        original behaviour.

    "per_asset"
        Each asset has an independent sentiment signal derived from its own
        Wyckoff phase sequence (which is phase-offset relative to the global
        timeline).  The signal leads the price by `sentiment_lag` steps so
        that a positive correlation between sentiment_i and future return_i
        exists by construction — giving the agent a learnable signal.
        get_step_data() returns sentiment as an np.ndarray of shape (num_assets,).

        DreamEnv detects which mode is active via the `sentiment_mode` attribute
        and adjusts the observation space accordingly:
          - "macro":     obs_dim = 3 + 3 * N   (one global sentiment slot)
          - "per_asset": obs_dim = 2 + 4 * N   (one sentiment slot per asset)

    Phase offsets in per_asset mode
        Assets are assigned cyclic phase offsets so that, at any given step,
        different assets are typically in different Wyckoff phases.  With N=3
        the offsets are 0, ~85, ~170 steps (one full cycle ≈ 250 steps).
        This guarantees that the rotation signal is observable: asset A may be
        in Markup while B is in Distribution — the agent should favour A.
    """

    SENTIMENT_MODES = ("macro", "per_asset")

    def __init__(
        self,
        steps: int = 252,
        num_assets: int = 1,
        sentiment_mode: str = "macro",
        sentiment_lag: int = 1,
    ):
        """
        Args:
            steps:          Number of simulation time steps per episode.
            num_assets:     Number of correlated asset channels to simulate.
            sentiment_mode: "macro" or "per_asset" (see class docstring).
            sentiment_lag:  Steps by which per-asset sentiment leads price.
                            Only used when sentiment_mode="per_asset".
                            Range [1, 5] is sensible; default 2.
        """
        if sentiment_mode not in self.SENTIMENT_MODES:
            raise ValueError(
                f"sentiment_mode must be one of {self.SENTIMENT_MODES}, "
                f"got '{sentiment_mode}'."
            )
        if sentiment_lag < 1:
            raise ValueError("sentiment_lag must be >= 1.")

        self.steps = steps
        self.num_assets = num_assets
        self.sentiment_mode = sentiment_mode
        self.sentiment_lag = sentiment_lag

        # Pre-allocated buffers (populated in generate_new_market)
        self.prices: np.ndarray | None = None
        self.log_returns: np.ndarray | None = None
        # "macro"     → shape (steps,)           float scalar per step
        # "per_asset" → shape (steps, num_assets) float array per step
        self.sentiment: np.ndarray | None = None
        self.volatility: np.ndarray | None = None
        self.hidden_states: np.ndarray | None = None

        self.generate_new_market()

    # ------------------------------------------------------------------
    # Phase configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_phase_duration(state: int) -> int:
        """
        Samples the duration (in steps) for a given Wyckoff market phase.

        Args:
            state: 0=Accumulation, 1=Markup, 2=Distribution, 3=Markdown.

        Returns:
            Random integer duration drawn from phase-specific bounds.
        """
        durations = {
            0: (40, 90),
            1: (60, 120),
            2: (30, 60),
            3: (20, 50),
        }
        lo, hi = durations[state]
        return np.random.randint(lo, hi)

    @staticmethod
    def _raw_sentiment_for_state(state: int) -> float:
        """
        Returns a noisy sentiment draw for a given Wyckoff phase.
        Used by both macro and per-asset generation paths.
        """
        phase_params = {
            0: (-0.2, 0.2),   # Accumulation: neutral/negative
            1: ( 0.7, 0.2),   # Markup:       positive/euphoric
            2: ( 0.6, 0.3),   # Distribution: positive but erratic
            3: (-0.8, 0.2),   # Markdown:     negative/panic
        }
        mu, sigma = phase_params[state]
        return float(np.clip(np.random.normal(mu, sigma), -1.0, 1.0))

    # ------------------------------------------------------------------
    # Market generation
    # ------------------------------------------------------------------

    def _generate_phase_sequence(self) -> np.ndarray:
        """
        Generates one independent Wyckoff phase sequence of length self.steps.
        Initial state is sampled uniformly from {0,1,2,3} so that episodes
        do not always start at Accumulation.

        Returns:
            np.ndarray of shape (steps,), dtype int32.
        """
        state = int(np.random.randint(0, 4))
        seq = np.zeros(self.steps, dtype=np.int32)
        time_in_state = 0
        phase_duration = self._sample_phase_duration(state)
        for t in range(self.steps):
            seq[t] = state
            time_in_state += 1
            if time_in_state >= phase_duration:
                state = (state + 1) % 4
                time_in_state = 0
                phase_duration = self._sample_phase_duration(state)
        return seq

    def generate_new_market(self) -> None:
        """
        Simulates price paths and sentiment for N assets.

        "macro" mode
        ------------
        All assets share ONE global Wyckoff phase sequence for price
        generation.  A single scalar sentiment is derived from that shared
        sequence.  This mirrors EmpiricalDataFeed where FinBERT produces
        one macroeconomic signal regardless of the number of assets.

        "per_asset" mode
        ----------------
        Each asset has its OWN independent Wyckoff phase sequence with a
        random initial state.  Both the PRICE and the SENTIMENT of asset i
        are driven by asset i's own sequence.

        This is the only correct design: if asset i's price follows
        sequence A but its sentiment comes from a shifted version of
        sequence B (as in a roll-based offset approach), the sentiment
        does not predict that asset's returns — the two series are
        structurally decoupled.  By driving both price and sentiment from
        the same per-asset sequence, we guarantee:

            sentiment_i[t]  →  predicts  →  return_i[t + sentiment_lag]

        with a learnable, consistent relationship across all episodes.
        The agent cannot exploit asset index or fixed offsets — every
        episode reshuffles which asset is in which phase.
        """
        phase_params_price = {
            0: ( 0.000, 0.005),   # Accumulation
            1: ( 0.003, 0.010),   # Markup
            2: ( 0.000, 0.025),   # Distribution
            3: (-0.006, 0.035),   # Markdown
        }

        if self.num_assets > 1:
            asset_vol_offsets = 0.15 * (
                np.arange(self.num_assets) - (self.num_assets - 1) / 2.0
            )
        else:
            asset_vol_offsets = np.zeros(1)

        prices = np.zeros((self.steps, self.num_assets), dtype=np.float64)
        current_prices = np.random.uniform(50.0, 150.0, size=self.num_assets)
        prices[0, :] = current_prices

        if self.sentiment_mode == "macro":
            # ---- MACRO: one shared phase sequence -------------------------
            global_phase_seq = self._generate_phase_sequence()

            for t in range(1, self.steps):
                mu_p, sigma_p = phase_params_price[global_phase_seq[t]]
                shocks = np.random.normal(0.0, 1.0, size=self.num_assets)
                asset_sigmas = np.maximum(
                    sigma_p * (1.0 + asset_vol_offsets), 0.002
                )
                current_prices = current_prices * (
                    1.0 + mu_p + asset_sigmas * shocks
                )
                current_prices = np.maximum(current_prices, 1.0)
                prices[t, :] = current_prices

            sentiment    = self._generate_macro_sentiment(global_phase_seq)
            hidden_states = global_phase_seq

        else:
            # ---- PER_ASSET: each asset has its own phase sequence ---------
            # Price AND sentiment of asset i both follow asset_phase_seqs[i].
            # This guarantees sentiment_i predicts return_i with the correct lag.
            asset_phase_seqs = [
                self._generate_phase_sequence() for _ in range(self.num_assets)
            ]

            for t in range(1, self.steps):
                for i in range(self.num_assets):
                    mu_p, sigma_p = phase_params_price[asset_phase_seqs[i][t]]
                    sigma_i = max(
                        sigma_p * (1.0 + float(asset_vol_offsets[i])), 0.002
                    )
                    shock = float(np.random.normal(0.0, 1.0))
                    current_prices[i] = max(
                        current_prices[i] * (1.0 + mu_p + sigma_i * shock), 1.0
                    )
                prices[t, :] = current_prices

            sentiment    = self._generate_per_asset_sentiment(asset_phase_seqs)
            hidden_states = asset_phase_seqs[0]   # asset 0 as reference

        # ---- Log returns --------------------------------------------------
        log_returns = np.zeros_like(prices)
        log_returns[1:, :] = np.log(prices[1:, :] / prices[:-1, :])

        # ---- Rolling volatility (vectorized) ------------------------------
        vol_window = 20
        oracle_rmse = 0.0212
        padded_returns = np.pad(
            log_returns, pad_width=((vol_window - 1, 0), (0, 0)), mode="edge"
        )
        windows = sliding_window_view(padded_returns, window_shape=vol_window, axis=0)
        vol_realised = np.std(windows, axis=2, ddof=1) * np.sqrt(252)
        oracle_noise = np.random.normal(0.0, oracle_rmse, size=vol_realised.shape)
        volatility = np.clip(vol_realised + oracle_noise, 0.0, 2.0).astype(np.float32)

        # ---- Commit atomically -------------------------------------------
        self.prices      = prices.astype(np.float32)
        self.log_returns = log_returns.astype(np.float32)
        self.volatility  = volatility
        self.sentiment   = sentiment
        self.hidden_states = hidden_states

    # ------------------------------------------------------------------
    # Sentiment generation helpers
    # ------------------------------------------------------------------

    def _generate_macro_sentiment(self, phase_seq: np.ndarray) -> np.ndarray:
        """
        Generates a single macroeconomic sentiment scalar per step.

        Identical to the original WyckoffMockData behaviour.

        Returns:
            np.ndarray of shape (steps,), dtype float32.
        """
        macro = np.array(
            [self._raw_sentiment_for_state(int(s)) for s in phase_seq],
            dtype=np.float32,
        )
        leading = np.roll(macro, -self.sentiment_lag)
        leading[-self.sentiment_lag:] = leading[-(self.sentiment_lag + 1)]  # evitar wrap-around artefacts
        return leading

    def _generate_per_asset_sentiment(self, asset_phase_seqs: list) -> np.ndarray:
        """
        Generates an independent sentiment signal per asset.

        Each asset's sentiment is derived from its OWN phase sequence
        (the same one that drives its price), shifted backwards by
        sentiment_lag steps so that the observable value at time t
        corresponds to the phase the asset will be in at t + sentiment_lag.

        This coupling guarantees:
            sentiment_i[t]  ->  predicts  ->  return_i[t + sentiment_lag]

        Args:
            asset_phase_seqs: List of N np.ndarray of shape (steps,),
                              one independent sequence per asset.

        Returns:
            np.ndarray of shape (steps, num_assets), dtype float32.
        """
        sentiment_matrix = np.zeros((self.steps, self.num_assets), dtype=np.float32)

        for i, phase_seq in enumerate(asset_phase_seqs):
            raw_sentiment = np.array(
                [self._raw_sentiment_for_state(int(s)) for s in phase_seq],
                dtype=np.float32,
            )
            # Shift backwards by sentiment_lag so the agent sees the sentiment
            # that corresponds to the phase at t + sentiment_lag.
            leading_sentiment = np.roll(raw_sentiment, -self.sentiment_lag)
            # Replace wrap-around tail artefacts with last valid value.
            leading_sentiment[-self.sentiment_lag:] = leading_sentiment[
                -(self.sentiment_lag + 1)
            ]
            sentiment_matrix[:, i] = leading_sentiment

        return sentiment_matrix

    # ------------------------------------------------------------------
    # Step data accessor
    # ------------------------------------------------------------------

    def get_step_data(
        self, step: int
    ) -> tuple[np.ndarray, np.ndarray, float | np.ndarray, np.ndarray]:
        """
        Returns the market snapshot for a given simulation step.

        Args:
            step: Integer index into the current episode timeline.

        Returns:
            Tuple (prices, log_returns, sentiment, volatility):
            - prices:      np.ndarray shape (num_assets,)
            - log_returns: np.ndarray shape (num_assets,)
            - sentiment:
                "macro"     → float scalar in [-1, 1]
                "per_asset" → np.ndarray shape (num_assets,), values in [-1, 1]
            - volatility:  np.ndarray shape (num_assets,), annualised
        """
        if self.sentiment_mode == "macro":
            sentiment_out = float(self.sentiment[step])
        else:
            sentiment_out = self.sentiment[step, :]   # shape (num_assets,)

        return (
            self.prices[step, :],
            self.log_returns[step, :],
            sentiment_out,
            self.volatility[step, :],
        )


class EmpiricalDataFeed:
    """
    Historical multi-asset data pipeline for RL training on real market data.

    Parses a pre-processed CSV file containing aligned cross-sectional asset
    features and exposes randomised episode windows for Domain Randomization.

    Expected CSV columns:
        Date                       — trading date (parsed as datetime)
        ticker                     — asset identifier string
        Close                      — adjusted closing price
        log_return                 — daily log return (pre-computed)
        predicted_volatility_t1    — forward-looking volatility forecast
        sentiment_zscore           — normalised macro/news sentiment signal

    Sentiment mode
    --------------
    EmpiricalDataFeed always operates in "macro" mode.  The sentiment_zscore
    column is a single macroeconomic signal produced by the FinBERT / rsLoRA
    module in Phase 1 and is identical across all assets on a given date.
    The sentiment_mode attribute is exposed so that DreamEnv can query it
    with the same interface used for WyckoffMockData.

    Episode selection logic:
        1. Sample `num_assets` tickers at random (or use a fixed list).
        2. Find all dates where every selected ticker has data (inner join).
        3. Sample a contiguous window of `steps` days from the aligned range.
    """

    # Always macro: one FinBERT z-score per day, shared across all assets.
    sentiment_mode: str = "macro"

    def __init__(
        self,
        csv_path: str = "../data/train_agent_dataset.csv",
        steps: int = 252,
        tickers: list[str] | str | None = None,
        num_assets: int = 1,
        randomize: bool = True,
    ):
        """
        Args:
            csv_path:   Path to the pre-processed historical CSV dataset.
            steps:      Maximum episode length (in trading days).
            tickers:    Optional fixed ticker list. If None, tickers are sampled
                        randomly at each episode reset (Domain Randomization).
            num_assets: Number of assets per episode. Ignored if tickers given.
            randomize:  If True, sample random tickers and windows each reset.
        """
        self.csv_path = csv_path
        self.steps = steps
        self.randomize = randomize

        if tickers is not None:
            self.fixed_tickers: list[str] | None = (
                tickers if isinstance(tickers, list) else [tickers]
            )
            self.num_assets = len(self.fixed_tickers)
        else:
            self.fixed_tickers = None
            self.num_assets = num_assets

        self.current_tickers: list[str] | None = None
        self.current_market_steps: int = 0

        self.prices: np.ndarray | None = None
        self.log_returns: np.ndarray | None = None
        self.sentiment: np.ndarray | None = None   # shape (steps,), float32
        self.volatility: np.ndarray | None = None

        self._full_df = pd.read_csv(csv_path)
        self._full_df["Date"] = pd.to_datetime(self._full_df["Date"])
        self.all_tickers: np.ndarray = self._full_df["ticker"].unique()

        self.generate_new_market()

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------

    def generate_new_market(self) -> None:
        """
        Samples a new episode and populates the data buffers.
        Called once at construction and once per environment reset.
        """
        # --- Step 1: Ticker selection ---
        if self.fixed_tickers is not None:
            self.current_tickers = self.fixed_tickers
        elif self.randomize:
            self.current_tickers = list(
                np.random.choice(self.all_tickers, size=self.num_assets, replace=False)
            )
        else:
            self.current_tickers = list(self.all_tickers[: self.num_assets])

        # --- Step 2: Inner join on dates ---
        ticker_mask = self._full_df["ticker"].isin(self.current_tickers)
        ticker_data = self._full_df[ticker_mask]

        date_counts = ticker_data["Date"].value_counts()
        synchronized_dates = date_counts[
            date_counts == self.num_assets
        ].index.sort_values()

        effective_steps = min(self.steps, len(synchronized_dates))
        max_start_idx = len(synchronized_dates) - effective_steps

        # --- Step 3: Time window ---
        if self.randomize and max_start_idx > 0:
            start_idx = np.random.randint(0, max_start_idx)
        else:
            start_idx = 0

        chosen_dates = synchronized_dates[start_idx : start_idx + effective_steps]

        # --- Step 4: Populate buffers ---
        prices = np.zeros((effective_steps, self.num_assets), dtype=np.float32)
        log_returns = np.zeros_like(prices)
        volatility = np.zeros_like(prices)
        sentiment = np.zeros(effective_steps, dtype=np.float32)

        date_mask = ticker_data["Date"].isin(chosen_dates)
        filtered = ticker_data[date_mask]

        for i, ticker in enumerate(self.current_tickers):
            asset_df = (
                filtered[filtered["ticker"] == ticker]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            prices[:, i] = asset_df["Close"].values
            log_returns[:, i] = asset_df["log_return"].values
            volatility[:, i] = asset_df["predicted_volatility_t1"].values
            if i == 0:
                sentiment = asset_df["sentiment_zscore"].values.astype(np.float32)

        self.prices = prices
        self.log_returns = log_returns
        self.volatility = volatility
        self.sentiment = sentiment
        self.current_market_steps = effective_steps

    # ------------------------------------------------------------------
    # Step data accessor
    # ------------------------------------------------------------------

    def get_step_data(
        self, step: int
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Returns the historical market snapshot for a given episode step.

        Returns:
            Tuple (prices, log_returns, sentiment, volatility):
            - prices:      np.ndarray shape (num_assets,)
            - log_returns: np.ndarray shape (num_assets,)
            - sentiment:   float scalar, z-scored macro sentiment
            - volatility:  np.ndarray shape (num_assets,)
        """
        return (
            self.prices[step, :],
            self.log_returns[step, :],
            float(self.sentiment[step]),
            self.volatility[step, :],
        )
