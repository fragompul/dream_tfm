"""
D.R.E.A.M. Evaluation Engine.

Evaluates a trained PPO agent using Monte Carlo rollouts over synthetic
or empirical market data.  Compatible with both sentiment modes
("macro" and "per_asset") and multi-asset DreamEnv.

Four evaluation dimensions:
    1. Financial   — Sharpe, Sortino, MDD, Win Rate vs B&H, total return.
    2. Behavioural — average exposure, turnover, correlation of portfolio
                     weights with sentiment signals.
    3. Signal use  — per-asset correlation between asset weight and asset
                     sentiment (per_asset mode only).
    4. Policy behaviour — weight evolution over time, action distributions,
                          sentiment-weight alignment, and action stability.

Usage:
    evaluator = DreamEvaluator(
        model_path="test_modelos/model.zip",
        num_assets=3,
        sentiment_mode="per_asset",
        num_runs=30,
        steps=1000,
    )
    df_fin, df_beh = evaluator.run_monte_carlo()
    evaluator.plot_dashboard()
    evaluator.plot_behavior()
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless rendering (Kaggle / server environments)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agent.data_feed import WyckoffMockData, EmpiricalDataFeed
from src.agent.dream_env import DreamEnv
from src.agent.config import RANDOM_SEED

# =============================================================================
# Evaluation configuration — edit these to match the experiment being evaluated
# =============================================================================

MODEL_PATH   = "test_modelos/dream_empirical_macro_N3_noCashPen.zip"
VECNORM_PATH = None
NUM_ASSETS = 3
SENTIMENT_MODE:  str = "macro"   # "macro" | "per_asset"
SENTIMENT_LAG:   int = 1
NUM_RUNS:        int = 30
EVAL_STEPS:      int = 126
DATA_MODE:       str = "empirical"   # "synthetic" | "empirical"
CSV_PATH:        str = "data/backtest_agent_dataset.csv"
OUTPUT_IMG:      str = "test_modelos/eval_exp4_empirical_macro_N3_noCashPen.png"

# Experiment guide
# ------------------------------------------
# Exp 1 — pipeline sanity check (synthetic macro, N=2):
#     MODEL_PATH   = "test_modelos/dream_synthetic_macro_N2_v6.zip"
#     VECNORM_PATH = "test_modelos/dream_synthetic_macro_N2_v6_vecnorm.pkl"
#     NUM_ASSETS = 2;  SENTIMENT_MODE = "macro";  DATA_MODE = "synthetic"
#
# Exp 2 — per-asset signal (synthetic per_asset, N=3):
#     MODEL_PATH   = "test_modelos/dream_synthetic_per_asset_N3_v6.zip"
#     VECNORM_PATH = "test_modelos/dream_synthetic_per_asset_N3_v6_vecnorm.pkl"
#     NUM_ASSETS = 3;  SENTIMENT_MODE = "per_asset";  DATA_MODE = "synthetic"
#
# Exp 3 — macro multi-asset (synthetic macro, N=3):
#     MODEL_PATH   = "test_modelos/dream_synthetic_macro_N3_v6.zip"
#     VECNORM_PATH = "test_modelos/dream_synthetic_macro_N3_v6_vecnorm.pkl"
#     NUM_ASSETS = 3;  SENTIMENT_MODE = "macro";  DATA_MODE = "synthetic"
#
# Exp 4 — empirical backtest (real data):
#     MODEL_PATH   = "test_modelos/dream_empirical_macro_N3_v6.zip"
#     VECNORM_PATH = "test_modelos/dream_empirical_macro_N3_v6_vecnorm.pkl"
#     NUM_ASSETS = 3;  SENTIMENT_MODE = "macro";  DATA_MODE = "empirical"
#     CSV_PATH     = "data/backtest_agent_dataset.csv"


# =============================================================================
# Stand-alone metric helpers
# =============================================================================

def calculate_financial_metrics(
    portfolio_values: list[float],
    bh_values: list[float],
) -> dict:
    """
    Computes standard quantitative finance KPIs for a single episode.

    Args:
        portfolio_values: List of portfolio values at each step (including t=0).
        bh_values:        Equal-weight buy-and-hold values at each step.

    Returns:
        Dictionary with keys: Sharpe, Sortino, Calmar, MDD, WinRate,
        Total_Ret_Agent, Total_Ret_BH.
    """
    pv = np.array(portfolio_values, dtype=float)
    bh = np.array(bh_values, dtype=float)

    port_returns = np.diff(pv) / np.where(pv[:-1] > 0, pv[:-1], 1.0)
    bh_returns   = np.diff(bh) / np.where(bh[:-1] > 0, bh[:-1], 1.0)

    mean_ret = float(np.mean(port_returns))
    std_ret  = float(np.std(port_returns))

    # Sharpe (annualised, risk-free = 0)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-10 else 0.0

    # Sortino (downside deviation relative to 0, annualised)
    downside = port_returns[port_returns < 0]
    if len(port_returns) > 0:
        down_var = np.sum(downside ** 2) / len(port_returns)
        down_std = float(np.sqrt(down_var))
    else:
        down_std = 1e-10
    down_std = max(down_std, 1e-10)
    sortino = (mean_ret / down_std) * np.sqrt(252)

    # Maximum Drawdown
    cum = pv / pv[0]
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / np.where(running_max > 0, running_max, 1.0)
    mdd = float(abs(np.min(drawdowns)) * 100.0)

    # Calmar ratio (annualised return / MDD)
    ann_return = float(mean_ret * 252)
    calmar = ann_return / (mdd / 100.0) if mdd > 0 else 0.0

    # Win rate vs B&H (fraction of steps where agent beats B&H)
    outperform = float(np.sum(port_returns > bh_returns))
    win_rate = (outperform / len(port_returns) * 100.0) if len(port_returns) > 0 else 0.0

    total_ret_agent = float((pv[-1] / pv[0] - 1.0) * 100.0)
    total_ret_bh    = float((bh[-1] / bh[0] - 1.0) * 100.0)

    return {
        "Sharpe":           sharpe,
        "Sortino":          sortino,
        "Calmar":           calmar,
        "MDD":              mdd,
        "WinRate":          win_rate,
        "Total_Ret_Agent":  total_ret_agent,
        "Total_Ret_BH":     total_ret_bh,
    }


# =============================================================================
# Main evaluator class
# =============================================================================

class DreamEvaluator:
    """
    Monte Carlo evaluator for the D.R.E.A.M. multi-asset PPO agent.

    Compatible with both WyckoffMockData sentiment modes.
    """

    def __init__(
        self,
        model_path: str,
        num_assets: int = 3,
        sentiment_mode: str = "per_asset",
        sentiment_lag: int = 2,
        num_runs: int = 30,
        steps: int = 500,
        commission_rate: float = 0.001,
        initial_balance: float = 10_000.0,
        data_mode: str = "synthetic",
        csv_path: str = "data/backtest_agent_dataset.csv",
        vecnorm_path: str | None = None,
    ):
        """
        Args:
            model_path:      Path to the serialised PPO model (.zip).
            num_assets:      Number of assets (must match the trained model).
            sentiment_mode:  "macro" or "per_asset".
            sentiment_lag:   Lag used during training (only affects per_asset).
            num_runs:        Number of independent Monte Carlo episodes.
            steps:           Episode length in trading days.
            commission_rate: Transaction cost rate (must match training env).
            initial_balance: Starting portfolio value.
        """
        self.num_assets      = num_assets
        self.sentiment_mode  = sentiment_mode
        self.sentiment_lag   = sentiment_lag
        self.num_runs        = num_runs
        self.steps           = steps
        self.commission_rate = commission_rate
        self.initial_balance = initial_balance
        self.data_mode    = data_mode
        self.csv_path     = csv_path
        self.vecnorm_path = vecnorm_path

        # Build throw-away env for PPO.load space matching.
        # For Phase 6 models, wrap with VecNormalize using saved
        # running stats so the reward scale matches training.
        _data = WyckoffMockData(
            steps=self.steps,
            num_assets=self.num_assets,
            sentiment_mode=self.sentiment_mode,
            sentiment_lag=self.sentiment_lag,
        )
        _raw = DreamEnv(_data, initial_balance=self.initial_balance)

        if vecnorm_path and os.path.exists(vecnorm_path):
            _venv = DummyVecEnv([lambda: _raw])
            _env  = VecNormalize.load(vecnorm_path, _venv)
            _env.training = False
            _env.norm_reward = False
            print(f"VecNormalize loaded from {vecnorm_path}")
        else:
            _env = _raw

        print(f"Loading PPO model from {model_path} …")
        self.model = PPO.load(model_path, env=_env, device="cpu")

        # Results populated by run_monte_carlo()
        self.agent_paths: list[np.ndarray] = []
        self.bh_paths:    list[np.ndarray] = []
        self.df_fin:  pd.DataFrame | None = None
        self.df_beh:  pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Monte Carlo rollouts
    # ------------------------------------------------------------------

    def run_monte_carlo(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs num_runs independent rollouts on freshly generated markets.

        Returns:
            (df_fin, df_beh) — financial and behavioural metric DataFrames,
            one row per Monte Carlo run.
        """
        np.random.seed(RANDOM_SEED)

        if self.data_mode == "empirical":
            test_data = EmpiricalDataFeed(
                csv_path=self.csv_path,
                steps=self.steps,
                num_assets=self.num_assets,
                randomize=True,
            )
            data_label = f"empirical ({self.csv_path.split('/')[-1]})"
        else:
            test_data = WyckoffMockData(
                steps=self.steps,
                num_assets=self.num_assets,
                sentiment_mode=self.sentiment_mode,
                sentiment_lag=self.sentiment_lag,
            )
            data_label = f"synthetic/{self.sentiment_mode}"

        eval_env = DreamEnv(test_data, initial_balance=self.initial_balance)

        all_fin, all_beh = [], []
        self.agent_paths, self.bh_paths = [], []

        # Per-episode telemetry for behaviour analysis
        all_weights_episodes    = []
        all_sentiments_episodes = []
        all_exposures_episodes  = []

        print(f"Running Monte Carlo ({self.num_runs} episodes, "
              f"data={data_label}, N={self.num_assets}) …")

        for run in range(self.num_runs):
            obs, _ = eval_env.reset()

            # ---- Tracking buffers ----------------------------------------
            portfolio_values = [self.initial_balance]
            bh_initial_prices = test_data.prices[0, :].copy()  # shape (N,)
            bh_values         = [self.initial_balance]

            # Per-step telemetry for behavioural analysis
            total_exposures: list[float] = [0.0]
            # shape: (steps, N) for per-asset weight tracking
            asset_weights_history: list[np.ndarray] = [
                np.zeros(self.num_assets)
            ]
            # Sentiment history: scalar per step (macro) or array (per_asset)
            sentiment_history: list = []

            # ---- Episode rollout -----------------------------------------
            for _ in range(self.steps - 1):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)

                step_idx = eval_env.current_step
                cur_prices = test_data.prices[step_idx, :]

                portfolio_values.append(float(info["portfolio_value"]))

                # Equal-weight B&H: price return of each asset, averaged
                bh_price_returns = cur_prices / bh_initial_prices
                bh_portfolio = self.initial_balance * float(
                    np.mean(bh_price_returns)
                ) * (1.0 - self.commission_rate)
                bh_values.append(bh_portfolio)

                # Current asset weights
                pv = float(info["portfolio_value"])
                if pv > 0:
                    w = (eval_env.position_sizes * cur_prices) / pv
                else:
                    w = np.zeros(self.num_assets)
                asset_weights_history.append(w.copy())
                total_exposures.append(float(np.sum(w)))

                # Sentiment at this step
                _, _, sent, _ = test_data.get_step_data(step_idx)
                sentiment_history.append(sent)

                if terminated or truncated:
                    break

            # ---- Store episode telemetry for behaviour analysis ----------
            all_weights_episodes.append(np.array(asset_weights_history))
            all_sentiments_episodes.append(list(sentiment_history))
            all_exposures_episodes.append(np.array(total_exposures))

            # ---- Metrics for this run ------------------------------------
            fin = calculate_financial_metrics(portfolio_values, bh_values)
            beh = self._calculate_behavioural_metrics(
                asset_weights_history=np.array(asset_weights_history),
                total_exposures=np.array(total_exposures),
                sentiment_history=sentiment_history,
            )
            all_fin.append(fin)
            all_beh.append(beh)

            # Normalised paths for plotting
            pv_arr = np.array(portfolio_values)
            bh_arr = np.array(bh_values)
            self.agent_paths.append(pv_arr / pv_arr[0])
            self.bh_paths.append(bh_arr / bh_arr[0])

        self.df_fin = pd.DataFrame(all_fin)
        self.df_beh = pd.DataFrame(all_beh)

        # Store per-episode telemetry for behaviour plotting.
        # _all_weights[i]:    np.ndarray shape (T, N) for episode i
        # _all_sentiments[i]: list of sentiment values for episode i
        # _all_exposures[i]:  np.ndarray shape (T,) for episode i
        self._all_weights    = all_weights_episodes
        self._all_sentiments = all_sentiments_episodes
        self._all_exposures  = all_exposures_episodes

        self._print_summary()
        return self.df_fin, self.df_beh

    # ------------------------------------------------------------------
    # Behavioural metrics
    # ------------------------------------------------------------------

    def _calculate_behavioural_metrics(
        self,
        asset_weights_history: np.ndarray,   # shape (T, N)
        total_exposures: np.ndarray,          # shape (T,)
        sentiment_history: list,
    ) -> dict:
        """
        Computes behavioural KPIs capturing how the agent uses its signals.

        Always computed:
            Avg_Exposure       — mean fraction of portfolio invested
            Turnover           — mean |Δweight| per step per asset
            Corr_Macro_Sent    — correlation between total exposure and sentiment
                                 (meaningful in both modes)

        Only in per_asset mode:
            Corr_Asset_{i}     — Pearson r between weight_i and sentiment_i.
                                 A positive value means the agent overweights
                                 assets with improving sentiment — the core
                                 rotation hypothesis.
        """
        result: dict = {}

        T = len(total_exposures)
        avg_exposure = float(np.mean(total_exposures)) * 100.0
        result["Avg_Exposure"] = avg_exposure

        # Turnover: mean absolute weight change per step (all assets combined)
        if asset_weights_history.shape[0] > 1:
            weight_deltas = np.diff(asset_weights_history, axis=0)
            turnover = float(np.mean(np.sum(np.abs(weight_deltas), axis=1)))
        else:
            turnover = 0.0
        result["Turnover"] = turnover

        # Macro correlation: total exposure vs sentiment
        if self.sentiment_mode == "macro":
            # sentiment_history is a list of floats
            sent_arr = np.array(sentiment_history, dtype=float)
            if len(sent_arr) == T:
                exp_arr = total_exposures.copy()
            else:
                # Trim to matching length (can differ by 1 at episode end)
                n = min(len(sent_arr), T)
                sent_arr = sent_arr[:n]
                exp_arr  = total_exposures[:n]

            # Add tiny noise to avoid degenerate constant arrays
            noise = np.random.normal(0, 1e-7, len(exp_arr))
            corr_macro, _ = pearsonr(exp_arr + noise, sent_arr)
            result["Corr_Macro_Sent"] = float(corr_macro)

        else:
            # per_asset mode: sentiment_history is a list of np.ndarray (N,)
            if len(sentiment_history) == 0:
                result["Corr_Macro_Sent"] = float("nan")
            else:
                sent_matrix = np.array(sentiment_history)   # shape (T-1, N)
                # Macro correlation: mean asset sentiment vs total exposure
                mean_sent = sent_matrix.mean(axis=1)        # shape (T-1,)
                n = min(len(mean_sent), T)
                noise = np.random.normal(0, 1e-7, n)
                corr_macro, _ = pearsonr(
                    total_exposures[:n] + noise, mean_sent[:n]
                )
                result["Corr_Macro_Sent"] = float(corr_macro)

                # Per-asset correlations: weight_i vs sentiment_i
                for i in range(self.num_assets):
                    w_i   = asset_weights_history[1:n+1, i]  # skip t=0
                    s_i   = sent_matrix[:n, i]
                    noise_i = np.random.normal(0, 1e-7, len(w_i))
                    if np.std(w_i) < 1e-8 or np.std(s_i) < 1e-8:
                        corr_i = float("nan")
                    else:
                        corr_i, _ = pearsonr(w_i + noise_i, s_i)
                    result[f"Corr_Asset_{i}"] = float(corr_i)

        return result

    # ------------------------------------------------------------------
    # Summary printer
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Prints a concise statistical summary to stdout."""
        if self.df_fin is None or self.df_beh is None:
            return

        print("\n" + "=" * 60)
        print("  D.R.E.A.M. MONTE CARLO EVALUATION SUMMARY")
        print("=" * 60)

        # Financial
        for col in ["Sharpe", "Sortino", "MDD", "WinRate",
                    "Total_Ret_Agent", "Total_Ret_BH"]:
            if col in self.df_fin.columns:
                m = self.df_fin[col].mean()
                s = self.df_fin[col].std()
                print(f"  {col:<22} {m:+.3f} ± {s:.3f}")

        print()

        # Behavioural
        beh_cols = [c for c in self.df_beh.columns if "Corr" in c or
                    c in ("Avg_Exposure", "Turnover")]
        for col in beh_cols:
            m = self.df_beh[col].mean()
            s = self.df_beh[col].std()
            print(f"  {col:<22} {m:+.3f} ± {s:.3f}")

        print("=" * 60)

        # Highlight per-asset correlations if present
        corr_asset_cols = [c for c in self.df_beh.columns if c.startswith("Corr_Asset_")]
        if corr_asset_cols:
            print("\n  Signal-use test (per-asset sentiment → weight correlation):")
            print("  Positive values confirm the agent rotates toward assets")
            print("  with improving sentiment (core D.R.E.A.M. hypothesis).")
            for col in corr_asset_cols:
                m = self.df_beh[col].mean()
                flag = "✅" if m > 0.1 else ("⚠️ " if m >= 0 else "❌")
                print(f"    {flag} {col:<20} mean r = {m:+.3f}")
            print()

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def plot_dashboard(
        self,
        output_img: str = "test_modelos/dream_eval_dashboard.png",
    ) -> None:
        """
        Renders a 3×2 evaluation dashboard and saves it to disk.

        Panels:
            [0,0] Capital paths with ±1 std bands (agent vs B&H)
            [0,1] Sharpe ratio distribution
            [1,0] MDD boxplot
            [1,1] Avg exposure vs Sharpe scatter
            [2,0] Financial metrics summary (text)
            [2,1] Behavioural metrics summary (text)
        """
        if self.df_fin is None or self.df_beh is None:
            raise RuntimeError("Call run_monte_carlo() before plot_dashboard().")

        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))

        agent_arr = np.array(self.agent_paths)
        bh_arr    = np.array(self.bh_paths)
        T = agent_arr.shape[1]
        t = np.arange(T)

        # ---- [0,0] Capital paths -----------------------------------------
        mean_ag = agent_arr.mean(axis=0)
        std_ag  = agent_arr.std(axis=0)
        mean_bh = bh_arr.mean(axis=0)
        std_bh  = bh_arr.std(axis=0)

        axes[0, 0].plot(t, mean_ag, color="green", lw=2.5, label="D.R.E.A.M. (mean)")
        axes[0, 0].fill_between(
            t, mean_ag - std_ag, mean_ag + std_ag, color="green", alpha=0.15,
            label="±1 std"
        )
        axes[0, 0].plot(t, mean_bh, color="darkorange", ls="--", lw=2,
                        label="Equal-weight B&H")
        axes[0, 0].fill_between(
            t, mean_bh - std_bh, mean_bh + std_bh, color="darkorange", alpha=0.1
        )
        axes[0, 0].axhline(1.0, color="grey", ls=":", lw=1)
        axes[0, 0].set_title("Capital Paths — Monte Carlo", fontweight="bold")
        axes[0, 0].set_ylabel("Normalised Value (base = 1.0)")
        axes[0, 0].legend(loc="upper left", fontsize=9)

        # ---- [0,1] Sharpe distribution ------------------------------------
        sns.histplot(self.df_fin["Sharpe"], ax=axes[0, 1], kde=True,
                     color="steelblue", bins=12)
        axes[0, 1].axvline(self.df_fin["Sharpe"].mean(), color="steelblue",
                           ls="--", lw=2,
                           label=f"Mean: {self.df_fin['Sharpe'].mean():.2f}")
        axes[0, 1].axvline(1.0, color="red", ls=":", lw=1.5,
                           label="TFM target (>1.0)")
        axes[0, 1].set_title("Sharpe Ratio Distribution (out-of-sample)",
                             fontweight="bold")
        axes[0, 1].set_xlabel("Annualised Sharpe Ratio")
        axes[0, 1].legend(fontsize=9)

        # ---- [1,0] MDD boxplot -------------------------------------------
        sns.boxplot(x=self.df_fin["MDD"], ax=axes[1, 0], color="salmon")
        axes[1, 0].axvline(20.0, color="red", ls=":", lw=1.5,
                           label="TFM target (<20%)")
        axes[1, 0].set_title("Maximum Drawdown Distribution (%)",
                             fontweight="bold")
        axes[1, 0].set_xlabel("Max Drawdown (%)")
        axes[1, 0].legend(fontsize=9)

        # ---- [1,1] Exposure vs Sharpe scatter ----------------------------
        sns.scatterplot(
            x=self.df_beh["Avg_Exposure"],
            y=self.df_fin["Sharpe"],
            ax=axes[1, 1], color="purple", alpha=0.7,
        )
        axes[1, 1].set_title("Avg Exposure vs Sharpe", fontweight="bold")
        axes[1, 1].set_xlabel("Mean Market Exposure (%)")
        axes[1, 1].set_ylabel("Sharpe Ratio")

        # ---- [2,0] Financial metrics text box ----------------------------
        axes[2, 0].axis("off")
        fin_lines = [r"$\bf{Financial\ Metrics}$"]
        for col, label in [
            ("Sharpe",          "Sharpe Ratio     "),
            ("Sortino",         "Sortino Ratio    "),
            ("Calmar",          "Calmar Ratio     "),
            ("MDD",             "Max Drawdown (%) "),
            ("WinRate",         "Win Rate vs B&H  "),
            ("Total_Ret_Agent", "Agent Return (%) "),
            ("Total_Ret_BH",    "B&H Return (%)   "),
        ]:
            if col in self.df_fin.columns:
                m = self.df_fin[col].mean()
                s = self.df_fin[col].std()
                fin_lines.append(f"{label}: {m:+.2f} ± {s:.2f}")
        axes[2, 0].text(
            0.05, 0.95, "\n".join(fin_lines),
            va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            transform=axes[2, 0].transAxes,
        )

        # ---- [2,1] Behavioural metrics text box --------------------------
        axes[2, 1].axis("off")
        beh_lines = [r"$\bf{Behavioural\ Metrics}$"]
        for col in ["Avg_Exposure", "Turnover", "Corr_Macro_Sent"]:
            if col in self.df_beh.columns:
                m = self.df_beh[col].mean()
                s = self.df_beh[col].std()
                beh_lines.append(f"{col:<22}: {m:+.3f} ± {s:.3f}")

        if self.sentiment_mode == "per_asset":
            beh_lines.append("")
            beh_lines.append(r"$\bf{Per-Asset\ Signal\ Correlation}$")
            beh_lines.append("(weight_i vs sentiment_i — positive = agent")
            beh_lines.append(" rotates toward improving-sentiment assets)")
            for col in [c for c in self.df_beh.columns if c.startswith("Corr_Asset_")]:
                m = self.df_beh[col].mean()
                flag = "✓" if m > 0.1 else ("~" if m >= 0 else "✗")
                beh_lines.append(f"  [{flag}] {col}: {m:+.3f}")

        axes[2, 1].text(
            0.05, 0.95, "\n".join(beh_lines),
            va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            transform=axes[2, 1].transAxes,
        )

        data_label = (
            f"empirical" if self.data_mode == "empirical"
            else f"synthetic/{self.sentiment_mode}"
        )
        plt.suptitle(
            f"D.R.E.A.M. Monte Carlo Evaluation  |  "
            f"N={self.num_assets} assets  |  data={data_label}  |  "
            f"{self.num_runs} runs × {self.steps} steps",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_img), exist_ok=True)
        plt.savefig(output_img, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\nDashboard saved → {output_img}")


    # ------------------------------------------------------------------
    # Behaviour dashboard
    # ------------------------------------------------------------------

    def plot_behavior(
        self,
        output_img: str = "test_modelos/dream_behavior_dashboard.png",
    ) -> None:
        """
        Renders a 2x2 behaviour dashboard showing how the agent acts.

        Panels:
            [0,0] Stacked area: portfolio weight allocation over time
                  (best Sharpe episode).  Shows asset weights + cash.
            [0,1] Histogram: distribution of per-asset weights across
                  all episodes.  A concentrated distribution indicates
                  a deterministic policy; a spread distribution indicates
                  active management.
            [1,0] Overlay: asset weight vs sentiment signal over time.
                  For per_asset mode: weight_i vs sentiment_i per asset.
                  For macro mode: total exposure vs macro sentiment.
            [1,1] Policy stability metrics (text): action autocorrelation,
                  mean/std of weights per asset, cash fraction statistics.

        Must be called after run_monte_carlo().
        """
        if self.df_fin is None or not hasattr(self, '_all_weights'):
            raise RuntimeError("Call run_monte_carlo() before plot_behavior().")

        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Select the episode with the best Sharpe for detailed visualisation
        best_idx = int(self.df_fin["Sharpe"].idxmax())
        best_w   = self._all_weights[best_idx]       # shape (T, N)
        best_s   = self._all_sentiments[best_idx]     # list of T-1 values
        best_e   = self._all_exposures[best_idx]       # shape (T,)
        T        = best_w.shape[0]
        t_range  = np.arange(T)

        # ---- [0,0] Stacked area: weight allocation over time -------------
        ax = axes[0, 0]
        cash_w = 1.0 - best_w.sum(axis=1)
        cash_w = np.clip(cash_w, 0.0, 1.0)

        colors = plt.cm.Set2(np.linspace(0, 1, self.num_assets + 1))
        labels = [f"Asset {i}" for i in range(self.num_assets)] + ["Cash"]
        all_series = [best_w[:, i] for i in range(self.num_assets)] + [cash_w]

        ax.stackplot(t_range, *all_series, labels=labels, colors=colors, alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title("Portfolio Allocation Over Time (Best Sharpe Episode)",
                     fontweight="bold")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Step")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

        # ---- [0,1] Histogram: weight distributions -----------------------
        ax = axes[0, 1]
        all_w_flat = np.concatenate(self._all_weights, axis=0)  # (total_T, N)
        for i in range(self.num_assets):
            ax.hist(all_w_flat[:, i], bins=30, alpha=0.5,
                    label=f"Asset {i}", density=True)
        all_cash = [1.0 - w.sum(axis=1) for w in self._all_weights]
        all_cash_flat = np.clip(np.concatenate(all_cash), 0, 1)
        ax.hist(all_cash_flat, bins=30, alpha=0.5,
                label="Cash", color="grey", density=True)
        ax.set_title("Weight Distribution (All Episodes)", fontweight="bold")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # ---- [1,0] Weight vs sentiment overlay ---------------------------
        ax = axes[1, 0]
        if self.sentiment_mode == "per_asset" and len(best_s) > 0:
            sent_arr = np.array(best_s)  # (T-1, N)
            n_plot = min(len(t_range) - 1, len(sent_arr))
            for i in range(self.num_assets):
                color = colors[i]
                ax.plot(t_range[1:n_plot+1], best_w[1:n_plot+1, i],
                        color=color, lw=1.5, label=f"w_{i}")
                ax.plot(t_range[1:n_plot+1], sent_arr[:n_plot, i] * 0.3 + 0.5,
                        color=color, ls="--", lw=1, alpha=0.6,
                        label=f"sent_{i} (scaled)")
            ax.set_title("Asset Weights vs Sentiment (Best Episode)",
                         fontweight="bold")
        else:
            # Macro mode: total exposure vs sentiment
            if len(best_s) > 0:
                sent_arr = np.array(best_s, dtype=float)
                n_plot = min(len(best_e) - 1, len(sent_arr))
                ax.plot(t_range[1:n_plot+1], best_e[1:n_plot+1],
                        color="green", lw=2, label="Total Exposure")
                ax2 = ax.twinx()
                ax2.plot(t_range[1:n_plot+1], sent_arr[:n_plot],
                         color="darkorange", ls="--", lw=1.5, alpha=0.7,
                         label="Macro Sentiment")
                ax2.set_ylabel("Sentiment", color="darkorange")
                ax2.legend(loc="upper right", fontsize=8)
            ax.set_title("Exposure vs Macro Sentiment (Best Episode)",
                         fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Weight / Exposure")
        ax.legend(loc="upper left", fontsize=8)

        # ---- [1,1] Policy stability metrics (text) -----------------------
        ax = axes[1, 1]
        ax.axis("off")

        # Compute action autocorrelation per asset
        autocorr_lines = []
        for i in range(self.num_assets):
            all_w_i = np.concatenate([w[1:, i] for w in self._all_weights])
            all_w_i_lag = np.concatenate([w[:-1, i] for w in self._all_weights])
            if len(all_w_i) > 10 and np.std(all_w_i) > 1e-8:
                ac = float(np.corrcoef(all_w_i, all_w_i_lag)[0, 1])
            else:
                ac = float("nan")
            autocorr_lines.append(f"  Asset {i}: autocorr = {ac:+.3f}")

        # Cash statistics
        all_cash_vals = np.clip(
            np.concatenate([1.0 - w.sum(axis=1) for w in self._all_weights]),
            0, 1
        )

        # Mean weights per asset
        mean_w_lines = []
        for i in range(self.num_assets):
            m = float(all_w_flat[:, i].mean())
            s = float(all_w_flat[:, i].std())
            mean_w_lines.append(f"  Asset {i}: {m:.3f} +/- {s:.3f}")

        lines = [
            r"$f{Policy\ Stability\ Metrics}$",
            "",
            "Action autocorrelation (1-step):",
            "  High (>0.9) = patient, Low (<0.5) = noisy",
        ] + autocorr_lines + [
            "",
            "Mean portfolio weights:",
        ] + mean_w_lines + [
            f"  Cash:    {all_cash_vals.mean():.3f} +/- {all_cash_vals.std():.3f}",
            "",
            f"Episodes evaluated: {len(self._all_weights)}",
            f"Best Sharpe episode: #{best_idx} "
            f"(Sharpe={self.df_fin.loc[best_idx, 'Sharpe']:.2f})",
        ]

        ax.text(
            0.05, 0.95, "\n".join(lines),
            va="top", fontsize=11, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            transform=ax.transAxes,
        )

        data_label = (
            "empirical" if self.data_mode == "empirical"
            else f"synthetic/{self.sentiment_mode}"
        )
        plt.suptitle(
            f"D.R.E.A.M. Policy Behaviour  |  "
            f"N={self.num_assets} assets  |  data={data_label}  |  "
            f"{self.num_runs} runs",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_img) or ".", exist_ok=True)
        plt.savefig(output_img, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Behaviour dashboard saved -> {output_img}")

    # ------------------------------------------------------------------
    # Individual panel export
    # ------------------------------------------------------------------

    def save_individual_panels(self, output_dir: str) -> None:
        """
        Saves every dashboard panel as an independent high-resolution figure.

        Produces two subfolders:
            <output_dir>/financial/   — panels from plot_dashboard()
            <output_dir>/behavior/    — panels from plot_behavior()

        Financial panels saved:
            capital_paths.png
            sharpe_distribution.png
            mdd_distribution.png
            exposure_vs_sharpe.png

        Behavior panels saved:
            weight_allocation.png
            weight_distribution.png
            sentiment_overlay.png
            stability_metrics.png

        Must be called after run_monte_carlo().
        """
        if self.df_fin is None or not hasattr(self, '_all_weights'):
            raise RuntimeError("Call run_monte_carlo() before save_individual_panels().")

        fin_dir = os.path.join(output_dir, "financial")
        beh_dir = os.path.join(output_dir, "behavior")
        os.makedirs(fin_dir, exist_ok=True)
        os.makedirs(beh_dir, exist_ok=True)

        sns.set_theme(style="darkgrid")

        data_label = (
            "empirical" if self.data_mode == "empirical"
            else f"synthetic/{self.sentiment_mode}"
        )
        base_title = (
            f"D.R.E.A.M.  |  N={self.num_assets} assets  |  "
            f"data={data_label}  |  {self.num_runs} runs × {self.steps} steps"
        )

        agent_arr = np.array(self.agent_paths)
        bh_arr    = np.array(self.bh_paths)
        T = agent_arr.shape[1]
        t = np.arange(T)

        # ----------------------------------------------------------------
        # FINANCIAL panels
        # ----------------------------------------------------------------

        # 1. Capital paths
        fig, ax = plt.subplots(figsize=(10, 5))
        mean_ag = agent_arr.mean(axis=0)
        std_ag  = agent_arr.std(axis=0)
        mean_bh = bh_arr.mean(axis=0)
        std_bh  = bh_arr.std(axis=0)
        ax.plot(t, mean_ag, color="green", lw=2.5, label="D.R.E.A.M. (mean)")
        ax.fill_between(t, mean_ag - std_ag, mean_ag + std_ag,
                        color="green", alpha=0.15, label="±1 std")
        ax.plot(t, mean_bh, color="darkorange", ls="--", lw=2,
                label="Equal-weight B&H")
        ax.fill_between(t, mean_bh - std_bh, mean_bh + std_bh,
                        color="darkorange", alpha=0.1)
        ax.axhline(1.0, color="grey", ls=":", lw=1)
        ax.set_title("Capital Paths — Monte Carlo", fontsize=14, fontweight="bold")
        ax.set_ylabel("Normalised Value (base = 1.0)", fontsize=12)
        ax.set_xlabel("Step", fontsize=12)
        ax.legend(fontsize=11)
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(fin_dir, "capital_paths.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # 2. Sharpe distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(self.df_fin["Sharpe"], ax=ax, kde=True,
                     color="steelblue", bins=12)
        ax.axvline(self.df_fin["Sharpe"].mean(), color="steelblue", ls="--", lw=2,
                   label=f"Mean: {self.df_fin['Sharpe'].mean():.2f}")
        ax.axvline(1.0, color="red", ls=":", lw=1.5, label="TFM target (>1.0)")
        ax.set_title("Sharpe Ratio Distribution (out-of-sample)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Annualised Sharpe Ratio", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend(fontsize=11)
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(fin_dir, "sharpe_distribution.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # 3. MDD boxplot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=self.df_fin["MDD"], ax=ax, color="salmon")
        ax.axvline(20.0, color="red", ls=":", lw=1.5, label="TFM target (<20%)")
        ax.set_title("Maximum Drawdown Distribution (%)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Max Drawdown (%)", fontsize=12)
        ax.legend(fontsize=11)
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(fin_dir, "mdd_distribution.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # 4. Exposure vs Sharpe scatter
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x=self.df_beh["Avg_Exposure"], y=self.df_fin["Sharpe"],
                        ax=ax, color="purple", alpha=0.7, s=80)
        ax.set_title("Avg Exposure vs Sharpe", fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean Market Exposure (%)", fontsize=12)
        ax.set_ylabel("Sharpe Ratio", fontsize=12)
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(fin_dir, "exposure_vs_sharpe.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # ----------------------------------------------------------------
        # BEHAVIOR panels
        # ----------------------------------------------------------------

        best_idx = int(self.df_fin["Sharpe"].idxmax())
        best_w   = self._all_weights[best_idx]
        best_s   = self._all_sentiments[best_idx]
        best_e   = self._all_exposures[best_idx]
        T_b      = best_w.shape[0]
        t_range  = np.arange(T_b)
        colors   = plt.cm.Set2(np.linspace(0, 1, self.num_assets + 1))

        # 5. Weight allocation (stacked area)
        fig, ax = plt.subplots(figsize=(10, 5))
        cash_w = np.clip(1.0 - best_w.sum(axis=1), 0.0, 1.0)
        labels     = [f"Asset {i}" for i in range(self.num_assets)] + ["Cash"]
        all_series = [best_w[:, i] for i in range(self.num_assets)] + [cash_w]
        ax.stackplot(t_range, *all_series, labels=labels, colors=colors, alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Portfolio Allocation Over Time  "
            f"(Best Sharpe Episode — #{best_idx}, "
            f"Sharpe={self.df_fin.loc[best_idx,'Sharpe']:.2f})",
            fontsize=13, fontweight="bold"
        )
        ax.set_ylabel("Weight", fontsize=12)
        ax.set_xlabel("Step", fontsize=12)
        ax.legend(loc="upper right", fontsize=10, ncol=2)
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(beh_dir, "weight_allocation.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # 6. Weight distribution (histograms)
        fig, ax = plt.subplots(figsize=(10, 5))
        all_w_flat = np.concatenate(self._all_weights, axis=0)
        for i in range(self.num_assets):
            ax.hist(all_w_flat[:, i], bins=30, alpha=0.5,
                    label=f"Asset {i}", density=True, color=colors[i])
        all_cash_flat = np.clip(
            np.concatenate([1.0 - w.sum(axis=1) for w in self._all_weights]), 0, 1
        )
        ax.hist(all_cash_flat, bins=30, alpha=0.5,
                label="Cash", color="grey", density=True)
        ax.set_title("Weight Distribution (All Episodes)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Weight", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(beh_dir, "weight_distribution.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # 7. Sentiment overlay
        fig, ax = plt.subplots(figsize=(10, 5))
        if self.sentiment_mode == "per_asset" and len(best_s) > 0:
            sent_arr = np.array(best_s)
            n_plot = min(len(t_range) - 1, len(sent_arr))
            for i in range(self.num_assets):
                ax.plot(t_range[1:n_plot+1], best_w[1:n_plot+1, i],
                        color=colors[i], lw=2, label=f"w_{i}")
                ax.plot(t_range[1:n_plot+1], sent_arr[:n_plot, i] * 0.3 + 0.5,
                        color=colors[i], ls="--", lw=1.2, alpha=0.65,
                        label=f"sent_{i} (scaled)")
            ax.set_title("Asset Weights vs Per-Asset Sentiment (Best Episode)",
                         fontsize=13, fontweight="bold")
            ax.set_ylabel("Weight / Sentiment (scaled)", fontsize=12)
        else:
            if len(best_s) > 0:
                sent_arr = np.array(best_s, dtype=float)
                n_plot = min(len(best_e) - 1, len(sent_arr))
                ax.plot(t_range[1:n_plot+1], best_e[1:n_plot+1],
                        color="green", lw=2.5, label="Total Exposure")
                ax2 = ax.twinx()
                ax2.plot(t_range[1:n_plot+1], sent_arr[:n_plot],
                         color="darkorange", ls="--", lw=1.8, alpha=0.75,
                         label="Macro Sentiment")
                ax2.set_ylabel("Sentiment", color="darkorange", fontsize=12)
                ax2.tick_params(axis="y", labelcolor="darkorange")
                ax2.legend(loc="upper right", fontsize=10)
            ax.set_title("Total Exposure vs Macro Sentiment (Best Episode)",
                         fontsize=13, fontweight="bold")
            ax.set_ylabel("Market Exposure", fontsize=12)
        ax.set_xlabel("Step", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(beh_dir, "sentiment_overlay.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # 8. Stability metrics (text panel → clean figure)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")

        autocorr_lines = []
        for i in range(self.num_assets):
            all_w_i     = np.concatenate([w[1:, i] for w in self._all_weights])
            all_w_i_lag = np.concatenate([w[:-1, i] for w in self._all_weights])
            if len(all_w_i) > 10 and np.std(all_w_i) > 1e-8:
                ac = float(np.corrcoef(all_w_i, all_w_i_lag)[0, 1])
            else:
                ac = float("nan")
            autocorr_lines.append(f"  Asset {i}: autocorr = {ac:+.3f}")

        all_w_flat_stab = np.concatenate(self._all_weights, axis=0)
        all_cash_stab   = np.clip(
            np.concatenate([1.0 - w.sum(axis=1) for w in self._all_weights]), 0, 1
        )
        mean_w_lines = []
        for i in range(self.num_assets):
            m = float(all_w_flat_stab[:, i].mean())
            s = float(all_w_flat_stab[:, i].std())
            mean_w_lines.append(f"  Asset {i}: {m:.3f} +/- {s:.3f}")

        # Behavioural KPIs
        beh_summary = []
        for col in ["Avg_Exposure", "Turnover", "Corr_Macro_Sent"]:
            if col in self.df_beh.columns:
                m = self.df_beh[col].mean()
                s = self.df_beh[col].std()
                beh_summary.append(f"  {col}: {m:+.3f} ± {s:.3f}")
        if self.sentiment_mode == "per_asset":
            for col in [c for c in self.df_beh.columns if c.startswith("Corr_Asset_")]:
                m = self.df_beh[col].mean()
                flag = "✓" if m > 0.1 else ("~" if m >= 0 else "✗")
                beh_summary.append(f"  [{flag}] {col}: {m:+.3f}")

        lines = (
            ["Behavioural KPIs:", ""]
            + beh_summary
            + ["", "Action autocorrelation (1-step):",
               "  High (>0.9) = patient  |  Low (<0.5) = noisy"]
            + autocorr_lines
            + ["", "Mean portfolio weights:"]
            + mean_w_lines
            + [f"  Cash:  {all_cash_stab.mean():.3f} +/- {all_cash_stab.std():.3f}",
               "",
               f"Episodes evaluated: {len(self._all_weights)}",
               f"Best Sharpe: episode #{best_idx}  "
               f"(Sharpe = {self.df_fin.loc[best_idx,'Sharpe']:.2f})"]
        )
        ax.text(
            0.05, 0.97, "\n".join(lines),
            va="top", fontsize=11, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            transform=ax.transAxes,
        )
        ax.set_title("Policy Stability Metrics", fontsize=14, fontweight="bold")
        plt.suptitle(base_title, fontsize=11, y=1.01)
        plt.tight_layout()
        path = os.path.join(beh_dir, "stability_metrics.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        print(f"\nAll individual panels saved to: {output_dir}/")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    evaluator = DreamEvaluator(
        model_path=MODEL_PATH,
        num_assets=NUM_ASSETS,
        sentiment_mode=SENTIMENT_MODE,
        sentiment_lag=SENTIMENT_LAG,
        num_runs=NUM_RUNS,
        steps=EVAL_STEPS,
        data_mode=DATA_MODE,
        csv_path=CSV_PATH,
        vecnorm_path=VECNORM_PATH,
    )
    evaluator.run_monte_carlo()
    evaluator.plot_dashboard(output_img=OUTPUT_IMG)

    behavior_img = OUTPUT_IMG.replace(".png", "_behavior.png")
    evaluator.plot_behavior(output_img=behavior_img)

    # Save every panel individually for the thesis
    exp_name = os.path.splitext(os.path.basename(OUTPUT_IMG))[0]
    panels_dir = os.path.join(os.path.dirname(OUTPUT_IMG), "figures", exp_name)
    print(f"\nSaving individual panels to {panels_dir} …")
    evaluator.save_individual_panels(output_dir=panels_dir)
