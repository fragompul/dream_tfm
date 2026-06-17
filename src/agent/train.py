"""
D.R.E.A.M. Reinforcement Learning Training Pipeline — Phase 6

Changes vs Phase 5
------------------
1. Separate actor/critic networks (net_arch=dict(pi=..., vf=...))
   In Phase 5 the policy used shared base layers, meaning the critic's
   noisy gradients contaminated the actor.  Separate networks allow each
   head to specialise independently.  This is the most likely cause of
   the ev ~= 0 collapse observed in macro-mode experiments.

2. VecNormalize for reward normalisation
   Enabled for synthetic training only.
   Controlled by USE_VECNORM flag.
   Set True for synthetic (high reward variance across Wyckoff regimes).
   Set False for empirical fine-tuning: with a small dataset the
   running reward stats converge to near-zero std, normalising every
   reward to ~0 and causing the policy to collapse to all-cash.

3. B&H baseline adapted for N+1 softmax action space
   The equal-weight static action now emits N+1 logits where the first
   N are equal and the cash logit is set to a low value, so softmax
   produces approximately equal asset weights with minimal cash.

4. Action space changed from Box([-1,1]^N) to Box([-inf,inf]^(N+1))
   Matches the new DreamEnv Phase 6 softmax action space.

Experiment guide
----------------
Exp 1 — Pipeline sanity check:
    DATA_MODE = "synthetic", SENTIMENT_MODE = "macro"
    NUM_ASSETS = 2, TOTAL_TIMESTEPS = 300_000, TURNOVER_COEF = 0.03

Exp 2 — Per-asset signal learning:
    DATA_MODE = "synthetic", SENTIMENT_MODE = "per_asset"
    NUM_ASSETS = 3, TOTAL_TIMESTEPS = 700_000, TURNOVER_COEF = 0.01

Exp 3 — Macro-only (simulates real-data conditions):
    DATA_MODE = "synthetic", SENTIMENT_MODE = "macro"
    NUM_ASSETS = 3, TOTAL_TIMESTEPS = 700_000, TURNOVER_COEF = 0.03

Exp 4 — Real data transfer (fine-tune from Exp 3 or from scratch):
    DATA_MODE = "empirical", SENTIMENT_MODE = "macro"
    NUM_ASSETS = 3, TOTAL_TIMESTEPS = 500_000, TURNOVER_COEF = 0.01
    CASH_PENALTY = 0.1, MIN_EXPOSURE = 0.5
    USE_VECNORM = False, EPISODE_STEPS = 126
    LR_INITIAL = 3e-4, LR_FINAL = 3e-5
"""

import os
import sys
import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.agent.data_feed import WyckoffMockData, EmpiricalDataFeed
from src.agent.dream_env import DreamEnv
from src.agent.config import RANDOM_SEED

# =============================================================================
# Training configuration
# =============================================================================

DATA_MODE: str      = "empirical"
SENTIMENT_MODE: str = "macro"
SENTIMENT_LAG: int  = 1
NUM_ASSETS: int     = 3
EPISODE_STEPS: int  = 126           # empirical: 126 | synthetic: 1000
TOTAL_TIMESTEPS: int = 300_000      # finetune: 300_000 | train: 700_000

TURNOVER_COEF: float = 0.01   # per_asset: 0.01 | macro: 0.03
CASH_PENALTY:  float = 0.    # synthetic: 0.0  | empirical: 0.1
MIN_EXPOSURE:  float = 0.    # minimum investment mandate (50%)

FINETUNE_FROM: str | None = "test_modelos/dream_synthetic_macro_N3.zip" # "test_modelos/dream_synthetic_macro_N3.zip"

# When fine-tuning from a synthetic checkpoint into empirical data,
# the VecNormalize running stats from synthetic training are incompatible
# with the empirical reward scale and must be reset.
# True  — reset stats (use when crossing synthetic -> empirical boundary)
# False — keep stats (use when continuing empirical from empirical checkpoint)
# Controls whether VecNormalize reward normalisation is applied.
# True  — recommended for synthetic training (high reward variance).
# False — recommended for empirical fine-tuning (stable reward scale;
#         running stats converge to near-zero std and collapse policy).
USE_VECNORM: bool = False   # synthetic: True | empirical: False
RESET_VECNORM: bool = True

# Path to saved VecNormalize stats (.pkl) to load when RESET_VECNORM=False.
# Use when continuing an empirical run from an empirical checkpoint so
# the normaliser resumes with the accumulated stats from that run.
# Set to None when RESET_VECNORM=True (stats will be reset anyway).
VECNORM_PATH: str | None = None   # e.g. "test_modelos/dream_empirical_macro_N3_v6_vecnorm.pkl"

LR_INITIAL: float = 3e-4
LR_FINAL:   float = 3e-5

def _linear_lr_schedule(initial: float, final: float):
    def schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining
    return schedule


PPO_HYPERPARAMETERS: dict = {
    "learning_rate": _linear_lr_schedule(LR_INITIAL, LR_FINAL),
    "n_steps":       2048,      # empirical: 2048 | synthetic: 2000
    "batch_size":    252,       # empirical: 252  | synthetic: 200
    "n_epochs":      10,
    "gamma":         0.98,
    "gae_lambda":    0.95,
    "clip_range":    0.20,
    "ent_coef":      0.005,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    # Separate actor and critic networks.
    # Prevents noisy critic gradients from contaminating the actor,
    # which was a likely cause of ev ~= 0 in macro-mode experiments.
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[256, 256],   # actor
            vf=[256, 256],   # critic
        )
    ),
}

# =============================================================================
# Logging
# =============================================================================

class _TeeStream:
    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._original.flush()
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    return log_path

# =============================================================================
# Checkpoint + compact log callback
# =============================================================================

class CheckpointAndLogCallback(BaseCallback):
    def __init__(
        self,
        save_every_steps: int = 100_000,
        log_every_iterations: int = 25,
        checkpoint_dir: str = "./test_modelos/checkpoints",
        baseline_reward: float = 0.0,
    ):
        super().__init__(verbose=0)
        self.save_every_steps      = save_every_steps
        self.log_every_iterations  = log_every_iterations
        self.checkpoint_dir        = checkpoint_dir
        self.baseline_reward       = baseline_reward
        self._last_checkpoint_step = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        steps_since = self.num_timesteps - self._last_checkpoint_step
        if steps_since >= self.save_every_steps:
            path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{self.num_timesteps:_d}",
            )
            self.model.save(path)
            print(f"\n  Checkpoint -> {path}.zip  ({self.num_timesteps:,} steps)\n")
            self._last_checkpoint_step = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        if self.n_calls % self.log_every_iterations != 0:
            return
        buf = self.model.ep_info_buffer
        if not buf:
            return

        ep_rewards = [ep["r"] for ep in buf]
        ep_lengths = [ep["l"] for ep in buf]
        mean_rew = float(np.mean(ep_rewards))
        mean_len = float(np.mean(ep_lengths))
        rew_step = mean_rew / mean_len if mean_len > 0 else 0.0
        gap      = mean_rew - self.baseline_reward

        ev  = self.logger.name_to_value.get("train/explained_variance", float("nan"))
        std = self.logger.name_to_value.get("train/std",                float("nan"))
        vl  = self.logger.name_to_value.get("train/value_loss",         float("nan"))

        print(
            f"[{self.num_timesteps:>9,}] "
            f"ep_rew={mean_rew:+7.1f}  ep_len={mean_len:5.0f}  "
            f"rew/step={rew_step:+.3f}  gap_vs_bnh={gap:+.1f}  "
            f"ev={ev:+.4f}  std={std:.3f}  vl={vl:.2f}"
        )

# =============================================================================
# Buy-and-hold baseline
# =============================================================================

def evaluate_buy_and_hold_baseline(
    data_generator, initial_balance: float = 10_000.0, n_episodes: int = 20
) -> float:
    """
    Equal-weight B&H adapted for the N+1 softmax action space.

    We emit N+1 logits where the first N are equal (value=2.0) and the
    cash logit is low (value=-4.0).  After softmax this gives approximately
    equal allocation to all assets with minimal cash.
    """
    env = DreamEnv(
        data_generator=data_generator,
        initial_balance=initial_balance,
        turnover_coef=TURNOVER_COEF,
        cash_penalty=CASH_PENALTY,
        min_exposure=MIN_EXPOSURE,
    )
    num_assets = env.num_assets

    # N equal asset logits + 1 low cash logit
    static_action = np.array(
        [2.0] * num_assets + [-4.0], dtype=np.float32
    )

    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=RANDOM_SEED)
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(static_action)
            total += r
            done = terminated or truncated
        rewards.append(total)

    mean, std = float(np.mean(rewards)), float(np.std(rewards))
    print(f"  B&H baseline — mean reward: {mean:.2f} +/- {std:.2f}")
    return mean

# =============================================================================
# Main entry point
# =============================================================================

def train_dream_agent() -> PPO:
    import random
    import torch

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    log_dir  = "./test_modelos/logs"
    log_path = setup_logging(log_dir)

    print("=" * 70)
    print("      D.R.E.A.M. RL PIPELINE — Phase 6")
    print("=" * 70)
    print(f"  Data:      {DATA_MODE.upper()}")
    print(f"  Sentiment: {SENTIMENT_MODE}")
    print(f"  Assets:    {NUM_ASSETS}")
    print(f"  Steps/ep:  {EPISODE_STEPS:,}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    if FINETUNE_FROM:
        print(f"  Fine-tune: {FINETUNE_FROM}")
    print(f"  Turnover:  {TURNOVER_COEF}")
    print(f"  Cash pen:  {CASH_PENALTY} (min_exp={MIN_EXPOSURE})")
    print(f"  Log:       {log_path}")

    # ------------------------------------------------------------------
    # 1. Data feed
    # ------------------------------------------------------------------
    if DATA_MODE == "synthetic":
        data_generator = WyckoffMockData(
            steps=EPISODE_STEPS,
            num_assets=NUM_ASSETS,
            sentiment_mode=SENTIMENT_MODE,
            sentiment_lag=SENTIMENT_LAG,
        )
    elif DATA_MODE == "empirical":
        csv_path = "./data/train_agent_dataset.csv"
        data_generator = EmpiricalDataFeed(
            csv_path=csv_path,
            steps=EPISODE_STEPS,
            num_assets=NUM_ASSETS,
            randomize=True,
        )
    else:
        raise ValueError(f"Unknown DATA_MODE: '{DATA_MODE}'")

    # ------------------------------------------------------------------
    # 2. Baseline
    # ------------------------------------------------------------------
    print("\nEvaluating B&H baseline (20 episodes) ...")
    baseline_reward = evaluate_buy_and_hold_baseline(data_generator)
    print(f"  Baseline: {baseline_reward:.2f}")

    # ------------------------------------------------------------------
    # 3. Environment + VecNormalize
    # ------------------------------------------------------------------
    # Monitor must wrap DreamEnv BEFORE VecNormalize so that episode
    # rewards (ep_rew_mean) are recorded in the un-normalised scale and
    # appear in TensorBoard and the compact callback log.
    # Order: DreamEnv -> Monitor -> DummyVecEnv -> VecNormalize
    def make_env():
        env = DreamEnv(
            data_generator=data_generator,
            turnover_coef=TURNOVER_COEF,
            cash_penalty=CASH_PENALTY,
            min_exposure=MIN_EXPOSURE,
        )
        return Monitor(env)

    raw_env = DreamEnv(
        data_generator=data_generator,
        turnover_coef=TURNOVER_COEF,
        cash_penalty=CASH_PENALTY,
        min_exposure=MIN_EXPOSURE,
    )
    print("\nRunning Gymnasium integrity check ...")
    check_env(raw_env, warn=True)
    obs_dim = raw_env.observation_space.shape[0]
    act_dim = raw_env.action_space.shape[0]
    print(f"  Passed.  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"(mode={raw_env.sentiment_mode})")

    # VecNormalize: enabled for synthetic, disabled for empirical.
    # Synthetic episodes vary widely in reward scale across Wyckoff
    # regimes, so normalisation stabilises the critic.
    # Empirical fine-tuning uses a small dataset: with enough steps
    # the running reward std converges to near-zero, normalising every
    # reward to ~0 and collapsing the policy to all-cash. Empirical
    # reward is stable enough without normalisation.
    venv = DummyVecEnv([make_env])
    if USE_VECNORM:
        venv = VecNormalize(
            venv,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=PPO_HYPERPARAMETERS["gamma"],
        )
        print("  VecNormalize: enabled (synthetic mode)")
    else:
        print("  VecNormalize: disabled (empirical mode)")

    # ------------------------------------------------------------------
    # 4. Directories
    # ------------------------------------------------------------------
    model_dir      = "./test_modelos"
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(model_dir,      exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. PPO model
    # ------------------------------------------------------------------
    print("\nBuilding PPO model ...")

    if FINETUNE_FROM and os.path.exists(FINETUNE_FROM):
        # For synthetic fine-tune with RESET_VECNORM=False, optionally
        # load saved VecNorm stats from a previous run.
        if USE_VECNORM and not RESET_VECNORM:
            if VECNORM_PATH and os.path.exists(VECNORM_PATH):
                raw_venv = DummyVecEnv([make_env])
                venv = VecNormalize.load(VECNORM_PATH, raw_venv)
                venv.training = True
                print(f"  VecNormalize stats loaded from {VECNORM_PATH}.")
            else:
                print("  VecNormalize stats retained (RESET_VECNORM=False).")

        print(f"  Loading weights from: {FINETUNE_FROM}")
        model = PPO.load(
            FINETUNE_FROM,
            env=venv,
            device="cpu",
            custom_objects={
                "learning_rate": PPO_HYPERPARAMETERS["learning_rate"],
                "clip_range":    PPO_HYPERPARAMETERS["clip_range"],
            },
        )
        model.ent_coef = PPO_HYPERPARAMETERS["ent_coef"]
    else:
        model = PPO(
            policy="MlpPolicy",
            env=venv,
            verbose=1,
            device="cpu",
            seed=RANDOM_SEED,
            tensorboard_log=log_dir,
            **PPO_HYPERPARAMETERS,
        )

    print("\nHyperparameters:")
    for k, v in PPO_HYPERPARAMETERS.items():
        if k == "policy_kwargs":
            print(f"    {'policy_kwargs':<20} net_arch pi={v['net_arch']['pi']} vf={v['net_arch']['vf']}")
        else:
            val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
            print(f"    {k:<20} {val_str}")

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    callback = CheckpointAndLogCallback(
        save_every_steps=max(50_000, TOTAL_TIMESTEPS // 10),
        log_every_iterations=25,
        checkpoint_dir=checkpoint_dir,
        baseline_reward=baseline_reward,
    )

    print(f"\nStarting training ({TOTAL_TIMESTEPS:,} timesteps) ...")
    print("-" * 70)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    print("-" * 70)

    # ------------------------------------------------------------------
    # 7. Save model + VecNormalize stats
    # ------------------------------------------------------------------
    model_name = f"dream_{DATA_MODE}_{SENTIMENT_MODE}_N{NUM_ASSETS}"
    save_path  = os.path.join(model_dir, model_name)
    model.save(save_path)

    print(f"\nModel saved    -> {save_path}.zip")
    if USE_VECNORM:
        vecnorm_path = os.path.join(model_dir, f"{model_name}_vecnorm.pkl")
        venv.save(vecnorm_path)
        print(f"VecNorm saved  -> {vecnorm_path}")
    print(f"Log            -> {log_path}")
    print("Training complete.")
    print("=" * 70)

    return model


if __name__ == "__main__":
    train_dream_agent()
