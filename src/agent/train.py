"""
D.R.E.A.M. Reinforcement Learning Training Pipeline — Phase 5

Key changes from Phase 4:
    - SENTIMENT_MODE parameter selects "macro" or "per_asset" data generation.
    - SENTIMENT_LAG controls how many steps per-asset sentiment leads price.
    - Observation space dimension is set automatically by DreamEnv based on
      the sentiment mode detected from the data feed.
    - TOTAL_TIMESTEPS reduced for synthetic experiments (500k is enough to
      validate convergence; empirical experiments can be run longer).
    - ent_coef reduced from 0.02 → 0.005 to prevent the std blow-up observed
      in Phase 4 long runs.
    - Linear learning rate decay added: starts at LR_INITIAL and decays to
      LR_FINAL over the training budget, preventing late-stage std drift.

Experiment guide
----------------
Exp 1 — Pipeline sanity check:
    DATA_MODE      = "synthetic"
    SENTIMENT_MODE = "macro"
    NUM_ASSETS     = 2
    TOTAL_TIMESTEPS = 300_000

Exp 2 — Per-asset signal learning:
    DATA_MODE      = "synthetic"
    SENTIMENT_MODE = "per_asset"
    NUM_ASSETS     = 3
    TOTAL_TIMESTEPS = 700_000

Exp 3 — Macro-only (simulates real-data conditions):
    DATA_MODE      = "synthetic"
    SENTIMENT_MODE = "macro"
    NUM_ASSETS     = 3
    TOTAL_TIMESTEPS = 700_000

Exp 4 — Real data transfer (eval only, or fine-tune from Exp 3 checkpoint):
    DATA_MODE      = "empirical"
    SENTIMENT_MODE = "macro"   (EmpiricalDataFeed is always macro)
    NUM_ASSETS     = 3
    TOTAL_TIMESTEPS = 300_000  (fine-tune budget)
"""

import os
import sys
import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from src.agent.data_feed import WyckoffMockData, EmpiricalDataFeed
from src.agent.dream_env import DreamEnv
from src.agent.config import RANDOM_SEED

# =============================================================================
# Training configuration — edit these for each experiment
# =============================================================================

DATA_MODE: str      = "empirical"    # "synthetic" | "empirical"
SENTIMENT_MODE: str = "macro"    # "macro"     | "per_asset"
SENTIMENT_LAG: int  = 2              # steps sentiment leads price (per_asset)
NUM_ASSETS: int     = 3
EPISODE_STEPS: int  = 512
TOTAL_TIMESTEPS: int = 300_000

# Fine-tune from an existing checkpoint? Set path or None.
FINETUNE_FROM: str | None = "test_modelos/dream_synthetic_macro_N3_v5.zip"   # e.g. "test_modelos/checkpoints/checkpoint_700_000.zip"

# Learning rate schedule: linear decay from LR_INITIAL → LR_FINAL
LR_INITIAL: float = 2e-4
LR_FINAL:   float = 4e-5

def _linear_lr_schedule(initial: float, final: float):
    """Returns a callable that linearly decays from initial to final."""
    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 at start → 0.0 at end
        return final + (initial - final) * progress_remaining
    return schedule


PPO_HYPERPARAMETERS: dict = {
    "learning_rate": _linear_lr_schedule(LR_INITIAL, LR_FINAL),
    "n_steps":       1024,
    "batch_size":    128,
    "n_epochs":      10,
    "gamma":         0.98,
    "gae_lambda":    0.95,
    "clip_range":    0.20,
    "ent_coef":      0.005,   # ↓ from 0.02 — prevents std blow-up in long runs
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
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
    """
    Saves checkpoints every save_every_steps and prints a one-line summary
    every log_every_iterations rollout updates.
    """

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
            print(f"\n  💾 Checkpoint → {path}.zip  ({self.num_timesteps:,} steps)\n")
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
    env = DreamEnv(data_generator=data_generator, initial_balance=initial_balance)
    num_assets   = env.num_assets
    equal_weight = 1.0 / num_assets
    static_action = np.full(num_assets, 2.0 * equal_weight - 1.0, dtype=np.float32)

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
    print(f"  B&H baseline — mean reward: {mean:.2f} ± {std:.2f}")
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
    print("      D.R.E.A.M. RL PIPELINE — Phase 5")
    print("=" * 70)
    print(f"  Data:      {DATA_MODE.upper()}")
    print(f"  Sentiment: {SENTIMENT_MODE}")
    print(f"  Assets:    {NUM_ASSETS}")
    print(f"  Steps/ep:  {EPISODE_STEPS:,}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    if FINETUNE_FROM:
        print(f"  Fine-tune: {FINETUNE_FROM}")
    print(f"  Log:       {log_path}")

    # ------------------------------------------------------------------
    # 1. Data feed
    # ------------------------------------------------------------------
    if DATA_MODE == "synthetic":
        print(f"\nInitialising WyckoffMockData "
              f"(mode={SENTIMENT_MODE}, lag={SENTIMENT_LAG}) …")
        data_generator = WyckoffMockData(
            steps=EPISODE_STEPS,
            num_assets=NUM_ASSETS,
            sentiment_mode=SENTIMENT_MODE,
            sentiment_lag=SENTIMENT_LAG,
        )
    elif DATA_MODE == "empirical":
        csv_path = "./data/train_agent_dataset.csv"
        print(f"\nLoading empirical data from {csv_path} …")
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
    print("\nEvaluating B&H baseline (20 episodes) …")
    baseline_reward = evaluate_buy_and_hold_baseline(data_generator)
    print(f"  Baseline: {baseline_reward:.2f}")

    # ------------------------------------------------------------------
    # 3. Environment check
    # ------------------------------------------------------------------
    env = DreamEnv(data_generator=data_generator)
    print("\nRunning Gymnasium integrity check …")
    check_env(env, warn=True)

    obs_dim = env.observation_space.shape[0]
    print(f"  ✅ Passed.  obs_dim={obs_dim}  "
          f"(mode={env.sentiment_mode})")

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
    print("\nBuilding PPO model …")

    if FINETUNE_FROM and os.path.exists(FINETUNE_FROM):
        print(f"  Loading weights from checkpoint: {FINETUNE_FROM}")
        model = PPO.load(
            FINETUNE_FROM,
            env=env,
            device="cpu",
            custom_objects={
                "learning_rate": PPO_HYPERPARAMETERS["learning_rate"],
                "clip_range":    PPO_HYPERPARAMETERS["clip_range"],
            },
        )
        # Override ent_coef for fine-tuning (avoid re-exploration explosion)
        model.ent_coef = PPO_HYPERPARAMETERS["ent_coef"]
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cpu",
            tensorboard_log=log_dir,
            seed=RANDOM_SEED,
            **PPO_HYPERPARAMETERS,
        )

    print("\nHyperparameters:")
    for k, v in PPO_HYPERPARAMETERS.items():
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

    print(f"\nStarting training ({TOTAL_TIMESTEPS:,} timesteps) …")
    print("-" * 70)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    print("-" * 70)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    model_name = (
        f"dream_{DATA_MODE}_{SENTIMENT_MODE}_N{NUM_ASSETS}_v5"
    )
    save_path = os.path.join(model_dir, model_name)
    model.save(save_path)

    print(f"\nModel saved → {save_path}.zip")
    print(f"Log         → {log_path}")
    print("✅ Training complete.")
    print("=" * 70)

    return model


if __name__ == "__main__":
    train_dream_agent()
