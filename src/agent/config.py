"""
D.R.E.A.M. — Shared configuration and normalisation constants.

This module is the single source of truth for any value that must be
consistent across training, evaluation, and inference.  Import from here
rather than hardcoding values in individual modules.

Sentiment normalisation
-----------------------
The empirical sentiment signal (sentiment_zscore) was normalised using
statistics computed over the full training split of the dataset.  The
same constants are applied to the synthetic generator so that both data
sources produce signals in the same numerical space.  This is essential
for synthetic-to-real transfer: the agent must see the same distribution
of sentiment values during synthetic training as it will during empirical
evaluation and live inference.

How the empirical stats were computed (from 10_data_agent.ipynb):

    train_df = pd.read_csv("data/train_agent_dataset.csv")
    SENTIMENT_MEAN = train_df["sentiment_raw"].mean()   # 0.143970
    SENTIMENT_STD  = train_df["sentiment_raw"].std()    # 0.114915

    # Applied to both train and test splits:
    df["sentiment_zscore"] = (
        df["sentiment_raw"] - SENTIMENT_MEAN
    ) / SENTIMENT_STD

After this transform, sentiment_zscore in the CSV has µ ≈ 0, σ ≈ 1.

Synthetic generator calibration
---------------------------------
The Wyckoff phase sentiment parameters (mu_s, sigma_s) were derived by:

    1. Choosing target z-score means per phase that preserve the Wyckoff
       narrative (Markup positive, Markdown negative) and are centred so
       that the weighted mean across phase durations equals zero.

    2. Solving for sigma_within so that the combined (between-phase +
       within-phase) variance equals 1.0.

    3. Converting back to the original FinBERT scale:
           mu_orig    = SENTIMENT_MEAN + z_target * SENTIMENT_STD
           sigma_orig = sigma_within_z  * SENTIMENT_STD

    Validated with a 300k-sample Monte Carlo:
        µ = +0.0007  (target: 0.00)
        σ =  0.9987  (target: 1.00)
        p5 = -1.60   p95 = +1.46

Target z-scores per Wyckoff phase (post-normalisation):
    Accumulation :  -0.762
    Markup       :  +0.985
    Distribution :  +0.199
    Markdown     :  -1.373
"""

# ---------------------------------------------------------------------------
# Sentiment normalisation constants
# Source: mean and std of raw FinBERT output in train_agent_dataset.csv
# Computed in: notebooks/10_data_agent.ipynb
# ---------------------------------------------------------------------------

SENTIMENT_MEAN: float = 0.143970
SENTIMENT_STD:  float = 0.114915


# ---------------------------------------------------------------------------
# Wyckoff phase sentiment parameters  (original / pre-normalisation scale)
# Format: {phase_index: (mu_sentiment, sigma_sentiment)}
#   0 = Accumulation, 1 = Markup, 2 = Distribution, 3 = Markdown
# ---------------------------------------------------------------------------

WYCKOFF_SENTIMENT_PARAMS: dict[int, tuple[float, float]] = {
    0: ( 0.056443, 0.048754),   # Accumulation  → z ≈ -0.76
    1: ( 0.257114, 0.048754),   # Markup        → z ≈ +0.99
    2: ( 0.166812, 0.048754),   # Distribution  → z ≈ +0.20
    3: (-0.013792, 0.048754),   # Markdown      → z ≈ -1.37
}

RANDOM_SEED: int = 42