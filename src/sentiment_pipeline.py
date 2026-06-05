#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global Financial Sentiment Analysis & Data Acquisition Pipeline.

Fetches news from Alpaca Markets and RSS Feeds, runs a FinBERT model
fine-tuned with rsLoRA to infer the sentiment of each article, and
computes a time-decay weighted global sentiment score.

Weighting scheme:
  - Today's articles receive the maximum weight (1.0).
  - Articles from 7 days ago receive the minimum weight (1/7 ≈ 0.143).
  - Weight for an article published N days ago = 1 / (N + 1), normalised
    so the final score is the weighted mean.

Sentiment score per article:
  Expected value of the sentiment distribution using temperature-scaled
  softmax (T=2) over FinBERT logits:
      score = P_scaled(pos)*1 + P_scaled(neu)*0 + P_scaled(neg)*(-1)
  where P_scaled(i) = softmax(logits / T)[i].
"""

import os
import time
import requests
import pandas as pd
import feedparser # need to be installed separately via pip install feedparser
from datetime import date, timedelta, datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel # need to be installed separately via pip install peft


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Alpaca Markets credentials
ALPACA_KEY    = os.getenv("ALPACA_KEY")  # placeholder value
ALPACA_SECRET = os.getenv("ALPACA_SECRET")  # placeholder value

# Search window: last 7 days
END_DATE   = date.today()
START_DATE = END_DATE - timedelta(days=7)

# Tickers to query on Alpaca
ALPACA_SYMBOLS = {
    "SP500":   "SPY",
    "NASDAQ":  "QQQ",
    "GOLD":    "GLD",
    "OIL":     "USO",
    "BITCOIN": "BTCUSD",
}

# Financial news RSS feeds
RSS_FEEDS = {
    "Yahoo Finance":  "https://finance.yahoo.com/news/rss/",
    "MarketWatch":    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Markets":   "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "Seeking Alpha":  "https://seekingalpha.com/feed.xml",
    "Investing.com":  "https://www.investing.com/rss/news.rss",
}

# Path to the rsLoRA adapter weights (adjust to match your environment)
MODEL_PATH      = "../model/finbert_rslora_finetuned"
BASE_MODEL_NAME = "ProsusAI/finbert"

# FinBERT label mapping: 0=negative, 1=neutral, 2=positive
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

# Softmax temperature for expected-value computation
TEMPERATURE = 1.0


# ==============================================================================
# 1. DATA ACQUISITION
# ==============================================================================

def fetch_alpaca_news(
    ticker_map: dict,
    api_key: str,
    api_secret: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch financial news from the Alpaca Markets News API.

    Parameters
    ----------
    ticker_map : dict
        Mapping of descriptive label → ticker symbol (e.g. {"SP500": "SPY"}).
    api_key : str
        Alpaca API key ID.
    api_secret : str
        Alpaca API secret key.
    start_date : str
        ISO-format start date (inclusive), e.g. "2025-05-11".
    end_date : str
        ISO-format end date (inclusive), e.g. "2025-05-18".

    Returns
    -------
    pd.DataFrame
        Columns: source, ticker, date, text.
    """
    print("[INFO] Fetching news from Alpaca Markets...")

    headers = {
        "APCA-API-KEY-ID":     api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    all_results = []

    for label, symbol in ticker_map.items():
        print(f"  -> Querying {symbol} ({label})...")
        params = {
            "symbols": symbol,
            "start":   f"{start_date}T00:00:00Z",
            "end":     f"{end_date}T23:59:59Z",
            "limit":   20,   # max per request
            "sort":    "desc",
        }
        try:
            r = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers=headers,
                params=params,
                timeout=15,
            )
            r.raise_for_status()
            for article in r.json().get("news", []):
                pub_date = article.get("created_at", "")[:10]   # "YYYY-MM-DD"
                text = (
                    f"{article.get('headline', '')}. "
                    f"{article.get('summary', '')}"
                ).strip()
                if len(text) > 15:
                    all_results.append(
                        {
                            "source": "Alpaca/Benzinga",
                            "ticker": label,
                            "date":   pub_date,
                            "text":   text,
                        }
                    )
        except Exception as exc:
            print(f"  [ERROR] Failed to retrieve {symbol}: {exc}")

        time.sleep(0.5)

    df = pd.DataFrame(all_results).drop_duplicates(subset=["text"])
    print(f"[Alpaca] Total articles retrieved: {len(df)}")
    return df


def fetch_rss_news(feeds_dict: dict) -> pd.DataFrame:
    """
    Fetch financial news from RSS feeds.

    Parameters
    ----------
    feeds_dict : dict
        Mapping of feed name → RSS URL.

    Returns
    -------
    pd.DataFrame
        Columns: source, ticker, date, text.
    """
    print("\n[INFO] Fetching news from RSS feeds...")
    results = []

    for name, url in feeds_dict.items():
        print(f"  -> Parsing {name}...")
        feed = feedparser.parse(url)

        for entry in feed.entries[:20]:
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            pub     = entry.get("published", "")          # raw date string
            text    = f"{title}. {summary}".strip()
            text    = text.split("<")[0]                  # strip inline HTML

            if len(text) > 15:
                results.append(
                    {
                        "source": name,
                        "ticker": "GENERAL",
                        "date":   pub,
                        "text":   text,
                    }
                )

    df = pd.DataFrame(results).drop_duplicates(subset=["text"])
    print(f"[RSS] Total articles retrieved: {len(df)}")
    return df


# ==============================================================================
# 2. MODEL LOADING
# ==============================================================================

def load_rslora_model(
    base_name: str,
    lora_path: str,
) -> tuple:
    """
    Load the base FinBERT model and apply rsLoRA adapter weights.

    Parameters
    ----------
    base_name : str
        HuggingFace model identifier for the base model
        (e.g. "ProsusAI/finbert").
    lora_path : str
        Local filesystem path to the directory containing the rsLoRA
        adapter weights and tokenizer.

    Returns
    -------
    tuple
        (model, tokenizer, device) on success, or (None, None, None) on
        failure.
    """
    print(f"\n[INFO] Loading FinBERT base model and rsLoRA weights from: {lora_path}")
    try:
        tokenizer  = AutoTokenizer.from_pretrained(lora_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_name, num_labels=3
        )
        model  = PeftModel.from_pretrained(base_model, lora_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        print(f"[INFO] Model loaded successfully on {device}.")
        return model, tokenizer, device

    except Exception as exc:
        print(f"[ERROR] Could not load the model: {exc}")
        print("[INFO] Make sure MODEL_PATH points to the correct directory.")
        return None, None, None


# ==============================================================================
# 3. SENTIMENT INFERENCE
# ==============================================================================

def temperature_softmax(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Apply temperature-scaled softmax to a batch of logits.

    Dividing logits by T > 1 flattens (softens) the distribution,
    making the model express less confidence in any single class.
    This produces a probability vector whose expected value is a more
    calibrated estimate of the overall sentiment.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model output, shape (batch_size, num_classes).
    temperature : float
        Scaling factor T. T=1 → standard softmax; T>1 → softer.

    Returns
    -------
    torch.Tensor
        Probability vectors, shape (batch_size, num_classes).
    """
    return torch.softmax(logits / temperature, dim=-1)


def analyze_sentiment(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    temperature: float = TEMPERATURE,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Run inference on every article in *df* and attach sentiment columns.

    The expected sentiment score for each article is the expected value
    of its temperature-scaled probability distribution:

        score = P_scaled(positive)*1
              + P_scaled(neutral)*0
              + P_scaled(negative)*(-1)

    Values close to +1 are strongly bullish; values close to -1 are
    strongly bearish.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a "text" column.
    model : PeftModel
        Fine-tuned FinBERT model.
    tokenizer : AutoTokenizer
        Matching tokenizer.
    device : torch.device
        Target compute device.
    temperature : float
        Softmax temperature (default: TEMPERATURE = 2.0).
    batch_size : int
        Number of texts processed per forward pass.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with:
        - prob_negative, prob_neutral, prob_positive  (scaled probabilities)
        - sentiment_score                              (expected value)
    """
    print(f"\n[INFO] Running sentiment inference (temperature={temperature})...")
    texts     = df["text"].tolist()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            enc    = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits                    # (B, 3)
            probs  = temperature_softmax(logits, temperature).cpu()
            all_probs.extend(probs.numpy())

    # FinBERT standard label order: 0=negative, 1=neutral, 2=positive
    df = df.copy()
    df["prob_negative"] = [p[0] for p in all_probs]
    df["prob_neutral"]  = [p[1] for p in all_probs]
    df["prob_positive"] = [p[2] for p in all_probs]

    # E[sentiment] = P(pos)·(+1) + P(neu)·(0) + P(neg)·(−1)
    df["sentiment_score"] = df["prob_positive"] - df["prob_negative"]

    return df


# ==============================================================================
# 4. TIME-DECAY WEIGHTING
# ==============================================================================

def assign_time_weights(df: pd.DataFrame, reference_date: date) -> pd.DataFrame:
    """
    Assign a recency weight to each article.

    The weight is inversely proportional to the article's age:

        weight(article) = 1 / (days_ago + 1)

    where days_ago = (reference_date − article_date).days.

    An article published today (days_ago=0) receives weight 1.0.
    An article from 7 days ago (days_ago=7) receives weight 1/8 ≈ 0.125.

    Articles whose date cannot be parsed are assigned days_ago=7
    (minimum weight).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a "date" column (ISO-format strings "YYYY-MM-DD" or
        RFC 2822 date strings from RSS feeds).
    reference_date : date
        The "today" reference point (normally date.today()).

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus a "time_weight" column.
    """
    df = df.copy()
    weights = []

    for raw_date in df["date"]:
        try:
            # Try ISO format first (Alpaca)
            parsed = date.fromisoformat(str(raw_date)[:10])
        except (ValueError, TypeError):
            try:
                # Fallback: RFC 2822 / HTTP date (RSS feeds)
                parsed = datetime.strptime(
                    str(raw_date), "%a, %d %b %Y %H:%M:%S %z"
                ).date()
            except Exception:
                parsed = reference_date - timedelta(days=7)   # worst case

        days_ago = max((reference_date - parsed).days, 0)
        days_ago = min(days_ago, 7)                            # clamp to window
        weights.append(1.0 / (days_ago + 1))

    df["time_weight"] = weights
    return df


# ==============================================================================
# 5. GLOBAL SCORE COMPUTATION
# ==============================================================================

def compute_global_sentiment(df: pd.DataFrame) -> float:
    """
    Compute the time-decay weighted mean sentiment score.

        global_score = Σ(score_i · weight_i) / Σ(weight_i)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "sentiment_score" and "time_weight" columns.

    Returns
    -------
    float
        Weighted mean sentiment in the range [−1, +1].
    """
    total_weight = df["time_weight"].sum()
    if total_weight == 0:
        return 0.0
    weighted_sum = (df["sentiment_score"] * df["time_weight"]).sum()
    return float(weighted_sum / total_weight)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Data collection
    # ------------------------------------------------------------------
    df_alpaca = fetch_alpaca_news(
        ALPACA_SYMBOLS,
        ALPACA_KEY,
        ALPACA_SECRET,
        START_DATE.isoformat(),
        END_DATE.isoformat(),
    )
    df_rss = fetch_rss_news(RSS_FEEDS)

    df_all = pd.concat([df_alpaca, df_rss], ignore_index=True)

    if df_all.empty:
        print("[WARNING] No articles found. Exiting.")
        return

    print(f"\n[INFO] Unified dataset: {len(df_all)} articles.")

    # ------------------------------------------------------------------
    # 2. Assign time-decay weights
    # ------------------------------------------------------------------
    df_all = assign_time_weights(df_all, reference_date=END_DATE)

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    model, tokenizer, device = load_rslora_model(BASE_MODEL_NAME, MODEL_PATH)

    if model is None:
        print(
            "[WARNING] Inference skipped — no model found at the configured path.\n"
            "          Set MODEL_PATH to the directory containing your rsLoRA weights."
        )
        return

    # ------------------------------------------------------------------
    # 4. Sentiment inference
    # ------------------------------------------------------------------
    df_analyzed = analyze_sentiment(df_all, model, tokenizer, device, TEMPERATURE)

    # ------------------------------------------------------------------
    # 5. Weighted global score
    # ------------------------------------------------------------------
    global_score = compute_global_sentiment(df_analyzed)

    print("\n" + "=" * 60)
    print("                        RESULTS")
    print("=" * 60)
    print(f"Total articles analysed       : {len(df_analyzed)}")
    print(f"Softmax temperature           : {TEMPERATURE}")
    print(f"GLOBAL SENTIMENT SCORE (wmean): {global_score:.4f}")

    if global_score > 0.15:
        interpretation = "Clearly Bullish"
    elif global_score < -0.15:
        interpretation = "Clearly Bearish"
    else:
        interpretation = "Neutral / Mixed"

    print(f"Interpretation                : {interpretation}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    output_file = "global_sentiment_analysis.csv"
    df_analyzed.to_csv(output_file, index=False)
    print(f"\n[INFO] Results saved to '{output_file}'")


if __name__ == "__main__":
    main()