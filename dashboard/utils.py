import time
import random
import pandas as pd
import numpy as np

def mock_inference_nlp(text):
    """Mocks the FinBERT rsLoRA NLP phase."""
    time.sleep(1.5)  # Simulate inference time
    score = random.uniform(-1, 1)
    
    # Generate fake SHAP-like data for words
    words = text.split()
    shap_values = {word: random.uniform(-0.5, 0.5) for word in words}
    
    return {
        "score": score,
        "sentiment": "Bullish" if score > 0.2 else ("Bearish" if score < -0.2 else "Neutral"),
        "shap_values": shap_values
    }

def mock_inference_volatility(ticker):
    """Mocks the Volatility Simple Ensemble prediction."""
    time.sleep(2.0)  # Simulate inference time
    
    # Dummy historical data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    hist_vol = np.random.normal(loc=0.02, scale=0.005, size=30)
    
    # Predicted risk t+1
    pred_risk = hist_vol[-1] + random.uniform(-0.005, 0.005)
    
    df = pd.DataFrame({"Date": dates, "Volatility": hist_vol})
    
    return {
        "prediction_t1": max(0, pred_risk),
        "historical_df": df
    }

def mock_agent_step(ticker, current_portfolio_value):
    """Mocks a single decision step of the D.R.E.A.M. Agent."""
    time.sleep(1.0)
    
    actions = ["Buy", "Sell", "Hold"]
    action = random.choices(actions, weights=[0.4, 0.3, 0.3])[0]
    confidence = random.uniform(0.6, 0.99)
    
    # Simulate return
    simulated_return = random.uniform(-0.02, 0.03)
    new_value = current_portfolio_value * (1 + simulated_return)
    
    return {
        "action": action,
        "confidence": confidence,
        "new_value": new_value,
        "simulated_return": simulated_return
    }
