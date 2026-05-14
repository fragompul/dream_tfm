import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import joblib
import warnings

# Ignore yfinance and sklearn warnings for clean output
warnings.filterwarnings('ignore')

# Set device dynamically
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. DEEP LEARNING ARCHITECTURES (From Notebook)
# ==========================================
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(VolatilityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros on the correct device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class VolatilityGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(VolatilityGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. UTILS & DEPENDENCY LOADING
# ==========================================
def load_compiled_state_dict(model, path, device):
    """
    Loads PyTorch weights, handling the '_orig_mod.' prefix added by torch.compile()
    during training to prevent KeyErrors during inference.
    """
    state_dict = torch.load(path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    return model

def load_artifacts(models_dir="../models"):
    """Loads scalers and the 6 ensemble models."""
    
    # 1. Scalers
    ml_scaler = joblib.load(os.path.join(models_dir, "ml_feature_scaler.pkl"))
    dl_scaler = joblib.load(os.path.join(models_dir, "dl_feature_scaler.pkl"))

    # 2. ML Models
    ml_models = {
        "Ridge": joblib.load(os.path.join(models_dir, "ridge.pkl")),
        "RandomForest": joblib.load(os.path.join(models_dir, "randomforest.pkl")),
        "XGBoost": joblib.load(os.path.join(models_dir, "xgboost.pkl")),
        "LightGBM": joblib.load(os.path.join(models_dir, "lightgbm.pkl"))
    }

    # 3. DL Models (Input size is 8: 2 numerical + 6 OHE columns)
    INPUT_SIZE = 8
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3

    lstm = VolatilityLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    lstm = load_compiled_state_dict(lstm, os.path.join(models_dir, "lstm_volatility.pth"), DEVICE)
    lstm.eval()

    gru = VolatilityGRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    gru = load_compiled_state_dict(gru, os.path.join(models_dir, "gru_volatility.pth"), DEVICE)
    gru.eval()

    dl_models = {
        "LSTM": lstm,
        "GRU": gru
    }

    return ml_scaler, dl_scaler, ml_models, dl_models

# ==========================================
# 3. DATA FETCHING & FEATURE ENGINEERING
# ==========================================
def fetch_and_prepare_data(ticker, asset_class):
    """Downloads live data and engineers the necessary technical and lag features."""
    
    df = yf.download(ticker, period="60d", progress=False)
    
    if df.empty:
        raise ValueError(f"Could not fetch data for ticker {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Base Features
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['historical_volatility_20d'] = df['log_return'].rolling(window=20).std()

    # Lag Features for ML models
    df['log_return_lag1'] = df['log_return'].shift(1)
    df['volatility_lag1'] = df['historical_volatility_20d'].shift(1)

    df = df.dropna()

    if len(df) < 20:
        raise ValueError("Not enough data points after cleaning NaNs to build a 20-day DL sequence.")

    # Guaranteed Static One-Hot Encoding mimicking `drop_first=True`
    # 'Commodities' is omitted deliberately as it is dropped by drop_first=True
    expected_classes = [
        'Cryptocurrency', 'Equities_Global', 'Equities_US_Broad', 
        'Fixed_Income', 'Sectors_US'
    ]
    
    for cls in expected_classes:
        df[f'asset_class_{cls}'] = 1.0 if cls == asset_class else 0.0

    return df

# ==========================================
# 4. TENSOR PREPARATION
# ==========================================
def build_inputs(df, ml_scaler, dl_scaler):
    """Splits and scales features for ML (1D) and DL (3D) models."""
    
    ohe_cols = [
        'asset_class_Cryptocurrency', 'asset_class_Equities_Global', 
        'asset_class_Equities_US_Broad', 'asset_class_Fixed_Income', 'asset_class_Sectors_US'
    ]

    # Exact column order from training dataframe
    ml_features = ['Volume', 'log_return', 'historical_volatility_20d', 'log_return_lag1', 'volatility_lag1'] + ohe_cols
    dl_features = ['Volume', 'log_return', 'historical_volatility_20d'] + ohe_cols

    # --- ML INPUT (Last available row only, 10 features) ---
    ml_data = df[ml_features].iloc[-1:].values
    ml_input_scaled = ml_scaler.transform(ml_data)

    # --- DL INPUT (Last 20 days sequence, 8 features) ---
    dl_data = df[dl_features].iloc[-20:].values
    dl_data_scaled = dl_scaler.transform(dl_data)
    
    # Add batch dimension: [batch_size=1, seq_len=20, features=8]
    dl_input_tensor = torch.tensor(dl_data_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    return ml_input_scaled, dl_input_tensor

# ==========================================
# 5. MAIN RISK ORACLE FUNCTION
# ==========================================
def predict_volatility(ticker, asset_class, models_dir="../models"):
    """
    Main function to be called by the Phase 3 DRL Agent.
    Fetches recent data and outputs the ensemble's predicted volatility for t+1.
    """
    # 1. Load weights and scalers
    ml_scaler, dl_scaler, ml_models, dl_models = load_artifacts(models_dir)
    
    # 2. Fetch live data
    df = fetch_and_prepare_data(ticker, asset_class)
    
    # 3. Build appropriate tensors
    ml_input, dl_input = build_inputs(df, ml_scaler, dl_scaler)
    
    # 4. Inference
    predictions = {}
    
    # ML Predictions
    for name, model in ml_models.items():
        pred = model.predict(ml_input)[0]
        predictions[name] = float(pred)
        
    # DL Predictions
    with torch.no_grad():
        for name, model in dl_models.items():
            pred = model(dl_input).item()
            predictions[name] = pred

    # 5. Simple Ensemble (Arithmetic Mean of all 6 models)
    ensemble_risk = np.mean(list(predictions.values()))
    
    return {
        "ticker": ticker,
        "asset_class": asset_class,
        "predicted_volatility_t1": float(ensemble_risk),
        "details": predictions
    }

# ==========================================
# LOCAL TESTING
# ==========================================
if __name__ == "__main__":
    # Test execution block
    TEST_TICKER = "SPY"
    TEST_ASSET_CLASS = "Equities_US_Broad"
    
    try:
        print(f"[{TEST_TICKER}] Initializing Risk Oracle on {DEVICE}...")
        # Path assumes the script is executed from the 'src' folder
        result = predict_volatility(ticker=TEST_TICKER, asset_class=TEST_ASSET_CLASS, models_dir="../models")
        
        print("\n--- T+1 INFERENCE REPORT ---")
        print(f"Asset: {result['ticker']} ({result['asset_class']})")
        print(f"Expected Volatility (t+1): {result['predicted_volatility_t1']:.6f}\n")
        print("Model Breakdown:")
        for model_name, pred_val in result['details'].items():
            print(f" - {model_name}: {pred_val:.6f}")
            
    except Exception as e:
        print(f"Inference failed: {e}")