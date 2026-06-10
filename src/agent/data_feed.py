import numpy as np
import pandas as pd

class WyckoffMockData:
    """
    Synthetic market data generator utilizing a Hidden Markov Model (HMM) 
    to simulate classic Wyckoff market phases (Accumulation, Markup, Distribution, Markdown).
    """
    def __init__(self, steps=1000, cycle_length=None):
        self.steps = steps
        self.prices = None
        self.sentiment = None
        self.volatility = None
        self.log_returns = None
        self.hidden_states = None
        self.generate_new_market()

    def generate_new_market(self):
        """
        Executes a simulation run to generate price paths, log returns, macro sentiment, 
        and an annualized volatility vector overlaid with Phase 2 predictive noise.
        """
        self.prices = np.zeros(self.steps)
        self.sentiment = np.zeros(self.steps)
        self.volatility = np.zeros(self.steps)
        self.log_returns = np.zeros(self.steps)  
        self.hidden_states = np.zeros(self.steps, dtype=int)
        
        current_price = np.random.uniform(50.0, 150.0)
        self.prices[0] = current_price
        
        state = 0
        
        def _get_phase_duration(current_state):
            if current_state == 0: return np.random.randint(40, 90)    # Accumulation
            elif current_state == 1: return np.random.randint(60, 120) # Markup (Bullish)
            elif current_state == 2: return np.random.randint(30, 60)  # Distribution
            elif current_state == 3: return np.random.randint(20, 50)  # Markdown (Bearish)
        
        time_in_state = 0
        duration = _get_phase_duration(state)
        
        # Step 1: Structural Price and Sentiment Path Generation
        for t in range(self.steps):
            self.hidden_states[t] = state
            
            if state == 0: 
                mu_price, sigma_price = 0.000, 0.005
                mu_sent, sigma_sent = -0.2, 0.2
            elif state == 1: 
                mu_price, sigma_price = 0.003, 0.010
                mu_sent, sigma_sent = 0.7, 0.2
            elif state == 2: 
                mu_price, sigma_price = 0.000, 0.025
                mu_sent, sigma_sent = 0.6, 0.3 
            elif state == 3: 
                mu_price, sigma_price = -0.006, 0.035
                mu_sent, sigma_sent = -0.8, 0.2
            
            if t > 0:
                shock = np.random.normal(0, 1)
                current_price = current_price * (1.0 + mu_price + sigma_price * shock)
                current_price = max(current_price, 1.0)
                self.prices[t] = current_price
            
            raw_sent = np.random.normal(mu_sent, sigma_sent)
            self.sentiment[t] = np.clip(raw_sent, -1.0, 1.0)
            
            time_in_state += 1
            if time_in_state >= duration:
                state = (state + 1) % 4  
                time_in_state = 0
                duration = _get_phase_duration(state)

        # Step 2: Continuous Log Returns Calculation
        self.log_returns[1:] = np.log(self.prices[1:] / self.prices[:-1])
        
        # Step 3: Rolling 20-Day Annualized Volatility with Embedded Predictive Error
        window = 20
        rmse_oracle = 0.0212  
        
        for t in range(self.steps):
            if t < window:
                sub_returns = self.log_returns[:t+1]
                if len(sub_returns) > 1:
                    vol_real = np.std(sub_returns, ddof=1) * np.sqrt(252)
                else:
                    vol_real = 0.1 
            else:
                sub_returns = self.log_returns[t-window+1:t+1]
                vol_real = np.std(sub_returns, ddof=1) * np.sqrt(252)
            
            oracle_noise = np.random.normal(0, rmse_oracle)
            self.volatility[t] = np.clip(vol_real + oracle_noise, 0.0, 2.0)

    def get_step_data(self, step):
        """
        Fetches variables corresponding to the current simulated timeline increment.
        Returns: Tuple (price, log_return, sentiment, volatility)
        """
        return self.prices[step], self.log_returns[step], self.sentiment[step], self.volatility[step]


class EmpiricalDataFeed:
    """
    Empirical multi-asset historical data parser designed to stream localized observations 
    while natively utilizing Domain Randomization features for robust agent exploration.
    """
    def __init__(self, csv_path="../data/dataset_train.csv", steps=1000, ticker=None, randomize=True):
        self.csv_path = csv_path
        self.steps = steps
        self.fixed_ticker = ticker
        self.randomize = randomize
        self.current_ticker = None
        self.current_market_steps = 0
        
        self.prices = None
        self.log_returns = None
        self.sentiment = None
        self.volatility = None
        
        # Load unified dataset profile
        self.full_df = pd.read_csv(csv_path)
        self.full_df['Date'] = pd.to_datetime(self.full_df['Date'])
        self.all_tickers = self.full_df['ticker'].unique()
        
        self.generate_new_market()

    def generate_new_market(self):
        """
        Dynamically extracts an asset class profile and maps a rolling timeline window slice 
        to array attributes upon environment reset events.
        """
        # 1. Asset Isolation Segment (Domain Randomization)
        if self.randomize and self.fixed_ticker is None:
            self.current_ticker = np.random.choice(self.all_tickers)
        else:
            self.current_ticker = self.fixed_ticker if self.fixed_ticker else self.all_tickers[0]
            
        ticker_df = self.full_df[self.full_df['ticker'] == self.current_ticker].sort_values('Date').reset_index(drop=True)
        effective_steps = min(self.steps, len(ticker_df))
        
        # 2. Window Offset Slice Identification
        max_start_idx = len(ticker_df) - effective_steps
        
        if self.randomize and max_start_idx > 0:
            start_idx = np.random.randint(0, max_start_idx)
            slice_df = ticker_df.iloc[start_idx : start_idx + effective_steps].reset_index(drop=True)
        else:
            slice_df = ticker_df.iloc[:effective_steps].reset_index(drop=True)
            
        # 3. Micro-Optimization: Explicit Conversion to Highly Linear NumPy Storage
        self.prices = slice_df['Close'].values
        self.log_returns = slice_df['log_return'].values
        self.sentiment = slice_df['sentiment_zscore'].values         
        self.volatility = slice_df['predicted_volatility_t1'].values  
        
        self.current_market_steps = len(slice_df)

    def get_step_data(self, step):
        """
        Fetches variables corresponding to the requested slice matrix location index.
        Returns: Tuple (price, log_return, sentiment, volatility)
        """
        return self.prices[step], self.log_returns[step], self.sentiment[step], self.volatility[step]