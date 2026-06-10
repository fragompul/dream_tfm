import numpy as np
import pandas as pd

class WyckoffMockData:
    """
    Synthetic multi-asset market data generator driven by a global systemic core.
    
    Utilizes a single Hidden Markov Model (HMM) to simulate overarching macroeconomic 
    phases (Accumulation, Markup, Distribution, Markdown). While all assets share the 
    same global phase and macro sentiment profile, each asset path maintains independent 
    stochastic tracking noise.
    """
    def __init__(self, steps=252, num_assets=1):
        self.steps = steps
        self.num_assets = num_assets
        
        self.prices = None
        self.log_returns = None
        self.sentiment = None
        self.volatility = None
        self.hidden_states = None
        
        self.generate_new_market()

    def generate_new_market(self):
        """
        Simulates a synchronized macroeconomic timeline across N distinct asset channels.
        """
        self.prices = np.zeros((self.steps, self.num_assets))
        self.log_returns = np.zeros((self.steps, self.num_assets))
        self.volatility = np.zeros((self.steps, self.num_assets))
        self.sentiment = np.zeros(self.steps)  # Systemic Macro Sentiment (Shared)
        self.hidden_states = np.zeros(self.steps, dtype=int)
        
        # Initialize independent random starting prices for each asset
        current_prices = np.random.uniform(50.0, 150.0, size=self.num_assets)
        self.prices[0, :] = current_prices
        
        state = 0
        time_in_state = 0
        
        def _get_phase_duration(current_state):
            if current_state == 0: return np.random.randint(40, 90)    # Accumulation
            elif current_state == 1: return np.random.randint(60, 120) # Markup
            elif current_state == 2: return np.random.randint(30, 60)  # Distribution
            elif current_state == 3: return np.random.randint(20, 50)  # Markdown
            
        duration = _get_phase_duration(state)
        
        # Step 1: Generate Coordinated Macro Timeline and Asset Prices
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
            
            # Update individual asset pricing under global structural parameters
            if t > 0:
                for i in range(self.num_assets):
                    asset_shock = np.random.normal(0, 1)
                    # Introduce slight structural dispersion between assets
                    asset_sigma = sigma_price * (1.0 + 0.15 * (i - (self.num_assets - 1) / 2)) if self.num_assets > 1 else sigma_price
                    asset_sigma = max(0.002, asset_sigma)
                    
                    current_prices[i] *= (1.0 + mu_price + asset_sigma * asset_shock)
                    current_prices[i] = max(current_prices[i], 1.0)
                    self.prices[t, i] = current_prices[i]
            
            # Populate single systemic macro sentiment stream
            raw_sent = np.random.normal(mu_sent, sigma_sent)
            self.sentiment[t] = np.clip(raw_sent, -1.0, 1.0)
            
            time_in_state += 1
            if time_in_state >= duration:
                state = (state + 1) % 4  
                time_in_state = 0
                duration = _get_phase_duration(state)

        # Step 2: Compute Log Returns per Asset Column
        self.log_returns[1:, :] = np.log(self.prices[1:, :] / self.prices[:-1, :])
        
        # Step 3: Compute Rolling Volatility Channels with Predictive Oracle Noise
        window = 20
        rmse_oracle = 0.0212  
        
        for i in range(self.num_assets):
            for t in range(self.steps):
                if t < window:
                    sub_returns = self.log_returns[:t+1, i]
                    vol_real = np.std(sub_returns, ddof=1) * np.sqrt(252) if len(sub_returns) > 1 else 0.1
                else:
                    sub_returns = self.log_returns[t-window+1:t+1, i]
                    vol_real = np.std(sub_returns, ddof=1) * np.sqrt(252)
                
                oracle_noise = np.random.normal(0, rmse_oracle)
                self.volatility[t, i] = np.clip(vol_real + oracle_noise, 0.0, 2.0)

    def get_step_data(self, step):
        """
        Fetches synchronous metrics corresponding to the requested simulation increment.
        
        Returns:
            Tuple containing:
            - price (np.array): Shape (num_assets,)
            - log_return (np.array): Shape (num_assets,)
            - sentiment (float): Systemic global macro sentiment scalar
            - volatility (np.array): Shape (num_assets,)
        """
        return self.prices[step, :], self.log_returns[step, :], float(self.sentiment[step]), self.volatility[step, :]


class EmpiricalDataFeed:
    """
    Empirical multi-asset historical data pipeline designed to parse and align 
    cross-sectional asset profiles onto a synchronized time matrix.
    
    Supports both single-asset Domain Randomization and multi-asset concurrent portfolio 
    allocation strategies. Slices are dynamically bounded by mutual date intersection.
    """
    def __init__(self, csv_path="../data/train_agent_dataset.csv", steps=252, tickers=None, num_assets=1, randomize=True):
        self.csv_path = csv_path
        self.steps = steps
        self.num_assets = num_assets
        self.randomize = randomize
        
        # Normalize explicit or dynamic list configuration
        if tickers is not None:
            self.selected_tickers = tickers if isinstance(tickers, list) else [tickers]
            self.num_assets = len(self.selected_tickers)
        else:
            self.selected_tickers = None
            
        self.current_tickers = None
        self.current_market_steps = 0
        
        self.prices = None
        self.log_returns = None
        self.sentiment = None
        self.volatility = None
        
        # Load unified storage layer
        self.full_df = pd.read_csv(csv_path)
        self.full_df['Date'] = pd.to_datetime(self.full_df['Date'])
        self.all_tickers = self.full_df['ticker'].unique()
        
        self.generate_new_market()

    def generate_new_market(self):
        """
        Extracts a temporally aligned cross-sectional data matrix for the current episode.
        """
        # 1. Coordinate Portfolio Ticker Profiles
        if self.selected_tickers is not None:
            self.current_tickers = self.selected_tickers
        else:
            if self.randomize:
                # Stochastic Cluster Allocation (Domain Randomization)
                self.current_tickers = list(np.random.choice(self.all_tickers, size=self.num_assets, replace=False))
            else:
                self.current_tickers = list(self.all_tickers[:self.num_assets])
                
        # 2. Enforce Strict Chronological Synchronization
        ticker_data = self.full_df[self.full_df['ticker'].isin(self.current_tickers)]
        date_counts = ticker_data['Date'].value_counts()
        synchronized_dates = date_counts[date_counts == self.num_assets].index.sort_values()
        
        effective_steps = min(self.steps, len(synchronized_dates))
        max_start_idx = len(synchronized_dates) - effective_steps
        
        # Identify window offset boundaries
        if self.randomize and max_start_idx > 0:
            start_idx = np.random.randint(0, max_start_idx)
            chosen_dates = synchronized_dates[start_idx : start_idx + effective_steps]
        else:
            chosen_dates = synchronized_dates[:effective_steps]
            
        # 3. Populate Native Multi-Dimensional Array Buffers
        self.prices = np.zeros((effective_steps, self.num_assets))
        self.log_returns = np.zeros((effective_steps, self.num_assets))
        self.volatility = np.zeros((effective_steps, self.num_assets))
        self.sentiment = np.zeros(effective_steps)  # Systemic Macro Vector
        
        # Map localized DataFrame subsets to continuous NumPy blocks
        for i, ticker in enumerate(self.current_tickers):
            asset_df = ticker_data[(ticker_data['ticker'] == ticker) & (ticker_data['Date'].isin(chosen_dates))].sort_values('Date')
            
            self.prices[:, i] = asset_df['Close'].values
            self.log_returns[:, i] = asset_df['log_return'].values
            self.volatility[:, i] = asset_df['predicted_volatility_t1'].values
            
            # Track sentiment globally via the shared temporal matrix anchor
            if i == 0:
                self.sentiment = asset_df['sentiment_zscore'].values
                
        self.current_market_steps = effective_steps

    def get_step_data(self, step):
        """
        Fetches synchronous historical metrics corresponding to the requested matrix index.
        
        Returns:
            Tuple containing:
            - price (np.array): Shape (num_assets,)
            - log_return (np.array): Shape (num_assets,)
            - sentiment (float): Systemic global macro sentiment scalar
            - volatility (np.array): Shape (num_assets,)
        """
        return self.prices[step, :], self.log_returns[step, :], float(self.sentiment[step]), self.volatility[step, :]