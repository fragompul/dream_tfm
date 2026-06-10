import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DreamEnv(gym.Env):
    """
    D.R.E.A.M. Portfolio Management Environment (Phase 3).
    
    A unified, highly vectorized trading simulation framework designed to support
    both single-asset trading (via Domain Randomization) and multi-asset dynamic 
    portfolio allocation models. Operates under a strict non-leverage cash constraint.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_generator, initial_balance=10000.0):
        super(DreamEnv, self).__init__()
        
        self.data_gen = data_generator
        self.initial_balance = initial_balance
        self.num_assets = getattr(self.data_gen, "num_assets", 1)
        
        # Determine continuous maximum timeline steps dynamically
        self.max_steps = self.data_gen.steps - 1
        
        # Action Space: Target weight bounds mapping directly to portfolio allocation vectors
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        
        # Observation Space Setup (3 Global Features + 3 Asset Features per Channel)
        # Flattened Array Profile: [norm_balance, norm_portfolio, macro_sentiment] + N * [log_return, volatility, norm_position]
        low_global = [0.0, 0.0, -3.0]
        high_global = [np.inf, np.inf, 3.0]
        
        low_assets = [-np.inf, 0.0, 0.0] * self.num_assets
        high_assets = [np.inf, 2.0, np.inf] * self.num_assets
        
        self.observation_space = spaces.Box(
            low=np.array(low_global + low_assets, dtype=np.float32),
            high=np.array(high_global + high_assets, dtype=np.float32),
            dtype=np.float32
        )
        
        # State tracking matrices initialization
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_sizes = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.current_mdd = 0.0
        
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the accounting structures and requests a fresh matrix stream from the data feed.
        """
        super().reset(seed=seed)
        
        # Generate new market conditions (Stochastic ticker group and time window)
        self.data_gen.generate_new_market()
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_sizes = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.current_mdd = 0.0 
        
        # Dynamic synchronization with actual extracted timeline slice lengths
        if hasattr(self.data_gen, "current_market_steps"):
            self.max_steps = self.data_gen.current_market_steps - 1
        else:
            self.max_steps = self.data_gen.steps - 1
            
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Compiles and flattens global accounting parameters and individual asset profiles 
        into a stationary and scale-invariant observation array.
        """
        prices, log_returns, sentiment, volatility = self.data_gen.get_step_data(self.current_step)
        
        norm_balance = self.balance / self.initial_balance
        norm_portfolio = self.portfolio_value / self.initial_balance
        
        # Build systemic observation core
        obs = [norm_balance, norm_portfolio, float(sentiment)]
        
        # Append cross-sectional asset profiles
        for i in range(self.num_assets):
            norm_position = (self.position_sizes[i] * prices[i]) / self.initial_balance
            obs.extend([log_returns[i], volatility[i], norm_position])
            
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Executes trading actions under an enforced Sells-First sequence to free capital, 
        updates portfolio valuations, and calculates the multi-objective reward function.
        """
        prices, log_returns, sentiment, volatility = self.data_gen.get_step_data(self.current_step)
        
        # 1. Action Array Mapping to Raw Target Weights
        raw_target_weights = (action + 1.0) / 2.0
        
        # Apply Operational Dead-Zones per channel
        for i in range(self.num_assets):
            if raw_target_weights[i] < 0.05:
                raw_target_weights[i] = 0.0
            elif raw_target_weights[i] > 0.95:
                raw_target_weights[i] = 1.0
                
        # Enforce strict non-leveraged sum constraints
        weight_sum = np.sum(raw_target_weights)
        if weight_sum > 1.0:
            target_weights = raw_target_weights / weight_sum
        else:
            target_weights = raw_target_weights
            
        # 2. Extract Portfolio Matrix Snapshots Before Reallocation
        prev_portfolio_value = self.balance + np.sum(self.position_sizes * prices)
        current_weights = (self.position_sizes * prices) / prev_portfolio_value if prev_portfolio_value > 0 else np.zeros(self.num_assets)
        
        delta_weights = target_weights - current_weights
        commission_rate = 0.001
        
        # 3. CRITICAL TRANSACTION PIPELINE: SELLS FIRST (To release liquid capital)
        for i in range(self.num_assets):
            if delta_weights[i] < -0.02:
                sell_value = abs(delta_weights[i]) * prev_portfolio_value
                assets_to_sell = sell_value / prices[i]
                actual_sell_assets = min(assets_to_sell, self.position_sizes[i])
                
                if actual_sell_assets > 0:
                    self.position_sizes[i] -= actual_sell_assets
                    gross_revenue = actual_sell_assets * prices[i]
                    self.balance += gross_revenue * (1.0 - commission_rate)
                    
        # 4. CRITICAL TRANSACTION PIPELINE: BUYS SECOND (Allocating available cash buffer)
        for i in range(self.num_assets):
            if delta_weights[i] > 0.02:
                invest_amount = delta_weights[i] * prev_portfolio_value
                actual_invest = min(invest_amount, self.balance)
                
                if actual_invest > 0:
                    self.balance -= actual_invest
                    self.position_sizes[i] += (actual_invest * (1.0 - commission_rate)) / prices[i]
                    
        # 5. Continuous Portfolio Mark-to-Market Evaluation
        self.portfolio_value = self.balance + np.sum(self.position_sizes * prices)
        
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
            
        prev_mdd = self.current_mdd
        self.current_mdd = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        mdd_penalty = max(0.0, self.current_mdd - prev_mdd)
        
        step_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0.0
        exposure_ratios = (self.position_sizes * prices) / self.portfolio_value if self.portfolio_value > 0 else np.zeros(self.num_assets)
        total_exposure_ratio = np.sum(exposure_ratios)
        
        # 6. Multi-Objective Reward Vectorization
        reward = step_return * 100.0 
        
        # Systemic risk penalty derived across all exposure fields
        risk_penalty = 2.0 * np.sum(volatility * exposure_ratios) * mdd_penalty
        reward -= risk_penalty
        
        # Macro sentiment structural synchronization bonus
        alignment_bonus = sentiment * total_exposure_ratio * 0.5
        reward += alignment_bonus
        
        # Quadratic Turnover Penalty aggregated across the full asset array
        turnover_penalty = 2.0 * np.sum(delta_weights ** 2)
        reward -= turnover_penalty
        
        # Structural positive reinforcement for maintaining profitable market exposure
        if step_return > 0 and total_exposure_ratio > 0.1:
            reward += 0.5
            
        # Global market-wide panic cash cushion preservation bonus
        mean_volatility = np.mean(volatility)
        if mean_volatility > 0.8 and total_exposure_ratio < 0.05:
            reward += 0.1

        # 7. Environment Lifecycle Increment Step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "portfolio_value": self.portfolio_value,
            "mdd": self.current_mdd,
            "tickers": self.data_gen.current_tickers if hasattr(self.data_gen, "current_tickers") else "SYNTHETIC"
        }
        
        return self._get_observation(), reward, terminated, truncated, info