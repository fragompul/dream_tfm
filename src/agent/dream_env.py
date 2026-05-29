import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DreamEnv(gym.Env):
    """
    Entorno D.R.E.A.M. Fase 3.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_generator, initial_balance=10000.0):
        super(DreamEnv, self).__init__()
        
        self.data_gen = data_generator
        self.initial_balance = initial_balance
        self.max_steps = self.data_gen.steps - 1
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Espacio de Estados: [Sentimiento, Volatilidad, Saldo, Posición, Valor de Cartera]
        # Se normalizan los límites para estabilizar la convergencia de PPO
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -np.inf, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # FIX: Generar un mercado completamente nuevo en cada episodio
        self.data_gen.generate_new_market()
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.current_mdd = 0.0 
        
        return self._get_observation(), {}

    def _get_observation(self):
        _, sentiment, volatility = self.data_gen.get_step_data(self.current_step)
        
        # FIX H1: Normalización estricta de las variables financieras
        # Dividimos por el balance inicial para mantener los valores cercanos a 1.0
        norm_balance = self.balance / self.initial_balance
        norm_portfolio = self.portfolio_value / self.initial_balance
        
        # El tamaño de la posición también debe escalarse (Position Value / Initial Balance)
        current_price = self.data_gen.prices[self.current_step]
        norm_position = (self.position_size * current_price) / self.initial_balance
        
        return np.array([
            sentiment, 
            volatility, 
            norm_balance, 
            norm_position, 
            norm_portfolio
        ], dtype=np.float32)

    def step(self, action):
        current_price, sentiment, volatility = self.data_gen.get_step_data(self.current_step)
        
        # 1. Mapeo de la Acción al Peso Objetivo (Target Weight)
        target_weight = (action[0] + 1.0) / 2.0 
        
        # Filtros matemáticos de protección (Zona Muerta)
        # Forzamos los extremos para asegurar la liquidez o exposición total
        if target_weight < 0.05:
            target_weight = 0.0
        elif target_weight > 0.95:
            target_weight = 1.0
            
        # 2. Cálculo del Estado Actual
        prev_portfolio_value = self.balance + (self.position_size * current_price)
        current_weight = (self.position_size * current_price) / prev_portfolio_value if prev_portfolio_value > 0 else 0.0
        
        # 3. Cálculo del Diferencial (Delta Weight)
        delta_weight = target_weight - current_weight
        commission_rate = 0.001
        
        # 4. Lógica de Ejecución por Diferenciales
        # Umbral de tolerancia: Ignoramos reajustes menores al 2% para ahorrar comisiones
        if abs(delta_weight) > 0.02:
            if delta_weight > 0:  
                # Comprar
                invest_amount = delta_weight * prev_portfolio_value
                actual_invest = min(invest_amount, self.balance) 
                
                if actual_invest > 0:
                    self.balance -= actual_invest
                    self.position_size += (actual_invest * (1.0 - commission_rate)) / current_price
                    
            elif delta_weight < 0: 
                # Vender
                sell_value = abs(delta_weight) * prev_portfolio_value
                assets_to_sell = sell_value / current_price
                actual_sell_assets = min(assets_to_sell, self.position_size)
                
                if actual_sell_assets > 0:
                    self.position_size -= actual_sell_assets
                    gross_revenue = actual_sell_assets * current_price
                    self.balance += gross_revenue * (1.0 - commission_rate)
                    
        # 5. Actualización de Cartera y Métricas de Riesgo
        self.portfolio_value = self.balance + (self.position_size * current_price)
        
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
            
        prev_mdd = self.current_mdd
        self.current_mdd = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        mdd_penalty = max(0.0, self.current_mdd - prev_mdd)
        
        step_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0.0
        exposure_ratio = (self.position_size * current_price) / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        # 6. Función de Recompensa Continua (Re-calibrada)
        reward = step_return * 100.0 
        
        # Reducimos ligeramente el pánico al riesgo para que la red se atreva a explorar
        risk_penalty = 2.0 * volatility * exposure_ratio * mdd_penalty
        reward -= risk_penalty
        
        # FIX: Multiplicador x10 en la "miga de pan"
        # Ahora, si el sentimiento es 0.8 y la exposición es 0.5, el bono es +0.4.
        # Esto vence holgadamente el -0.1 de la comisión inicial.
        alignment_bonus = sentiment * exposure_ratio * 1.0
        reward += alignment_bonus
        
        # NUEVO: Penalización Cuadrática por Rotación (Quadratic Turnover Penalty)
        # Un salto de 1.0 resta -2.0. Un salto de 0.1 resta solo -0.02.
        turnover_penalty = 2.0 * (delta_weight ** 2)
        reward -= turnover_penalty
        
        # Refuerzo positivo: Si el agente está invertido y GANANDO dinero real, lo premiamos extra.
        # Esto le enseña que mantener posiciones ganadoras es la verdadera meta.
        if step_return > 0 and exposure_ratio > 0.1:
            reward += 0.5
            
        # Bono de liquidez reajustado (Solo en pánico extremo)
        if volatility > 0.8 and exposure_ratio < 0.05:
            reward += 0.1

        # 7. Gestión del Ciclo de Vida del Entorno
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "portfolio_value": self.portfolio_value,
            "mdd": self.current_mdd
        }
        
        return self._get_observation(), reward, terminated, truncated, info