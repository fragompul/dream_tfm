import numpy as np
from stable_baselines3 import PPO

class DreamInferenceEngine:
    def __init__(self, model_path, initial_balance=10000.0, commission_rate=0.001):
        self.model = PPO.load(model_path, device="cpu")
        # El initial_balance solo sirve para normalizar el vector de estado (dividimos por él).
        # Ya no usamos variables internas para rastrear el dinero real.
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate

    def _build_observation(self, current_price, sentiment, volatility, current_balance, current_position):
        """Genera el vector de estado inyectando la verdad de la cuenta."""
        portfolio_value = current_balance + (current_position * current_price)
        
        norm_balance = current_balance / self.initial_balance
        norm_position = (current_position * current_price) / self.initial_balance
        norm_portfolio = portfolio_value / self.initial_balance
        
        return np.array([
            sentiment, 
            volatility, 
            norm_balance, 
            norm_position, 
            norm_portfolio
        ], dtype=np.float32), portfolio_value

    def process_market_tick(self, current_price: float, sentiment: float, volatility: float, 
                            current_balance: float, current_position: float):
        """
        El agente recibe toda la realidad desde fuera (Mercado + Cuenta Bróker)
        y devuelve la orden a ejecutar.
        """
        # 1. Construcción del Estado con los inputs externos
        obs, prev_portfolio_value = self._build_observation(
            current_price, sentiment, volatility, current_balance, current_position
        )
        
        # 2. Inferencia determinista
        action, _ = self.model.predict(obs, deterministic=True)
        
        # 3. Mapeo a peso objetivo y zona muerta
        target_weight = (action[0] + 1.0) / 2.0 
        if target_weight < 0.05: target_weight = 0.0
        elif target_weight > 0.95: target_weight = 1.0
            
        current_weight = (current_position * current_price) / prev_portfolio_value if prev_portfolio_value > 0 else 0.0
        delta_weight = target_weight - current_weight
        
        # 4. Lógica de Ejecución Teórica (Lo que el orquestador DEBERÍA hacer)
        order_type = "MANTENER"
        executed_amount = 0.0
        
        if abs(delta_weight) > 0.02:
            if delta_weight > 0:
                order_type = "COMPRAR"
                invest_amount = delta_weight * prev_portfolio_value
                executed_amount = min(invest_amount, current_balance)
            elif delta_weight < 0:
                order_type = "VENDER"
                sell_value = abs(delta_weight) * prev_portfolio_value
                assets_to_sell = sell_value / current_price
                actual_sell_assets = min(assets_to_sell, current_position)
                executed_amount = actual_sell_assets * current_price
                    
        return {
            "action_raw": action[0],
            "target_weight": target_weight,
            "order_type": order_type,
            "executed_amount_usd": executed_amount
        }