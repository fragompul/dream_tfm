import numpy as np

class WyckoffMockData:
    def __init__(self, steps=1000, cycle_length=250):
        self.steps = steps
        self.cycle_length = cycle_length
        self.generate_new_market() # Genera el primer mercado al instanciar

    def generate_new_market(self):
        """Regenera toda la serie temporal estocástica para evitar overfitting."""
        t = np.arange(self.steps)
        
        # Inyectamos una variación aleatoria en la longitud del ciclo para que no sea siempre igual
        dynamic_cycle = self.cycle_length * np.random.uniform(0.8, 1.2)
        base_cycle = np.sin(2 * np.pi * t / dynamic_cycle)
        
        market_noise = np.random.standard_t(df=3, size=self.steps) * 0.8
        
        # Variamos el precio inicial aleatoriamente entre 50 y 150
        start_price = np.random.uniform(50.0, 150.0)
        self.prices = start_price + (base_cycle * 20) + np.cumsum(market_noise)
        self.prices = np.maximum(self.prices, 1.0) 
        
        returns = np.diff(self.prices, prepend=self.prices[0]) / self.prices[0]
        garch_effect = np.abs(returns) * 12
        synthetic_volatility = garch_effect + np.random.lognormal(-2.5, 0.4, self.steps)
        self.volatility = np.clip(synthetic_volatility, 0.0, 1.0)
        
        momentum = np.gradient(self.prices)
        raw_sentiment = np.tanh(momentum / 3.0)
        nlp_noise = np.random.normal(0, 0.5, self.steps) 
        self.sentiment = np.clip(raw_sentiment + nlp_noise, -1.0, 1.0)

    def get_step_data(self, step):
        return self.prices[step], self.sentiment[step], self.volatility[step]