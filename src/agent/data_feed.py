import numpy as np

class WyckoffMockData:
    def __init__(self, steps=1000, cycle_length=None):
        self.steps = steps
        self.generate_new_market()

    def generate_new_market(self):
        """
        Genera una serie temporal basada en un Proceso de Markov Oculto,
        calculando la volatilidad real a 20 días + el RMSE del Oráculo.
        """
        self.prices = np.zeros(self.steps)
        self.sentiment = np.zeros(self.steps)
        self.volatility = np.zeros(self.steps)
        self.hidden_states = np.zeros(self.steps, dtype=int)
        
        # Condiciones iniciales aleatorias
        current_price = np.random.uniform(50.0, 150.0)
        self.prices[0] = current_price
        
        state = 0
        
        def get_phase_duration(current_state):
            if current_state == 0: return np.random.randint(40, 90)    # Acumulación
            elif current_state == 1: return np.random.randint(60, 120) # Alcista
            elif current_state == 2: return np.random.randint(30, 60)  # Distribución
            elif current_state == 3: return np.random.randint(20, 50)  # Bajista
        
        time_in_state = 0
        duration = get_phase_duration(state)
        
        # --- PASO 1: Generar Precios y Sentimiento (HMM) ---
        for t in range(self.steps):
            self.hidden_states[t] = state
            
            # Parametrización del régimen
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
                duration = get_phase_duration(state)

        # --- PASO 2: Calcular Retornos Logarítmicos ---
        log_returns = np.zeros(self.steps)
        # r_t = ln(P_t / P_{t-1})
        log_returns[1:] = np.log(self.prices[1:] / self.prices[:-1])
        
        # --- PASO 3: Calcular Volatilidad 20d Anualizada + Ruido del Oráculo ---
        window = 20
        rmse_oracle = 0.0212  # Error del modelo Simple_All según TFM
        
        for t in range(self.steps):
            if t < window:
                # Ventana expansiva para los primeros días
                sub_returns = log_returns[:t+1]
                if len(sub_returns) > 1:
                    vol_real = np.std(sub_returns, ddof=1) * np.sqrt(252)
                else:
                    vol_real = 0.1 # Volatilidad base inicial
            else:
                # Cálculo de la ventana móvil estricta de 20 periodos
                sub_returns = log_returns[t-window+1:t+1]
                vol_real = np.std(sub_returns, ddof=1) * np.sqrt(252)
            
            # Inyectamos el error predictivo estocástico del modelo Fase 2
            oracle_noise = np.random.normal(0, rmse_oracle)
            
            # Recortamos a 0.0 por abajo para evitar volatilidades negativas
            # El límite superior se deja holgado (2.0) por si hay shocks extremos
            self.volatility[t] = np.clip(vol_real + oracle_noise, 0.0, 2.0)

    def get_step_data(self, step):
        return self.prices[step], self.sentiment[step], self.volatility[step]