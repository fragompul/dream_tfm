from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agente.src.data_feed import WyckoffMockData
from agente.src.dream_env import DreamEnv

def train_dream_agent():
    """
    Función principal de instanciación y entrenamiento a gran escala.
    """
    # Incrementamos los datos a 300,000 steps y los ciclos de mercado a 2,000 steps 
    # para permitir tendencias sostenidas donde el agente pueda mantener posiciones rentables.
    mock_data = WyckoffMockData(steps=1000, cycle_length=250)
    env = DreamEnv(mock_data)
    
    # Verificación de integridad
    check_env(env, warn=True)
    
    print("Desplegando Arquitectura PPO en CPU (Maximizando estabilidad)...")
    # Utilizamos device="cpu" para evitar cuellos de botella en memoria 
    # al procesar redes MLP con secuencias temporales tan largas.
    model = PPO("MlpPolicy", env, verbose=1, 
                    learning_rate=0.0003, 
                    n_steps=2048, 
                    device="cpu",
                    tensorboard_log="./dream_logs/")
    
    print("Comenzando Entrenamiento Intensivo (300k timesteps)...")
    model.learn(total_timesteps=300000)
    
    print("Entrenamiento finalizado. Agente guardado en buffer.")
    model.save("./agente/modelos/cont_dream_ppo_v3")
    return model

if __name__ == "__main__":
    trained_agent = train_dream_agent()