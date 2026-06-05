import os
import matplotlib
matplotlib.use('Agg')  # Inferencia headless para entornos como Kaggle/Servidores

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from data_feed import WyckoffMockData
from dream_env import DreamEnv

def calculate_academic_metrics(portfolio_values, prices):
    """
    Calcula las métricas cuantitativas bajo estrictos estándares académicos.
    """
    portfolio_values = np.array(portfolio_values)
    prices = np.array(prices)
    
    # 1. Rendimientos diarios
    port_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    bh_returns = np.diff(prices) / prices[:-1]
    
    # 2. Ratio de Sharpe Anualizado
    mean_return = np.mean(port_returns)
    std_return = np.std(port_returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
    
    # 3. Ratio de Sortino Académico (Semi-desviación inferior respecto a MAR=0)
    downside_returns = port_returns[port_returns < 0]
    if len(port_returns) > 0:
        # Dividimos por la longitud total del PnL para penalizar la frecuencia del riesgo
        downside_variance = np.sum(downside_returns ** 2) / len(port_returns)
        down_std = np.sqrt(downside_variance)
    else:
        down_std = 1e-6
    down_std = 1e-6 if down_std == 0 else down_std
    sortino_ratio = (mean_return / down_std) * np.sqrt(252)
    
    # 4. Máximo Drawdown (MDD)
    cum_returns = portfolio_values / portfolio_values[0]
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    mdd = abs(np.min(drawdowns)) * 100.0
    
    # 5. Win Rate vs Buy & Hold (Frecuencia de batir al mercado)
    outperforming_periods = np.sum(port_returns > bh_returns)
    win_rate_bh = (outperforming_periods / len(port_returns)) * 100.0 if len(port_returns) > 0 else 0.0
    
    # Rendimientos totales porcentuales
    total_return_agent = ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100
    total_return_bh = ((prices[-1] / prices[0]) - 1) * 100
    
    return {
        "Sharpe": sharpe_ratio,
        "Sortino": sortino_ratio,
        "MDD": mdd,
        "WinRate_vs_BH": win_rate_bh,
        "Total_Ret_Agent": total_return_agent,
        "Total_Ret_BH": total_return_bh
    }

def run_monte_carlo_evaluation(model_path, num_runs=30, steps=500):
    """
    Ejecuta una simulación de Monte Carlo evaluando al agente sobre N curvas independientes.
    """
    # Fijamos semilla base para reproducibilidad estadística de la muestra
    np.random.seed(42)
    
    print(f"Cargando agente PPO continuo desde {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    # Instanciamos datos y entorno base (el reset cambiará el mercado en cada iteración)
    test_data = WyckoffMockData(steps=steps)
    eval_env = DreamEnv(test_data)
    
    # Matrices para guardar los caminos temporales normalizados (empezando en 1.0)
    agent_paths = []
    bh_paths = []
    
    # Lista para almacenar los KPIs finales de cada ejecución
    all_metrics = []
    
    print(f"Iniciando simulación de Monte Carlo ({num_runs} ejecuciones independientes)...")
    for run in range(num_runs):
        obs, _ = eval_env.reset()
        
        current_agent_path = [1.0]
        current_bh_path = [1.0]
        
        # Guardamos precios iniciales para el cálculo del benchmark
        initial_price = test_data.prices[0]
        commission_rate = 0.001
        
        prices_run = [initial_price]
        portfolio_values_run = [eval_env.initial_balance]
        
        for _ in range(steps - 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            p_t = test_data.prices[eval_env.current_step]
            prices_run.append(p_t)
            portfolio_values_run.append(info['portfolio_value'])
            
            # Trayectorias relativas normalizadas
            current_agent_path.append(info['portfolio_value'] / eval_env.initial_balance)
            # El B&H asume la misma penalización transaccional en el paso 0 para ser un rival justo
            current_bh_path.append((1.0 - commission_rate) * (p_t / initial_price))
            
            if terminated or truncated:
                break
                
        # Calcular KPIs de esta curva concreta
        run_kpis = calculate_academic_metrics(portfolio_values_run, prices_run)
        all_metrics.append(run_kpis)
        
        agent_paths.append(current_agent_path)
        bh_paths.append(current_bh_path)
        
    return np.array(agent_paths), np.array(bh_paths), pd.DataFrame(all_metrics)

def generate_advanced_dashboard(agent_paths, bh_paths, df_metrics, output_img):
    """
    Genera un panel visual de nivel académico con intervalos de varianza y distribuciones.
    """
    print("Renderizando dashboard estadístico de Monte Carlo...")
    sns.set_theme(style="darkgrid")
    
    # Creamos un layout de 2x2 para mostrar análisis de caminos y distribuciones de riesgo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ejes temporales compartidos
    time_steps = range(agent_paths.shape[1])
    
    # --- 1. GRÁFICO SUPERIOR IZQUIERDO: Evolución Temporal con Bandas de Varianza ---
    mean_agent = np.mean(agent_paths, axis=0)
    std_agent = np.std(agent_paths, axis=0)
    mean_bh = np.mean(bh_paths, axis=0)
    std_bh = np.std(bh_paths, axis=0)
    
    axes[0, 0].plot(time_steps, mean_agent, color='green', linewidth=2.5, label='Media Cartera D.R.E.A.M.')
    axes[0, 0].fill_between(time_steps, mean_agent - std_agent, mean_agent + std_agent, color='green', alpha=0.15, label='±1 Desv. Est. (Agente)')
    
    axes[0, 0].plot(time_steps, mean_bh, color='darkorange', linestyle='--', linewidth=2, label='Media Buy & Hold')
    axes[0, 0].fill_between(time_steps, mean_bh - std_bh, mean_bh + std_bh, color='darkorange', alpha=0.1, label='±1 Desv. Est. (B&H)')
    
    axes[0, 0].set_title('Análisis de Caminos de Inversión (Normalizado)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Multiplicador de Capital (Base 1.0)', fontweight='bold')
    axes[0, 0].legend(loc='upper left')
    
    # --- 2. GRÁFICO SUPERIOR DERECHO: Distribución del Ratio de Sharpe ---
    sns.histplot(df_metrics['Sharpe'], ax=axes[0, 1], kde=True, color='blue', bins=10)
    axes[0, 1].axvline(df_metrics['Sharpe'].mean(), color='blue', linestyle='--', linewidth=2, label=f"Media: {df_metrics['Sharpe'].mean():.2f}")
    axes[0, 1].axvline(1.00, color='red', linestyle=':', label="Objetivo TFM (>1.00)")
    axes[0, 1].set_title('Distribución del Ratio de Sharpe en Out-of-Sample', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Ratio de Sharpe Anualizado')
    axes[0, 1].legend()
    
    # --- 3. GRÁFICO INFERIOR IZQUIERDO: Distribución del Máximo Drawdown ---
    sns.boxplot(x=df_metrics['MDD'], ax=axes[1, 0], color='salmon')
    axes[1, 0].axvline(20.0, color='red', linestyle=':', label="Techo Objetivo TFM (<20%)")
    axes[1, 0].set_title('Dispersión del Máximo Drawdown (Preservación de Capital)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Máximo Drawdown (%)')
    axes[1, 0].legend()
    
    # --- 4. GRÁFICO INFERIOR DERECHO: Caja de Resumen Estadístico Completo ---
    axes[1, 1].axis('off') # Apagamos los ejes para usarlo de lienzo de texto
    text_summary = [
        r"$\bf{RESUMEN\ ESTADÍSTICO\ DE\ MONTE\ CARLO}$",
        f"Número de Mercados Evaluados: {len(df_metrics)}",
        "",
        f"Ratio de Sharpe Medio:  {df_metrics['Sharpe'].mean():.2f} ± {df_metrics['Sharpe'].std():.2f}  (Obj: >1.00)",
        f"Ratio de Sortino Medio: {df_metrics['Sortino'].mean():.2f} ± {df_metrics['Sortino'].std():.2f} (Obj: >1.20)",
        f"Máximo Drawdown Medio:  {df_metrics['MDD'].mean():.2f}% ± {df_metrics['MDD'].std():.2f}% (Obj: <20%)",
        f"Win Rate vs B&H Medio:  {df_metrics['WinRate_vs_BH'].mean():.2f}% ± {df_metrics['WinRate_vs_BH'].std():.2f}%",
        "",
        f"Retorno Medio Agente:   {df_metrics['Total_Ret_Agent'].mean():.2f}% ± {df_metrics['Total_Ret_Agent'].std():.2f}%",
        f"Retorno Medio B&H:      {df_metrics['Total_Ret_BH'].mean():.2f}% ± {df_metrics['Total_Ret_BH'].std():.2f}%"
    ]
    axes[1, 1].text(0.05, 0.95, "\n".join(text_summary), fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"Dashboard de Monte Carlo guardado con éxito en: {output_img}")

if __name__ == "__main__":
    # Configuración de rutas
    model_path = "models/dream_agent.zip"
    output_img = "test_modelos/backtest_monte_carlo_tfm.png"
    
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    
    # Ejecución del motor estadístico
    agent_paths, bh_paths, df_metrics = run_monte_carlo_evaluation(model_path, num_runs=100, steps=252*5)
    
    # Generación de gráficos finales
    generate_advanced_dashboard(agent_paths, bh_paths, df_metrics, output_img)