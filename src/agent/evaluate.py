import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from stable_baselines3 import PPO

from src.agent.data_feed import WyckoffMockData
from src.agent.dream_env import DreamEnv

def calculate_financial_metrics(portfolio_values, prices):
    """
    Calcula las métricas cuantitativas exigidas en los objetivos del TFM.
    Asume 252 periodos por año para la anualización.
    """
    # 1. Rendimientos diarios (Returns)
    port_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    bh_returns = np.diff(prices) / prices[:-1]
    
    # 2. Ratio de Sharpe (Rendimiento ajustado al riesgo total)
    mean_return = np.mean(port_returns)
    std_return = np.std(port_returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
    
    # 3. Ratio de Sortino (Rendimiento ajustado a la volatilidad negativa)
    downside_returns = port_returns[port_returns < 0]
    down_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
    sortino_ratio = (mean_return / down_std) * np.sqrt(252)
    
    # 4. Máximo Drawdown (MDD)
    cum_returns = np.array(portfolio_values) / portfolio_values[0]
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    mdd = abs(np.min(drawdowns)) * 100.0  # En porcentaje
    
    # 5. Batir al Mercado (Win Rate vs Buy & Hold)
    # Contamos cuántos periodos el agente tuvo un retorno superior al Buy & Hold
    outperforming_periods = np.sum(port_returns > bh_returns)
    win_rate_bh = (outperforming_periods / len(port_returns)) * 100.0
    
    # Rendimiento Acumulado Total
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

def evaluate_and_save_plot(model_path, output_img="agente/resultados/backtest_eval_continuo.png"):
    """
    Módulo de inferencia adaptado para acciones continuas.
    Genera un dashboard dual con comparativa directa frente a Buy & Hold.
    """
    print(f"Cargando agente PPO continuo desde {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    print("Generando datos Out-of-Sample para Backtesting...")
    test_data = WyckoffMockData(steps=500, cycle_length=200)
    eval_env = DreamEnv(test_data)
    
    obs, _ = eval_env.reset()
    
    portfolio_values = [eval_env.initial_balance]
    prices = [test_data.prices[0]]
    target_weights = [0.0]  
    
    for _ in range(test_data.steps - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        w_t = (action[0] + 1.0) / 2.0
        
        prices.append(test_data.prices[eval_env.current_step])
        portfolio_values.append(info['portfolio_value'])
        target_weights.append(w_t)
        
        if terminated or truncated:
            break
            
    print("Calculando Métricas de Rendimiento Cuantitativo...")
    metrics = calculate_financial_metrics(portfolio_values, prices)
    
    # Simulación de la cartera Buy & Hold para el gráfico
    bh_portfolio_values = [eval_env.initial_balance * (p / prices[0]) for p in prices]

    print("\n" + "="*40)
    print("RESULTADOS DEL BACKTESTING (TFM KPIs)")
    print("="*40)
    print(f"Ratio de Sharpe:       {metrics['Sharpe']:.2f} (Objetivo: > 1.00)")
    print(f"Ratio de Sortino:      {metrics['Sortino']:.2f} (Objetivo: > 1.20)")
    print(f"Máximo Drawdown (MDD): {metrics['MDD']:.2f}% (Objetivo: < 20.0%)")
    print(f"Win Rate vs B&H:       {metrics['WinRate_vs_BH']:.2f}% (Objetivo: > 60.0%)")
    print(f"Retorno Total Agente:  {metrics['Total_Ret_Agent']:.2f}%")
    print(f"Retorno Total B&H:     {metrics['Total_Ret_BH']:.2f}%")
    print("="*40 + "\n")

    print("Renderizando dashboard de inferencia en memoria...")
    sns.set_theme(style="darkgrid")
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- GRÁFICO SUPERIOR: Cartera D.R.E.A.M. vs Cartera Buy & Hold ---
    ax1.plot(portfolio_values, color='green', label='Valor Cartera D.R.E.A.M.', linewidth=2)
    ax1.plot(bh_portfolio_values, color='darkorange', linestyle='--', label='Estrategia Buy & Hold', linewidth=1.5)
    ax1.set_ylabel('Valor de Cartera ($)', color='black', fontweight='bold')
    
    # Mantenemos el precio original en un eje secundario muy tenue por referencia
    ax2 = ax1.twinx()
    ax2.plot(prices, color='gray', alpha=0.3, label='Precio Bruto del Activo')
    ax2.set_ylabel('Precio del Activo ($)', color='gray', fontweight='bold')
    
    ax1.set_title('Backtesting de Inferencia: D.R.E.A.M. vs Buy & Hold', fontsize=14, fontweight='bold')
    
    # Unificar Leyendas
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Insertar CAJA DE MÉTRICAS en el gráfico
    textstr = '\n'.join((
        r'$\bf{Métricas\ TFM}$',
        f"Sharpe: {metrics['Sharpe']:.2f}",
        f"Sortino: {metrics['Sortino']:.2f}",
        f"Max Drawdown: {metrics['MDD']:.2f}%",
        f"Win Rate B&H: {metrics['WinRate_vs_BH']:.2f}%",
        f"Retorno Agente: {metrics['Total_Ret_Agent']:.2f}%",
        f"Retorno B&H: {metrics['Total_Ret_BH']:.2f}%"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.02, 0.55, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, color='black')
    
    # --- GRÁFICO INFERIOR: Exposición Dinámica de la Cartera ---
    ax3.fill_between(range(len(target_weights)), 0, target_weights, color='blue', alpha=0.3)
    ax3.plot(target_weights, color='blue', linewidth=1.5)
    ax3.set_ylabel('Exposición Objetivo', color='blue', fontweight='bold')
    ax3.set_xlabel('Pasos de Tiempo (Steps)', fontweight='bold')
    ax3.set_ylim(-0.05, 1.05) 
    
    fig.tight_layout()
    
    plt.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"Evaluación finalizada. Gráfica guardada en disco como: {output_img}")

if __name__ == "__main__":
    model_path = "agente/modelos/cont_dream_ppo_v2"
    output_img = "agente/resultados/backtest_dream_cont_v2_bis.png"
    evaluate_and_save_plot(model_path, output_img)