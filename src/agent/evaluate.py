import os
import matplotlib
matplotlib.use('Agg')  # Inferencia headless para entornos como Kaggle/Servidores

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from scipy.stats import pearsonr

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

class DreamEvaluator:
    def __init__(self, model_path, num_runs=30, steps=600, commission_rate=0.001):
        self.model_path = model_path
        self.num_runs = num_runs
        self.steps = steps
        self.commission_rate = commission_rate
        
        print(f"Cargando agente PPO continuo desde {self.model_path}...")
        self.model = PPO.load(self.model_path, device="cpu")

    def _calculate_financials(self, portfolio_values, prices):
        """Dimensión 1: Métricas Financieras (Alpha y Riesgo)"""
        port_values = np.array(portfolio_values)
        price_values = np.array(prices)
        
        port_returns = np.diff(port_values) / port_values[:-1]
        bh_returns = np.diff(price_values) / price_values[:-1]
        
        # Sharpe
        mean_return = np.mean(port_returns)
        std_return = np.std(port_returns)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
        
        # Sortino Académico (MAR = 0)
        downside_returns = port_returns[port_returns < 0]
        if len(port_returns) > 0:
            downside_variance = np.sum(downside_returns ** 2) / len(port_returns)
            down_std = np.sqrt(downside_variance)
        else:
            down_std = 1e-6
        down_std = 1e-6 if down_std == 0 else down_std
        sortino = (mean_return / down_std) * np.sqrt(252)
        
        # MDD
        cum_returns = port_values / port_values[0]
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        mdd = abs(np.min(drawdowns)) * 100.0
        
        # Calmar Ratio
        annualized_return = (mean_return * 252) * 100 
        calmar = annualized_return / mdd if mdd > 0 else 0.0
        
        # Win Rate vs B&H
        outperforming_periods = np.sum(port_returns > bh_returns)
        win_rate = (outperforming_periods / len(port_returns)) * 100.0 if len(port_returns) > 0 else 0.0
        
        total_ret_agent = ((port_values[-1] / port_values[0]) - 1) * 100
        total_ret_bh = ((price_values[-1] / price_values[0]) - 1) * 100
        
        return {
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MDD": mdd,
            "Calmar": calmar,
            "WinRate": win_rate,
            "Total_Ret_Agent": total_ret_agent,
            "Total_Ret_BH": total_ret_bh
        }

    def _calculate_behavior(self, exposures, sentiments, volatilities):
        """Dimensión 2: Comportamiento (Psicología de la Política)"""
        exposures_arr = np.array(exposures)
        sentiments_arr = np.array(sentiments)
        vols_arr = np.array(volatilities)
        
        # 1. Exposición Media al Mercado
        avg_exposure = np.mean(exposures_arr) * 100.0
        
        # 2. Tasa de Rotación (Turnover): Suma de cambios absolutos en exposición
        turnover = np.sum(np.abs(np.diff(exposures_arr))) / len(exposures_arr)
        
        # 3. Correlaciones (¿Hace caso a los sensores?)
        # Añadimos un pequeño ruido para evitar errores si la exposición es constante
        corr_sent, _ = pearsonr(exposures_arr + np.random.normal(0, 1e-6, len(exposures_arr)), sentiments_arr)
        corr_vol, _ = pearsonr(exposures_arr + np.random.normal(0, 1e-6, len(exposures_arr)), vols_arr)
        
        return {
            "Avg_Exposure": avg_exposure,
            "Turnover": turnover,
            "Corr_Sentiment": corr_sent,
            "Corr_Volatility": corr_vol
        }

    def run_monte_carlo(self):
        """Dimensión 3: Evaluación de Robustez sobre N Mercados Ocultos"""
        np.random.seed(42)
        
        test_data = WyckoffMockData(steps=self.steps)
        eval_env = DreamEnv(test_data)
        
        self.agent_paths = []
        self.bh_paths = []
        self.all_financials = []
        self.all_behaviors = []
        
        print(f"Iniciando simulación de Monte Carlo ({self.num_runs} ejecuciones)...")
        for run in range(self.num_runs):
            obs, _ = eval_env.reset()
            
            portfolio_values = [eval_env.initial_balance]
            prices = [test_data.prices[0]]
            
            exposures = [0.0]
            sentiments = [obs[0]]
            volatilities = [obs[1]]
            
            for _ in range(self.steps - 1):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                # Extracción de telemetría en cada step
                current_price, current_sent, current_vol = test_data.get_step_data(eval_env.current_step)
                current_exposure = (eval_env.position_size * current_price) / info['portfolio_value'] if info['portfolio_value'] > 0 else 0.0
                
                prices.append(current_price)
                portfolio_values.append(info['portfolio_value'])
                exposures.append(current_exposure)
                sentiments.append(current_sent)
                volatilities.append(current_vol)
                
                if terminated or truncated:
                    break
                    
            # Análisis de la trayectoria (Run)
            fin_metrics = self._calculate_financials(portfolio_values, prices)
            beh_metrics = self._calculate_behavior(exposures, sentiments, volatilities)
            
            self.all_financials.append(fin_metrics)
            self.all_behaviors.append(beh_metrics)
            
            # Normalización para gráficos
            norm_agent = np.array(portfolio_values) / eval_env.initial_balance
            norm_bh = (1.0 - self.commission_rate) * (np.array(prices) / prices[0])
            self.agent_paths.append(norm_agent)
            self.bh_paths.append(norm_bh)
            
        self.df_fin = pd.DataFrame(self.all_financials)
        self.df_beh = pd.DataFrame(self.all_behaviors)
        return self.df_fin, self.df_beh

    def plot_dashboard(self, output_img="test_modelos/dream_eval_dashboard.png"):
        """Renderiza el panel de control integral con métricas financieras y conductuales."""
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        time_steps = range(np.array(self.agent_paths).shape[1])
        
        # 1. Caminos de Inversión
        mean_ag = np.mean(self.agent_paths, axis=0)
        std_ag = np.std(self.agent_paths, axis=0)
        mean_bh = np.mean(self.bh_paths, axis=0)
        std_bh = np.std(self.bh_paths, axis=0)
        
        axes[0, 0].plot(time_steps, mean_ag, color='green', lw=2, label='D.R.E.A.M.')
        axes[0, 0].fill_between(time_steps, mean_ag - std_ag, mean_ag + std_ag, color='green', alpha=0.15)
        axes[0, 0].plot(time_steps, mean_bh, color='darkorange', ls='--', lw=2, label='Buy & Hold')
        axes[0, 0].fill_between(time_steps, mean_bh - std_bh, mean_bh + std_bh, color='darkorange', alpha=0.1)
        axes[0, 0].set_title('Evolución de Capital (Monte Carlo)', fontweight='bold')
        axes[0, 0].legend()

        # 2. Distribución Sharpe
        sns.histplot(self.df_fin['Sharpe'], ax=axes[0, 1], kde=True, color='blue')
        axes[0, 1].axvline(1.0, color='red', ls=':', label="Objetivo > 1.0")
        axes[0, 1].set_title('Distribución Ratio de Sharpe', fontweight='bold')
        axes[0, 1].legend()

        # 3. Dispersión MDD
        sns.boxplot(x=self.df_fin['MDD'], ax=axes[1, 0], color='salmon')
        axes[1, 0].axvline(20.0, color='red', ls=':', label="Techo < 20%")
        axes[1, 0].set_title('Dispersión Máximo Drawdown (%)', fontweight='bold')
        axes[1, 0].legend()

        # 4. Comportamiento: Exposición y Correlación
        sns.scatterplot(x=self.df_beh['Avg_Exposure'], y=self.df_fin['Sharpe'], ax=axes[1, 1], color='purple')
        axes[1, 1].set_title('Comportamiento: Exposición Media vs Sharpe', fontweight='bold')
        axes[1, 1].set_xlabel('Exposición Media (%)')
        
        # 5. Caja de Métricas Financieras
        axes[2, 0].axis('off')
        txt_fin = (
            r"$\bf{MÉTRICAS\ FINANCIERAS\ (Dimensión\ 1\ &\ 3)}$" + "\n"
            f"Sharpe Medio:    {self.df_fin['Sharpe'].mean():.2f} ± {self.df_fin['Sharpe'].std():.2f}\n"
            f"Sortino Medio:   {self.df_fin['Sortino'].mean():.2f} ± {self.df_fin['Sortino'].std():.2f}\n"
            f"Calmar Ratio:    {self.df_fin['Calmar'].mean():.2f} ± {self.df_fin['Calmar'].std():.2f}\n"
            f"Máximo Drawdown: {self.df_fin['MDD'].mean():.2f}% ± {self.df_fin['MDD'].std():.2f}%\n"
            f"Win Rate (B&H):  {self.df_fin['WinRate'].mean():.2f}%\n"
            f"Retorno Agente:  {self.df_fin['Total_Ret_Agent'].mean():.2f}%\n"
            f"Retorno B&H:     {self.df_fin['Total_Ret_BH'].mean():.2f}%"
        )
        axes[2, 0].text(0.1, 0.9, txt_fin, fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white'))

        # 6. Caja de Métricas de Comportamiento
        axes[2, 1].axis('off')
        txt_beh = (
            r"$\bf{MÉTRICAS\ DE\ COMPORTAMIENTO\ (Dimensión\ 2)}$" + "\n"
            f"Exposición Media:      {self.df_beh['Avg_Exposure'].mean():.2f}%\n"
            f"Turnover (Rotación):   {self.df_beh['Turnover'].mean():.4f} por step\n\n"
            r"$\bf{Correlación\ de\ la\ Política\ (Fusión\ de\ Datos)}$" + "\n"
            f"Corr vs Sentimiento:   {self.df_beh['Corr_Sentiment'].mean():.2f} (Esperado: > 0)\n"
            f"Corr vs Volatilidad:   {self.df_beh['Corr_Volatility'].mean():.2f} (Esperado: < 0)"
        )
        axes[2, 1].text(0.1, 0.9, txt_beh, fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white'))

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_img), exist_ok=True)
        plt.savefig(output_img, dpi=300)
        plt.close(fig)
        print(f"Evaluación completada. Dashboard guardado en: {output_img}")


if __name__ == "__main__":
    # # Configuración de rutas
    # model_path = "models/dream_agent.zip"
    # output_img = "test_modelos/backtest_monte_carlo_tfm.png"
    
    # # Asegurar que el directorio de salida existe
    # os.makedirs(os.path.dirname(output_img), exist_ok=True)
    
    # # Ejecución del motor estadístico
    # agent_paths, bh_paths, df_metrics = run_monte_carlo_evaluation(model_path, num_runs=100, steps=252*5)
    
    # # Generación de gráficos finales
    # generate_advanced_dashboard(agent_paths, bh_paths, df_metrics, output_img)
    
    evaluator = DreamEvaluator(model_path="models/dream_agent.zip", num_runs=30, steps=600)
    evaluator.run_monte_carlo()
    evaluator.plot_dashboard()