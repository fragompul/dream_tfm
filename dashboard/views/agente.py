import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agent.inference import DreamInferenceEngine
from src.agent.dream_env import DreamEnv
from src.agent.data_feed import WyckoffMockData, EmpiricalDataFeed

st.title("🤖 Fase 3: Agente Autónomo DRL")
st.markdown("Arquitectura de agente autónomo DRL para gestionar una cartera multiactivo (hasta 3 activos), basada en el algoritmo PPO.")

st.divider()

with st.expander("ℹ️ ¿Cómo funciona este módulo?", expanded=False):
    st.info("""
    * 🧠 **Arquitectura:** El motor de inferencia Fase 6 procesa un estado de mercado complejo compuesto por precios, volatilidades proyectadas y sentimiento procesado por FinBERT.
    * 🎯 **Decisión:** En lugar de reglas duras, el agente emite **logits Softmax** continuos que definen el peso objetivo exacto (% de cartera) para $N$ activos más la liquidez (Cash).
    * ⚙️ **Recompensa:** El agente ha sido entrenado usando un **Information Ratio** maximizando el retorno ajustado por riesgo contra un benchmark Equal-Weight, e incurre en penalizaciones por retener liquidez injustificada durante mercados alcistas.
    """)

tab_flash, tab_synth, tab_emp = st.tabs([
    "⚡ Consulta Flash", 
    "🧪 Simulación Sintética", 
    "📈 Simulación Empírica"
])

# ==========================================
# CONSTANTES Y RUTAS
# ==========================================
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
EMPIRICAL_MODEL = os.path.join(MODELS_DIR, "dream_empirical_macro_N3_noCashPen.zip")
SYNTH_MODEL = os.path.join(MODELS_DIR, "dream_synthetic_per_asset_N3.zip")
SYNTH_VECNORM = os.path.join(MODELS_DIR, "dream_synthetic_per_asset_N3_vecnorm.pkl")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

TICKERS_UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "URTH", "XLF", "XLV", "XLE", "XLK", "GLD", "SLV", "USO", "DBA", "TLT", "AGG", "BTC-USD", "ETH-USD", "BNB-USD"]

# ==========================================
# TAB 1: CONSULTA FLASH (1 Activo)
# ==========================================
with tab_flash:
    st.subheader("Inferencia Inmediata en Vivo")
    st.markdown("Proporciona el estado actual de tu activo. El motor multiactivo procesará la orden aislándolo y devolviéndote la asignación óptima entre ese activo y liquidez.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        flash_ticker = st.selectbox("Activo a consultar:", TICKERS_UNIVERSE, index=0)
        flash_price = st.number_input("Precio Actual ($)", value=450.0, min_value=0.1)
    with col2:
        flash_balance = st.number_input("Capital en Cuenta ($)", value=10000.0, min_value=0.0)
        flash_vol = st.number_input("Volatilidad Proyectada (Fase 2)", value=0.15, min_value=0.0, max_value=2.0)
    with col3:
        flash_units = st.number_input("Unidades en propiedad (Cantidad)", value=0.0, min_value=0.0)
        flash_sent = st.slider("Sentimiento Macro FinBERT", min_value=0.0, max_value=1.0, value=0.6)
        
    btn_flash = st.button("Consultar Agente", type="primary")
    
    if btn_flash:
        try:
            # Instanciar el motor empirical N=3
            engine = DreamInferenceEngine(
                model_path=EMPIRICAL_MODEL,
                vecnorm_path=None,
                num_assets=3,
                sentiment_mode="macro",
                sentiment_normalized=False
            )
            engine.reset()
            
            # Arrays padding: metemos los datos reales en index 0, y ruido estanco en 1 y 2
            prices_arr = np.array([flash_price, 100.0, 100.0])
            vols_arr = np.array([flash_vol, 0.15, 0.15])
            pos_arr = np.array([flash_units, 0.0, 0.0])
            
            # Ejecutar dos steps para que se compute un log_return válido. 
            # El primero siempre inicializa log_ret a 0.
            _ = engine.process_market_tick(prices_arr, vols_arr, flash_sent, pos_arr, flash_balance)
            result = engine.process_market_tick(prices_arr, vols_arr, flash_sent, pos_arr, flash_balance)
            
            # Extraer resultados solo para nuestro activo de interés (index 0)
            target_weight_asset = result["target_weights"][0]
            cash_weight = result["cash_weight"]
            
            # Recalcular allocation simplificado (sólo Activo vs Cash)
            total_sub_weight = target_weight_asset + cash_weight
            if total_sub_weight == 0: total_sub_weight = 1e-9
            
            normalized_asset_w = target_weight_asset / total_sub_weight
            normalized_cash_w = cash_weight / total_sub_weight
            
            current_value_asset = flash_price * flash_units
            total_portfolio_value = current_value_asset + flash_balance
            current_weight_asset = current_value_asset / total_portfolio_value if total_portfolio_value > 0 else 0
            
            st.divider()
            st.subheader(f"Decisión de la Red Neuronal para {flash_ticker}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Peso Actual del Activo", f"{current_weight_asset*100:.1f}%")
            c2.metric("Peso Objetivo (Red Neuronal)", f"{normalized_asset_w*100:.1f}%", f"{(normalized_asset_w - current_weight_asset)*100:+.1f}%")
            c3.metric("Liquidez Recomendada", f"{normalized_cash_w*100:.1f}%")
            
            # Gráfica de Donut
            fig_donut = px.pie(
                names=[flash_ticker, "Liquidez"],
                values=[normalized_asset_w, normalized_cash_w],
                hole=0.6,
                color_discrete_sequence=["#1B9C85", "#262730"]
            )
            fig_donut.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            st.plotly_chart(fig_donut, theme="streamlit", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error cargando el modelo: {e}")

# ==========================================
# FUNCIONES AUXILIARES DE SIMULACIÓN
# ==========================================
def run_simulation(data_generator, engine, horizon, initial_balance, ticker_names):
    env = DreamEnv(data_generator=data_generator, initial_balance=initial_balance)
    obs, _ = env.reset()
    engine.reset()
    
    history_dream = [initial_balance]
    history_bnh = [initial_balance]
    weights_history = []
    days = [0]
    
    bnh_val = initial_balance
    
    progress_bar = st.progress(0, text="Iterando a lo largo del tiempo...")
    chart_placeholder = st.empty()
    weights_placeholder = st.empty()
    kpi_placeholder = st.empty()
    
    for step in range(1, horizon):
        prices, _, sentiment, volatility = data_generator.get_step_data(env.current_step)
        
        # Apply VecNormalize if the engine loaded it (critical for synthetic models)
        norm_obs = engine._vecnorm.normalize_obs(obs.reshape(1, -1)).flatten() if engine._vecnorm else obs
        action, _ = engine.model.predict(norm_obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Guardar equity del Agente
        dream_val = info['portfolio_value']
        history_dream.append(dream_val)
        
        # Calcular equity B&H (Equal-Weight de los activos en N=3)
        next_prices, _, _, _ = data_generator.get_step_data(env.current_step if env.current_step < data_generator.steps else data_generator.steps - 1)
        bnh_returns = (next_prices - prices) / prices
        bnh_return_mean = np.mean(bnh_returns)
        bnh_val = bnh_val * (1 + bnh_return_mean)
        history_bnh.append(bnh_val)
        
        days.append(step)
        
        # Guardar pesos reales del entorno
        if dream_val > 0:
            current_weights = (env.position_sizes * prices) / dream_val
            cash_w = env.balance / dream_val
        else:
            current_weights = np.zeros(engine.num_assets)
            cash_w = 1.0
            
        weights = list(current_weights) + [cash_w]
        weights_history.append(weights)
        
        if step % 10 == 0 or step == horizon - 1:
            df_sim = pd.DataFrame({
                "Día": days,
                "D.R.E.A.M.": history_dream,
                "Buy & Hold (Eq. Weight)": history_bnh
            })
            
            fig = px.line(df_sim, x="Día", y=["D.R.E.A.M.", "Buy & Hold (Eq. Weight)"], 
                          color_discrete_map={"D.R.E.A.M.": "#1B9C85", "Buy & Hold (Eq. Weight)": "#FF4B4B"})
            fig.update_layout(title="Curva de Equidad: Agente PPO vs Mercado", hovermode="x unified", height=350, margin=dict(b=0))
            fig.update_xaxes(range=[0, horizon])
            chart_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True)
            
            # Graficar áreas apiladas para los pesos
            if len(weights_history) > 0:
                cols = ticker_names + ["Liquidez"]
                df_weights = pd.DataFrame(weights_history, columns=cols)
                df_weights["Día"] = days[1:] # offset
                
                fig_w = px.area(df_weights, x="Día", y=cols, title="Evolución de la Asignación de Cartera", color_discrete_sequence=px.colors.qualitative.Vivid)
                fig_w.update_traces(stackgroup=None, fill='tozeroy', opacity=0.35, line=dict(width=2))
                fig_w.update_layout(height=250, margin=dict(t=30, b=0), hovermode="x unified")
                fig_w.update_xaxes(range=[0, horizon])
                fig_w.update_yaxes(range=[0, 1.05])
                weights_placeholder.plotly_chart(fig_w, theme="streamlit", use_container_width=True)
            
            with kpi_placeholder.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Equity Agente", f"${dream_val:,.0f}", f"{(dream_val/initial_balance - 1)*100:.1f}%")
                k2.metric("Equity B&H", f"${bnh_val:,.0f}", f"{(bnh_val/initial_balance - 1)*100:.1f}%", delta_color="off")
                k3.metric("Max Drawdown", f"{info['mdd']*100:.2f}%")
                # Aproximacion simple de Sharpe 
                returns_series = pd.Series(history_dream).pct_change().dropna()
                sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() > 0 else 0
                k4.metric("Sharpe Ratio Anual", f"{sharpe:.2f}")
                
            progress_bar.progress(step / horizon, text=f"Día {step}/{horizon}")
            
        time.sleep(0.005) # Animación rápida
        
        if terminated or truncated:
            break
            
    progress_bar.empty()
    st.toast('✅ Simulación Completada!')


# ==========================================
# TAB 2: SIMULACIÓN SINTÉTICA (N=3)
# ==========================================
with tab_synth:
    st.subheader("Backtest sobre Generador Wyckoff")
    st.markdown("Pone a prueba la capacidad del agente multiactivo para aprender las 4 fases del mercado sintético y asignar pesos dinámicamente.")
    
    sc1, sc2, sc3 = st.columns(3)
    synth_horizon = sc1.number_input("Horizonte Temporal (Días)", value=365, min_value=50, max_value=2000, key="sh")
    synth_capital = sc2.number_input("Capital Inicial ($)", value=100_000.0, key="sc")
    synth_tickers = sc3.multiselect("Activos Seleccionados (máx 3):", TICKERS_UNIVERSE, default=["SPY", "QQQ", "GLD"], max_selections=3)
    
    if st.button("Lanzar Simulación Sintética", type="primary", key="btn_synth"):
        if len(synth_tickers) != 3:
            st.error("⚠️ Debes seleccionar exactamente 3 activos para el modelo.")
        else:
            engine_synth = DreamInferenceEngine(
                model_path=SYNTH_MODEL,
                vecnorm_path=SYNTH_VECNORM,
                num_assets=3,
                sentiment_mode="per_asset",
                sentiment_normalized=True # Wyckoff ya devuelve z-scores
            )
            data_synth = WyckoffMockData(steps=synth_horizon, num_assets=3, sentiment_mode="per_asset")
            data_synth.generate_new_market()
            
            run_simulation(data_synth, engine_synth, synth_horizon, synth_capital, synth_tickers)


# ==========================================
# TAB 3: SIMULACIÓN EMPÍRICA (N=3)
# ==========================================
with tab_emp:
    st.subheader("Backtest con Datos Reales")
    st.markdown("Valida el agente en condiciones reales de mercado usando el dataset preprocesado.")
    
    ec1, ec2, ec3 = st.columns(3)
    emp_horizon = ec1.number_input("Horizonte Temporal (Días)", value=365, min_value=50, max_value=2000, key="eh")
    emp_capital = ec2.number_input("Capital Inicial ($)", value=100_000.0, key="ec")
    emp_tickers = ec3.multiselect("Activos seleccionados (máx 3):", TICKERS_UNIVERSE, default=["SPY", "TLT", "BTC-USD"], max_selections=3)
    
    if st.button("Lanzar Simulación Empírica", type="primary", key="btn_emp"):
        if len(emp_tickers) != 3:
            st.error("⚠️ Debes seleccionar exactamente 3 activos para el modelo.")
        else:
            csv_path = os.path.join(DATA_DIR, "train_agent_dataset.csv")
            if not os.path.exists(csv_path):
                st.error(f"No se encuentra el dataset en: {csv_path}. Verifica que la carpeta 'data/' contenga el CSV.")
            else:
                engine_emp = DreamInferenceEngine(
                    model_path=EMPIRICAL_MODEL,
                    vecnorm_path=None,
                    num_assets=3,
                    sentiment_mode="macro",
                    sentiment_normalized=False
                )
                data_emp = EmpiricalDataFeed(
                    csv_path=csv_path,
                    steps=emp_horizon,
                    tickers=emp_tickers,
                    randomize=False
                )
                # Obtenemos los steps reales disponibles en el inner join de fechas
                real_horizon = min(500, data_emp.steps) 
                
                run_simulation(data_emp, engine_emp, real_horizon, emp_capital, emp_tickers)
