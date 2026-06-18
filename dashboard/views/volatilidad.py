import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Asegurar que el path alcance 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.inference_volatility import predict_volatility, fetch_and_prepare_data

# Diccionario maestro de los 20 activos y sus clases
ASSET_UNIVERSE = {
    "Equities_US_Broad": ["SPY", "QQQ", "DIA", "IWM"],
    "Equities_Global": ["EFA", "EEM", "URTH"],
    "Sectors_US": ["XLF", "XLV", "XLE", "XLK"],
    "Commodities": ["GLD", "SLV", "USO", "DBA"],
    "Fixed_Income": ["TLT", "AGG"],
    "Cryptocurrency": ["BTC-USD", "ETH-USD", "BNB-USD"]
}

# Mapeo inverso: Ticker -> Clase de Activo
TICKER_TO_CLASS = {ticker: category for category, tickers in ASSET_UNIVERSE.items() for ticker in tickers}
ALL_TICKERS = sorted(list(TICKER_TO_CLASS.keys()))

st.title("📊 Fase 2: Predicción de Volatilidad")
st.markdown("Pronóstico de riesgo a un día vista para protección de cartera e identificación de fases bajistas.")

st.divider()

with st.expander("ℹ️ ¿Cómo funciona este módulo?", expanded=False):
    st.info("""
    * 🎯 **Objetivo:** Funciona como el oráculo de riesgo cuantitativo del sistema, centrado en medir la turbulencia futura y no la dirección del precio.
    * ⚙️ **Tecnología:** Procesa ventanas móviles con las características numéricas del mercado (precio, volumen y retornos logarítmicos). El sistema utiliza un Ensamble Simple tras haber realizado un exhaustivo benchmarking contra modelos clásicos (Ridge, Random Forest, XGBoost,...) y arquitecturas profundas (LSTM, GRU, TFT). El ensamble demostró ser la mejor técnica para aislar la señal frente al ruido estocástico a corto plazo.
    * 📈 **Salida:** Devuelve una estimación unificada de la volatilidad esperada para el instante $t+1$, proporcionando una métrica prospectiva del riesgo inminente.
    """)

tab_single, tab_compare = st.tabs(["📌 Análisis Individual", "📈 Comparativa de Activos"])

# ==========================================
# PESTAÑA 1: ANÁLISIS INDIVIDUAL
# ==========================================
with tab_single:
    st.subheader("Predicción Individual de la Volatilidad")
    
    # Inputs: Fila con selectbox y botón
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_ticker = st.selectbox("Selecciona un activo para hacer la predicción en $t+1$:", ALL_TICKERS, index=ALL_TICKERS.index("SPY") if "SPY" in ALL_TICKERS else 0)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_predict_single = st.button("Ejecutar Inferencia", type="primary", use_container_width=True)
        
    if btn_predict_single:
        auto_asset_class = TICKER_TO_CLASS[selected_ticker]
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        
        with st.spinner(f"Evaluando {selected_ticker}..."):
            try:
                result_vol = predict_volatility(selected_ticker, auto_asset_class, models_dir=models_dir)
                df_hist = fetch_and_prepare_data(selected_ticker, auto_asset_class)
                
                pred_t1 = result_vol["predicted_volatility_t1"]
                current_vol = df_hist.iloc[-1]["historical_volatility_20d"]
                delta_vol = pred_t1 - current_vol
                current_price = df_hist.iloc[-1]["Close"]
                
                # KPIs (Semáforo de Riesgo)
                st.divider()
                st.markdown("### Semáforo de Riesgo")
                kpi1, kpi2, kpi3 = st.columns(3)
                
                with kpi1:
                    st.metric(label="Riesgo Actual (Vol 20d)", value=f"{current_vol:.4f}")
                with kpi2:
                    st.metric(label="Riesgo Predicho (t+1)", value=f"{pred_t1:.4f}")
                with kpi3:
                    st.metric(label="Delta (Variación)", 
                              value=f"{delta_vol:+.4f}", 
                              delta=f"{delta_vol:+.4f}", 
                              delta_color="inverse")
                              
                # Gráficos (Termómetro y Evolución)
                col_gauge, col_chart = st.columns([1, 2])
                
                with col_gauge:
                    # Termómetro de Turbulencia (Ajustado)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_t1,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Termómetro de Turbulencia", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, max(0.4, pred_t1 * 1.5)], 'tickwidth': 1},
                            'bar': {'color': "rgba(0,0,0,0.5)"},
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 0.15], 'color': "#1B9C85"},       # Bajo Riesgo (Verde)
                                {'range': [0.15, 0.25], 'color': "#F1D00A"},    # Normal (Amarillo)
                                {'range': [0.25, max(1.0, pred_t1 * 1.5)], 'color': "#FF4B4B"} # Alto Riesgo (Rojo)
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': pred_t1
                            }
                        }
                    ))
                    fig_gauge.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=300)
                    st.plotly_chart(fig_gauge, theme="streamlit", use_container_width=True)
                    
                with col_chart:
                    # Gráfico Temporal (Precio vs Volatilidad) de doble eje Y
                    df_plot = df_hist.reset_index()
                    if 'Date' not in df_plot.columns:
                        df_plot = df_plot.rename(columns={'index': 'Date'})
                        
                    fig_temporal = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Eje principal: Precio
                    fig_temporal.add_trace(
                        go.Scatter(x=df_plot["Date"], y=df_plot["Close"], name="Precio de Cierre", line=dict(color="blue")),
                        secondary_y=False,
                    )
                    
                    # Eje secundario: Volatilidad Histórica
                    fig_temporal.add_trace(
                        go.Scatter(x=df_plot["Date"], y=df_plot["historical_volatility_20d"], name="Volatilidad (20d)", line=dict(color="orange", dash="dot")),
                        secondary_y=True,
                    )
                    
                    # Marcador final (Predicción t+1) unido con línea
                    last_date = df_plot.iloc[-1]["Date"]
                    pred_date = last_date + pd.Timedelta(days=1)
                    
                    # Línea conectora
                    fig_temporal.add_trace(
                        go.Scatter(
                            x=[last_date, pred_date], 
                            y=[current_vol, pred_t1], 
                            mode="lines+markers",
                            name="Predicción t+1",
                            text=["", "t+1"],
                            textposition="top center",
                            line=dict(color="red" if delta_vol > 0 else "green", width=2, dash="dash"),
                            marker=dict(size=[0, 12], symbol="star", color="red" if delta_vol > 0 else "green")
                        ),
                        secondary_y=True,
                    )
                    
                    fig_temporal.update_layout(
                        title=f"Evolución: Precio vs Riesgo ({selected_ticker})",
                        hovermode="x unified",
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    fig_temporal.update_yaxes(title_text="Precio ($)", secondary_y=False)
                    fig_temporal.update_yaxes(title_text="Volatilidad", secondary_y=True)
                    
                    st.plotly_chart(fig_temporal, theme="streamlit", use_container_width=True)
                    
                st.divider()
                with st.expander("📖 Guía de Interpretación de Resultados", expanded=False):
                    st.info("""
                    * 🚦 **Semáforo de Riesgo:** Mide la salud a corto plazo del activo. Un *Delta* rojo indica una aceleración inminente del riesgo, lo que normalmente precede a fases de distribución o caídas pronunciadas.
                    * 🌡️ **Termómetro de Turbulencia:** 
                        * **0.0 - 0.15 (Verde):** Condiciones óptimas. Riesgo ordenado, típico de mercados alcistas sanos (fase de mark-up).
                        * **0.15 - 0.25 (Amarillo):** Volatilidad latente. Precaución con posiciones sobre-apalancadas.
                        * **> 0.25 (Rojo):** Turbulencia severa. Mercados altamente inestables y erráticos. Sugiere retirada a liquidez.
                    * 📈 **Evolución: Precio vs Riesgo:** Un comportamiento sano muestra el precio subiendo suavemente con la volatilidad bajando o lateral. Si la curva punteada (riesgo) se dispara hacia arriba acompañada de una estrella verde oscura o roja intensa, indica miedo en el mercado.
                    """)
                    
            except Exception as e:
                st.error(f"Error durante la inferencia: {str(e)}")


# ==========================================
# PESTAÑA 2: RADAR MULTIACTIVO
# ==========================================
with tab_compare:
    st.subheader("Predicción Multiactivo de la Volatilidad")
    
    # Inputs: Multiselect y botón para todos
    if "selected_compare_tickers" not in st.session_state:
        st.session_state["selected_compare_tickers"] = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]
        
    col_sel_multi, col_btn_all, col_btn_run = st.columns([5, 1, 1])

    with col_sel_multi:
        compare_tickers = st.multiselect(
            "Selecciona todos los activos a escanear:", 
            ALL_TICKERS, 
            default=st.session_state["selected_compare_tickers"]
        )
        
    with col_btn_all:
        st.markdown("<br>", unsafe_allow_html=True) # Alineación vertical con el input
        if st.button("Seleccionar Todos"):
            st.session_state["selected_compare_tickers"] = ALL_TICKERS
            st.rerun()
            
    with col_btn_run:
        st.markdown("<br>", unsafe_allow_html=True) # Alineación vertical con el input
        # Movemos la definición de btn_compare dentro de esta nueva columna
        btn_compare = st.button("Ejecutar Inferencia", type="primary")
    
    if btn_compare and compare_tickers:
        st.session_state["selected_compare_tickers"] = compare_tickers # Actualizar state
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        fig_comp = go.Figure() # Figura para las curvas de todos los activos
        
        for i, tick in enumerate(compare_tickers):
            status_text.text(f"Evaluando {tick} ({i+1}/{len(compare_tickers)})...")
            try:
                a_class = TICKER_TO_CLASS[tick]
                res_v = predict_volatility(tick, a_class, models_dir=models_dir)
                df_h = fetch_and_prepare_data(tick, a_class)
                
                pred_t1 = res_v["predicted_volatility_t1"]
                curr_vol = df_h.iloc[-1]["historical_volatility_20d"]
                curr_price = df_h.iloc[-1]["Close"]
                
                results.append({
                    "Ticker": tick,
                    "Precio Actual": curr_price,
                    "Vol. Actual": curr_vol,
                    "Vol. Predicha": pred_t1
                })
                
                # Curva temporal multi-activo
                df_plot = df_h.reset_index()
                if 'Date' not in df_plot.columns:
                    df_plot = df_plot.rename(columns={'index': 'Date'})
                    
                fig_comp.add_trace(go.Scatter(
                    x=df_plot["Date"], 
                    y=df_plot["historical_volatility_20d"], 
                    mode='lines',
                    name=f'{tick} (Hist)'
                ))
                
                last_date = df_plot.iloc[-1]["Date"]
                fig_comp.add_trace(go.Scatter(
                    x=[last_date + pd.Timedelta(days=1)], 
                    y=[pred_t1], 
                    mode='markers', 
                    marker=dict(size=8, symbol='star'),
                    name=f'{tick} (Pred t+1)'
                ))
                
            except Exception as e:
                st.toast(f"Error procesando {tick}: {e}")
                
            progress_bar.progress((i + 1) / len(compare_tickers))
            
        status_text.empty()
        progress_bar.empty()
        
        if len(results) > 0:
            df_results = pd.DataFrame(results)
            df_results["Delta"] = df_results["Vol. Predicha"] - df_results["Vol. Actual"]
            
            # 0. Gráfico Comparativo Principal
            st.divider()
            st.markdown("### Superposición de Volatilidad Histórica y Riesgo Proyectado")
            fig_comp.update_layout(
                yaxis_title="Volatilidad Anualizada", 
                xaxis_title="Fecha",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig_comp, theme="streamlit", use_container_width=True)
            
            # 1. Ranking de Turbulencia (Gráfico de Barras Horizontales)
            st.divider()
            st.markdown("### Ranking de Turbulencia Proyectada")
            df_sorted = df_results.sort_values(by="Vol. Predicha", ascending=True)
            
            fig_bar = px.bar(
                df_sorted, 
                x="Vol. Predicha", 
                y="Ticker", 
                orientation="h",
                color="Vol. Predicha",
                color_continuous_scale="Reds",
                text_auto='.4f',
                title="Clasificación por Riesgo Inminente (t+1)"
            )
            fig_bar.update_layout(height=100 + len(df_sorted)*40, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, theme="streamlit", use_container_width=True)
            
            # 2. Matriz de Transición de Riesgo y Tabla de Datos Crudos (Mismo ancho)
            st.divider()
            col_scatter, col_table = st.columns(2, gap="large")
            
            with col_scatter:
                st.markdown("### Matriz de Transición de Riesgo")
                
                # Asegurar un rango dinámico basado en los datos para la línea y=x
                max_val = max(df_results["Vol. Actual"].max(), df_results["Vol. Predicha"].max()) * 1.1
                min_val = min(df_results["Vol. Actual"].min(), df_results["Vol. Predicha"].min()) * 0.9
                
                fig_scatter = px.scatter(
                    df_results, 
                    x="Vol. Actual", 
                    y="Vol. Predicha", 
                    text="Ticker",
                    color="Delta",
                    color_continuous_scale=["green", "gray", "red"],
                    color_continuous_midpoint=0,
                    size=[10]*len(df_results),
                    title="Volatilidad Actual vs Predicción"
                )
                
                fig_scatter.update_traces(textposition='top center')
                
                # Línea de referencia diagonal (y=x)
                fig_scatter.add_shape(
                    type="line", 
                    x0=min_val, y0=min_val, 
                    x1=max_val, y1=max_val,
                    line=dict(color="white", width=2, dash="dash")
                )
                
                fig_scatter.update_layout(
                    xaxis_title="Volatilidad Actual (Histórica)",
                    yaxis_title="Volatilidad Predicha (t+1)",
                    height=450,
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_scatter, theme="streamlit", use_container_width=True)
                
            with col_table:
                st.markdown("### Datos Crudos")
                st.markdown("Tabla resumen de las métricas extraídas.")
                
                def color_delta_text(val):
                    color = '#FF4B4B' if val > 0 else '#1B9C85'
                    return f'color: {color}'
                    
                format_dict = {
                    "Precio Actual": "${:.2f}",
                    "Vol. Actual": "{:.4f}",
                    "Vol. Predicha": "{:.4f}",
                    "Delta": "{:+.4f}"
                }
                
                st.dataframe(
                    df_results.style.map(color_delta_text, subset=['Delta']).format(format_dict),
                    use_container_width=True,
                    height=450,
                    hide_index=True
                )
                
            st.divider()
            with st.expander("📖 Guía de Interpretación del Radar Multiactivo", expanded=False):
                st.info("""
                * 🌐 **Superposición de Volatilidad:** Permite identificar cómo los activos reaccionan a los mismos shocks de mercado. Activos con curvas correlacionadas son vulnerables a eventos sistémicos, mientras que los descorrelacionados aportan resiliencia.
                * 📊 **Ranking de Turbulencia:** Es una herramienta táctica. Permite descartar instantáneamente activos cuya volatilidad t+1 es demasiado alta, y priorizar aquellos con bajo riesgo (barras más cortas).
                * 🎯 **Matriz de Transición:** 
                    * Puntos **POR ENCIMA** de la línea diagonal implican que el ensamble prevé que **la volatilidad va a subir** (Delta Riesgo > 0, representados en tonos rojizos).
                    * Puntos **POR DEBAJO** indican que el mercado de ese activo **se está calmando** (Delta Riesgo < 0, representados en tonos verdes).
                * 📝 **Datos Crudos:** Proporciona los valores absolutos generados para análisis manuales o volcado de datos.
                """)
