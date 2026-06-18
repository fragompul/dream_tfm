import streamlit as st
import pandas as pd
import os
import sys

st.title("📈 Resultados y Conclusiones del Estudio")
st.markdown("Síntesis del benchmarking predictivo y evaluación del ecosistema D.R.E.A.M.")

st.divider()

# Definir ruta base de las imágenes
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "img")

tab_fase1, tab_fase2, tab_fase3 = st.tabs([
    "🧠 Fase 1: Sentimiento", 
    "📊 Fase 2: Volatilidad", 
    "🤖 Fase 3: Agente DRL"
])

with tab_fase1:
    st.info("Resultados de la Fase 1 en desarrollo. Próximamente se incluirá el rendimiento de FinBERT y rsLoRA.")

with tab_fase3:
    st.info("Resultados de la Fase 3 en desarrollo. Próximamente se incluirán las curvas de equity y ratios de Sharpe.")

with tab_fase2:
    st.header("Benchmarking Predictivo: Oráculo de Volatilidad")
    st.markdown("""
    El objetivo central de la **Fase 2** es determinar qué algoritmo (o combinación) posee la mayor capacidad de generalización sobre datos no vistos para estimar el riesgo a un día vista ($t+1$). El ganador será el oráculo de riesgo del agente DRL.
    """)
    
    # 1. MÉTRICAS GLOBALES
    st.subheader("1. Rendimiento Global y Selección de Modelo")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Mejor Modelo (RMSE)", "Ensamble Simple", "0.0212", delta_color="inverse")
    col_met2.metric("Mejor Modelo (MAE)", "Ensamble Ponderado", "0.0097", delta_color="inverse")
    col_met3.metric("Mejor Modelo (R²)", "Ensamble Simple", "0.9849", delta_color="normal")
    
    st.markdown("""
    La evaluación demostró que la **aproximación por ensamble es superior**, logrando batir a todos los estimadores base. Curiosamente, la regresión **Ridge** clásica demostró ser extraordinariamente competitiva, superando a arquitecturas complejas de *Deep Learning* (LSTM, GRU, TFT) que mostraron propensión a sobreajustarse frente al ruido estocástico del corto plazo.
    """)
    
    col_img1, col_img2 = st.columns([1, 1])
    with col_img1:
        img_rmse = os.path.join(IMG_DIR, "fig_rmse_barplot.png")
        if os.path.exists(img_rmse):
            st.image(img_rmse, caption="Benchmarking global de RMSE por modelo.")
    with col_img2:
        # Reconstruimos la tabla para darle un toque premium
        df_bench = pd.DataFrame({
            "Modelo": ["Simple_All", "Weighted_All", "Regresión Ridge", "Top_2", "Random Forest", "XGBoost", "LightGBM", "TFT", "LSTM", "GRU"],
            "Familia": ["Ensamble", "Ensamble", "ML Lineal", "Ensamble", "ML (Bagging)", "ML (Boosting)", "ML (Boosting)", "DL (Atencional)", "DL (Secuencial)", "DL (Secuencial)"],
            "RMSE": [0.0212, 0.0213, 0.0226, 0.0226, 0.0228, 0.0233, 0.0236, 0.0294, 0.0297, 0.0314],
            "R²": [0.9849, 0.9848, 0.9829, 0.9828, 0.9825, 0.9817, 0.9812, 0.9737, 0.9703, 0.9668]
        })
        st.dataframe(df_bench.style.highlight_min(subset=["RMSE"], color="lightgreen").highlight_max(subset=["R²"], color="lightgreen"), use_container_width=True)

    st.divider()

    # 2. ANÁLISIS POR ACTIVO Y TEMPORALIDAD
    st.subheader("2. Robustez Transversal: Activos y Ciclos Temporales")
    st.markdown("""
    El error absoluto está dictado en gran medida por la naturaleza del activo. Como era esperable, el modelo sufre una **mayor degradación frente a los activos de alta volatilidad** (como Ethereum o Bitcoin), y logra métricas excepcionales en instrumentos refugio (como bonos TLT o AGG). Sin embargo, el Ensamble se posiciona como el mejor de forma transversal.
    """)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        img_heat = os.path.join(IMG_DIR, "fig_heatmap_assets.png")
        if os.path.exists(img_heat):
            st.image(img_heat, caption="Mapa de calor del RMSE por activo. Patrón de bandas horizontales.")
            
    with col_t2:
        img_tracking = os.path.join(IMG_DIR, "fig_volatility_tracking.png")
        if os.path.exists(img_tracking):
            st.image(img_tracking, caption="Seguimiento visual a 150 días sobre ETH-USD. Tracking efectivo con ligero lag (sin lookahead bias).")

    st.divider()

    # 3. DISPERSIÓN Y RESIDUOS
    st.subheader("3. Anatomía del Error (Residuos)")
    st.markdown("""
    La combinación analítica consigue **corregir los sesgos de sobreestimación y subestimación** de los modelos individuales. El ensamble presenta una distribución leptocúrtica centrada en cero, evidenciando **ausencia de sesgos direccionales** y exhibiendo colas más delgadas. Esto es crucial en *trading*: significa menor probabilidad de estimaciones defectuosas que causen pérdidas catastróficas.
    """)
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        img_scatter = os.path.join(IMG_DIR, "fig_scatter.png")
        if os.path.exists(img_scatter):
            st.image(img_scatter, caption="Dispersión: El ensamble logra una concentración más estrecha.")
    with col_r2:
        img_kde = os.path.join(IMG_DIR, "fig_residuals_kde.png")
        if os.path.exists(img_kde):
            st.image(img_kde, caption="Distribución de Residuos: Reducción drástica de colas extremas.")

    st.divider()

    # 4. EXPLICABILIDAD XAI
    st.subheader("4. Inteligencia Artificial Explicable (XAI)")
    st.markdown("""
    Para cumplir con las normativas (AI Act) y asegurar que el oráculo no es una "caja negra", se aplicó **SHAP** y **Feature Importance**. 
    El modelo revela un dominio absoluto de la **volatilidad histórica a 20 días** (0.999), confirmando económicamente el *fenómeno de agrupamiento de volatilidad de Bollerslev (1986)*.
    """)
    
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        img_feat = os.path.join(IMG_DIR, "fig_feature_importance.png")
        if os.path.exists(img_feat):
            st.image(img_feat, caption="Dominio predictivo absoluto de la volatilidad pasada.")
    with col_x2:
        img_shap = os.path.join(IMG_DIR, "fig_shap_summary.png")
        if os.path.exists(img_shap):
            st.image(img_shap, caption="Valores SHAP: Relación fuertemente asimétrica ante shocks.")

    st.success("**VEREDICTO FINAL:** El **Ensamble Simple** es seleccionado oficialmente como el modelo ganador. Su arquitectura se congela para operar como el Oráculo Predictivo de la Fase 3, alimentando diariamente al Agente DRL.")
