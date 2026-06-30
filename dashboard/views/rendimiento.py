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
    st.header("Benchmarking NLP: Análisis de Sentimiento Financiero")
    st.markdown("""
    El objetivo central de la **Fase 1** es determinar qué estrategia de adaptación sobre el modelo `ProsusAI/finbert` es la más idónea para el ecosistema D.R.E.A.M. La prioridad no es solo la precisión global, sino la **detección fiable de la clase negativa** (riesgo de mercado) y la viabilidad técnica para un entorno en producción.
    """)
    
    # 1. MÉTRICAS GLOBALES
    st.subheader("1. Rendimiento Global y Selección de Modelo")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Mejor Modelo (F1 Negativo)", "rsLoRA", "0.84", delta_color="normal")
    col_met2.metric("Mejor Modelo (Accuracy)", "Full Fine-Tuning", "85.3%", delta_color="normal")
    col_met3.metric("Mejor Eficiencia (Tamaño)", "1.8 MB (rsLoRA)", delta="-416 MB", delta_color="inverse")
    
    st.markdown("""
    El modelo base sin ajustar fracasa en este dominio (Accuracy 16.8%). Las aproximaciones de **Full Fine-Tuning** consiguen un salto de rendimiento masivo, pero **rsLoRA** se posiciona como el mejor modelo operativo. Aunque cede un 2.7% en accuracy global, consigue una mejora fundamental de 11 puntos en el F1 de la clase negativa, reduciendo drásticamente el tamaño del modelo.
    """)
    
    col_img1, col_img2 = st.columns([1, 1])
    with col_img1:
        img_f1 = os.path.join(IMG_DIR, "f1_class_comparison.png")
        if os.path.exists(img_f1):
            st.image(img_f1, caption="Comparativa de F1 por clase y tamaño de modelo en disco.")
    with col_img2:
        df_bench_nlp = pd.DataFrame({
            "Modelo": ["Base sin ajuste", "Full FT (Exp 1)", "Full FT (Exp 2)", "rsLoRA (Exp 3)"],
            "Accuracy": ["16.8%", "85.3%", "84.9%", "82.6%"],
            "F1 Negativo": [0.05, 0.73, 0.72, 0.84],
            "Parámetros": ["0%", "100%", "100%", "2.4%"],
            "Tamaño": ["-", "~418 MB", "~418 MB", "1.8 MB"]
        })
        st.dataframe(df_bench_nlp.style.highlight_max(subset=["F1 Negativo"], color="lightgreen"), use_container_width=True)

    st.divider()

    # 2. CONVERGENCIA Y MATRICES DE CONFUSIÓN
    st.subheader("2. Reequilibrio Operativo y Clasificación")
    st.markdown("""
    El *Full Fine-Tuning* presentó una tendencia a sobreajustar e infrapredictar la clase negativa, desplazando muchos falsos negativos a la clase *neutral*. Por su parte, la combinación de **rsLoRA** con rebalanceo del dataset, *label smoothing* (0.1) y pesos de clase corrigió este sesgo. Alcanzó un **recall del 89% en eventos negativos**, logrando desplazar fuertemente las predicciones a la diagonal principal.
    """)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        img_cm1 = os.path.join(IMG_DIR, "confussion_matrix_exp1.png")
        if os.path.exists(img_cm1):
            st.image(img_cm1, caption="Matriz Full FT (Exp 1): Fuerte peso y atracción hacia la clase neutral.")
            
    with col_t2:
        img_cm3 = os.path.join(IMG_DIR, "confussion_matrix_exp3.png")
        if os.path.exists(img_cm3):
            st.image(img_cm3, caption="Matriz rsLoRA (Exp 3): Desplazamiento de los falsos negativos a la diagonal.")

    st.divider()

    # 3. VALIDACIÓN EN PRODUCCIÓN
    st.subheader("3. Prueba de Estrés: Alpha Vantage API")
    st.markdown("""
    El modelo fue contrastado contra la API comercial de **Alpha Vantage** sobre datos en vivo. Aunque se obtuvo un acuerdo del **52.9%** ($r = 0.5063$), las discrepancias demuestran que Alpha Vantage clasifica lingüísticamente, etiquetando caídas explícitas (ej. *'SPY ETF Falls 1.2%'*) como neutrales. D.R.E.A.M., por el contrario, detecta el riesgo direccional de mercado.
    """)
    
    col_r1, col_r2 = st.columns([1.3, 1])
    with col_r1:
        img_api = os.path.join(IMG_DIR, "api_comparison_table.png")
        if os.path.exists(img_api):
            st.image(img_api, caption="Comparativa artículo a artículo. D.R.E.A.M. identifica mejor el riesgo bajista.")
    with col_r2:
        st.info("""
        **Divergencia Operativa:**
        - **Alpha Vantage:** Modelo de propósito general. Mide polaridad lingüística.
        - **D.R.E.A.M.:** Modelo enfocado en *Trading*. Evalúa la direccionalidad del riesgo.
        
        Esta cualidad es clave para proteger al agente DRL en etapas bajistas.
        """)

    st.success("**VEREDICTO FINAL:** El modelo **rsLoRA (Exp 3)** es seleccionado oficialmente como el modelo de producción. Su alta capacidad para detectar riesgos de mercado (Recall 89%), ausencia de sobreajuste y enorme eficiencia paramétrica (1.8 MB) lo hacen ideal para actuar como el Oráculo de Sentimiento en tiempo real.")

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

with tab_fase3:
    st.header("Rendimiento del Agente DRL y Análisis Sim-to-Real")
    st.markdown("""
    La **Fase 3** evalúa la capacidad algorítmica del agente mediante simulaciones Monte Carlo (30 episodios). Se analiza primero el rendimiento en un entorno sintético ideal (donde las señales tienen un poder predictivo garantizado) y posteriormente se cuantifica la brecha de transferencia (*sim-to-real gap*) al aplicar el modelo sobre el mercado real (2019-2020).
    """)

    # 1. VALIDACIÓN SINTÉTICA
    st.subheader("1. Validación en el Dominio Sintético (El Techo Teórico)")
    st.markdown("""
    En el entorno sintético, la arquitectura demuestra una capacidad de aprendizaje óptima. Dependiendo de la granularidad de la señal, el agente aprendió dos comportamientos exitosos: una **política de rotación** (Exp 1: maximiza retornos) y una **política de timing de mercado** (Exp 2: minimiza el riesgo entrando a liquidez).
    """)

    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Mejor Sharpe (Sintético)", "2.02", "Modo Macro (Exp 2)")
    col_met2.metric("Retorno Medio Máximo", "+112.94%", "Modo Per Asset (Exp 1)")
    col_met3.metric("Mejor MDD (Riesgo)", "10.65%", "Modo Macro (Exp 2)")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        img_exp1 = os.path.join(IMG_DIR, "exp1_capital_paths.png")
        if os.path.exists(img_exp1):
            st.image(img_exp1, caption="Exp 1: El agente supera consistentemente al benchmark B&H.")
    with col_img2:
        img_exp2 = os.path.join(IMG_DIR, "exp2_sentiment_overlay.png")
        if os.path.exists(img_exp2):
            st.image(img_exp2, caption="Exp 2: Correlación extrema (+0.712) entre exposición total y sentimiento global.")
            
    st.divider()

    # 2. TRANSFERENCIA EMPÍRICA
    st.subheader("2. Transferencia al Dominio Empírico")
    st.markdown("""
    La transferencia a datos reales de mercado constituyó la prueba de estrés definitiva. Se evaluaron tres enfoques metodológicos: **Zero-Shot** (transferencia directa sin reentrenar), **Fine-Tuning** y **Entrenamiento Directo**. 
    """)

    df_drl = pd.DataFrame({
        "Experimento": ["1 (Sint. Per-Asset)", "2 (Sint. Macro)", "3 (Emp. Zero-Shot)", "4 (Emp. Fine-Tuning)", "5 (Emp. Directo)"],
        "Sharpe": [1.88, 2.02, -0.12, 0.53, 0.53],
        "MDD (%)": [20.97, 10.65, 1.68, 11.72, 19.24],
        "Retorno (%)": [112.94, 65.85, 0.53, 1.84, 5.75],
        "Exposición (%)": [95.8, 42.6, 9.5, 72.6, 94.3],
        "Corr. Sentimiento": [0.370, 0.712, 0.144, -0.051, 0.148]
    })
    
    st.dataframe(df_drl.style.highlight_max(subset=["Sharpe", "Retorno (%)", "Corr. Sentimiento"], color="lightgreen").highlight_min(subset=["MDD (%)"], color="lightgreen"), use_container_width=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        img_exp3 = os.path.join(IMG_DIR, "exp3_capital_paths.png")
        if os.path.exists(img_exp3):
            st.image(img_exp3, caption="Exp 3 (Zero-Shot): Agente paralizado. Exposición del 9.5% por umbrales de activación.")
    with col_r2:
        img_exp4 = os.path.join(IMG_DIR, "exp4_weight_allocation.png")
        if os.path.exists(img_exp4):
            st.image(img_exp4, caption="Exp 4 (Fine-Tuning): Olvido catastrófico. Posiciones estáticas, perdiendo la política de timing.")

    st.divider()
    
    # 3. DIAGNÓSTICO DEL GAP
    st.subheader("3. Diagnóstico del Sim-to-Real Gap")
    st.markdown("""
    Se observó una caída del Ratio de Sharpe del **73.8%** (de 2.02 a 0.53) al cruzar al dominio empírico. El análisis experimental identifica las siguientes causas estructurales:
    
    * **Poder predictivo limitado de la señal FinBERT:** En el mundo real (2019-2020), la correlación entre las noticias financieras agregadas y el retorno inmediato de un criptoactivo individual es estadísticamente débil (< 0.15).
    * **Brecha distribucional de amplitud:** Las señales de FinBERT son más ruidosas y presentan menor variabilidad que las generadas matemáticamente en el entorno sintético. Esto causa que el agente *Zero-Shot* asuma que es ruido y se refugie en liquidez (90.5% del tiempo).
    * **Escasez de ventanas temporales (Overfitting Empírico):** La falta de un historial de mercado largo para todos los activos redujo el número de episodios de entrenamiento efectivos, provocando que el agente en el entrenamiento directo (Exp 5) memorizara patrones y mostrara un sesgo posicional extremo hacia el Activo 1.
    """)
    
    st.warning("**CONCLUSIÓN:** La limitación no reside en la capacidad de aprendizaje algorítmica de la arquitectura D.R.E.A.M. (sobradamente demostrada en los escenarios sintéticos), sino en el **techo informacional** de las señales de entrada empíricas.")
