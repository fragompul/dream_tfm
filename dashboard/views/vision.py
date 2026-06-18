import streamlit as st
import os
import sys

# Asegurar rutas de imágenes correctas relativas al directorio del dashboard
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "img")
IMG_PIPELINE = os.path.join(IMG_DIR, "pipeline.png")
IMG_WYCKOFF = os.path.join(IMG_DIR, "wyckoff.png")

# ==========================================
# SECCIÓN 1: CABECERA Y CONCEPTO
# ==========================================
IMG_LOGO = os.path.join(IMG_DIR, "logoDREAM.png")

# Logo más pequeño (0.8), margen extra (0.4), texto restante (3)
col_header_logo, spacer, col_header_text = st.columns([0.8, 0.4, 3])

with col_header_logo:
    if os.path.exists(IMG_LOGO):
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.image(IMG_LOGO, use_container_width=True) # Al reducir el peso de la columna a 0.8, la imagen será más pequeña
    else:
        st.warning("Logo no encontrado")

with col_header_text:
    st.markdown("<h1 style='text-align: left; font-size: 3.5rem;'>Bienvenido a D.R.E.A.M.</h1>", unsafe_allow_html=True)
    st.markdown(
        """<p style='text-align: left; font-size: 1.2rem; color: #6c757d; margin-bottom: 2rem;'>
        <b>D.R.E.A.M.</b> (<i>Deep Reinforcement Ensamble Agent Model</i>) es una arquitectura de trading algorítmico automatizado basada en un agente autónomo. Su objetivo principal es maximizar el retorno ajustado al riesgo mediante el paradigma de <b>Fusión de Datos</b>.
        </p>""", 
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-bottom: 2.5rem;'></div>", unsafe_allow_html=True) # Margen vertical extra

with st.container():
    st.info("""
    **🎯 El Problema y Nuestra Solución:**  
    Tradicionalmente, los sistemas cuantitativos evalúan series temporales pero ignoran el contexto macroeconómico y el sentimiento del mercado. Son capaces de medir cuánto se mueve un activo, pero son ciegos al motivo que origina dicho movimiento. 
    
    D.R.E.A.M. soluciona esto combinando el *"humor"* del mercado con estimaciones precisas de volatilidad, permitiendo evitar decisiones que comprometan la cartera en periodos de inestabilidad.
    """)

st.divider()

# ==========================================
# SECCIÓN 2: LA ARQUITECTURA (3 COLUMNAS)
# ==========================================
st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Ecosistema y Fases</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='text-align: center;'>📰 Fase 1: Percepción</h3>", unsafe_allow_html=True)
    st.markdown("""
    Utiliza un modelo **FinBERT** adaptado mediante **rsLoRA** para procesar texto financiero no estructurado (noticias y redes sociales). 
    
    Extrae y cuantifica el sentimiento del mercado, generando un indicador continuo de polaridad entre **-1 y 1**.
    """)

with col2:
    st.markdown("<h3 style='text-align: center;'>📊 Fase 2: Riesgo</h3>", unsafe_allow_html=True)
    st.markdown("""
    Emplea estrategias de ensamblado dinámico (**ensamble simple**) de modelos de Machine Learning y Deep Learning. 
    
    Su función es predecir la volatilidad a un día vista ($t+1$) operando como un **oráculo de riesgo cuantitativo**.
    """)

with col3:
    st.markdown("<h3 style='text-align: center;'>🤖 Fase 3: Decisión</h3>", unsafe_allow_html=True)
    st.markdown("""
    Actúa como el cerebro integrador utilizando el algoritmo **Proximal Policy Optimization (PPO)**. 
    
    Procesa el vector de estado multimodal (sentimiento cualitativo y volatilidad cuantitativa) para emitir recomendaciones operativas (comprar, vender o mantener). Está penalizado proactivamente por caídas continuadas a través de métricas como el Máximo Drawdown.
    """)

st.divider()

# ==========================================
# SECCIÓN 3: ESQUEMA VISUAL
# ==========================================
st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Pipeline Propuesto</h2>", unsafe_allow_html=True)

if os.path.exists(IMG_PIPELINE):
    st.image(IMG_PIPELINE, use_container_width=True, caption="Arquitectura General de Fusión de Datos del Sistema D.R.E.A.M.")
else:
    st.warning(f"No se encuentra la imagen de arquitectura en: {IMG_PIPELINE}")

st.divider()

# ==========================================
# SECCIÓN 4: TEORÍA DE WYCKOFF Y CICLOS
# ==========================================
st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Dinámica de Mercado: Ciclos de Wyckoff</h2>", unsafe_allow_html=True)

col_wyckoff_img, col_wyckoff_text = st.columns([1.2, 1])

with col_wyckoff_img:
    if os.path.exists(IMG_WYCKOFF):
        st.image(IMG_WYCKOFF, use_container_width=True, caption="Fases del Ciclo de Mercado (Wyckoff)")
    else:
        st.warning(f"No se encuentra la imagen de Wyckoff en: {IMG_WYCKOFF}")

with col_wyckoff_text:
    st.markdown("El paradigma de **fusión de datos** permite al sistema decodificar las cuatro fases del ciclo de mercado de Wyckoff:")
    
    st.markdown("""
    * 🟡 **Acumulación:** Sentimiento neutral/negativo con riesgo muy bajo.
    * 🟢 **Movimiento Alcista:** Momento de euforia con riesgo tendencial ordenado.
    * 🔴 **Distribución:** Divergencia crítica donde el sentimiento es de extrema codicia pero el modelo detecta picos de volatilidad y error estocástico.
    * 📉 **Movimiento Bajista:** Capitulación semántica (pánico) y alta turbulencia, forzando al agente a refugiarse en liquidez.
    """)
