import streamlit as st
from theme_utils import apply_theme, load_sidebar_branding

# Mover la configuración a la parte más alta de app.py
st.set_page_config(page_title="D.R.E.A.M. Terminal", page_icon="📈", layout="wide")

# 1. Limpieza y Estructura Estricta de la Barra Lateral
load_sidebar_branding()

# Definición de las páginas usando la nueva API st.navigation y st.Page
pg_vision = st.Page("views/vision.py", title="Visión General", icon="🏠")
pg_sentimiento = st.Page("views/sentimiento.py", title="Fase 1: Análisis de Sentimiento", icon="📰")
pg_volatilidad = st.Page("views/volatilidad.py", title="Fase 2: Predicción de Volatilidad", icon="📊")
pg_agente = st.Page("views/agente.py", title="Fase 3: Agente Autónomo DRL", icon="🤖")
pg_rendimiento = st.Page("views/rendimiento.py", title="Análisis y Evaluación", icon="📈")

pg = st.navigation([pg_vision, pg_sentimiento, pg_volatilidad, pg_agente, pg_rendimiento])

apply_theme()

# Ejecutar la página seleccionada
pg.run()
