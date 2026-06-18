import streamlit as st
import os

def load_sidebar_branding():
    """Carga el logo institucional en lo más alto de la barra lateral."""
    # st.logo() en Streamlit 1.36+ renderiza encima de st.navigation.
    logo_path = os.path.join(os.path.dirname(__file__), "..", "img", "logoDREAM.png")
    if hasattr(st, "logo") and os.path.exists(logo_path):
        st.logo(logo_path, icon_image=None)
    elif os.path.exists(logo_path):
        # Fallback para versiones antiguas
        st.sidebar.image(logo_path, use_container_width=True)

def apply_theme():
    """Fuerza permanentemente el Light Mode en el config.toml nativo sin opción a cambio."""
    config_dir = os.path.join(os.path.dirname(__file__), ".streamlit")
    config_path = os.path.join(config_dir, "config.toml")
    
    os.makedirs(config_dir, exist_ok=True)
    
    toml_content = """[theme]
base="light"
primaryColor="#1B9C85"
backgroundColor="#F0F2F6"
secondaryBackgroundColor="#FFFFFF"
textColor="#111111"
"""
    
    needs_update = True
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            content = f.read()
            if 'base="light"' in content:
                needs_update = False
                
    if needs_update:
        with open(config_path, "w") as f:
            f.write(toml_content)
        st.rerun() # Reiniciar la app inmediatamente para aplicar el tema nativo
